"""
Order Manager for coordinating trade execution.

Manages the lifecycle of orders including:
- Order placement with retry logic
- Order tracking and status updates
- Integration with risk management

Follows @python-pro skill patterns.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from src.risk import RiskLimits, TrailingStopManager
from src.risk.trailing_stop import PositionSide

from .binance_client import BinanceClient, BinanceAPIError


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order details."""

    id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float | None
    stop_price: float | None
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    binance_order_id: int | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    """Current position."""

    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    unrealized_pnl: float = 0.0
    leverage: int = 1


class OrderManager:
    """
    Manages order placement and position tracking.

    Features:
    - Order placement with validation
    - Position tracking
    - Integration with trailing stops
    - Risk limit checks before orders

    Example:
        ```python
        manager = OrderManager(binance_client, risk_limits, stop_manager)

        # Open position
        order = await manager.open_position(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            entry_price=45000,
        )

        # Close position
        await manager.close_position("BTCUSDT")
        ```
    """

    def __init__(
        self,
        client: BinanceClient,
        risk_limits: RiskLimits,
        stop_manager: TrailingStopManager,
    ) -> None:
        """
        Initialize order manager.

        Args:
            client: Binance API client.
            risk_limits: Risk limits manager.
            stop_manager: Trailing stop manager.
        """
        self._client = client
        self._risk = risk_limits
        self._stops = stop_manager

        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._order_counter = 0

    async def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float | None = None,
        atr: float | None = None,
        use_market: bool = False,
    ) -> Order | None:
        """
        Open a new position.

        Args:
            symbol: Trading pair.
            side: "BUY" for long, "SELL" for short.
            quantity: Position size.
            entry_price: Limit price (None for market).
            atr: Current ATR for stop calculation.
            use_market: Force market order.

        Returns:
            Order if successful, None if rejected.
        """
        # Check risk limits
        position_value = quantity * (entry_price or 0)
        if not self._risk.is_position_allowed(position_value, symbol):
            logger.warning(f"Position rejected by risk limits: {symbol}")
            return None

        # Determine order type
        order_type = "MARKET" if use_market or entry_price is None else "LIMIT"

        # Create order
        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=entry_price,
            stop_price=None,
            status=OrderStatus.PENDING,
        )

        try:
            # Place order
            if order_type == "MARKET":
                response = await self._client.place_market_order(symbol, side, quantity)
            else:
                response = await self._client.place_limit_order(
                    symbol, side, quantity, entry_price, post_only=True
                )

            order.binance_order_id = response.get("orderId")
            order.status = OrderStatus.SUBMITTED

            # Update for filled market orders
            if response.get("status") == "FILLED":
                order.status = OrderStatus.FILLED
                order.filled_quantity = float(response.get("executedQty", 0))
                order.avg_fill_price = float(response.get("avgPrice", entry_price or 0))

                # Create position
                await self._on_position_opened(
                    symbol=symbol,
                    side=side,
                    quantity=order.filled_quantity,
                    entry_price=order.avg_fill_price,
                    atr=atr,
                )

            self._orders[order_id] = order
            logger.info(f"Order placed: {order}")
            return order

        except BinanceAPIError as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected: {e}")
            return order

    async def close_position(
        self,
        symbol: str,
        use_market: bool = True,
    ) -> Order | None:
        """
        Close an existing position.

        Args:
            symbol: Trading pair.
            use_market: Use market order for immediate execution.

        Returns:
            Order if successful.
        """
        if symbol not in self._positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        position = self._positions[symbol]

        # Opposite side to close
        close_side = "SELL" if position.side == PositionSide.LONG else "BUY"

        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            side=close_side,
            order_type="MARKET" if use_market else "LIMIT",
            quantity=position.quantity,
            price=None,
            stop_price=None,
            status=OrderStatus.PENDING,
        )

        try:
            response = await self._client.place_market_order(
                symbol, close_side, position.quantity
            )

            order.binance_order_id = response.get("orderId")
            order.status = OrderStatus.FILLED
            order.filled_quantity = float(response.get("executedQty", 0))
            order.avg_fill_price = float(response.get("avgPrice", 0))

            # Calculate PnL
            pnl = self._calculate_pnl(position, order.avg_fill_price)

            # Update risk manager
            self._risk.update_pnl(pnl, symbol)
            self._risk.remove_position(symbol)

            # Remove trailing stop
            self._stops.close_position(symbol)

            # Remove position
            del self._positions[symbol]

            self._orders[order_id] = order
            logger.info(f"Position closed: {symbol}, PnL: {pnl:.2f}")
            return order

        except BinanceAPIError as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Close order rejected: {e}")
            return order

    async def _on_position_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        atr: float | None,
    ) -> None:
        """Handle position opened event."""
        position_side = PositionSide.LONG if side == "BUY" else PositionSide.SHORT

        # Create position record
        position = Position(
            symbol=symbol,
            side=position_side,
            quantity=quantity,
            entry_price=entry_price,
        )
        self._positions[symbol] = position

        # Update risk manager
        position_value = quantity * entry_price
        self._risk.add_position(symbol, position_value)

        # Create trailing stop if ATR provided
        if atr is not None:
            self._stops.create_stop(symbol, position_side, entry_price, atr)

            # Place stop-market order on exchange
            stop_price = self._stops.get_stop_price(symbol)
            if stop_price:
                stop_side = "SELL" if position_side == PositionSide.LONG else "BUY"
                await self._client.place_stop_market(
                    symbol, stop_side, quantity, stop_price, reduce_only=True
                )

        logger.info(f"Position opened: {symbol} {side} {quantity} @ {entry_price}")

    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate P&L for a closed position."""
        if position.side == PositionSide.LONG:
            return (exit_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - exit_price) * position.quantity

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter}_{int(datetime.utcnow().timestamp())}"

    async def check_stops(self, symbol: str, current_price: float) -> None:
        """
        Check if stop should be triggered and update trailing.

        Args:
            symbol: Trading pair.
            current_price: Current market price.
        """
        if symbol not in self._positions:
            return

        # Update trailing stop
        self._stops.update_stop(symbol, current_price)

        # Check if stop hit
        if self._stops.is_stopped(symbol, current_price):
            logger.warning(f"Stop triggered for {symbol} at {current_price}")
            await self.close_position(symbol, use_market=True)

    def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all open positions."""
        return self._positions.copy()

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def sync_positions(self) -> None:
        """Sync positions with exchange."""
        positions = await self._client.get_positions()

        for pos in positions:
            symbol = pos["symbol"]
            amount = float(pos["positionAmt"])

            if amount == 0:
                continue

            side = PositionSide.LONG if amount > 0 else PositionSide.SHORT
            entry_price = float(pos["entryPrice"])
            unrealized_pnl = float(pos["unRealizedProfit"])

            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=abs(amount),
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                leverage=int(pos.get("leverage", 1)),
            )

        logger.info(f"Synced {len(self._positions)} positions from exchange")
