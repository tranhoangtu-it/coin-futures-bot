"""
Slippage Control for large order execution.

Implements strategies to minimize market impact:
- TWAP (Time-Weighted Average Price)
- Iceberg orders
- Market depth analysis

Follows @quant-analyst skill patterns.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from src.infrastructure import OrderBookSynchronizer

from .binance_client import BinanceClient


@dataclass
class ExecutionResult:
    """Result of order execution."""

    total_quantity: float
    filled_quantity: float
    avg_price: float
    slippage_bps: float  # Basis points
    n_orders: int
    duration_seconds: float


class SlippageController:
    """
    Controls slippage for large order execution.

    Features:
    - TWAP algorithm for time-distributed execution
    - Iceberg orders for hidden liquidity
    - Market depth analysis before execution
    - Adaptive sizing based on liquidity

    Example:
        ```python
        controller = SlippageController(client, order_book)

        # Execute with TWAP
        result = await controller.execute_twap(
            symbol="BTCUSDT",
            side="BUY",
            total_quantity=1.0,
            duration_minutes=10,
            n_slices=10,
        )
        ```
    """

    def __init__(
        self,
        client: BinanceClient,
        order_book: OrderBookSynchronizer | None = None,
    ) -> None:
        """
        Initialize slippage controller.

        Args:
            client: Binance API client.
            order_book: Order book for depth analysis.
        """
        self._client = client
        self._order_book = order_book

    async def execute_twap(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        duration_minutes: int = 10,
        n_slices: int = 10,
        randomize: bool = True,
    ) -> ExecutionResult:
        """
        Execute using Time-Weighted Average Price (TWAP).

        Splits order into equal slices executed at regular intervals.

        Args:
            symbol: Trading pair.
            side: "BUY" or "SELL".
            total_quantity: Total quantity to execute.
            duration_minutes: Total execution window.
            n_slices: Number of order slices.
            randomize: Add randomization to timing/size.

        Returns:
            ExecutionResult with execution details.
        """
        start_time = datetime.utcnow()
        slice_size = total_quantity / n_slices
        interval_seconds = (duration_minutes * 60) / n_slices

        filled_quantity = 0.0
        total_value = 0.0
        orders_placed = 0

        # Get reference price for slippage calculation
        ticker = await self._client.get_ticker_price(symbol)
        reference_price = float(ticker["price"])

        logger.info(
            f"Starting TWAP: {symbol} {side} {total_quantity}, "
            f"{n_slices} slices over {duration_minutes}min"
        )

        for i in range(n_slices):
            # Skip if remaining quantity is too small
            remaining = total_quantity - filled_quantity
            if remaining < slice_size * 0.1:
                break

            current_slice = min(slice_size, remaining)

            # Add randomization
            if randomize:
                import random
                current_slice *= 0.8 + 0.4 * random.random()
                current_slice = min(current_slice, remaining)

            try:
                # Place market order for slice
                order = await self._client.place_market_order(
                    symbol, side, round(current_slice, 6)
                )

                if order.get("status") == "FILLED":
                    qty = float(order.get("executedQty", 0))
                    price = float(order.get("avgPrice", 0))
                    filled_quantity += qty
                    total_value += qty * price
                    orders_placed += 1

                    logger.debug(
                        f"TWAP slice {i+1}/{n_slices}: {qty} @ {price}"
                    )

            except Exception as e:
                logger.error(f"TWAP slice {i+1} failed: {e}")

            # Wait for next slice
            if i < n_slices - 1:
                wait_time = interval_seconds
                if randomize:
                    import random
                    wait_time *= 0.7 + 0.6 * random.random()
                await asyncio.sleep(wait_time)

        # Calculate results
        avg_price = total_value / filled_quantity if filled_quantity > 0 else 0
        slippage_bps = (
            (avg_price - reference_price) / reference_price * 10000
            if side == "BUY" else
            (reference_price - avg_price) / reference_price * 10000
        )
        duration = (datetime.utcnow() - start_time).total_seconds()

        result = ExecutionResult(
            total_quantity=total_quantity,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
            slippage_bps=slippage_bps,
            n_orders=orders_placed,
            duration_seconds=duration,
        )

        logger.info(
            f"TWAP complete: filled {filled_quantity:.6f} @ {avg_price:.2f}, "
            f"slippage={slippage_bps:.2f}bps"
        )

        return result

    async def execute_iceberg(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        visible_quantity: float,
        price: float,
        timeout_seconds: int = 300,
    ) -> ExecutionResult:
        """
        Execute using iceberg/hidden orders.

        Places small visible orders that are automatically refreshed.

        Args:
            symbol: Trading pair.
            side: "BUY" or "SELL".
            total_quantity: Total quantity.
            visible_quantity: Quantity shown at any time.
            price: Limit price.
            timeout_seconds: Maximum execution time.

        Returns:
            ExecutionResult.
        """
        start_time = datetime.utcnow()
        filled_quantity = 0.0
        total_value = 0.0
        orders_placed = 0

        reference_price = price

        logger.info(
            f"Starting iceberg: {symbol} {side} {total_quantity}, "
            f"visible={visible_quantity} @ {price}"
        )

        while filled_quantity < total_quantity:
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                logger.warning("Iceberg execution timeout")
                break

            remaining = total_quantity - filled_quantity
            current_qty = min(visible_quantity, remaining)

            try:
                # Place limit order
                order = await self._client.place_limit_order(
                    symbol, side, current_qty, price, post_only=True
                )
                order_id = order.get("orderId")
                orders_placed += 1

                # Wait for fill or timeout
                for _ in range(30):  # 30 seconds max per order
                    order_status = await self._client.get_order(symbol, order_id=order_id)
                    status = order_status.get("status")

                    if status == "FILLED":
                        qty = float(order_status.get("executedQty", 0))
                        avg_price = float(order_status.get("avgPrice", price))
                        filled_quantity += qty
                        total_value += qty * avg_price
                        logger.debug(f"Iceberg order filled: {qty} @ {avg_price}")
                        break
                    elif status in ["CANCELLED", "REJECTED", "EXPIRED"]:
                        logger.warning(f"Iceberg order {status}")
                        break

                    await asyncio.sleep(1)
                else:
                    # Cancel unfilled order
                    await self._client.cancel_order(symbol, order_id=order_id)
                    logger.debug("Iceberg order cancelled (timeout)")

            except Exception as e:
                logger.error(f"Iceberg order failed: {e}")
                await asyncio.sleep(5)

        # Calculate results
        avg_price = total_value / filled_quantity if filled_quantity > 0 else 0
        slippage_bps = (
            (avg_price - reference_price) / reference_price * 10000
            if side == "BUY" else
            (reference_price - avg_price) / reference_price * 10000
        )
        duration = (datetime.utcnow() - start_time).total_seconds()

        return ExecutionResult(
            total_quantity=total_quantity,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
            slippage_bps=slippage_bps,
            n_orders=orders_placed,
            duration_seconds=duration,
        )

    def estimate_slippage(
        self,
        side: str,
        quantity: float,
    ) -> float | None:
        """
        Estimate slippage from order book.

        Args:
            side: "BUY" or "SELL".
            quantity: Order quantity.

        Returns:
            Estimated slippage in basis points.
        """
        if self._order_book is None or not self._order_book.is_synchronized:
            return None

        ob = self._order_book.order_book
        mid_price = ob.get_mid_price()

        if mid_price is None:
            return None

        remaining = quantity
        total_cost = 0.0

        # Walk through order book levels
        if side == "BUY":
            levels = ob.get_asks(100)
            for level in levels:
                if remaining <= 0:
                    break
                fill_qty = min(float(level.quantity), remaining)
                total_cost += fill_qty * float(level.price)
                remaining -= fill_qty
        else:
            levels = ob.get_bids(100)
            for level in levels:
                if remaining <= 0:
                    break
                fill_qty = min(float(level.quantity), remaining)
                total_cost += fill_qty * float(level.price)
                remaining -= fill_qty

        if remaining > 0:
            logger.warning("Insufficient liquidity for slippage estimate")
            return None

        avg_price = total_cost / quantity
        slippage_bps = abs(avg_price - float(mid_price)) / float(mid_price) * 10000

        return slippage_bps

    def get_available_liquidity(
        self,
        side: str,
        max_slippage_bps: float = 10,
    ) -> float:
        """
        Get available liquidity within slippage tolerance.

        Args:
            side: "BUY" or "SELL".
            max_slippage_bps: Maximum acceptable slippage.

        Returns:
            Available quantity within slippage limit.
        """
        if self._order_book is None or not self._order_book.is_synchronized:
            return 0.0

        ob = self._order_book.order_book
        mid_price = ob.get_mid_price()

        if mid_price is None:
            return 0.0

        max_price_deviation = float(mid_price) * (max_slippage_bps / 10000)

        available = 0.0

        if side == "BUY":
            levels = ob.get_asks(100)
            for level in levels:
                if float(level.price) > float(mid_price) + max_price_deviation:
                    break
                available += float(level.quantity)
        else:
            levels = ob.get_bids(100)
            for level in levels:
                if float(level.price) < float(mid_price) - max_price_deviation:
                    break
                available += float(level.quantity)

        return available
