"""
ATR-based Trailing Stop Manager.

Implements dynamic trailing stops that adapt to market volatility.
Uses Average True Range (ATR) for stop distance calculation.

Follows @risk-manager skill patterns.
"""

from dataclasses import dataclass
from enum import Enum

from loguru import logger


class PositionSide(Enum):
    """Position direction."""

    LONG = "long"
    SHORT = "short"


@dataclass
class StopLevel:
    """Current stop level for a position."""

    symbol: str
    side: PositionSide
    entry_price: float
    current_stop: float
    highest_price: float  # For long
    lowest_price: float  # For short
    atr_at_entry: float
    atr_multiplier: float

    def get_risk_distance(self) -> float:
        """Get distance from entry to stop."""
        if self.side == PositionSide.LONG:
            return self.entry_price - self.current_stop
        return self.current_stop - self.entry_price

    def get_risk_pct(self) -> float:
        """Get risk as percentage of entry price."""
        return self.get_risk_distance() / self.entry_price


class TrailingStopManager:
    """
    Manages ATR-based trailing stops for positions.

    Features:
    - Initial stop based on ATR at entry
    - Trailing stop that locks in profits
    - Supports both long and short positions
    - Configurable ATR multiplier

    Example:
        ```python
        stop_mgr = TrailingStopManager(atr_multiplier=2.0)

        # Open position
        stop_level = stop_mgr.create_stop(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=45000,
            current_atr=500,
        )

        # Update on price change
        stop_level = stop_mgr.update_stop("BTCUSDT", current_price=46000)

        # Check if stop hit
        if stop_mgr.is_stopped("BTCUSDT", current_price=44000):
            # Execute stop loss
            pass
        ```
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        min_profit_to_trail: float = 0.5,
        trail_step_atr: float = 0.5,
    ) -> None:
        """
        Initialize trailing stop manager.

        Args:
            atr_multiplier: Multiple of ATR for initial stop distance.
            min_profit_to_trail: Minimum profit (in ATR) before trailing starts.
            trail_step_atr: How much price must move for stop to trail.
        """
        self.atr_multiplier = atr_multiplier
        self.min_profit_to_trail = min_profit_to_trail
        self.trail_step_atr = trail_step_atr

        self._stops: dict[str, StopLevel] = {}

    def create_stop(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        current_atr: float,
    ) -> StopLevel:
        """
        Create initial stop for a new position.

        Args:
            symbol: Trading pair symbol.
            side: Position side (LONG or SHORT).
            entry_price: Entry price.
            current_atr: Current ATR value.

        Returns:
            StopLevel with initial stop configuration.
        """
        stop_distance = current_atr * self.atr_multiplier

        if side == PositionSide.LONG:
            initial_stop = entry_price - stop_distance
        else:
            initial_stop = entry_price + stop_distance

        stop_level = StopLevel(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_stop=initial_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
            atr_at_entry=current_atr,
            atr_multiplier=self.atr_multiplier,
        )

        self._stops[symbol] = stop_level

        logger.info(
            f"Created {side.value} stop for {symbol}: "
            f"entry={entry_price:.2f}, stop={initial_stop:.2f}, "
            f"ATR={current_atr:.2f}"
        )

        return stop_level

    def update_stop(
        self,
        symbol: str,
        current_price: float,
    ) -> StopLevel | None:
        """
        Update trailing stop based on current price.

        For long positions: stop trails up when price increases
        For short positions: stop trails down when price decreases

        Args:
            symbol: Trading pair symbol.
            current_price: Current market price.

        Returns:
            Updated StopLevel or None if no position.
        """
        if symbol not in self._stops:
            return None

        stop = self._stops[symbol]
        stop_distance = stop.atr_at_entry * self.atr_multiplier
        min_profit_distance = stop.atr_at_entry * self.min_profit_to_trail

        if stop.side == PositionSide.LONG:
            # Update highest price
            if current_price > stop.highest_price:
                stop.highest_price = current_price

                # Check if we should trail
                profit = current_price - stop.entry_price
                if profit > min_profit_distance:
                    new_stop = current_price - stop_distance
                    if new_stop > stop.current_stop:
                        old_stop = stop.current_stop
                        stop.current_stop = new_stop
                        logger.debug(
                            f"{symbol} trailing stop moved up: "
                            f"{old_stop:.2f} -> {new_stop:.2f}"
                        )
        else:
            # Short position
            if current_price < stop.lowest_price:
                stop.lowest_price = current_price

                # Check if we should trail
                profit = stop.entry_price - current_price
                if profit > min_profit_distance:
                    new_stop = current_price + stop_distance
                    if new_stop < stop.current_stop:
                        old_stop = stop.current_stop
                        stop.current_stop = new_stop
                        logger.debug(
                            f"{symbol} trailing stop moved down: "
                            f"{old_stop:.2f} -> {new_stop:.2f}"
                        )

        return stop

    def is_stopped(
        self,
        symbol: str,
        current_price: float,
    ) -> bool:
        """
        Check if price has hit the stop level.

        Args:
            symbol: Trading pair symbol.
            current_price: Current market price.

        Returns:
            True if stop is triggered.
        """
        if symbol not in self._stops:
            return False

        stop = self._stops[symbol]

        if stop.side == PositionSide.LONG:
            return current_price <= stop.current_stop
        else:
            return current_price >= stop.current_stop

    def get_stop_price(self, symbol: str) -> float | None:
        """Get current stop price for a symbol."""
        if symbol not in self._stops:
            return None
        return self._stops[symbol].current_stop

    def get_stop_level(self, symbol: str) -> StopLevel | None:
        """Get full stop level details for a symbol."""
        return self._stops.get(symbol)

    def close_position(self, symbol: str) -> None:
        """Remove stop when position is closed."""
        if symbol in self._stops:
            del self._stops[symbol]
            logger.info(f"Closed stop for {symbol}")

    def get_all_stops(self) -> dict[str, StopLevel]:
        """Get all active stop levels."""
        return self._stops.copy()

    def calculate_stop_price(
        self,
        side: PositionSide,
        entry_price: float,
        atr: float,
    ) -> float:
        """
        Calculate stop price without creating a stop.

        Useful for order preview.

        Args:
            side: Position side.
            entry_price: Planned entry price.
            atr: Current ATR.

        Returns:
            Stop price.
        """
        stop_distance = atr * self.atr_multiplier

        if side == PositionSide.LONG:
            return entry_price - stop_distance
        return entry_price + stop_distance

    def calculate_take_profit(
        self,
        side: PositionSide,
        entry_price: float,
        atr: float,
        risk_reward_ratio: float = 2.0,
    ) -> float:
        """
        Calculate take profit price based on risk/reward ratio.

        Args:
            side: Position side.
            entry_price: Entry price.
            atr: Current ATR.
            risk_reward_ratio: Desired R:R ratio.

        Returns:
            Take profit price.
        """
        stop_distance = atr * self.atr_multiplier
        tp_distance = stop_distance * risk_reward_ratio

        if side == PositionSide.LONG:
            return entry_price + tp_distance
        return entry_price - tp_distance
