"""
Risk Limits and Position Management.

Implements portfolio-level risk constraints:
- Maximum drawdown limits
- Daily loss limits
- Position concentration limits

Follows @risk-manager skill patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class RiskState:
    """Current risk state of the portfolio."""

    # Account values
    account_balance: float = 0.0
    peak_balance: float = 0.0
    starting_day_balance: float = 0.0

    # Current metrics
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    open_position_value: float = 0.0

    # Limits status
    is_max_drawdown_breached: bool = False
    is_daily_limit_breached: bool = False
    is_position_limit_breached: bool = False

    # Last update
    last_update: datetime = field(default_factory=datetime.utcnow)


class RiskLimits:
    """
    Portfolio risk limit manager.

    Enforces risk constraints at the portfolio level to prevent
    catastrophic losses.

    Example:
        ```python
        risk = RiskLimits(
            max_drawdown_pct=0.15,  # 15% max drawdown
            daily_loss_limit_pct=0.05,  # 5% daily loss limit
            max_position_pct=0.1,  # 10% max per position
        )

        # Initialize with balance
        risk.set_account_balance(10000)

        # Check before trading
        if risk.can_trade():
            # Execute trade
            risk.update_pnl(trade_pnl)

        # Check position size
        if risk.is_position_allowed(2000, "BTCUSDT"):
            # Open position
            pass
        ```
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.15,
        daily_loss_limit_pct: float = 0.05,
        max_position_pct: float = 0.1,
        max_total_exposure_pct: float = 0.5,
        cooldown_after_limit_minutes: int = 60,
    ) -> None:
        """
        Initialize risk limits.

        Args:
            max_drawdown_pct: Maximum drawdown from peak (e.g., 0.15 = 15%).
            daily_loss_limit_pct: Maximum daily loss (e.g., 0.05 = 5%).
            max_position_pct: Maximum single position size.
            max_total_exposure_pct: Maximum total portfolio exposure.
            cooldown_after_limit_minutes: Cooldown period after limit hit.
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.cooldown_minutes = cooldown_after_limit_minutes

        self._state = RiskState()
        self._positions: dict[str, float] = {}  # symbol -> position value
        self._pnl_history: list[dict[str, Any]] = []
        self._limit_breach_time: datetime | None = None

    @property
    def state(self) -> RiskState:
        """Get current risk state."""
        return self._state

    def set_account_balance(self, balance: float) -> None:
        """
        Set current account balance.

        Call this at startup and when balance changes.

        Args:
            balance: Current account balance.
        """
        self._state.account_balance = balance

        # Update peak if needed
        if balance > self._state.peak_balance:
            self._state.peak_balance = balance

        # Reset daily balance if new day
        if self._is_new_day():
            self._state.starting_day_balance = balance
            self._state.daily_pnl = 0.0

        self._update_drawdown()

    def update_pnl(self, pnl: float, symbol: str | None = None) -> None:
        """
        Update P&L after a trade.

        Args:
            pnl: Profit/loss amount.
            symbol: Optional symbol for the trade.
        """
        # Update balance
        self._state.account_balance += pnl
        self._state.daily_pnl += pnl

        # Update peak if new high
        if self._state.account_balance > self._state.peak_balance:
            self._state.peak_balance = self._state.account_balance

        # Record history
        self._pnl_history.append(
            {
                "timestamp": datetime.utcnow(),
                "pnl": pnl,
                "symbol": symbol,
                "balance": self._state.account_balance,
            }
        )

        self._update_drawdown()
        self._check_limits()

    def _update_drawdown(self) -> None:
        """Calculate current drawdown from peak."""
        if self._state.peak_balance > 0:
            self._state.current_drawdown = (
                self._state.peak_balance - self._state.account_balance
            ) / self._state.peak_balance
        else:
            self._state.current_drawdown = 0.0

    def _check_limits(self) -> None:
        """Check if any risk limits are breached."""
        # Check max drawdown
        if self._state.current_drawdown >= self.max_drawdown_pct:
            if not self._state.is_max_drawdown_breached:
                logger.warning(
                    f"MAX DRAWDOWN LIMIT BREACHED: {self._state.current_drawdown:.2%}"
                )
                self._state.is_max_drawdown_breached = True
                self._limit_breach_time = datetime.utcnow()

        # Check daily loss limit
        daily_loss_pct = abs(min(0, self._state.daily_pnl)) / self._state.starting_day_balance
        if daily_loss_pct >= self.daily_loss_limit_pct:
            if not self._state.is_daily_limit_breached:
                logger.warning(
                    f"DAILY LOSS LIMIT BREACHED: {daily_loss_pct:.2%}"
                )
                self._state.is_daily_limit_breached = True
                self._limit_breach_time = datetime.utcnow()

        self._state.last_update = datetime.utcnow()

    def can_trade(self) -> bool:
        """
        Check if trading is allowed under current risk state.

        Returns:
            True if trading is allowed.
        """
        # Check limits
        if self._state.is_max_drawdown_breached:
            logger.debug("Trading blocked: max drawdown breached")
            return False

        if self._state.is_daily_limit_breached:
            logger.debug("Trading blocked: daily limit breached")
            return False

        # Check cooldown
        if self._limit_breach_time is not None:
            cooldown_end = self._limit_breach_time + timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                logger.debug("Trading blocked: in cooldown period")
                return False

        return True

    def is_position_allowed(
        self,
        position_value: float,
        symbol: str,
    ) -> bool:
        """
        Check if a position size is allowed.

        Args:
            position_value: Value of the proposed position.
            symbol: Trading symbol.

        Returns:
            True if position is within limits.
        """
        if not self.can_trade():
            return False

        # Check single position limit
        max_single_position = self._state.account_balance * self.max_position_pct
        if position_value > max_single_position:
            logger.warning(
                f"Position {symbol} size {position_value:.2f} exceeds limit {max_single_position:.2f}"
            )
            return False

        # Check total exposure
        current_exposure = sum(self._positions.values())
        new_total_exposure = current_exposure + position_value
        max_exposure = self._state.account_balance * self.max_total_exposure_pct

        if new_total_exposure > max_exposure:
            logger.warning(
                f"Total exposure {new_total_exposure:.2f} would exceed limit {max_exposure:.2f}"
            )
            return False

        return True

    def add_position(self, symbol: str, value: float) -> None:
        """Record a new position."""
        self._positions[symbol] = value
        self._state.open_position_value = sum(self._positions.values())
        logger.debug(f"Added position {symbol}: {value:.2f}")

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        if symbol in self._positions:
            del self._positions[symbol]
            self._state.open_position_value = sum(self._positions.values())
            logger.debug(f"Removed position {symbol}")

    def update_position_value(self, symbol: str, value: float) -> None:
        """Update value of an existing position."""
        if symbol in self._positions:
            self._positions[symbol] = value
            self._state.open_position_value = sum(self._positions.values())

    def reset_daily_limits(self) -> None:
        """Reset daily limits (call at day start)."""
        self._state.starting_day_balance = self._state.account_balance
        self._state.daily_pnl = 0.0
        self._state.is_daily_limit_breached = False
        logger.info("Daily risk limits reset")

    def reset_all_limits(self) -> None:
        """Reset all limits (use with caution)."""
        self._state.is_max_drawdown_breached = False
        self._state.is_daily_limit_breached = False
        self._limit_breach_time = None
        logger.warning("All risk limits reset")

    def get_remaining_daily_budget(self) -> float:
        """Get remaining loss budget for today."""
        max_daily_loss = self._state.starting_day_balance * self.daily_loss_limit_pct
        used_budget = abs(min(0, self._state.daily_pnl))
        return max(0, max_daily_loss - used_budget)

    def get_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        return {
            "account_balance": self._state.account_balance,
            "peak_balance": self._state.peak_balance,
            "current_drawdown": self._state.current_drawdown,
            "daily_pnl": self._state.daily_pnl,
            "open_exposure": self._state.open_position_value,
            "can_trade": self.can_trade(),
            "remaining_daily_budget": self.get_remaining_daily_budget(),
            "positions": dict(self._positions),
        }

    def _is_new_day(self) -> bool:
        """Check if it's a new trading day."""
        if self._state.last_update.date() < datetime.utcnow().date():
            return True
        return False

    def get_pnl_history(self, days: int = 7) -> pd.DataFrame:
        """Get P&L history as DataFrame."""
        if not self._pnl_history:
            return pd.DataFrame()

        df = pd.DataFrame(self._pnl_history)
        cutoff = datetime.utcnow() - timedelta(days=days)
        return df[df["timestamp"] >= cutoff]
