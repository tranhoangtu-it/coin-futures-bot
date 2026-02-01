"""
Kelly Criterion for optimal position sizing.

Implements the fractional Kelly formula for risk-adjusted position scaling.
Adapts position size based on edge and risk tolerance.

Follows @risk-manager skill patterns.
"""

import numpy as np
import pandas as pd
from loguru import logger


class KellyCriterion:
    """
    Kelly Criterion calculator for position sizing.

    The Kelly Criterion determines the optimal fraction of capital to risk
    based on the probability of winning and the win/loss ratio.

    Formula: f* = (bp - q) / b
    Where:
        f* = optimal fraction
        b = win/loss ratio (avg_win / avg_loss)
        p = probability of winning
        q = probability of losing (1 - p)

    Example:
        ```python
        kelly = KellyCriterion(fractional_factor=0.3)

        # From historical trades
        fraction = kelly.calculate_from_trades(trades_df)

        # From model predictions
        fraction = kelly.calculate_from_probability(
            win_prob=0.55,
            avg_win=0.02,
            avg_loss=0.01,
        )

        # Apply to position
        position_size = kelly.get_position_size(
            account_balance=10000,
            entry_price=45000,
        )
        ```
    """

    def __init__(
        self,
        fractional_factor: float = 0.3,
        max_position_pct: float = 0.1,
        min_trades_for_calculation: int = 30,
    ) -> None:
        """
        Initialize Kelly Criterion calculator.

        Args:
            fractional_factor: Fraction of Kelly to use (0.2-0.5 recommended).
            max_position_pct: Maximum position size as % of account.
            min_trades_for_calculation: Minimum trades for reliable Kelly.
        """
        self.fractional_factor = fractional_factor
        self.max_position_pct = max_position_pct
        self.min_trades_for_calculation = min_trades_for_calculation

        self._last_kelly: float = 0.0
        self._last_full_kelly: float = 0.0

    @property
    def last_kelly(self) -> float:
        """Get last calculated fractional Kelly."""
        return self._last_kelly

    @property
    def last_full_kelly(self) -> float:
        """Get last calculated full Kelly."""
        return self._last_full_kelly

    def calculate(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly fraction from parameters.

        Args:
            win_prob: Probability of winning (0 to 1).
            avg_win: Average winning trade return (positive).
            avg_loss: Average losing trade return (positive, will convert to loss).

        Returns:
            Fractional Kelly position size (0 to max_position_pct).
        """
        if win_prob <= 0 or win_prob >= 1:
            logger.warning(f"Invalid win_prob: {win_prob}")
            return 0.0

        if avg_loss == 0:
            logger.warning("avg_loss is zero, cannot calculate Kelly")
            return 0.0

        # b = win/loss ratio
        b = avg_win / avg_loss

        # p = win probability, q = loss probability
        p = win_prob
        q = 1 - p

        # Full Kelly: f* = (bp - q) / b
        full_kelly = (b * p - q) / b

        self._last_full_kelly = full_kelly

        # If negative edge, don't trade
        if full_kelly <= 0:
            logger.info(f"Negative edge: Kelly = {full_kelly:.4f}")
            self._last_kelly = 0.0
            return 0.0

        # Apply fractional factor
        fractional_kelly = full_kelly * self.fractional_factor

        # Cap at max position
        final_kelly = min(fractional_kelly, self.max_position_pct)

        self._last_kelly = final_kelly

        logger.debug(
            f"Kelly: full={full_kelly:.4f}, fractional={fractional_kelly:.4f}, "
            f"final={final_kelly:.4f}"
        )

        return final_kelly

    def calculate_from_trades(
        self,
        trades: pd.DataFrame,
        pnl_column: str = "pnl",
    ) -> float:
        """
        Calculate Kelly from historical trade data.

        Args:
            trades: DataFrame with trade history.
            pnl_column: Column name for P&L values.

        Returns:
            Fractional Kelly position size.
        """
        if len(trades) < self.min_trades_for_calculation:
            logger.warning(
                f"Insufficient trades ({len(trades)}) for reliable Kelly. "
                f"Need at least {self.min_trades_for_calculation}."
            )
            return 0.0

        pnl = trades[pnl_column]

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        if len(wins) == 0 or len(losses) == 0:
            logger.warning("Need both winning and losing trades for Kelly")
            return 0.0

        win_prob = len(wins) / len(pnl)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        logger.info(
            f"Trade stats: win_rate={win_prob:.2%}, "
            f"avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}"
        )

        return self.calculate(win_prob, avg_win, avg_loss)

    def calculate_from_probability(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly from model-predicted probability.

        Alias for calculate() with clearer naming.

        Args:
            win_prob: Model's predicted win probability.
            avg_win: Expected average win.
            avg_loss: Expected average loss.

        Returns:
            Fractional Kelly position size.
        """
        return self.calculate(win_prob, avg_win, avg_loss)

    def get_position_size(
        self,
        account_balance: float,
        entry_price: float,
        kelly_fraction: float | None = None,
    ) -> float:
        """
        Calculate position size in units (e.g., BTC).

        Args:
            account_balance: Total account balance in quote currency.
            entry_price: Entry price per unit.
            kelly_fraction: Kelly fraction to use. If None, uses last calculated.

        Returns:
            Position size in base currency units.
        """
        fraction = kelly_fraction if kelly_fraction is not None else self._last_kelly

        if fraction <= 0:
            return 0.0

        position_value = account_balance * fraction
        position_size = position_value / entry_price

        return position_size

    def get_position_size_with_leverage(
        self,
        account_balance: float,
        entry_price: float,
        leverage: int,
        kelly_fraction: float | None = None,
    ) -> float:
        """
        Calculate position size accounting for leverage.

        With leverage, the margin required is reduced, but risk per unit
        of account stays the same.

        Args:
            account_balance: Total account balance.
            entry_price: Entry price.
            leverage: Leverage multiplier.
            kelly_fraction: Kelly fraction to use.

        Returns:
            Position size in base currency units.
        """
        fraction = kelly_fraction if kelly_fraction is not None else self._last_kelly

        if fraction <= 0:
            return 0.0

        # Position value (notional)
        position_value = account_balance * fraction * leverage
        position_size = position_value / entry_price

        return position_size

    def get_edge(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate expected edge (expected return per trade).

        Edge = (win_prob * avg_win) - (loss_prob * avg_loss)

        Args:
            win_prob: Win probability.
            avg_win: Average win amount.
            avg_loss: Average loss amount.

        Returns:
            Expected edge per trade.
        """
        return (win_prob * avg_win) - ((1 - win_prob) * avg_loss)

    def get_r_multiple(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate R-multiple (reward/risk ratio).

        Args:
            win_prob: Win probability.
            avg_win: Average win.
            avg_loss: Average loss.

        Returns:
            R-multiple.
        """
        if avg_loss == 0:
            return 0.0
        return avg_win / avg_loss

    def calculate_expectancy(
        self,
        trades: pd.DataFrame,
        pnl_column: str = "pnl",
    ) -> dict[str, float]:
        """
        Calculate trading system expectancy metrics.

        Args:
            trades: Trade history DataFrame.
            pnl_column: P&L column name.

        Returns:
            Dictionary with expectancy metrics.
        """
        pnl = trades[pnl_column]

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        n_trades = len(pnl)
        n_wins = len(wins)
        n_losses = len(losses)

        win_rate = n_wins / n_trades if n_trades > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Profit factor = Gross Profits / Gross Losses
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "n_trades": n_trades,
            "n_wins": n_wins,
            "n_losses": n_losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "r_multiple": avg_win / avg_loss if avg_loss > 0 else 0,
        }
