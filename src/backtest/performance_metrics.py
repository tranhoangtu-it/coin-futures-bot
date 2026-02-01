"""
Performance metrics for trading strategy evaluation.

Comprehensive metrics calculation including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Trade statistics

Follows @risk-metrics-calculation skill patterns.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DrawdownInfo:
    """Drawdown period information."""

    start: pd.Timestamp
    end: pd.Timestamp
    recovery: pd.Timestamp | None
    depth: float
    duration_days: int
    recovery_days: int | None


class PerformanceMetrics:
    """
    Trading performance metrics calculator.

    Features:
    - Risk-adjusted return metrics
    - Drawdown analysis
    - Trade-level statistics
    - Rolling metrics

    Example:
        ```python
        metrics = PerformanceMetrics()

        # From equity curve
        stats = metrics.calculate(equity_curve)

        # Drawdown analysis
        drawdowns = metrics.analyze_drawdowns(equity_curve)

        # Risk metrics
        var = metrics.calculate_var(returns, confidence=0.95)
        ```
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            periods_per_year: Trading periods per year (252 for daily).
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate(
        self,
        equity: pd.Series | None = None,
        returns: pd.Series | None = None,
        trades: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            equity: Equity curve series.
            returns: Returns series.
            trades: DataFrame with trade history.

        Returns:
            Dictionary of performance metrics.
        """
        metrics = {}

        # Calculate returns from equity if needed
        if returns is None and equity is not None:
            returns = equity.pct_change().dropna()

        if returns is not None and len(returns) > 0:
            metrics.update(self._return_metrics(returns))
            metrics.update(self._risk_metrics(returns))

        if equity is not None and len(equity) > 0:
            metrics.update(self._drawdown_metrics(equity))

        if trades is not None and len(trades) > 0:
            metrics.update(self._trade_metrics(trades))

        return metrics

    def _return_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate return-based metrics."""
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)

        # Annualized return
        annual_return = (1 + total_return) ** (self.periods_per_year / n_periods) - 1

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "mean_return": returns.mean(),
            "median_return": returns.median(),
            "best_period": returns.max(),
            "worst_period": returns.min(),
            "positive_periods": (returns > 0).sum() / len(returns),
        }

    def _risk_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate risk metrics."""
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(self.periods_per_year)

        # Sharpe ratio
        daily_rf = self.risk_free_rate / self.periods_per_year
        excess_returns = returns - daily_rf
        sharpe = (
            excess_returns.mean() / returns.std() * np.sqrt(self.periods_per_year)
            if returns.std() > 0 else 0
        )

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)
            sortino = (returns.mean() * self.periods_per_year - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino = 0

        # Skewness and kurtosis
        skew = returns.skew()
        kurtosis = returns.kurtosis()

        return {
            "volatility": annual_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "skewness": skew,
            "kurtosis": kurtosis,
        }

    def _drawdown_metrics(self, equity: pd.Series) -> dict[str, float]:
        """Calculate drawdown metrics."""
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        max_drawdown = drawdown.min()

        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = in_drawdown.astype(int).groupby((~in_drawdown).cumsum())

        if len(drawdown_periods) > 0:
            max_drawdown_duration = max(len(g) for _, g in drawdown_periods if len(g) > 0)
        else:
            max_drawdown_duration = 0

        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Calmar ratio
        equity_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        annual_return = (1 + equity_return) ** (self.periods_per_year / len(equity)) - 1
        calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "calmar": calmar,
            "ulcer_index": np.sqrt((drawdown ** 2).mean()),
        }

    def _trade_metrics(self, trades: pd.DataFrame) -> dict[str, float]:
        """Calculate trade-level metrics."""
        if "pnl" not in trades.columns:
            return {}

        n_trades = len(trades)
        pnl = trades["pnl"]

        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        win_rate = len(wins) / n_trades if n_trades > 0 else 0

        # Average win/loss
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = pnl.mean()

        # Payoff ratio
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Consecutive wins/losses
        win_streak = self._max_consecutive(pnl > 0)
        loss_streak = self._max_consecutive(pnl <= 0)

        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "payoff_ratio": payoff_ratio,
            "max_win_streak": win_streak,
            "max_loss_streak": loss_streak,
            "total_pnl": pnl.sum(),
        }

    def _max_consecutive(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive True values."""
        groups = (~condition).cumsum()
        return condition.groupby(groups).sum().max() if len(condition) > 0 else 0

    def analyze_drawdowns(
        self,
        equity: pd.Series,
        top_n: int = 5,
    ) -> list[DrawdownInfo]:
        """
        Analyze drawdown periods.

        Args:
            equity: Equity curve.
            top_n: Number of worst drawdowns to return.

        Returns:
            List of DrawdownInfo for worst drawdowns.
        """
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        drawdowns = []

        # Find drawdown periods
        in_drawdown = drawdown < 0
        dd_groups = (~in_drawdown).cumsum()

        for _, group in drawdown.groupby(dd_groups):
            if (group < 0).any():
                start = group.index[0]
                depth = group.min()
                end = group.idxmin()

                # Find recovery
                remaining = equity.loc[end:]
                peak = running_max.loc[start]
                recovered = remaining >= peak

                if recovered.any():
                    recovery = recovered.idxmax()
                    recovery_days = (recovery - end).days
                else:
                    recovery = None
                    recovery_days = None

                duration_days = (end - start).days

                drawdowns.append(DrawdownInfo(
                    start=start,
                    end=end,
                    recovery=recovery,
                    depth=depth,
                    duration_days=duration_days,
                    recovery_days=recovery_days,
                ))

        # Sort by depth and return top N
        drawdowns.sort(key=lambda x: x.depth)
        return drawdowns[:top_n]

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Return series.
            confidence: Confidence level (e.g., 0.95 for 95%).
            method: "historical" or "parametric".

        Returns:
            VaR as a positive number (expected loss).
        """
        if method == "historical":
            return -np.percentile(returns, (1 - confidence) * 100)
        else:
            # Parametric (assuming normal distribution)
            from scipy import stats
            z = stats.norm.ppf(1 - confidence)
            return -(returns.mean() + z * returns.std())

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Return series.
            confidence: Confidence level.

        Returns:
            CVaR as a positive number.
        """
        var = self.calculate_var(returns, confidence)
        tail_returns = returns[returns <= -var]
        return -tail_returns.mean() if len(tail_returns) > 0 else var

    def rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            returns: Return series.
            window: Rolling window size.

        Returns:
            DataFrame with rolling metrics.
        """
        rolling = pd.DataFrame(index=returns.index)

        # Rolling Sharpe
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        daily_rf = self.risk_free_rate / self.periods_per_year
        rolling["sharpe"] = (rolling_mean - daily_rf) / rolling_std * np.sqrt(self.periods_per_year)

        # Rolling volatility
        rolling["volatility"] = rolling_std * np.sqrt(self.periods_per_year)

        # Rolling return
        rolling["return"] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )

        # Rolling max drawdown
        def rolling_max_dd(x):
            equity = (1 + x).cumprod()
            running_max = equity.cummax()
            dd = (equity - running_max) / running_max
            return dd.min()

        rolling["max_drawdown"] = returns.rolling(window).apply(rolling_max_dd, raw=False)

        return rolling
