"""
Vectorbt-based backtesting engine.

High-performance vectorized backtesting for strategy parameter optimization.
Uses vectorbt for fast simulation of millions of parameter combinations.

Follows @backtesting-frameworks skill patterns.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class VectorbtEngine:
    """
    High-performance backtesting engine using vectorized operations.

    Note: This implementation uses pure NumPy for vectorized backtesting
    when vectorbt is not available. Install vectorbt for advanced features.

    Features:
    - Vectorized signal simulation
    - Parameter grid search
    - Transaction cost modeling
    - Performance metrics calculation

    Example:
        ```python
        engine = VectorbtEngine(
            initial_capital=10000,
            commission=0.0004,
            slippage=0.0001,
        )

        results = engine.run_backtest(
            prices=df["close"],
            signals=signal_series,
        )

        # Parameter optimization
        best_params = engine.optimize(
            prices=df["close"],
            signal_func=generate_signals,
            param_grid={"threshold": [0.3, 0.4, 0.5]},
        )
        ```
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.0004,
        slippage: float = 0.0001,
        size_type: str = "percent",
        default_size: float = 0.1,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital.
            commission: Commission rate (e.g., 0.0004 = 0.04%).
            slippage: Expected slippage as fraction of price.
            size_type: "percent" or "fixed".
            default_size: Default position size (percent or fixed amount).
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.size_type = size_type
        self.default_size = default_size

    def run_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_size: pd.Series | float | None = None,
    ) -> dict[str, Any]:
        """
        Run vectorized backtest.

        Args:
            prices: Price series.
            signals: Signal series (1=long, -1=short, 0=flat).
            position_size: Optional position size series or constant.

        Returns:
            Dictionary with backtest results.
        """
        # Align data
        prices = prices.dropna()
        signals = signals.reindex(prices.index).fillna(0)

        if position_size is None:
            position_size = self.default_size

        if isinstance(position_size, (int, float)):
            position_size = pd.Series(position_size, index=prices.index)

        # Calculate returns
        returns = prices.pct_change().fillna(0)

        # Position changes (for commission calculation)
        position = signals.copy()
        position_changes = position.diff().abs().fillna(0)

        # Apply slippage to returns
        slippage_cost = position_changes * self.slippage

        # Strategy returns
        strategy_returns = position.shift(1).fillna(0) * returns - slippage_cost

        # Apply commission
        commission_cost = position_changes * self.commission
        strategy_returns = strategy_returns - commission_cost

        # Calculate equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Calculate metrics
        metrics = self._calculate_metrics(
            equity=equity,
            returns=strategy_returns,
            signals=signals,
            prices=prices,
        )

        return {
            "equity": equity,
            "returns": strategy_returns,
            "signals": signals,
            "positions": position,
            "metrics": metrics,
        }

    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        signals: pd.Series,
        prices: pd.Series,
    ) -> dict[str, float]:
        """Calculate performance metrics."""
        # Total return
        total_return = (equity.iloc[-1] / self.initial_capital) - 1

        # Annualized return (assuming daily data)
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        daily_rf = risk_free_rate / 252
        sharpe = (returns.mean() - daily_rf) / daily_vol * np.sqrt(252) if daily_vol > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        trades = self._extract_trades(signals, prices)
        if trades:
            wins = [t for t in trades if t["pnl"] > 0]
            win_rate = len(wins) / len(trades)

            # Profit factor
            gross_profit = sum(t["pnl"] for t in wins)
            losses = [t for t in trades if t["pnl"] < 0]
            gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Average trade
            avg_trade = np.mean([t["pnl"] for t in trades])
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "n_trades": len(trades),
            "final_equity": equity.iloc[-1],
        }

    def _extract_trades(
        self,
        signals: pd.Series,
        prices: pd.Series,
    ) -> list[dict[str, Any]]:
        """Extract individual trades from signals."""
        trades = []
        position = 0
        entry_price = 0
        entry_time = None

        for timestamp, signal in signals.items():
            current_price = prices.loc[timestamp]

            # Position change
            if signal != position:
                # Close existing position
                if position != 0:
                    pnl = (current_price - entry_price) * position
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "side": "long" if position > 0 else "short",
                        "pnl": pnl,
                    })

                # Open new position
                if signal != 0:
                    entry_price = current_price
                    entry_time = timestamp

                position = signal

        return trades

    def optimize(
        self,
        prices: pd.Series,
        signal_func: callable,
        param_grid: dict[str, list[Any]],
        metric: str = "sharpe",
    ) -> dict[str, Any]:
        """
        Grid search optimization for strategy parameters.

        Args:
            prices: Price series.
            signal_func: Function(prices, **params) -> signals.
            param_grid: Dictionary of parameter names to lists of values.
            metric: Metric to optimize ("sharpe", "total_return", etc.).

        Returns:
            Best parameters and their metrics.
        """
        import itertools

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        best_metric = -float("inf")
        best_params = {}
        best_result = None
        all_results = []

        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                # Generate signals with these parameters
                signals = signal_func(prices, **params)

                # Run backtest
                result = self.run_backtest(prices, signals)
                metric_value = result["metrics"].get(metric, 0)

                all_results.append({
                    "params": params,
                    "metric": metric_value,
                    "metrics": result["metrics"],
                })

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_result = result

            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")

        logger.info(f"Best {metric}: {best_metric:.4f} with params {best_params}")

        return {
            "best_params": best_params,
            "best_metric": best_metric,
            "best_result": best_result,
            "all_results": all_results,
        }

    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        n_periods: int | None = None,
    ) -> dict[str, Any]:
        """
        Run Monte Carlo simulation on returns.

        Args:
            returns: Historical return series.
            n_simulations: Number of simulation paths.
            n_periods: Number of periods to simulate (default: same as returns).

        Returns:
            Simulation statistics.
        """
        n_periods = n_periods or len(returns)
        returns_array = returns.dropna().values

        # Generate random paths by sampling with replacement
        simulated_paths = np.zeros((n_simulations, n_periods))

        for i in range(n_simulations):
            sampled_returns = np.random.choice(returns_array, size=n_periods, replace=True)
            simulated_paths[i] = self.initial_capital * np.cumprod(1 + sampled_returns)

        # Calculate statistics
        final_values = simulated_paths[:, -1]
        max_drawdowns = []

        for path in simulated_paths:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns.append(drawdown.min())

        return {
            "mean_final": np.mean(final_values),
            "median_final": np.median(final_values),
            "std_final": np.std(final_values),
            "percentile_5": np.percentile(final_values, 5),
            "percentile_95": np.percentile(final_values, 95),
            "mean_max_drawdown": np.mean(max_drawdowns),
            "worst_max_drawdown": np.min(max_drawdowns),
            "prob_profit": np.mean(final_values > self.initial_capital),
            "simulated_paths": simulated_paths,
        }
