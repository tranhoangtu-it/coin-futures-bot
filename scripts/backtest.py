"""
Backtesting script for strategy evaluation.

Runs walk-forward validation and generates performance reports.

Usage:
    python -m scripts.backtest --symbol BTCUSDT --days 180
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.infrastructure import get_settings
from src.execution import BinanceClient
from src.features import TechnicalIndicators
from src.backtest import WalkForwardValidator, VectorbtEngine, PerformanceMetrics
from src.mlops import MLflowManager


async def fetch_historical_data(
    client: BinanceClient,
    symbol: str,
    days: int,
    interval: str = "1h",
) -> pd.DataFrame:
    """Fetch historical klines data."""
    all_data = []
    limit = 1000

    logger.info(f"Fetching {days} days of {symbol} data...")

    klines = await client.get_klines(symbol, interval, limit=min(days * 24, 1500))

    for k in klines:
        all_data.append({
            "timestamp": pd.Timestamp(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")

    logger.info(f"Fetched {len(df)} candles")
    return df


def generate_signals(prices: pd.Series, rsi_period: int = 14, rsi_oversold: int = 30, rsi_overbought: int = 70) -> pd.Series:
    """Generate signals based on RSI."""
    ti = TechnicalIndicators()

    # Calculate RSI
    rsi = ti.rsi(prices, period=rsi_period)

    # Generate signals: 1 = long, -1 = short, 0 = flat
    signals = pd.Series(0, index=prices.index)
    signals[rsi < rsi_oversold] = 1  # Oversold -> Buy
    signals[rsi > rsi_overbought] = -1  # Overbought -> Sell

    return signals


async def backtest(args: argparse.Namespace) -> None:
    """Main backtest function."""
    settings = get_settings()

    # Initialize
    client = BinanceClient(settings)
    await client.initialize()

    mlflow_mgr = MLflowManager(settings)
    mlflow_mgr.setup()

    try:
        # Fetch data
        df = await fetch_historical_data(client, args.symbol, args.days)
        prices = df["close"]

        # Run vectorized backtest
        logger.info("Running vectorized backtest...")

        engine = VectorbtEngine(
            initial_capital=10000,
            commission=0.0004,
            slippage=0.0001,
        )

        # Generate signals
        signals = generate_signals(prices)

        # Run backtest
        result = engine.run_backtest(prices, signals)
        metrics = result["metrics"]

        logger.info("=" * 50)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

        # Run walk-forward validation
        logger.info("\nRunning walk-forward validation...")

        validator = WalkForwardValidator(
            n_folds=5,
            train_ratio=0.7,
            purge_gap=24,  # 24 hours gap
        )

        def train_model(train_data: pd.DataFrame):
            """Train model stub."""
            return {"rsi_period": 14, "threshold": 30}

        def evaluate(model, test_data: pd.DataFrame) -> dict:
            """Evaluate model."""
            signals = generate_signals(test_data["close"])
            bt_result = engine.run_backtest(test_data["close"], signals)
            return bt_result["metrics"]

        wf_result = validator.validate(df, train_model, evaluate)

        logger.info("\nWalk-Forward Results:")
        logger.info(f"Folds: {wf_result.n_folds}")
        logger.info(f"Avg Test Sharpe: {wf_result.avg_test_metrics.get('sharpe', 0):.4f}")
        logger.info(f"Stability: {wf_result.overall_stability:.2%}")

        # Calculate additional metrics
        pm = PerformanceMetrics()
        equity = result["equity"]
        detailed_metrics = pm.calculate(equity=equity)

        # Drawdown analysis
        drawdowns = pm.analyze_drawdowns(equity, top_n=3)
        logger.info("\nTop 3 Drawdowns:")
        for dd in drawdowns:
            logger.info(f"  {dd.start} - {dd.end}: {dd.depth:.2%}")

        # Monte Carlo simulation
        logger.info("\nRunning Monte Carlo simulation...")
        mc_result = engine.monte_carlo_simulation(
            result["returns"],
            n_simulations=1000,
        )
        logger.info(f"Mean final equity: {mc_result['mean_final']:.2f}")
        logger.info(f"5th percentile: {mc_result['percentile_5']:.2f}")
        logger.info(f"95th percentile: {mc_result['percentile_95']:.2f}")
        logger.info(f"Probability of profit: {mc_result['prob_profit']:.2%}")

        # Log to MLflow
        with mlflow_mgr.start_run(run_name=f"backtest_{args.symbol}_{datetime.now().strftime('%Y%m%d')}"):
            mlflow_mgr.log_params({
                "symbol": args.symbol,
                "days": args.days,
                "strategy": "RSI",
            })
            mlflow_mgr.log_metrics(metrics)
            mlflow_mgr.log_metrics({
                "wf_stability": wf_result.overall_stability,
                "mc_prob_profit": mc_result["prob_profit"],
            })

        logger.info("\nBacktest complete!")

    finally:
        await client.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=180, help="Days of history")

    args = parser.parse_args()

    logger.info(f"Starting backtest for {args.symbol}")
    asyncio.run(backtest(args))


if __name__ == "__main__":
    main()
