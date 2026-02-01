"""
Quick backtest script with minimal dependencies.
Tests strategy on BTCUSDT using Binance public API.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import aiohttp
from loguru import logger


async def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch historical klines from Binance public API."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
    
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Generate trading signals based on RSI."""
    rsi = calculate_rsi(df["close"])
    
    signals = pd.Series(0, index=df.index)
    signals[rsi < 30] = 1   # Oversold -> Buy
    signals[rsi > 70] = -1  # Overbought -> Sell
    
    return signals


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 10000,
    commission: float = 0.0004,
) -> dict:
    """Run vectorized backtest."""
    returns = prices.pct_change().fillna(0)
    position = signals.shift(1).fillna(0)
    position_changes = position.diff().abs().fillna(0)
    
    # Strategy returns
    strategy_returns = position * returns - position_changes * commission
    
    # Equity curve
    equity = initial_capital * (1 + strategy_returns).cumprod()
    
    # Metrics
    total_return = (equity.iloc[-1] / initial_capital) - 1
    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 * 24 / n_days) - 1  # Hourly data
    
    # Sharpe
    daily_returns = strategy_returns.resample("D").sum()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    trades = []
    in_position = False
    entry_price = 0
    
    for i in range(1, len(signals)):
        if signals.iloc[i] != 0 and not in_position:
            in_position = True
            entry_price = prices.iloc[i]
        elif signals.iloc[i] == 0 and in_position:
            exit_price = prices.iloc[i]
            pnl = (exit_price - entry_price) / entry_price if signals.iloc[i-1] > 0 else (entry_price - exit_price) / entry_price
            trades.append(pnl)
            in_position = False
    
    win_rate = len([t for t in trades if t > 0]) / len(trades) if trades else 0
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_trades": len(trades),
        "final_equity": equity.iloc[-1],
    }


async def main():
    """Main backtest function."""
    logger.info("=" * 60)
    logger.info("TradingBot Quick Backtest")
    logger.info("=" * 60)
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    all_results = []
    
    for symbol in symbols:
        logger.info(f"\nFetching data for {symbol}...")
        df = await fetch_klines(symbol, limit=1000)
        logger.info(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Generate signals
        signals = generate_signals(df)
        n_signals = (signals != 0).sum()
        logger.info(f"Generated {n_signals} trading signals")
        
        # Run backtest
        result = run_backtest(df["close"], signals)
        result["symbol"] = symbol
        all_results.append(result)
        
        logger.info(f"\n{symbol} Results:")
        logger.info(f"  Total Return:    {result['total_return']:+.2%}")
        logger.info(f"  Annual Return:   {result['annual_return']:+.2%}")
        logger.info(f"  Sharpe Ratio:    {result['sharpe']:.2f}")
        logger.info(f"  Max Drawdown:    {result['max_drawdown']:.2%}")
        logger.info(f"  Win Rate:        {result['win_rate']:.2%}")
        logger.info(f"  Number of Trades: {result['n_trades']}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PORTFOLIO SUMMARY")
    logger.info("=" * 60)
    
    avg_return = np.mean([r["total_return"] for r in all_results])
    avg_sharpe = np.mean([r["sharpe"] for r in all_results])
    avg_drawdown = np.mean([r["max_drawdown"] for r in all_results])
    
    logger.info(f"Average Return:    {avg_return:+.2%}")
    logger.info(f"Average Sharpe:    {avg_sharpe:.2f}")
    logger.info(f"Average Drawdown:  {avg_drawdown:.2%}")
    
    # Risk assessment
    logger.info("\n" + "=" * 60)
    logger.info("RISK ASSESSMENT")
    logger.info("=" * 60)
    
    if avg_sharpe < 0.5:
        logger.warning("⚠️  Low Sharpe ratio - strategy needs improvement")
    elif avg_sharpe < 1.0:
        logger.info("⚡ Moderate Sharpe - acceptable but room for improvement")
    else:
        logger.info("✅ Good Sharpe ratio")
    
    if abs(avg_drawdown) > 0.2:
        logger.warning("⚠️  High drawdown risk - reduce position sizes")
    
    # Recommendation
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)
    
    if avg_sharpe > 0.5 and avg_return > 0:
        logger.info("📊 Strategy shows promise. Recommend proceeding with TESTNET.")
    else:
        logger.warning("📊 Strategy needs optimization before live trading.")
    
    return all_results


if __name__ == "__main__":
    asyncio.run(main())
