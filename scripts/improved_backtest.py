"""
Improved trading strategy with multiple indicators.
Uses trend-following approach with confirmation signals.
"""

import asyncio
import sys
from pathlib import Path

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


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multiple technical indicators."""
    result = df.copy()
    
    # EMAs for trend
    result["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    result["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    result["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # MACD
    result["macd"] = result["ema_fast"] - result["ema_slow"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    result["rsi"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    result["bb_upper"] = sma + 2 * std
    result["bb_lower"] = sma - 2 * std
    result["bb_position"] = (df["close"] - sma) / (2 * std)
    
    # ATR for volatility
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result["atr"] = tr.rolling(14).mean()
    
    # Volume ratio
    result["volume_sma"] = df["volume"].rolling(20).mean()
    result["volume_ratio"] = df["volume"] / result["volume_sma"]
    
    # Trend strength
    result["trend"] = np.where(df["close"] > result["ema_200"], 1, -1)
    
    return result


def generate_improved_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals using multi-indicator approach.
    
    Entry conditions for LONG:
    1. Price above EMA200 (uptrend)
    2. MACD histogram positive and rising
    3. RSI between 40-70 (momentum without overbought)
    4. Volume above average
    
    Entry conditions for SHORT:
    1. Price below EMA200 (downtrend)
    2. MACD histogram negative and falling
    3. RSI between 30-60 (momentum without oversold)
    4. Volume above average
    """
    indicators = calculate_indicators(df)
    signals = pd.Series(0, index=df.index)
    
    # Long conditions
    long_trend = indicators["close"] > indicators["ema_200"]
    long_macd = (indicators["macd_hist"] > 0) & (indicators["macd_hist"] > indicators["macd_hist"].shift(1))
    long_rsi = (indicators["rsi"] > 40) & (indicators["rsi"] < 70)
    long_volume = indicators["volume_ratio"] > 1.0
    long_entry = long_trend & long_macd & long_rsi & long_volume
    
    # Short conditions
    short_trend = indicators["close"] < indicators["ema_200"]
    short_macd = (indicators["macd_hist"] < 0) & (indicators["macd_hist"] < indicators["macd_hist"].shift(1))
    short_rsi = (indicators["rsi"] > 30) & (indicators["rsi"] < 60)
    short_volume = indicators["volume_ratio"] > 1.0
    short_entry = short_trend & short_macd & short_rsi & short_volume
    
    # Set signals
    signals[long_entry] = 1
    signals[short_entry] = -1
    
    # Hold position until opposite signal or RSI extreme
    current_pos = 0
    for i in range(len(signals)):
        if signals.iloc[i] != 0:
            current_pos = signals.iloc[i]
        else:
            # Exit on RSI extremes
            if current_pos == 1 and indicators["rsi"].iloc[i] > 75:
                signals.iloc[i] = 0
                current_pos = 0
            elif current_pos == -1 and indicators["rsi"].iloc[i] < 25:
                signals.iloc[i] = 0
                current_pos = 0
            else:
                signals.iloc[i] = current_pos
    
    return signals


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 10000,
    commission: float = 0.0004,
    leverage: float = 1.0,
) -> dict:
    """Run vectorized backtest with proper position tracking."""
    returns = prices.pct_change().fillna(0)
    position = signals.shift(1).fillna(0)
    position_changes = position.diff().abs().fillna(0)
    
    # Strategy returns (with leverage)
    strategy_returns = position * returns * leverage - position_changes * commission
    
    # Equity curve
    equity = initial_capital * (1 + strategy_returns).cumprod()
    
    # Metrics
    total_return = (equity.iloc[-1] / initial_capital) - 1
    n_periods = len(returns)
    annual_factor = 252 * 24 / n_periods  # Hourly data
    annual_return = (1 + total_return) ** annual_factor - 1
    
    # Sharpe
    daily_returns = strategy_returns.resample("D").sum()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
    
    # Sortino
    downside = daily_returns[daily_returns < 0]
    sortino = daily_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate and trades
    position_diff = signals.diff()
    entries = (position_diff != 0) & (signals != 0)
    exits = (position_diff != 0) & (signals == 0)
    
    trades = []
    entry_price = 0
    entry_direction = 0
    
    for i in range(1, len(signals)):
        if entries.iloc[i]:
            entry_price = prices.iloc[i]
            entry_direction = signals.iloc[i]
        elif entry_direction != 0 and (exits.iloc[i] or signals.iloc[i] != entry_direction):
            exit_price = prices.iloc[i]
            pnl = (exit_price - entry_price) / entry_price * entry_direction
            trades.append(pnl)
            if signals.iloc[i] != 0 and signals.iloc[i] != entry_direction:
                entry_price = prices.iloc[i]
                entry_direction = signals.iloc[i]
            else:
                entry_direction = 0
    
    win_rate = len([t for t in trades if t > 0]) / len(trades) if trades else 0
    avg_win = np.mean([t for t in trades if t > 0]) if [t for t in trades if t > 0] else 0
    avg_loss = abs(np.mean([t for t in trades if t < 0])) if [t for t in trades if t < 0] else 0
    profit_factor = avg_win * win_rate / (avg_loss * (1 - win_rate)) if avg_loss > 0 and win_rate < 1 else 0
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": len(trades),
        "final_equity": equity.iloc[-1],
        "equity_curve": equity,
        "signals": signals,
    }


async def main():
    """Main backtest function."""
    logger.info("=" * 60)
    logger.info("TradingBot Improved Strategy Backtest")
    logger.info("=" * 60)
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    all_results = []
    
    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'='*40}")
        
        try:
            df = await fetch_klines(symbol, limit=1000)
            logger.info(f"Data: {df.index[0]} to {df.index[-1]} ({len(df)} candles)")
            
            # Generate signals
            signals = generate_improved_signals(df)
            n_signals = (signals != 0).sum()
            logger.info(f"Signals: {n_signals} active periods")
            
            # Run backtest without leverage
            result = run_backtest(df["close"], signals, leverage=1.0)
            result["symbol"] = symbol
            all_results.append(result)
            
            logger.info(f"\nResults (1x leverage):")
            logger.info(f"  Total Return:    {result['total_return']:+.2%}")
            logger.info(f"  Annual Return:   {result['annual_return']:+.2%}")
            logger.info(f"  Sharpe Ratio:    {result['sharpe']:.2f}")
            logger.info(f"  Sortino Ratio:   {result['sortino']:.2f}")
            logger.info(f"  Max Drawdown:    {result['max_drawdown']:.2%}")
            logger.info(f"  Win Rate:        {result['win_rate']:.2%}")
            logger.info(f"  Profit Factor:   {result['profit_factor']:.2f}")
            logger.info(f"  Trades:          {result['n_trades']}")
            
        except Exception as e:
            logger.error(f"Error testing {symbol}: {e}")
    
    # Portfolio summary
    logger.info("\n" + "=" * 60)
    logger.info("PORTFOLIO SUMMARY")
    logger.info("=" * 60)
    
    avg_return = np.mean([r["total_return"] for r in all_results])
    avg_sharpe = np.mean([r["sharpe"] for r in all_results])
    avg_sortino = np.mean([r["sortino"] for r in all_results])
    avg_drawdown = np.mean([r["max_drawdown"] for r in all_results])
    avg_winrate = np.mean([r["win_rate"] for r in all_results])
    
    logger.info(f"Average Return:    {avg_return:+.2%}")
    logger.info(f"Average Sharpe:    {avg_sharpe:.2f}")
    logger.info(f"Average Sortino:   {avg_sortino:.2f}")
    logger.info(f"Average Drawdown:  {avg_drawdown:.2%}")
    logger.info(f"Average Win Rate:  {avg_winrate:.2%}")
    
    # Risk assessment
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY ASSESSMENT")
    logger.info("=" * 60)
    
    if avg_sharpe > 1.0:
        logger.info("✅ Excellent Sharpe ratio (>1.0)")
        strategy_grade = "A"
    elif avg_sharpe > 0.5:
        logger.info("⚡ Good Sharpe ratio (0.5-1.0)")
        strategy_grade = "B"
    elif avg_sharpe > 0:
        logger.info("⚠️  Marginal Sharpe ratio (0-0.5)")
        strategy_grade = "C"
    else:
        logger.warning("❌ Negative Sharpe - strategy loses money")
        strategy_grade = "F"
    
    if avg_return > 0:
        logger.info("✅ Positive total return")
    else:
        logger.warning("❌ Negative total return")
    
    if avg_winrate > 0.5:
        logger.info(f"✅ Win rate above 50% ({avg_winrate:.1%})")
    else:
        logger.info(f"⚡ Win rate below 50% ({avg_winrate:.1%}) - needs good R:R")
    
    logger.info(f"\nOverall Strategy Grade: {strategy_grade}")
    
    # Recommendation
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)
    
    if strategy_grade in ["A", "B"] and avg_return > 0:
        logger.info("✅ Strategy shows promise!")
        logger.info("📊 Recommended: Proceed with testnet trading")
        return True
    elif strategy_grade == "C" and avg_return > 0:
        logger.info("⚡ Strategy is marginal but profitable")
        logger.info("📊 Recommended: More optimization before testnet")
        return False
    else:
        logger.warning("❌ Strategy needs significant improvement")
        logger.warning("📊 Do NOT proceed with live trading")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
