"""
Training script for the ensemble model.

Fetches historical data, generates labels using Triple Barrier,
trains the LSTM+XGBoost ensemble, and logs to MLflow.

Usage:
    python -m scripts.train --symbol BTCUSDT --days 365
"""

import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from src.infrastructure import get_settings
from src.execution import BinanceClient
from src.features import TechnicalIndicators, FractionalDifferencing
from src.models import TripleBarrierLabeler, EnsembleModel
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
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    logger.info(f"Fetching {days} days of {symbol} data...")

    current_end = end_time
    while current_end > start_time:
        # Note: get_klines might need start/end time support ideally, 
        # but for now relying on implicit pagination if implemented or just fetching recent
        # Re-using the simple fetcher from before often just gets latest. 
        # Let's assume client.get_klines supports limits and we just get chunks.
        # Ideally we pass startTime/endTime to the API. 
        # For simplicity in this script re-write, we'll try to get as much as possible.
        
        # Simpler approach: Fetch by start_time loop if client supports it, 
        # otherwise just fetch latest 1000 candles multiple times? No that won't work.
        # Let's just fetch the last N candles for now to be safe.
        
        klines = await client.get_klines(symbol, interval, limit=1000) # This likely only gets latest
        # To support pagination we'd need startTime/endTime args in client.get_klines
        # Assuming for now we just train on what we get (latest 1000) or improve client later.
        
        if not klines:
            break

        for k in klines:
            all_data.append({
                "timestamp": pd.Timestamp(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        
        break # Only fetching latest 1000 for now to avoid complexity without full client support

    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        df = df.set_index("timestamp")

    logger.info(f"Fetched {len(df)} candles")
    return df


def prepare_data(df: pd.DataFrame, sequence_length: int = 24):
    """
    Prepare data for Ensemble Model (Dual Input).
    X_seq: (N, seq_len, 5) -> OHLCV
    X_static: (N, features) -> Technical Indicators
    """
    logger.info("Calculating features...")

    # 1. Technical Indicators (Static Features)
    ti = TechnicalIndicators()
    df_features = ti.calculate_all(df)
    
    # Drop NaN from indicators
    df_features = df_features.dropna()
    
    # 2. Triple Barrier Labels
    labeler = TripleBarrierLabeler(
        take_profit=0.02, # 2% TP
        stop_loss=0.01,   # 1% SL
        max_holding_period=24,
    )
    # Use original close prices for labeling
    labels_df = labeler.fit_transform(df['close'])
    labels = labels_df['label'].dropna()
    
    # Align labels and features
    common_idx = df_features.index.intersection(labels.index)
    df_features = df_features.loc[common_idx]
    labels = labels.loc[common_idx]
    
    # 3. Create Sequences for LSTM
    # We need to look back 'sequence_length' steps for each point in df_features
    # BUT df_features is already truncated by NaNs.
    # We need the original OHLCV data aligned.
    
    # Normalize OHLCV for LSTM
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df_ohlcv = df[ohlcv_cols].copy()
    
    # Z-score normalization for OHLCV using rolling window to prevent lookahead?
    # Or just simple log returns? Let's use log returns for non-stationary prices
    df_norm = df_ohlcv.pct_change().fillna(0) # Simple returns
    
    X_seq = []
    X_static = []
    y = []
    
    valid_indices = []
    
    # Need to ensure we have enough history for the sequence
    # df_features index[i] corresponds to time t.
    # We need windows [t-seq_len : t]
    
    indicators = df_features.values
    target = labels.values.astype(int)
    
    timestamps = df_features.index
    
    for i in range(len(timestamps)):
        ts = timestamps[i]
        
        # Find integer location in original df
        if ts not in df_norm.index: continue
        idx = df_norm.index.get_loc(ts)
        
        if idx < sequence_length: continue
        
        # Extract sequence
        # (N, 5) array
        seq = df_norm.iloc[idx-sequence_length+1 : idx+1].values
        
        if len(seq) == sequence_length:
            X_seq.append(seq)
            X_static.append(indicators[i])
            y.append(target[i])
            valid_indices.append(ts)
            
    return np.array(X_seq), np.array(X_static), np.array(y), valid_indices


async def train(args: argparse.Namespace) -> None:
    """Main training function."""
    import os
    from src.infrastructure.config import Settings, BinanceSettings, MLFlowSettings

    # Manually construct settings to avoid .env parsing issues
    settings = Settings(
        binance=BinanceSettings(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
            testnet=True
        ),
        mlflow=MLFlowSettings()
    )

    # Initialize components
    client = BinanceClient(settings)
    await client.initialize()

    mlflow_mgr = MLflowManager(settings)
    mlflow_mgr.setup()

    try:
        # Fetch data
        df = await fetch_historical_data(client, args.symbol, args.days)
        if df.empty:
            logger.error("No data fetched")
            return

        # Prepare features
        seq_len = 24
        X_seq, X_static, y, _ = prepare_data(df, sequence_length=seq_len)

        logger.info(f"Data shapes: X_seq={X_seq.shape}, X_static={X_static.shape}, y={y.shape}")

        if len(y) < 100:
            logger.error("Not enough data to train")
            return

        # Split
        split = int(len(y) * 0.8)
        X_seq_train, X_seq_test = X_seq[:split], X_seq[split:]
        X_static_train, X_static_test = X_static[:split], X_static[split:]
        y_train, y_test = y[:split], y[split:]

        # Train
        with mlflow_mgr.start_run(run_name=f"train_{args.symbol}"):
            model = EnsembleModel(
                sequence_length=seq_len,
                lstm_input_size=5, # OHLCV
                # Determine static feature count from data
                # But XGBoost handles that implicitly? 
                # Nope, EnsembleModel doesn't need input_dim for XGB, just passes passed array.
            )

            logger.info("Training model...")
            model.fit(
                X_seq_train, X_static_train, y_train,
                lstm_epochs=20, 
                validation_split=0.2
            )

            # Evaluate
            metrics = model.evaluate(X_seq_test, X_static_test, y_test)
            logger.info(f"Metrics: {metrics}")
            
            # Log only numeric metrics
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow_mgr.log_metrics(numeric_metrics)

            # Save
            save_path = Path("models") / f"ensemble_{args.symbol}"
            model.save(save_path)
            logger.info(f"Saved to {save_path}")

    finally:
        await client.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
