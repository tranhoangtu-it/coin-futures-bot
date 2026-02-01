# 🦅 PET Trading Bot (ML-Powered)

**Advanced Algorithmic Trading System for Binance Futures**  
*Optimized for Profit Maximization, Risk Management, and Hardware Acceleration.*

---

## 🚀 Overview

The **PET Trading Bot** is a state-of-the-art automated trading engine designed to scan the crypto markets, identify high-probability setups using Machine Learning, and execute trades with professional risk management.

It has been optimized for high-performance hardware (**AMD Ryzen 7800X3D** + **Nvidia A2000**) to ensure maximum speed and continuous learning.

## ⚡ Key Features

### 🧠 1. Hybrid Intelligence Engine
Combines the predictive power of **Deep Learning** with the safety of **Technical Analysis**.
- **ML Ensemble**:
    -   **LSTM (PyTorch)**: Extracts temporal patterns from price action.
    -   **XGBoost (GPU)**: Classifies market regimes (Buy/Sell/Hold) with high precision using `cuda` acceleration.
- **Hybrid Filtering ("The Consensus")**:
    -   **Trend Gate**: Requires `ADX > 20` to trade (avoids "chop").
    -   **Momentum Safety**: Checks RSI to prevent buying tops (>70) or selling bottoms (<30).

### 🌍 2. Multi-Timeframe Scanning
The bot doesn't just look at one chart. It scans multiple time horizons simultaneously:
-   **15m (Scalping)**: Captures short-term volatility.
-   **1h (Day Trading)**: Captures intraday trends.
-   **4h (Swing Trading)**: Captures major market moves.

### 💰 3. Dynamic Profit Maximization
-   **Dynamic Leverage**: auto-adjusts risk based on timeframe volatility:
    -   `15m` -> **10x**
    -   `1h`  -> **5x**
    -   `4h`  -> **3x**
-   **Dynamic Trailing Stop**:
    -   **Break-Even**: Moves SL to entry when price moves **1x ATR** in favor.
    -   **Lock Profit**: Moves SL to lock gains when price moves **2x ATR** in favor.
-   **Active Management**: Continuously re-evaluates open positions. If the ML signal flips (e.g., Bullish to Bearish), the bot closes the trade immediately to save capital.

### 🏎️ 4. Hardware Acceleration
Tunneled for specific hardware:
-   **CPU (Ryzen 7800X3D)**: Massive concurrency enabled (`asyncio` Semaphore = 50) to scan 50+ pairs instantly.
-   **GPU (Nvidia A2000)**: XGBoost uses `device="cuda"` to offload heavy training logic to VRAM.

### 🔄 5. Continuous Learning
-   **Auto-Retraining**: A background service (`auto_retrain.py`) runs every **6 hours**.
-   It downloads fresh market data, retrains the model on the latest patterns, and updates the bot's "brain" automatically.

---

## 🛠️ Installation & Setup

### Prerequisites
-   Python 3.10+
-   Nvidia Drivers (for GPU acceleration)
-   Binance Futures Account (Testnet or Real)

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TRADING_SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT",...]
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Ensure torch and xgboost are installed with CUDA support
```

---

## 🖥️ Usage

### 1. Start the Market Scanner (The Bot)
This is the main trading engine.
```bash
python scripts/market_scanner.py
```
*Output: You will see "Scanning 15m...", "Scanning 1h...", and trade logs.*

### 2. Start Continuous Training (The Brain)
Run this in a separate terminal to keep the model smart.
```bash
python scripts/auto_retrain.py
```
*Output: "Starting Retraining Routine...", "Model updated".*

---

## 📊 Strategy Logic

1.  **Scan**: Fetch OHLCV data for all targets on 15m, 1h, 4h.
2.  **Extract**: Generate features (RSI, MACD, Volatility, LSTM latent vectors).
3.  **Predict**: XGBoost predicts probability of BUY, SELL, HOLD.
4.  **Filter**:
    -   Is `Relative_Score > 65%`?
    -   Is `ADX > 20`? (Trend exists)
    -   Is `RSI` safe? (Not overbought/sold)
5.  **Execute**:
    -   Calculate position size (Kelly Criterion / Risk limit).
    -   Set Leverage (Dynamically).
    -   Send Order to Binance.
6.  **Manage**:
    -   Monitor PnL.
    -   Trail Stop Loss.
    -   Close if Signal Flips or Stop/Target hit.

---

## ⚠️ Disclaimer
*This software is for educational purposes. Cryptocurrency trading involves high risk. Use at your own risk.*
