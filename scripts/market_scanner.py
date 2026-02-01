"""
ML-Powered Market Scanner Bot for Binance Testnet.

Upgrades:
- Uses trained EnsembleModel (LSTM+XGBoost) for signals
- Generalizes BTC model to all pairs
- Maintains precision and margin safety
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
import os
import hashlib
import hmac
import time
import random
import re
import math
import warnings

# Suppress pandas/torch warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import aiohttp
from loguru import logger
from xgboost import XGBClassifier

from src.features.technical_indicators import TechnicalIndicators
from src.risk.kelly_criterion import KellyCriterion
from src.risk.trailing_stop import TrailingStopManager, PositionSide
import torch
from loguru import logger
from dotenv import load_dotenv

from src.features import TechnicalIndicators
from src.models import EnsembleModel

load_dotenv()


class MLScanner:
    """Scans market using ML Ensemble Model."""
    
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.base_url = "https://testnet.binancefuture.com"
        
        self.leverage = 3
        self.max_positions = 12
        self.risk_per_trade = 0.01
        
        self.positions = {}
        self.session = None
        self.symbol_info = {}
        # Hardware Optimization: Ryzen 7800X3D can handle massive concurrency
        self.sem = asyncio.Semaphore(50)
        
        # Load ML Model
        self.model = None
        self.ti = TechnicalIndicators()
        self.seq_len = 24
        self.model_path = Path("models") / "ensemble_BTCUSDT"
        self.state_file = Path("data") / "live_state.json"
        
        # Ensure data folder exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Dashboard Data
        self.recent_trades = []
        self.sym_logs = {} 
        self.start_balance = 0.0 # Will be set on init
        self.session_start = time.time()
        
        # Risk Management (Production Integration)
        self.kelly = KellyCriterion(fractional_factor=0.3, max_position_pct=0.2)
        self.stops_mgr = TrailingStopManager(atr_multiplier=2.0)
        
        # We need to feed recent trades to Kelly to establish a baseline
        # (Ideally this loads from DB, but for now we start fresh or from state)

    def save_state(self):
        """Save live state for dashboard."""
        try:
            # Calculate Unrealized PnL
            unrealized_pnl = 0.0
            for sym, pos in self.positions.items():
                # We need current price, but might not have it instantly.
                # Use entry price as placeholder if live price unknown, 
                # or better, dashboard calculates it. 
                # For now, just save positions as is.
                pass

            state = {
                "timestamp": time.time(),
                "balance": self.virtual_balance, # Show Virtual Balance
                "start_balance": 100.0,
                "positions": self.positions,
                "recent_trades": self.recent_trades[-50:], 
                "logs": self.sym_logs, 
                "active": True
            }
            
            # Atomic write
            import json
            temp = self.state_file.with_suffix(".tmp")
            with open(temp, "w") as f:
                json.dump(state, f, indent=2, default=str)
            temp.replace(self.state_file)
            
        except Exception as e:
            logger.error(f"State save failed: {e}")
    
    async def init(self):
        # Load Model
        try:
            if self.model_path.exists():
                logger.info(f"🧠 Loading ML Model from {self.model_path}...")
                self.model = EnsembleModel(sequence_length=self.seq_len, lstm_input_size=5)
                self.model.load(self.model_path)
                logger.info("✅ Model Loaded Successfully")
            else:
                logger.error(f"❌ Model not found at {self.model_path}. Please train first.")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

        self.session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": self.api_key})
        try:
            account = await self._req("GET", "/fapi/v2/account", signed=True)
            balance = float(account.get("totalWalletBalance", 0))
            self.balance = balance
            self.start_balance = balance # Set initial session balance
            logger.info(f"✅ Connected | Balance: {balance:.2f} USDT")
            self.save_state()
            
            info = await self._req("GET", "/fapi/v1/exchangeInfo")
            count = 0
            for s in info["symbols"]:
                if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and re.match(r"^[A-Z0-9]+$", s["symbol"]):
                    step = 0
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            break
                    if step > 0:
                        self.symbol_info[s["symbol"]] = {
                            "step_size": step,
                            "precision": int(round(-math.log(step, 10), 0))
                        }
                        count += 1
            
            # Boost API Concurrency for "Global Scan"
            self.sem = asyncio.Semaphore(100) 
            
            # Virtual Account Reset ($100 Start)
            self.virtual_balance = 100.0 
            self.start_balance = 100.0
            
            logger.info(f"🔍 Loaded precision for {count} pairs")
            return True
        except Exception as e:
            logger.error(f"Init error: {e}")
            return False

    def _sign(self, p):
        p["timestamp"] = int(time.time() * 1000)
        q = "&".join(f"{k}={v}" for k, v in p.items())
        p["signature"] = hmac.new(self.api_secret.encode(), q.encode(), hashlib.sha256).hexdigest()
        return p
    
    async def _req(self, m, e, params=None, signed=False):
        url = f"{self.base_url}{e}"
        p = params or {}
        if signed: p = self._sign(p)
        for attempt in range(3):
            try:
                async with self.session.request(m, url, params=p) as r:
                    if r.status == 429:
                        await asyncio.sleep(2 + attempt)
                        continue
                    d = await r.json()
                    if r.status != 200:
                        if attempt == 2: logger.error(f"API Error {d.get('code')}: {d.get('msg')}")
                        raise Exception(f"{d.get('msg')}")
                    return d
            except Exception as e:
                if attempt == 2: raise e
                await asyncio.sleep(0.5)

    def round_qty(self, symbol, qty):
        if symbol not in self.symbol_info: return qty
        step = self.symbol_info[symbol]["step_size"]
        precision = self.symbol_info[symbol]["precision"]
        return round(round(qty / step) * step, precision)

    async def get_klines(self, symbol, interval="15m", limit=500): 
        try:
            d = await self._req("GET", "/fapi/v1/klines", 
                params={"symbol": symbol, "interval": interval, "limit": limit})
            # DEBUG CHECK
            # logger.info(f"Got {len(d)} lines for {symbol}")
            
            # Explicitly naming columns standard names
            df = pd.DataFrame(d, columns=["t","open","high","low","close","volume","ct","qv","tr","tbv","tbqv","i"])
            for col in ["open","high","low","close","volume"]:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            # logger.error(f"Klines error {symbol}: {e}")
            return None

    def analyze(self, symbol, df):
        """Analyze using ML Model."""
        if df is None: return 0, 0
        if len(df) < 100: return 0, 0
        
        # DEBUG: Log columns if needed
        # logger.info(f"{symbol} cols: {df.columns.tolist()}")
        
        # 1. Prepare Features (Static)
        try:
            # Using defaults which are open, high, low, close...
            features = self.ti.calculate_all(df)
            current_features = features.iloc[-1].values.reshape(1, -1)
            
            if np.isnan(current_features).any():
                return 0, 0

            # 2. Prepare Sequence (LSTM)
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            df_norm = df[ohlcv_cols].pct_change().fillna(0)
            
            if len(df_norm) < self.seq_len: return 0, 0
            
            seq = df_norm.iloc[-self.seq_len:].values.reshape(1, self.seq_len, 5)
            
            # 3. Predict
            probs = self.model.predict_proba(seq, current_features)[0]
            p_sell, p_hold, p_buy = float(probs[0]), float(probs[1]), float(probs[2])
            
            # Relative Scoring (Balanced for Win Rate)
            sum_active = p_buy + p_sell
            ml_signal = 0
            
            # Threshold: 3% minimum active probability (Avoid noise)
            if sum_active > 0.03: 
                relative_buy = p_buy / sum_active
                if relative_buy > 0.70: ml_signal = 1   # High conviction BUY
                elif relative_buy < 0.30: ml_signal = -1 # High conviction SELL
            
            # 4. HYBRID FILTERS ("Combine Everything")
            # Extract latest indicators
            latest = features.iloc[-1]
            adx = latest.get("adx", 0)
            rsi = latest.get("rsi_14", 50)
            vol_curr = df["volume"].iloc[-1]
            vol_ma = df["volume"].rolling(20).mean().iloc[-1]
            
            final_signal = 0
            
            if ml_signal != 0:
                # Filter 1: Trend Strength (ADX)
                # Stronger Trend Requirement > 20
                if adx > 20: 
                    
                    # Filter 2: RSI Validation (Don't buy tops, Don't sell bottoms)
                    if ml_signal == 1: # BUY
                        if rsi < 70: # Room to grow
                             final_signal = 1
                    elif ml_signal == -1: # SELL
                        if rsi > 30: # Room to drop
                             final_signal = -1
                    
                    # Filter 3: Volume Confirmation (Optional but good)
                    # if vol_curr < vol_ma: final_signal = 0 # Strict enforcement
            
            msg = f"ML:{ml_signal} (RelBuy:{p_buy/(sum_active+1e-9):.2f}, SumAct:{sum_active:.4f}) | ADX:{adx:.1f} RSI:{rsi:.1f} -> {final_signal}"
            self.sym_logs[symbol] = msg
            logger.info(f"🔍 {symbol} {msg}")

            if final_signal != 0:
                 # Recalculate ATR
                 c = df["close"].values
                 h = df["high"].values
                 l = df["low"].values
                 tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
                 atr = np.mean(tr[-14:])
                 if atr == 0: atr = c[-1] * 0.01
                 
                 return final_signal, atr
            
            return 0, 0

        except Exception as e:
            logger.warning(f"Analysis failed for {symbol}: {e}")
            return 0, 0

    async def scan_symbol(self, symbol, interval):
        async with self.sem:
            # Unique key for position: symbol (assuming 1 pos per symbol for now, 
            # ideally would be symbol_tf, but keeping it simple to avoid hedging conflicts)
            if symbol in self.positions: return
            if len(self.positions) >= self.max_positions: return
            
            # Fetch data for specific timeframe
            df = await self.get_klines(symbol, interval=interval, limit=500)
            if df is None: return
            
            signal, atr = self.analyze(symbol, df)
            
            if signal != 0:
                # VIRTUAL BALANCE LOGIC
                # "Mặc kệ ví có bao nhiêu" -> Ignore real wallet, trade as if we only have $100
                avail_bal = self.virtual_balance 
                if avail_bal < 10: return # Hard floor $10
                
                current_price = df["close"].iloc[-1]
                if current_price == 0: return

                risk_amt = avail_bal * self.risk_per_trade
                stop_dist = atr * 2.0
                if stop_dist == 0: return
                
                # Dynamic Leverage based on Timeframe
                leverage_map = {
                    "1m": 25,   # Nitro Scalping
                    "3m": 20,   # Turbo Scalping
                    "5m": 15,   # Fast Scalping
                    "15m": 10,  # Scalping
                    "1h": 5,    # Day Trading
                    "4h": 3,    # Swing
                    "12h": 2,   # Trend
                    "1d": 2     # Macro
                }
                # Kelly Criterion Sizing (Virtual Wallet)
                # We need to estimate Win Prob. The model gives us probabilities.
                probs = self.model.predict_proba(seq, current_features)[0]
                p_buy = float(probs[2])
                p_sell = float(probs[0])
                win_prob = p_buy if side == "BUY" else p_sell
                
                # Mock average win/loss ratio (Reward/Risk) if no history
                # We target 2:1 RR
                avg_win = 0.02 # 2% target
                avg_loss = 0.01 # 1% stop
                
                # If we have trade history, refine this
                if len(self.recent_trades) > 10:
                    df_t = pd.DataFrame(self.recent_trades)
                    if "pnl" in df_t.columns:
                        wins = df_t[df_t["pnl"] > 0]["pnl"].mean()
                        losses = abs(df_t[df_t["pnl"] <= 0]["pnl"].mean())
                        if not np.isnan(wins) and not np.isnan(losses) and losses > 0:
                            avg_win = wins
                            avg_loss = losses
                
                # Calculate Kelly Fraction
                kelly_frac = self.kelly.calculate(win_prob, avg_win, avg_loss)
                
                # Get Size based on VIRTUAL Balance
                qty = self.kelly.get_position_size_with_leverage(
                    account_balance=self.virtual_balance,
                    entry_price=current_price,
                    leverage=leverage,
                    kelly_fraction=kelly_frac
                )
                
                # Fallback / Min Limit
                if qty * current_price < 10:
                     # If Kelly says 0 or too small, but signal is strong... 
                     # we respect Kelly and SKIP, unless it's just a "startup" issue.
                     # For now, force min size if signal is very strong (>80%)
                     if win_prob > 0.8:
                         qty = 11 / current_price
                     else:
                         return
                
                qty = self.round_qty(symbol, qty)
                if qty <= 0: return

                logger.info(f"🧠 ML Signal {symbol} [{interval}]: {side} (Kelly={kelly_frac:.2f}, Lev={leverage}x)")
                
                await self.execute_trade(symbol, side, qty, current_price, atr, leverage)

    async def execute_trade(self, symbol, side, qty, current_price, atr, leverage):
        try:
            await self._req("POST", "/fapi/v1/leverage", 
                params={"symbol": symbol, "leverage": leverage}, signed=True)
            
            r = await self._req("POST", "/fapi/v1/order",
                params={"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty},
                signed=True)
            
            fill_price = float(r.get("avgPrice", 0))
            if fill_price == 0: fill_price = current_price
            
            if fill_price > 0:
                tp_dist = atr * 4.0 
                sl_dist = atr * 2.0 
                
                tp_price = fill_price + tp_dist if side == "BUY" else fill_price - tp_dist
                sl_price = fill_price - sl_dist if side == "BUY" else fill_price + sl_dist
                
                # Register with Trailing Stop Manager
                pos_side = PositionSide.LONG if side == "BUY" else PositionSide.SHORT
                stop_level = self.stops_mgr.create_stop(symbol, pos_side, fill_price, atr)
                sl_price = stop_level.current_stop # Use the manager's calculated stop
                
                self.positions[symbol] = {
                    "side": side,
                    "entry": fill_price,
                    "qty": qty,
                    "tp": tp_price,
                    "sl": sl_price,
                    "leverage": leverage 
                }
                
                # Dashboard Log
                self.recent_trades.append({
                    "timestamp": time.time(),
                    "symbol": symbol,
                    "side": side,
                    "price": fill_price,
                    "qty": qty,
                    "leverage": leverage
                })
                self.save_state()
                
                logger.info(f"🤖 ML ENTRY {side} {symbol} @ {fill_price} [Lev {leverage}x] | Stop: {sl_price:.4f}")
            return True
        except Exception as e:
            logger.error(f"Trade failed {symbol}: {e}")
            return False

    async def monitor_positions(self):
        if not self.positions: return
        
        # Create tasks for concurrent monitoring
        tasks = [self._check_position(symbol) for symbol in list(self.positions.keys())]
        await asyncio.gather(*tasks)

    async def _check_position(self, symbol):
        async with self.sem:
            if symbol not in self.positions: return
            
            try:
                # Upgrade: Fetch full klines to re-evaluate with ML
                df = await self.get_klines(symbol, limit=500)
                if df is None: return
                
                # 1. Check PnL (TP/SL) & Trailing Stop
                curr = df["close"].iloc[-1]
                pos = self.positions[symbol]
                
                # Use stored leverage or default
                lev = pos.get("leverage", self.leverage)
                
                close_signal = False
                pnl = 0
                entry = pos["entry"]
                sl = pos["sl"]
                
                # Calculate ATR distance (original distance approx)
                # We can approximate current ATR from the dataframe if needed, 
                # or carry it. For now, let's infer 'stop_dist' from Entry/SL
                stop_dist = abs(entry - sl)
                
                # 1. Check PnL (TP/SL) & Trailing Stop
                if pos["side"] == "BUY":
                    pnl = (curr - entry) / entry * 100 * lev
                else: 
                    pnl = (entry - curr) / entry * 100 * lev
                
                # Update Trailing Stop Manager (Handles all "High Water Mark" logic)
                self.stops_mgr.update_stop(symbol, curr)
                
                # Check if Stop Hit
                if self.stops_mgr.is_stopped(symbol, curr):
                    logger.info(f"🛑 Trailing Stop hit for {symbol} at {curr}")
                    close_signal = True
                
                # Update SL visual on dashboard
                current_stop = self.stops_mgr.get_stop_price(symbol)
                if current_stop: 
                    self.positions[symbol]["sl"] = current_stop

                # Check TP (Hard TP is still valid as specific target)
                if (pos["side"] == "BUY" and curr >= pos["tp"]) or \
                   (pos["side"] == "SELL" and curr <= pos["tp"]):
                       logger.info(f"🎯 Take Profit hit for {symbol} at {curr}")
                       close_signal = True
                
                # 2. AI Re-evaluation (Signal Flip)
                if not close_signal:
                    ml_signal, _ = self.analyze(symbol, df)
                    if pos["side"] == "BUY" and ml_signal == -1:
                        logger.info(f"🔄 ML Flip {symbol}: BUY -> SELL Signal. Closing early.")
                        close_signal = True
                    elif pos["side"] == "SELL" and ml_signal == 1:
                        logger.info(f"🔄 ML Flip {symbol}: SELL -> BUY Signal. Closing early.")
                        close_signal = True

                # 3. Execute Close
                if close_signal:
                    side = "SELL" if pos["side"] == "BUY" else "BUY"
                    await self._req("POST", "/fapi/v1/order",
                        params={"symbol": symbol, "side": side, "type": "MARKET", "quantity": pos["qty"]},
                        signed=True)
                    
                    # Virtual PnL Calculation
                    # Profit = (Exit - Entry) * Qty if Long, (Entry - Exit) * Qty if Short
                    curr_price_float = float(klines.iloc[-1]["close"])
                    entry_price = float(pos["entry"])
                    qty_float = float(pos["qty"])
                    
                    if pos["side"] == "BUY":
                        profit = (curr_price_float - entry_price) * qty_float
                    else:
                        profit = (entry_price - curr_price_float) * qty_float
                        
                    self.virtual_balance += profit
                    
                    # Log Close Event for Dashboard
                    self.recent_trades.append({
                        "timestamp": time.time(),
                        "symbol": symbol,
                        "side": "CLOSE", 
                        "pnl": profit, 
                        "price": curr_price_float,
                        "qty": qty_float,
                        "leverage": pos["leverage"]
                    })
                    
                    icon = "✅" if profit > 0 else "🛑"
                    logger.info(f"{icon} CLOSE {symbol} | PnL: ${profit:.2f} ({(pnl):.2f}%) | V.Bal: ${self.virtual_balance:.2f}")
                    
                    # Cleanup Stop Manager
                    self.stops_mgr.close_position(symbol)
                    
                    if symbol in self.positions:
                        del self.positions[symbol]
                        
            except Exception as e:
                logger.error(f"Monitor error {symbol}: {e}")

    async def run_scan_loop(self, timeframes, loop_name, delay=0):
        """Generic loop for scanning specific timeframes."""
        while True:
            try:
                # Refresh targets periodically
                if loop_name == "SLOW":
                   tickers = await self._req("GET", "/fapi/v1/ticker/24hr")
                   valid = [t for t in tickers if t["symbol"] in self.symbol_info]
                   sorted_valid = sorted(valid, key=lambda x: float(x["quoteVolume"]), reverse=True)
                   self.targets = [t["symbol"] for t in sorted_valid] 
                
                # Use shared targets (start with top volume for fast, but shuffle)
                targets = self.targets[:] if hasattr(self, 'targets') else []
                if not targets:
                    logger.warning(f"{loop_name}: No targets yet...")
                    await asyncio.sleep(5)
                    continue

                random.shuffle(targets)
                logger.info(f"🚀 {loop_name} SCAN: {len(targets)} symbols | TFs: {timeframes}")

                for interval in timeframes:
                     # Check connection/positions first
                    if len(self.positions) >= self.max_positions: 
                        await asyncio.sleep(5)
                        break

                    logger.info(f"🔎 {loop_name} Layer taking {interval}...")
                    
                    # Split into chunks to avoid congestion even with sem=50
                    chunk_size = 50
                    for i in range(0, len(targets), chunk_size):
                        chunk = targets[i:i+chunk_size]
                        tasks = [self.scan_symbol(s, interval) for s in chunk]
                        await asyncio.gather(*tasks)
                        self.save_state() # Heartbeat
                    
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"{loop_name} Loop Error: {e}")
                await asyncio.sleep(5)

    async def run(self):
        logger.info("=" * 60)
        logger.info("🧠 ML-POWERED MARKET SCANNER - DUAL LAYER")
        logger.info("=" * 60)
        
        if not await self.init(): return
        
        # Initialize targets once
        tickers = await self._req("GET", "/fapi/v1/ticker/24hr")
        valid = [t for t in tickers if t["symbol"] in self.symbol_info]
        self.targets = [t["symbol"] for t in sorted(valid, key=lambda x: float(x["quoteVolume"]), reverse=True)]
        
        # Layer 1: Hyper-Speed (Scalping)
        fast_tfs = ["1m", "3m", "5m"]
        
        # Layer 2: Macro-Trend (Analysis)
        slow_tfs = ["15m", "1h", "4h", "12h", "1d"]
        
        # Launch concurrent loops
        await asyncio.gather(
            self.monitor_positions_loop(),
            self.run_scan_loop(fast_tfs, "FAST", delay=1),
            self.run_scan_loop(slow_tfs, "SLOW", delay=30)
        )

    async def monitor_positions_loop(self):
        """Dedicated loop for position management."""
        while True:
            try:
                await self.monitor_positions()
                await self.get_available_balance() # Keep balance updated
                await asyncio.sleep(1) # Check every second
            except Exception as e:
                logger.error(f"Monitor Loop Error: {e}")
                await asyncio.sleep(1)
        
    async def get_available_balance(self):
        try:
             account = await self._req("GET", "/fapi/v2/account", signed=True)
             self.balance = float(account.get("totalWalletBalance", 0))
             # Logging clutter reduction
             # logger.info(f"💰 Balance: {self.balance:.2f} | Open: {len(self.positions)}")
             return float(account.get("availableBalance", 0))
        except: return 0.0

if __name__ == "__main__":
    asyncio.run(MLScanner().run())
