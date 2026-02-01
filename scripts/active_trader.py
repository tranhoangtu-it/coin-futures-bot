"""
Active Testnet Trading Bot.

Uses momentum + breakout strategy for more frequent trades.
Designed to quickly find profitable setups on testnet.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
import os
import hashlib
import hmac
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import aiohttp
from loguru import logger

from dotenv import load_dotenv
load_dotenv()


class ActiveTrader:
    """Active testnet trader with momentum strategy."""
    
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.base_url = "https://testnet.binancefuture.com"
        
        self.symbols = ["BTCUSDT", "ETHUSDT"]
        self.leverage = 3
        self.position_size_pct = 0.03
        
        # Track performance
        self.trades = []
        self.total_pnl = 0.0
        self.starting_balance = 0.0
        self.current_positions = {}
        
        self.session = None
    
    async def init(self):
        """Initialize session and account."""
        self.session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": self.api_key})
        
        try:
            account = await self._request("GET", "/fapi/v2/account", signed=True)
            self.starting_balance = float(account.get("totalWalletBalance", 5000))
            logger.info(f"✅ Connected to TESTNET")
            logger.info(f"💰 Balance: {self.starting_balance:.2f} USDT")
            
            for symbol in self.symbols:
                await self._request("POST", "/fapi/v1/leverage", 
                    params={"symbol": symbol, "leverage": self.leverage}, signed=True)
            
            return True
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False
    
    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params
    
    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False):
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        if signed:
            params = self._sign(params)
        
        async with self.session.request(method, url, params=params) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise Exception(f"API Error: {data}")
            return data
    
    async def get_klines(self, symbol: str, interval: str = "5m", limit: int = 50) -> pd.DataFrame:
        data = await self._request("GET", "/fapi/v1/klines", 
            params={"symbol": symbol, "interval": interval, "limit": limit})
        
        df = pd.DataFrame(data, columns=[
            "ts", "o", "h", "l", "c", "v", "ct", "qv", "t", "tbv", "tbqv", "i"
        ])
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)
        return df
    
    async def get_price(self, symbol: str) -> float:
        data = await self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
        return float(data["price"])
    
    async def get_balance(self) -> float:
        account = await self._request("GET", "/fapi/v2/account", signed=True)
        return float(account.get("totalWalletBalance", 0))
    
    def analyze(self, df: pd.DataFrame) -> tuple:
        """
        Momentum breakout strategy.
        
        Returns: (signal, take_profit_pct, stop_loss_pct)
        """
        close = df["c"].values
        high = df["h"].values
        low = df["l"].values
        volume = df["v"].values
        
        # Recent price action
        last_close = close[-1]
        prev_close = close[-2]
        high_20 = np.max(high[-20:])
        low_20 = np.min(low[-20:])
        
        # Momentum
        momentum = (last_close - close[-5]) / close[-5] * 100
        
        # Volume spike
        vol_avg = np.mean(volume[-20:])
        vol_current = volume[-1]
        vol_spike = vol_current > vol_avg * 1.3
        
        # RSI (fast)
        delta = np.diff(close[-15:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        # EMA trend
        ema_8 = pd.Series(close).ewm(span=8).mean().iloc[-1]
        ema_21 = pd.Series(close).ewm(span=21).mean().iloc[-1]
        uptrend = ema_8 > ema_21
        
        signal = 0
        tp = 0.8
        sl = 0.5
        
        # Breakout long: price near 20-bar high + momentum + volume + uptrend
        if (last_close > high_20 * 0.998 and 
            momentum > 0.3 and 
            uptrend and
            40 < rsi < 70):
            signal = 1
            tp = 1.0
            sl = 0.6
        
        # Breakout short: price near 20-bar low + negative momentum + downtrend
        elif (last_close < low_20 * 1.002 and 
              momentum < -0.3 and 
              not uptrend and
              30 < rsi < 60):
            signal = -1
            tp = 1.0
            sl = 0.6
        
        # Momentum continuation
        elif momentum > 0.5 and uptrend and rsi < 65:
            signal = 1
            tp = 0.6
            sl = 0.4
        elif momentum < -0.5 and not uptrend and rsi > 35:
            signal = -1
            tp = 0.6
            sl = 0.4
        
        return signal, tp, sl
    
    async def trade(self, symbol: str, side: str, qty: float):
        """Execute trade."""
        try:
            result = await self._request("POST", "/fapi/v1/order", 
                params={"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}, 
                signed=True)
            logger.info(f"📈 {side} {qty} {symbol}")
            return result
        except Exception as e:
            logger.error(f"Trade failed: {e}")
            return None
    
    async def close_position(self, symbol: str, price: float = None):
        """Close position and record PnL."""
        if symbol not in self.current_positions:
            return
        
        pos = self.current_positions[symbol]
        side = "SELL" if pos["side"] == "BUY" else "BUY"
        
        result = await self.trade(symbol, side, pos["qty"])
        if not result:
            return
        
        exit_price = price or float(result.get("avgPrice", pos["entry"]))
        
        if pos["side"] == "BUY":
            pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100 * self.leverage
        else:
            pnl_pct = (pos["entry"] - exit_price) / pos["entry"] * 100 * self.leverage
        
        pnl_value = pos["qty"] * pos["entry"] * (pnl_pct / 100) / self.leverage
        
        self.trades.append({
            "symbol": symbol,
            "side": pos["side"],
            "pnl_pct": pnl_pct,
            "pnl_value": pnl_value,
        })
        
        self.total_pnl += pnl_value
        del self.current_positions[symbol]
        
        emoji = "✅" if pnl_pct > 0 else "❌"
        logger.info(f"{emoji} Closed {symbol}: {pnl_pct:+.2f}% ({pnl_value:+.2f} USDT)")
    
    async def run_cycle(self):
        """Run one trading cycle."""
        for symbol in self.symbols:
            try:
                df = await self.get_klines(symbol, "5m", 50)
                price = await self.get_price(symbol)
                
                # Check exits first
                if symbol in self.current_positions:
                    pos = self.current_positions[symbol]
                    
                    if pos["side"] == "BUY":
                        pnl = (price - pos["entry"]) / pos["entry"] * 100 * self.leverage
                    else:
                        pnl = (pos["entry"] - price) / pos["entry"] * 100 * self.leverage
                    
                    if pnl >= pos["tp"]:
                        logger.info(f"🎯 TP hit {symbol}")
                        await self.close_position(symbol, price)
                    elif pnl <= -pos["sl"]:
                        logger.info(f"🛑 SL hit {symbol}")
                        await self.close_position(symbol, price)
                    continue
                
                # Check entries
                signal, tp, sl = self.analyze(df)
                
                if signal != 0:
                    balance = await self.get_balance()
                    notional = balance * self.position_size_pct * self.leverage
                    qty = round(notional / price, 3)
                    
                    if qty >= 0.001:
                        side = "BUY" if signal == 1 else "SELL"
                        result = await self.trade(symbol, side, qty)
                        
                        if result:
                            self.current_positions[symbol] = {
                                "side": side,
                                "entry": price,
                                "qty": qty,
                                "tp": tp,
                                "sl": sl,
                            }
            
            except Exception as e:
                logger.error(f"Error {symbol}: {e}")
    
    async def run(self, max_iterations: int = 200):
        """Main run loop."""
        logger.info("=" * 50)
        logger.info("🚀 ACTIVE TESTNET TRADER")
        logger.info(f"📊 Symbols: {self.symbols}")
        logger.info(f"⚡ Leverage: {self.leverage}x")
        logger.info("=" * 50)
        
        if not await self.init():
            return
        
        try:
            for i in range(max_iterations):
                await self.run_cycle()
                
                balance = await self.get_balance()
                pnl_pct = (balance - self.starting_balance) / self.starting_balance * 100
                
                wins = len([t for t in self.trades if t["pnl_pct"] > 0])
                total = len(self.trades)
                wr = wins / total * 100 if total > 0 else 0
                
                logger.info(
                    f"[{i+1}] 💰 {balance:.2f} | PnL: {pnl_pct:+.2f}% | "
                    f"Trades: {total} | WR: {wr:.0f}% | Pos: {len(self.current_positions)}"
                )
                
                # Success condition
                if pnl_pct > 0 and total >= 3:
                    logger.info("\n" + "🎉" * 20)
                    logger.info("✅ POSITIVE PnL ACHIEVED!")
                    logger.info(f"Final PnL: {pnl_pct:+.2f}%")
                    logger.info(f"Total Trades: {total}")
                    logger.info(f"Win Rate: {wr:.1f}%")
                    logger.info("🎉" * 20)
                    break
                
                await asyncio.sleep(15)
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            for symbol in list(self.current_positions.keys()):
                await self.close_position(symbol)
            
            if self.session:
                await self.session.close()
            
            logger.info("\n" + "=" * 50)
            logger.info("📊 FINAL REPORT")
            logger.info("=" * 50)
            logger.info(f"Starting: {self.starting_balance:.2f} USDT")
            logger.info(f"Total PnL: {self.total_pnl:+.2f} USDT")
            logger.info(f"Trades: {len(self.trades)}")
            if self.trades:
                wins = len([t for t in self.trades if t["pnl_pct"] > 0])
                logger.info(f"Win Rate: {wins}/{len(self.trades)} ({wins/len(self.trades)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(ActiveTrader().run())
