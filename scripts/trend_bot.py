"""
Simple Trend Following Bot for Binance Testnet.

Follows the trend with tight risk management.
Uses EMA crossover for entries and ATR for stops.
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


class TrendBot:
    """Simple trend following bot."""
    
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.base_url = "https://testnet.binancefuture.com"
        
        self.symbol = "BTCUSDT"
        self.leverage = 5
        self.risk_per_trade = 0.01  # 1% risk per trade
        
        self.trades = []
        self.total_pnl = 0.0
        self.starting_balance = 0.0
        self.position = None
        self.session = None
    
    async def init(self):
        self.session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": self.api_key})
        try:
            account = await self._req("GET", "/fapi/v2/account", signed=True)
            self.starting_balance = float(account.get("totalWalletBalance", 5000))
            logger.info(f"✅ Trend Bot | Balance: {self.starting_balance:.2f} USDT")
            
            await self._req("POST", "/fapi/v1/leverage", 
                params={"symbol": self.symbol, "leverage": self.leverage}, signed=True)
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
        if signed:
            p = self._sign(p)
        async with self.session.request(m, url, params=p) as r:
            d = await r.json()
            if r.status != 200:
                raise Exception(f"Error: {d}")
            return d
    
    async def get_data(self, interval="5m", limit=50):
        d = await self._req("GET", "/fapi/v1/klines", 
            params={"symbol": self.symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(d, columns=["t","o","h","l","c","v","ct","qv","tr","tbv","tbqv","i"])
        for col in ["o","h","l","c","v"]:
            df[col] = df[col].astype(float)
        return df
    
    async def get_price(self):
        d = await self._req("GET", "/fapi/v1/ticker/price", params={"symbol": self.symbol})
        return float(d["price"])
    
    async def get_balance(self):
        a = await self._req("GET", "/fapi/v2/account", signed=True)
        return float(a.get("totalWalletBalance", 0))
    
    def analyze(self, df):
        """
        Simple trend analysis.
        
        Long: EMA8 > EMA21, price above EMA8, momentum positive
        Short: EMA8 < EMA21, price below EMA8, momentum negative
        
        Returns: (signal, stop_distance)
        """
        c = df["c"].values
        h = df["h"].values
        l = df["l"].values
        
        ema8 = pd.Series(c).ewm(span=8).mean().values
        ema21 = pd.Series(c).ewm(span=21).mean().values
        
        # ATR for stop distance
        tr = np.maximum(h[1:] - l[1:], 
                       np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = np.mean(tr[-14:])
        
        price = c[-1]
        e8 = ema8[-1]
        e21 = ema21[-1]
        
        # Momentum
        mom = (c[-1] - c[-5]) / c[-5] * 100
        
        signal = 0
        
        # Long: uptrend + price above EMA8 + positive momentum
        if e8 > e21 and price > e8 and mom > 0.01:
            signal = 1
        # Short: downtrend + price below EMA8 + negative momentum  
        elif e8 < e21 and price < e8 and mom < -0.01:
            signal = -1
        
        return signal, atr * 1.5  # 1.5x ATR stop
    
    async def trade(self, side, qty):
        try:
            r = await self._req("POST", "/fapi/v1/order",
                params={"symbol": self.symbol, "side": side, "type": "MARKET", "quantity": qty},
                signed=True)
            logger.info(f"📈 {side} {qty} {self.symbol}")
            return r
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return None
    
    async def close_position(self, price):
        if not self.position:
            return
        
        side = "SELL" if self.position["side"] == "BUY" else "BUY"
        r = await self.trade(side, self.position["qty"])
        
        if r:
            if self.position["side"] == "BUY":
                pnl = (price - self.position["entry"]) / self.position["entry"] * 100 * self.leverage
            else:
                pnl = (self.position["entry"] - price) / self.position["entry"] * 100 * self.leverage
            
            pnl_val = self.position["qty"] * self.position["entry"] * (pnl/100) / self.leverage
            
            self.trades.append({"pnl_pct": pnl, "pnl_val": pnl_val})
            self.total_pnl += pnl_val
            
            e = "✅" if pnl > 0 else "❌"
            logger.info(f"{e} Closed: {pnl:+.2f}% ({pnl_val:+.2f} USDT)")
            
            self.position = None
    
    async def cycle(self):
        df = await self.get_data("5m", 50)
        price = await self.get_price()
        balance = await self.get_balance()
        
        # Check exits
        if self.position:
            if self.position["side"] == "BUY":
                pnl = (price - self.position["entry"]) / self.position["entry"] * 100 * self.leverage
            else:
                pnl = (self.position["entry"] - price) / self.position["entry"] * 100 * self.leverage
            
            # Take profit at 2%
            if pnl >= 2.0:
                logger.info("🎯 Take Profit")
                await self.close_position(price)
            # Stop loss at 1%
            elif pnl <= -1.0:
                logger.info("🛑 Stop Loss")
                await self.close_position(price)
            # Trail stop - exit if price moves against by 0.5% from peak
            elif self.position.get("peak_pnl", 0) - pnl > 0.5:
                logger.info("📉 Trail Stop")
                await self.close_position(price)
            else:
                self.position["peak_pnl"] = max(self.position.get("peak_pnl", 0), pnl)
            return
        
        # Check entries
        signal, stop_dist = self.analyze(df)
        
        if signal != 0:
            # Position sizing based on stop distance
            risk_amount = balance * self.risk_per_trade
            qty = round(risk_amount / stop_dist, 3)
            qty = max(0.001, min(qty, balance * 0.05 * self.leverage / price))
            
            side = "BUY" if signal == 1 else "SELL"
            r = await self.trade(side, qty)
            
            if r:
                self.position = {
                    "side": side,
                    "entry": price,
                    "qty": qty,
                    "stop": stop_dist,
                    "peak_pnl": 0,
                }
    
    async def run(self, max_iter=300):
        logger.info("=" * 50)
        logger.info("📈 TREND BOT - TESTNET")
        logger.info("=" * 50)
        
        if not await self.init():
            return
        
        try:
            for i in range(max_iter):
                await self.cycle()
                
                bal = await self.get_balance()
                pnl = (bal - self.starting_balance) / self.starting_balance * 100
                
                n = len(self.trades)
                wins = len([t for t in self.trades if t["pnl_pct"] > 0])
                wr = wins/n*100 if n > 0 else 0
                
                pos = "📊 " + self.position["side"] if self.position else "No"
                
                logger.info(
                    f"[{i+1}] 💰{bal:.2f} | PnL:{pnl:+.2f}% | "
                    f"Trades:{n} | WR:{wr:.0f}% | Pos:{pos}"
                )
                
                if pnl > 0 and n >= 3:
                    logger.info("\n" + "🎉"*20)
                    logger.info("✅ POSITIVE PnL!")
                    logger.info(f"Final: {pnl:+.2f}% | Trades: {n}")
                    logger.info("🎉"*20)
                    break
                
                await asyncio.sleep(10)
        
        except KeyboardInterrupt:
            logger.info("Stopped")
        finally:
            if self.position:
                price = await self.get_price()
                await self.close_position(price)
            
            if self.session:
                await self.session.close()
            
            logger.info("\n" + "="*50)
            logger.info("📊 FINAL")
            logger.info(f"PnL: {self.total_pnl:+.2f} USDT | Trades: {len(self.trades)}")
            if self.trades:
                wins = len([t for t in self.trades if t["pnl_pct"] > 0])
                logger.info(f"Win Rate: {wins}/{len(self.trades)}")


if __name__ == "__main__":
    asyncio.run(TrendBot().run())
