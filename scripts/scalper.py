"""
Scalping Bot for Binance Testnet.

Trades on very short timeframes with tight TP/SL.
Uses simple EMA crosses and momentum for quick entries.
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


class Scalper:
    """Quick scalping bot."""
    
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.base_url = "https://testnet.binancefuture.com"
        
        self.symbols = ["BTCUSDT", "ETHUSDT"]
        self.leverage = 5
        self.position_pct = 0.02  # Small positions
        
        self.trades = []
        self.total_pnl = 0.0
        self.starting_balance = 0.0
        self.positions = {}
        self.session = None
        
        # Take profit and stop loss (in %)
        self.tp_pct = 0.8  # 0.8% take profit
        self.sl_pct = 0.4  # 0.4% stop loss
    
    async def init(self):
        self.session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": self.api_key})
        try:
            account = await self._req("GET", "/fapi/v2/account", signed=True)
            self.starting_balance = float(account.get("totalWalletBalance", 5000))
            logger.info(f"✅ Scalper connected | Balance: {self.starting_balance:.2f} USDT")
            
            for s in self.symbols:
                await self._req("POST", "/fapi/v1/leverage", 
                    params={"symbol": s, "leverage": self.leverage}, signed=True)
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
    
    async def get_data(self, symbol, interval="1m", limit=30):
        d = await self._req("GET", "/fapi/v1/klines", 
            params={"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(d, columns=["t","o","h","l","c","v","ct","qv","tr","tbv","tbqv","i"])
        for c in ["o","h","l","c","v"]:
            df[c] = df[c].astype(float)
        return df
    
    async def get_price(self, s):
        d = await self._req("GET", "/fapi/v1/ticker/price", params={"symbol": s})
        return float(d["price"])
    
    async def get_balance(self):
        a = await self._req("GET", "/fapi/v2/account", signed=True)
        return float(a.get("totalWalletBalance", 0))
    
    def get_signal(self, df):
        """
        Simple and sensitive signal generation.
        
        Long if: price above EMA5, EMA5 > EMA10, momentum positive
        Short if: price below EMA5, EMA5 < EMA10, momentum negative
        """
        c = df["c"].values
        
        ema5 = pd.Series(c).ewm(span=5).mean().values
        ema10 = pd.Series(c).ewm(span=10).mean().values
        
        price = c[-1]
        e5 = ema5[-1]
        e10 = ema10[-1]
        
        # Short-term momentum
        mom = (c[-1] - c[-3]) / c[-3] * 100
        
        # Trend direction
        ema5_slope = (ema5[-1] - ema5[-3]) / ema5[-3] * 100
        
        signal = 0
        
        # Long: price > ema5 > ema10 and positive momentum
        if price > e5 > e10 and mom > 0.02 and ema5_slope > 0:
            signal = 1
        # Short: price < ema5 < ema10 and negative momentum
        elif price < e5 < e10 and mom < -0.02 and ema5_slope < 0:
            signal = -1
        
        return signal
    
    async def trade(self, symbol, side, qty):
        try:
            r = await self._req("POST", "/fapi/v1/order",
                params={"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty},
                signed=True)
            logger.info(f"📈 {side} {qty} {symbol}")
            return r
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return None
    
    async def close(self, symbol, price=None):
        if symbol not in self.positions:
            return
        
        p = self.positions[symbol]
        side = "SELL" if p["side"] == "BUY" else "BUY"
        
        r = await self.trade(symbol, side, p["qty"])
        if not r:
            return
        
        exit_p = price or float(r.get("avgPrice", p["entry"]))
        
        if p["side"] == "BUY":
            pnl = (exit_p - p["entry"]) / p["entry"] * 100 * self.leverage
        else:
            pnl = (p["entry"] - exit_p) / p["entry"] * 100 * self.leverage
        
        pnl_val = p["qty"] * p["entry"] * (pnl/100) / self.leverage
        
        self.trades.append({"pnl_pct": pnl, "pnl_val": pnl_val})
        self.total_pnl += pnl_val
        del self.positions[symbol]
        
        e = "✅" if pnl > 0 else "❌"
        logger.info(f"{e} Closed {symbol}: {pnl:+.2f}% ({pnl_val:+.2f} USDT)")
    
    async def cycle(self):
        for s in self.symbols:
            try:
                df = await self.get_data(s, "1m", 30)
                price = await self.get_price(s)
                
                # Check position
                if s in self.positions:
                    p = self.positions[s]
                    if p["side"] == "BUY":
                        pnl = (price - p["entry"]) / p["entry"] * 100 * self.leverage
                    else:
                        pnl = (p["entry"] - price) / p["entry"] * 100 * self.leverage
                    
                    if pnl >= self.tp_pct:
                        logger.info(f"🎯 TP {s}")
                        await self.close(s, price)
                    elif pnl <= -self.sl_pct:
                        logger.info(f"🛑 SL {s}")
                        await self.close(s, price)
                    continue
                
                # New entry
                sig = self.get_signal(df)
                if sig != 0:
                    bal = await self.get_balance()
                    qty = round(bal * self.position_pct * self.leverage / price, 3)
                    
                    if qty >= 0.001:
                        side = "BUY" if sig == 1 else "SELL"
                        r = await self.trade(s, side, qty)
                        if r:
                            self.positions[s] = {"side": side, "entry": price, "qty": qty}
            
            except Exception as e:
                logger.error(f"Cycle error {s}: {e}")
    
    async def run(self, max_iter=300):
        logger.info("=" * 50)
        logger.info("🎯 SCALPER BOT - TESTNET")
        logger.info(f"TP: {self.tp_pct}% | SL: {self.sl_pct}%")
        logger.info("=" * 50)
        
        if not await self.init():
            return
        
        try:
            for i in range(max_iter):
                await self.cycle()
                
                bal = await self.get_balance()
                pnl = (bal - self.starting_balance) / self.starting_balance * 100
                
                w = len([t for t in self.trades if t["pnl_pct"] > 0])
                n = len(self.trades)
                wr = w/n*100 if n > 0 else 0
                
                logger.info(
                    f"[{i+1}] 💰{bal:.2f} | PnL:{pnl:+.2f}% | "
                    f"Trades:{n} | WR:{wr:.0f}% | Pos:{len(self.positions)}"
                )
                
                if pnl > 0 and n >= 3:
                    logger.info("\n" + "🎉"*20)
                    logger.info("✅ POSITIVE PnL!")
                    logger.info(f"Final: {pnl:+.2f}% | Trades: {n} | WR: {wr:.1f}%")
                    logger.info("🎉"*20)
                    break
                
                await asyncio.sleep(10)
        
        except KeyboardInterrupt:
            logger.info("Stopped")
        finally:
            for s in list(self.positions.keys()):
                await self.close(s)
            if self.session:
                await self.session.close()
            
            logger.info("\n" + "="*50)
            logger.info("📊 FINAL")
            logger.info(f"PnL: {self.total_pnl:+.2f} USDT | Trades: {len(self.trades)}")


if __name__ == "__main__":
    asyncio.run(Scalper().run())
