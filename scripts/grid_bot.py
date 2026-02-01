"""
Grid Trading Bot for Binance Testnet.

Uses a grid strategy that profits from price oscillation.
Buys at support levels, sells at resistance levels.
Works well in ranging markets.
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


class GridBot:
    """Grid trading bot for ranging markets."""
    
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.base_url = "https://testnet.binancefuture.com"
        
        self.symbol = "BTCUSDT"
        self.leverage = 3
        self.grid_size = 0.002  # 0.2% between grid levels
        self.num_grids = 5
        self.position_per_grid = 0.01  # 1% of account per grid
        
        self.trades = []
        self.total_pnl = 0.0
        self.starting_balance = 0.0
        self.grid_levels = {}
        self.session = None
    
    async def init(self):
        self.session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": self.api_key})
        try:
            account = await self._req("GET", "/fapi/v2/account", signed=True)
            self.starting_balance = float(account.get("totalWalletBalance", 5000))
            logger.info(f"✅ Grid Bot connected | Balance: {self.starting_balance:.2f} USDT")
            
            await self._req("POST", "/fapi/v1/leverage", 
                params={"symbol": self.symbol, "leverage": self.leverage}, signed=True)
            
            # Get current price and set grid
            price = await self.get_price()
            self.setup_grid(price)
            
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
    
    async def get_price(self):
        d = await self._req("GET", "/fapi/v1/ticker/price", params={"symbol": self.symbol})
        return float(d["price"])
    
    async def get_balance(self):
        a = await self._req("GET", "/fapi/v2/account", signed=True)
        return float(a.get("totalWalletBalance", 0))
    
    def setup_grid(self, center_price: float):
        """Setup grid levels around current price."""
        self.grid_levels = {}
        
        for i in range(-self.num_grids, self.num_grids + 1):
            level = center_price * (1 + i * self.grid_size)
            self.grid_levels[round(level, 2)] = {
                "type": "buy" if i < 0 else "sell",
                "filled": False,
                "position": None,
            }
        
        logger.info(f"Grid setup: {len(self.grid_levels)} levels around ${center_price:.2f}")
        logger.info(f"Range: ${min(self.grid_levels.keys()):.2f} - ${max(self.grid_levels.keys()):.2f}")
    
    async def trade(self, side: str, qty: float):
        try:
            r = await self._req("POST", "/fapi/v1/order",
                params={"symbol": self.symbol, "side": side, "type": "MARKET", "quantity": qty},
                signed=True)
            logger.info(f"📈 {side} {qty} {self.symbol}")
            return r
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return None
    
    async def cycle(self):
        """Check grid levels and execute trades."""
        price = await self.get_price()
        balance = await self.get_balance()
        
        for level, info in self.grid_levels.items():
            if info["filled"]:
                # Check if we should close this position
                if info["position"]:
                    pos = info["position"]
                    if info["type"] == "buy" and price >= level * (1 + self.grid_size):
                        # Take profit on buy position
                        await self.trade("SELL", pos["qty"])
                        pnl = (price - pos["entry"]) * pos["qty"]
                        self.trades.append({"pnl": pnl, "type": "grid_tp"})
                        self.total_pnl += pnl
                        info["filled"] = False
                        info["position"] = None
                        logger.info(f"✅ Grid TP: +${pnl:.2f}")
                    elif info["type"] == "sell" and price <= level * (1 - self.grid_size):
                        # Take profit on sell position
                        await self.trade("BUY", pos["qty"])
                        pnl = (pos["entry"] - price) * pos["qty"]
                        self.trades.append({"pnl": pnl, "type": "grid_tp"})
                        self.total_pnl += pnl
                        info["filled"] = False
                        info["position"] = None
                        logger.info(f"✅ Grid TP: +${pnl:.2f}")
            else:
                # Check if we should open at this level
                if info["type"] == "buy" and price <= level:
                    qty = round(balance * self.position_per_grid * self.leverage / price, 3)
                    if qty >= 0.001:
                        r = await self.trade("BUY", qty)
                        if r:
                            info["filled"] = True
                            info["position"] = {"entry": price, "qty": qty}
                            logger.info(f"📊 Grid BUY at ${price:.2f}")
                elif info["type"] == "sell" and price >= level:
                    qty = round(balance * self.position_per_grid * self.leverage / price, 3)
                    if qty >= 0.001:
                        r = await self.trade("SELL", qty)
                        if r:
                            info["filled"] = True
                            info["position"] = {"entry": price, "qty": qty}
                            logger.info(f"📊 Grid SELL at ${price:.2f}")
    
    async def run(self, max_iter=200):
        logger.info("=" * 50)
        logger.info("📊 GRID TRADING BOT - TESTNET")
        logger.info(f"Grid size: {self.grid_size*100:.1f}% | Grids: {self.num_grids*2+1}")
        logger.info("=" * 50)
        
        if not await self.init():
            return
        
        try:
            for i in range(max_iter):
                await self.cycle()
                
                bal = await self.get_balance()
                pnl = (bal - self.starting_balance) / self.starting_balance * 100
                
                filled = sum(1 for l in self.grid_levels.values() if l["filled"])
                n = len(self.trades)
                wins = len([t for t in self.trades if t["pnl"] > 0])
                wr = wins/n*100 if n > 0 else 0
                
                logger.info(
                    f"[{i+1}] 💰{bal:.2f} | PnL:{pnl:+.2f}% | "
                    f"Trades:{n} | WR:{wr:.0f}% | Grid:{filled}/{len(self.grid_levels)}"
                )
                
                if pnl > 0 and n >= 3:
                    logger.info("\n" + "🎉"*20)
                    logger.info("✅ POSITIVE PnL!")
                    logger.info(f"Final: {pnl:+.2f}% | Trades: {n} | WR: {wr:.1f}%")
                    logger.info("🎉"*20)
                    break
                
                # Recenter grid if price moved too far
                price = await self.get_price()
                grid_center = (max(self.grid_levels.keys()) + min(self.grid_levels.keys())) / 2
                if abs(price - grid_center) / grid_center > 0.02:  # 2% drift
                    logger.info("⚠️ Recentering grid...")
                    # Close all positions first
                    for level, info in self.grid_levels.items():
                        if info["position"]:
                            side = "SELL" if info["type"] == "buy" else "BUY"
                            await self.trade(side, info["position"]["qty"])
                    self.setup_grid(price)
                
                await asyncio.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("Stopped")
        finally:
            # Close all positions
            for level, info in self.grid_levels.items():
                if info["position"]:
                    side = "SELL" if info["type"] == "buy" else "BUY"
                    await self.trade(side, info["position"]["qty"])
            
            if self.session:
                await self.session.close()
            
            logger.info("\n" + "="*50)
            logger.info("📊 FINAL")
            logger.info(f"PnL: {self.total_pnl:+.2f} USDT | Trades: {len(self.trades)}")


if __name__ == "__main__":
    asyncio.run(GridBot().run())
