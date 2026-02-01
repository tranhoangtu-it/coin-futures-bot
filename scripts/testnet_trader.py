"""
Testnet Trading Bot with Adaptive Strategy.

Runs on Binance Futures Testnet and iteratively improves strategy.
Uses mean-reversion and momentum hybrid approach.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import aiohttp
from loguru import logger

# Load environment
from dotenv import load_dotenv
load_dotenv()


class TestnetTrader:
    """Testnet trading bot with adaptive strategy."""
    
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            raise ValueError("This script only runs on testnet!")
        
        self.symbols = ["BTCUSDT", "ETHUSDT"]
        self.leverage = 5
        self.position_size_pct = 0.05  # 5% of account per trade
        
        # Strategy parameters (will be optimized)
        self.strategy_params = {
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_threshold": 1.2,
            "trend_ema": 50,
        }
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0.0
        self.starting_balance = 0.0
        self.current_positions = {}
        
        self.session = None
    
    async def initialize(self):
        """Initialize trading session."""
        import hashlib
        import hmac
        import time
        
        self.session = aiohttp.ClientSession(
            headers={"X-MBX-APIKEY": self.api_key}
        )
        
        # Test connection
        try:
            account = await self._request("GET", "/fapi/v2/account", signed=True)
            self.starting_balance = float(account.get("totalWalletBalance", 10000))
            logger.info(f"Connected to {'TESTNET' if self.testnet else 'MAINNET'}")
            logger.info(f"Account balance: {self.starting_balance:.2f} USDT")
            
            # Set leverage
            for symbol in self.symbols:
                await self._request(
                    "POST", "/fapi/v1/leverage",
                    params={"symbol": symbol, "leverage": self.leverage},
                    signed=True
                )
                logger.info(f"Set {symbol} leverage to {self.leverage}x")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def _sign_request(self, params: dict) -> dict:
        """Sign request with API secret."""
        import hashlib
        import hmac
        import time
        
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params
    
    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False):
        """Make API request."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            params = self._sign_request(params)
        
        async with self.session.request(method, url, params=params) as response:
            data = await response.json()
            if response.status != 200:
                raise Exception(f"API Error: {data}")
            return data
    
    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch klines data."""
        data = await self._request(
            "GET", "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit}
        )
        
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        return df.set_index("timestamp")
    
    async def get_price(self, symbol: str) -> float:
        """Get current price."""
        data = await self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
        return float(data["price"])
    
    async def get_account_balance(self) -> float:
        """Get current account balance."""
        account = await self._request("GET", "/fapi/v2/account", signed=True)
        return float(account.get("totalWalletBalance", 0))
    
    def calculate_signal(self, df: pd.DataFrame) -> int:
        """
        Calculate trading signal using hybrid strategy.
        
        Strategy: Mean-reversion at extremes + trend confirmation
        - Buy when: RSI oversold + price below lower BB + uptrend forming
        - Sell when: RSI overbought + price above upper BB + downtrend forming
        
        Returns: 1 (long), -1 (short), 0 (no signal)
        """
        params = self.strategy_params
        close = df["close"]
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(params["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(params["rsi_period"]).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = close.rolling(params["bb_period"]).mean()
        std = close.rolling(params["bb_period"]).std()
        bb_upper = sma + params["bb_std"] * std
        bb_lower = sma - params["bb_std"] * std
        
        # Trend EMA
        ema = close.ewm(span=params["trend_ema"], adjust=False).mean()
        
        # Volume
        vol_sma = df["volume"].rolling(20).mean()
        vol_ratio = df["volume"] / vol_sma
        
        # Get latest values
        current_rsi = rsi.iloc[-1]
        current_close = close.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        current_bb_upper = bb_upper.iloc[-1]
        current_ema = ema.iloc[-1]
        current_vol = vol_ratio.iloc[-1]
        
        # Price momentum (short-term)
        momentum = close.pct_change(3).iloc[-1]
        
        signal = 0
        
        # Mean-reversion buy: oversold and starting to bounce
        if (current_rsi < params["rsi_oversold"] and 
            current_close < current_bb_lower and
            momentum > 0 and  # Starting to bounce
            current_vol > params["volume_threshold"]):
            signal = 1
            logger.debug(f"BUY signal: RSI={current_rsi:.1f}, BB position, momentum={momentum:.4f}")
        
        # Mean-reversion sell: overbought and starting to drop
        elif (current_rsi > params["rsi_overbought"] and 
              current_close > current_bb_upper and
              momentum < 0 and  # Starting to drop
              current_vol > params["volume_threshold"]):
            signal = -1
            logger.debug(f"SELL signal: RSI={current_rsi:.1f}, BB position, momentum={momentum:.4f}")
        
        return signal
    
    async def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Place market order."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }
        
        try:
            result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)
            logger.info(f"Order placed: {side} {quantity} {symbol}")
            return result
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None
    
    async def close_position(self, symbol: str) -> dict:
        """Close position for symbol."""
        if symbol not in self.current_positions:
            return None
        
        position = self.current_positions[symbol]
        side = "SELL" if position["side"] == "BUY" else "BUY"
        
        result = await self.place_order(symbol, side, position["quantity"])
        if result:
            exit_price = float(result.get("avgPrice", 0))
            entry_price = position["entry_price"]
            
            if position["side"] == "BUY":
                pnl = (exit_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - exit_price) / entry_price * 100
            
            pnl_value = position["quantity"] * exit_price * (pnl / 100)
            
            self.trades.append({
                "symbol": symbol,
                "side": position["side"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": position["quantity"],
                "pnl_pct": pnl,
                "pnl_value": pnl_value,
                "timestamp": datetime.utcnow(),
            })
            
            self.total_pnl += pnl_value
            del self.current_positions[symbol]
            
            logger.info(f"Closed {symbol}: PnL = {pnl:+.2f}% ({pnl_value:+.2f} USDT)")
        
        return result
    
    async def run_iteration(self):
        """Run one trading iteration."""
        for symbol in self.symbols:
            try:
                # Get data
                df = await self.get_klines(symbol, "15m", 100)
                current_price = await self.get_price(symbol)
                
                # Check for exit signals first
                if symbol in self.current_positions:
                    position = self.current_positions[symbol]
                    entry_price = position["entry_price"]
                    
                    # Take profit at 1.5%
                    if position["side"] == "BUY":
                        profit_pct = (current_price - entry_price) / entry_price * 100
                    else:
                        profit_pct = (entry_price - current_price) / entry_price * 100
                    
                    if profit_pct > 1.5:
                        logger.info(f"Take profit {symbol}: {profit_pct:.2f}%")
                        await self.close_position(symbol)
                    elif profit_pct < -1.0:
                        logger.info(f"Stop loss {symbol}: {profit_pct:.2f}%")
                        await self.close_position(symbol)
                    continue
                
                # Check for entry signals
                signal = self.calculate_signal(df)
                
                if signal != 0:
                    # Calculate position size
                    balance = await self.get_account_balance()
                    position_value = balance * self.position_size_pct * self.leverage
                    quantity = round(position_value / current_price, 3)
                    
                    if quantity < 0.001:
                        continue
                    
                    side = "BUY" if signal == 1 else "SELL"
                    result = await self.place_order(symbol, side, quantity)
                    
                    if result:
                        self.current_positions[symbol] = {
                            "side": side,
                            "entry_price": current_price,
                            "quantity": quantity,
                            "timestamp": datetime.utcnow(),
                        }
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    def optimize_parameters(self):
        """Adjust parameters based on performance."""
        if len(self.trades) < 5:
            return
        
        win_trades = [t for t in self.trades if t["pnl_pct"] > 0]
        win_rate = len(win_trades) / len(self.trades)
        
        logger.info(f"Performance: {len(self.trades)} trades, {win_rate:.1%} win rate")
        
        # If win rate is low, adjust parameters
        if win_rate < 0.4:
            # Make entries more selective
            self.strategy_params["rsi_oversold"] -= 2
            self.strategy_params["rsi_overbought"] += 2
            self.strategy_params["volume_threshold"] += 0.1
            logger.info(f"Adjusted params: RSI {self.strategy_params['rsi_oversold']}-{self.strategy_params['rsi_overbought']}")
        elif win_rate > 0.6:
            # Can be slightly more aggressive
            self.strategy_params["rsi_oversold"] += 1
            self.strategy_params["rsi_overbought"] -= 1
    
    async def run(self, iterations: int = 100, interval_seconds: int = 60):
        """Run trading bot."""
        logger.info("=" * 60)
        logger.info("Starting Testnet Trading Bot")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info("=" * 60)
        
        if not await self.initialize():
            logger.error("Failed to initialize. Exiting.")
            return
        
        try:
            for i in range(iterations):
                logger.info(f"\n--- Iteration {i+1}/{iterations} ---")
                
                await self.run_iteration()
                
                # Calculate current PnL
                current_balance = await self.get_account_balance()
                total_pnl_pct = (current_balance - self.starting_balance) / self.starting_balance * 100
                
                logger.info(f"Balance: {current_balance:.2f} USDT | PnL: {total_pnl_pct:+.2f}%")
                logger.info(f"Trades: {len(self.trades)} | Positions: {len(self.current_positions)}")
                
                # Optimize every 10 iterations
                if (i + 1) % 10 == 0:
                    self.optimize_parameters()
                
                # Check if PnL is positive
                if total_pnl_pct > 0 and len(self.trades) >= 5:
                    logger.info("=" * 60)
                    logger.info("🎉 POSITIVE PnL ACHIEVED!")
                    logger.info(f"Final PnL: {total_pnl_pct:+.2f}%")
                    logger.info(f"Total Trades: {len(self.trades)}")
                    break
                
                await asyncio.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Close all positions
            for symbol in list(self.current_positions.keys()):
                await self.close_position(symbol)
            
            await self.session.close()
            
            # Final report
            logger.info("\n" + "=" * 60)
            logger.info("TRADING SESSION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Starting Balance: {self.starting_balance:.2f} USDT")
            final_balance = await self.get_account_balance() if self.session and not self.session.closed else self.starting_balance + self.total_pnl
            logger.info(f"Final Balance: {final_balance:.2f} USDT")
            logger.info(f"Total PnL: {self.total_pnl:+.2f} USDT")
            logger.info(f"Total Trades: {len(self.trades)}")
            
            if self.trades:
                wins = [t for t in self.trades if t["pnl_pct"] > 0]
                logger.info(f"Win Rate: {len(wins)/len(self.trades):.1%}")


async def main():
    trader = TestnetTrader()
    await trader.run(iterations=50, interval_seconds=30)


if __name__ == "__main__":
    asyncio.run(main())
