"""
Close all open positions on Binance Futures Testnet.
"""
import asyncio
import os
from loguru import logger
from dotenv import load_dotenv
import aiohttp
import time
import hmac
import hashlib

load_dotenv()

async def close_all():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    base_url = "https://testnet.binancefuture.com"
    
    session = aiohttp.ClientSession(headers={"X-MBX-APIKEY": api_key})
    
    def sign(p):
        p["timestamp"] = int(time.time() * 1000)
        q = "&".join(f"{k}={v}" for k, v in p.items())
        return hmac.new(api_secret.encode(), q.encode(), hashlib.sha256).hexdigest()

    try:
        # Get Positions
        params = {"timestamp": int(time.time() * 1000)}
        params["signature"] = sign(params)
        async with session.get(f"{base_url}/fapi/v2/positionRisk", params=params) as r:
            positions = await r.json()
            
        tasks = []
        for p in positions:
            amt = float(p["positionAmt"])
            if amt != 0:
                symbol = p["symbol"]
                side = "SELL" if amt > 0 else "BUY"
                logger.info(f"Closing {symbol} ({amt})...")
                
                p2 = {
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": abs(amt),
                    "timestamp": int(time.time() * 1000)
                }
                p2["signature"] = sign(p2)
                tasks.append(session.post(f"{base_url}/fapi/v1/order", params=p2))
        
        if tasks:
            await asyncio.gather(*tasks)
            logger.info("All closed.")
        else:
            logger.info("No open positions.")
            
    finally:
        await session.close()

if __name__ == "__main__":
    asyncio.run(close_all())
