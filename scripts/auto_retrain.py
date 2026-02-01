"""
Continuous Training Service.
Runs the training loop periodically to ensure the model stays fresh.
"""
import sys
import time
import asyncio
from pathlib import Path
from loguru import logger
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import train

async def continuous_train_loop(interval_hours=6):
    logger.info(f"🚀 Starting Continuous Training Loop (Interval: {interval_hours}h)")
    
    while True:
        try:
            logger.info("🏋️‍♂️ Starting Retraining Routine...")
            
            # Create args object similar to argparse
            class Args:
                symbol = "BTCUSDT"
                days = 90 # Train on last 90 days
                
            args = Args()
            
            # Run training
            await train(args)
            
            logger.info("✅ Retraining Complete. Model updated.")
            logger.info(f"💤 Sleeping for {interval_hours} hours...")
            
            # Sleep (convert hours to seconds)
            await asyncio.sleep(interval_hours * 3600)
            
        except Exception as e:
            logger.error(f"❌ Training Loop Error: {e}")
            await asyncio.sleep(300) # Sleep 5m on error then retry

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(continuous_train_loop())
    except KeyboardInterrupt:
        pass
