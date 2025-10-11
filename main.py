#!/usr/bin/env python3
"""
Main entry point for the Coin Futures Trading Bot.
Orchestrates all modules and handles system lifecycle.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any

import click
from rich.console import Console
from rich.logging import RichHandler

from src.config import Config
from src.core.message_queue import MessageQueue
from src.modules.data_ingestion import DataIngestionModule
from src.modules.ai_core import AICoreModule
from src.modules.risk_management import RiskManagementModule
from src.modules.execution_engine import ExecutionEngineModule
from src.modules.monitoring import MonitoringModule
from src.modules.backtesting import BacktestingModule

console = Console()


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self.message_queue = None
        self.modules = {}
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(console=console, rich_tracebacks=True),
                logging.FileHandler(self.config.LOG_FILE, mode='a')
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize all modules and message queue."""
        self.logger.info("Initializing trading bot...")
        
        # Initialize message queue
        self.message_queue = MessageQueue(self.config)
        await self.message_queue.initialize()
        
        # Initialize modules
        self.modules = {
            'data_ingestion': DataIngestionModule(self.config, self.message_queue),
            'ai_core': AICoreModule(self.config, self.message_queue),
            'risk_management': RiskManagementModule(self.config, self.message_queue),
            'execution_engine': ExecutionEngineModule(self.config, self.message_queue),
            'monitoring': MonitoringModule(self.config, self.message_queue),
            'backtesting': BacktestingModule(self.config, self.message_queue),
        }
        
        # Initialize each module
        for name, module in self.modules.items():
            self.logger.info(f"Initializing {name} module...")
            await module.initialize()
            
        self.logger.info("All modules initialized successfully")
    
    async def start(self):
        """Start all modules."""
        self.logger.info("Starting trading bot...")
        self.running = True
        
        # Start all modules
        tasks = []
        for name, module in self.modules.items():
            self.logger.info(f"Starting {name} module...")
            task = asyncio.create_task(module.start())
            tasks.append(task)
        
        # Wait for all modules to complete
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in module execution: {e}")
            raise
    
    async def stop(self):
        """Stop all modules gracefully."""
        self.logger.info("Stopping trading bot...")
        self.running = False
        
        # Stop all modules
        for name, module in self.modules.items():
            self.logger.info(f"Stopping {name} module...")
            await module.stop()
        
        # Stop message queue
        if self.message_queue:
            await self.message_queue.stop()
        
        self.logger.info("Trading bot stopped")


async def run_bot(config: Config):
    """Run the trading bot."""
    bot = TradingBot(config)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        console.print(f"\n[red]Received signal {signum}, shutting down...[/red]")
        asyncio.create_task(bot.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await bot.initialize()
        await bot.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Received keyboard interrupt, shutting down...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        raise
    finally:
        await bot.stop()


@click.command()
@click.option('--config', '-c', default='config.env', help='Configuration file path')
@click.option('--mode', '-m', type=click.Choice(['live', 'paper', 'backtest']), 
              default='paper', help='Trading mode')
@click.option('--symbol', '-s', default='BTCUSDT', help='Trading symbol')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(config: str, mode: str, symbol: str, verbose: bool):
    """Coin Futures Trading Bot - AI-powered algorithmic trading system."""
    
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Configuration file not found: {config}[/red]")
        sys.exit(1)
    
    cfg = Config.from_file(config_path)
    cfg.TRADING_MODE = mode
    cfg.DEFAULT_SYMBOL = symbol
    
    if verbose:
        cfg.LOG_LEVEL = 'DEBUG'
    
    console.print(f"[green]Starting trading bot in {mode} mode for {symbol}[/green]")
    
    # Run the bot
    try:
        asyncio.run(run_bot(cfg))
    except Exception as e:
        console.print(f"[red]Failed to start trading bot: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
