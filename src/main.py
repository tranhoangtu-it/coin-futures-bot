"""
TradingBot Main Entry Point.

Orchestrates all components for live trading:
- Data collection via WebSocket
- Feature calculation
- ML signal generation
- Order execution with risk management

Usage:
    python -m src.main

Follows @python-pro skill patterns.
"""

import asyncio
import signal
import sys
from pathlib import Path

from loguru import logger

from src.infrastructure import Settings, WebSocketManager, OrderBookSynchronizer, get_settings
from src.data import DataPipeline
from src.features import TechnicalIndicators, MicrostructureFeatures
from src.risk import KellyCriterion, TrailingStopManager, RiskLimits
from src.risk.trailing_stop import PositionSide
from src.execution import BinanceClient, OrderManager
from src.monitoring import AlertManager
from src.monitoring.alerts import AlertLevel


class TradingBot:
    """
    Main trading bot orchestrator.

    Manages the lifecycle of all trading components and
    coordinates signal generation with execution.

    Example:
        ```python
        bot = TradingBot()
        await bot.run()
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize trading bot.

        Args:
            settings: Application settings.
        """
        self._settings = settings or get_settings()
        self._running = False

        # Components
        self._data_pipeline: DataPipeline | None = None
        self._binance_client: BinanceClient | None = None
        self._order_manager: OrderManager | None = None
        self._risk_limits: RiskLimits | None = None
        self._stop_manager: TrailingStopManager | None = None
        self._kelly: KellyCriterion | None = None
        self._alerts: AlertManager | None = None

        # Feature calculators
        self._technical = TechnicalIndicators()
        self._microstructure = MicrostructureFeatures()

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing TradingBot components...")

        # Initialize Binance client
        self._binance_client = BinanceClient(self._settings)
        await self._binance_client.initialize()

        # Initialize risk management
        self._risk_limits = RiskLimits(
            max_drawdown_pct=self._settings.trading.max_daily_loss_pct * 3,
            daily_loss_limit_pct=self._settings.trading.max_daily_loss_pct,
            max_position_pct=self._settings.trading.max_position_pct,
        )

        self._stop_manager = TrailingStopManager(
            atr_multiplier=self._settings.trading.atr_multiplier,
        )

        self._kelly = KellyCriterion(
            fractional_factor=self._settings.trading.fractional_kelly,
            max_position_pct=self._settings.trading.max_position_pct,
        )

        # Initialize order manager
        self._order_manager = OrderManager(
            self._binance_client,
            self._risk_limits,
            self._stop_manager,
        )

        # Initialize data pipeline
        self._data_pipeline = DataPipeline(
            self._settings.trading.symbols,
            self._settings,
        )

        # Initialize alert manager
        self._alerts = AlertManager()
        await self._alerts.initialize()

        # Get account balance
        try:
            account = await self._binance_client.get_account()
            balance = float(account.get("totalWalletBalance", 0))
            self._risk_limits.set_account_balance(balance)
            logger.info(f"Account balance: {balance:.2f} USDT")
        except Exception as e:
            logger.warning(f"Could not fetch account balance: {e}")
            self._risk_limits.set_account_balance(10000)  # Default for testnet

        # Sync positions
        await self._order_manager.sync_positions()

        # Set leverage for all symbols
        for symbol in self._settings.trading.symbols:
            try:
                await self._binance_client.set_leverage(
                    symbol, self._settings.trading.default_leverage
                )
                logger.info(f"Set {symbol} leverage to {self._settings.trading.default_leverage}x")
            except Exception as e:
                logger.warning(f"Could not set leverage for {symbol}: {e}")

        logger.info("TradingBot initialized successfully")

    async def run(self) -> None:
        """Run the trading bot."""
        await self.initialize()

        self._running = True

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        try:
            await self._alerts.send_alert(
                level=AlertLevel.INFO,
                title="TradingBot Started",
                message=f"Monitoring symbols: {', '.join(self._settings.trading.symbols)}",
            )

            # Start data pipeline (this blocks)
            await self._data_pipeline.start()

        except asyncio.CancelledError:
            logger.info("Trading bot cancelled")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
            await self._alerts.system_error("main", str(e))
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the trading bot gracefully."""
        if not self._running:
            return

        logger.info("Shutting down TradingBot...")
        self._running = False

        # Stop data pipeline
        if self._data_pipeline:
            await self._data_pipeline.stop()

        # Close clients
        if self._binance_client:
            await self._binance_client.close()

        if self._alerts:
            await self._alerts.send_alert(
                level=AlertLevel.INFO,
                title="TradingBot Stopped",
                message="Trading bot has been shut down",
            )
            await self._alerts.close()

        logger.info("TradingBot shutdown complete")


def setup_logging(settings: Settings) -> None:
    """Configure loguru logging."""
    log_settings = settings.logging

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_settings.format,
        level=log_settings.level,
        colorize=True,
    )

    # Add file handler
    log_path = Path(log_settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format=log_settings.format,
        level=log_settings.level,
        rotation=log_settings.rotation,
        retention=log_settings.retention,
    )

    logger.info("Logging configured")


def main() -> None:
    """Entry point."""
    settings = get_settings()
    setup_logging(settings)

    logger.info("=" * 50)
    logger.info("TradingBot - Binance Futures Automated Trading")
    logger.info("=" * 50)

    bot = TradingBot(settings)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
