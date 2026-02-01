"""
Configuration module for TradingBot.

Uses pydantic-settings for type-safe configuration management with environment variables.
Follows @python-pro skill patterns for modern Python configuration.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceSettings(BaseSettings):
    """Binance API configuration."""

    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    api_key: SecretStr = Field(..., description="Binance API Key")
    api_secret: SecretStr = Field(..., description="Binance API Secret")
    testnet: bool = Field(default=True, description="Use testnet for paper trading")

    # WebSocket endpoints
    ws_base_url: str = Field(
        default="wss://fstream.binance.com",
        description="Binance Futures WebSocket base URL",
    )
    ws_testnet_url: str = Field(
        default="wss://stream.binancefuture.com",
        description="Binance Futures Testnet WebSocket URL",
    )

    # REST API endpoints
    rest_base_url: str = Field(
        default="https://fapi.binance.com",
        description="Binance Futures REST API base URL",
    )
    rest_testnet_url: str = Field(
        default="https://testnet.binancefuture.com",
        description="Binance Futures Testnet REST API URL",
    )

    @property
    def ws_url(self) -> str:
        """Get the appropriate WebSocket URL based on testnet setting."""
        return self.ws_testnet_url if self.testnet else self.ws_base_url

    @property
    def rest_url(self) -> str:
        """Get the appropriate REST API URL based on testnet setting."""
        return self.rest_testnet_url if self.testnet else self.rest_base_url


class TradingSettings(BaseSettings):
    """Trading configuration."""

    model_config = SettingsConfigDict(env_prefix="TRADING_")

    # Trading pairs
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT"],
        description="List of trading symbols",
    )

    # Position sizing
    fractional_kelly: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Fractional Kelly factor for position sizing",
    )
    max_position_pct: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Maximum position size as percentage of account",
    )

    # Risk management
    max_daily_loss_pct: float = Field(
        default=0.05,
        ge=0.01,
        le=0.2,
        description="Maximum daily loss as percentage of account",
    )
    atr_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="ATR multiplier for trailing stop",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR calculation period",
    )

    # Leverage
    default_leverage: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default leverage for positions",
    )


class WebSocketSettings(BaseSettings):
    """WebSocket connection configuration."""

    model_config = SettingsConfigDict(env_prefix="WS_")

    # Reconnection settings
    reconnect_delay_base: float = Field(
        default=1.0,
        description="Base delay for exponential backoff (seconds)",
    )
    reconnect_delay_max: float = Field(
        default=60.0,
        description="Maximum delay for exponential backoff (seconds)",
    )
    reconnect_max_attempts: int = Field(
        default=10,
        description="Maximum reconnection attempts before giving up",
    )

    # Heartbeat settings
    heartbeat_interval: float = Field(
        default=30.0,
        description="Heartbeat interval (seconds)",
    )
    heartbeat_timeout: float = Field(
        default=10.0,
        description="Heartbeat response timeout (seconds)",
    )

    # Order book settings
    order_book_depth: int = Field(
        default=20,
        description="Order book depth levels to maintain",
    )
    snapshot_buffer_size: int = Field(
        default=1000,
        description="Buffer size for order book events during sync",
    )


class DatabaseSettings(BaseSettings):
    """Database configuration for QuestDB."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = Field(default="localhost", description="QuestDB host")
    port: int = Field(default=9009, description="QuestDB ILP port")
    http_port: int = Field(default=9000, description="QuestDB HTTP port")


class MLFlowSettings(BaseSettings):
    """MLflow configuration."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking URI",
    )
    artifact_location: str = Field(
        default="./mlflow-artifacts",
        description="MLflow artifact storage location",
    )
    experiment_name: str = Field(
        default="tradingbot",
        description="MLflow experiment name",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        description="Loguru format string",
    )
    rotation: str = Field(
        default="100 MB",
        description="Log file rotation size",
    )
    retention: str = Field(
        default="7 days",
        description="Log file retention period",
    )
    log_file: str = Field(
        default="logs/tradingbot.log",
        description="Log file path",
    )


class Settings(BaseSettings):
    """Main settings class aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLFlowSettings = Field(default_factory=MLFlowSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are loaded only once.

    Returns:
        Settings: The application settings.
    """
    return Settings()
