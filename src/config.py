"""
Configuration management for the trading bot.
Handles environment variables and configuration validation.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseSettings, Field, validator


class Config(BaseSettings):
    """Main configuration class for the trading bot."""
    
    # Binance API Configuration
    BINANCE_API_KEY: str = Field(..., env="BINANCE_API_KEY")
    BINANCE_SECRET_KEY: str = Field(..., env="BINANCE_SECRET_KEY")
    BINANCE_TESTNET: bool = Field(True, env="BINANCE_TESTNET")
    
    # Database Configuration
    TIMESCALEDB_URL: str = Field(..., env="TIMESCALEDB_URL")
    REDIS_URL: str = Field(..., env="REDIS_URL")
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    KAFKA_GROUP_ID: str = Field("trading_bot_group", env="KAFKA_GROUP_ID")
    
    # Trading Configuration
    DEFAULT_SYMBOL: str = Field("BTCUSDT", env="DEFAULT_SYMBOL")
    TRADING_MODE: str = Field("paper", env="TRADING_MODE")  # live, paper, backtest
    RISK_PERCENTAGE: float = Field(0.02, env="RISK_PERCENTAGE")
    MAX_POSITION_SIZE: float = Field(0.1, env="MAX_POSITION_SIZE")
    MAX_DAILY_DRAWDOWN: float = Field(0.05, env="MAX_DAILY_DRAWDOWN")
    
    # AI Model Configuration
    MODEL_UPDATE_FREQUENCY: int = Field(3600, env="MODEL_UPDATE_FREQUENCY")  # seconds
    ENABLE_PAPER_TRADING: bool = Field(True, env="ENABLE_PAPER_TRADING")
    ENABLE_LIVE_TRADING: bool = Field(False, env="ENABLE_LIVE_TRADING")
    
    # Monitoring Configuration
    GRAFANA_URL: str = Field("http://localhost:3000", env="GRAFANA_URL")
    PROMETHEUS_URL: str = Field("http://localhost:9090", env="PROMETHEUS_URL")
    PAGERDUTY_API_KEY: Optional[str] = Field(None, env="PAGERDUTY_API_KEY")
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")
    
    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field("logs/trading_bot.log", env="LOG_FILE")
    
    # Feature Engineering
    ENABLE_ORDER_BOOK_FEATURES: bool = Field(True, env="ENABLE_ORDER_BOOK_FEATURES")
    ENABLE_NLP_FEATURES: bool = Field(True, env="ENABLE_NLP_FEATURES")
    ENABLE_ONCHAIN_FEATURES: bool = Field(True, env="ENABLE_ONCHAIN_FEATURES")
    
    # External Data Sources
    GLASSNODE_API_KEY: Optional[str] = Field(None, env="GLASSNODE_API_KEY")
    SANTIMENT_API_KEY: Optional[str] = Field(None, env="SANTIMENT_API_KEY")
    CRYPTOPANIC_API_KEY: Optional[str] = Field(None, env="CRYPTOPANIC_API_KEY")
    
    # Data Ingestion Configuration
    WEBSOCKET_RECONNECT_DELAY: int = Field(5, env="WEBSOCKET_RECONNECT_DELAY")
    REST_API_RATE_LIMIT: int = Field(1200, env="REST_API_RATE_LIMIT")  # requests per minute
    DATA_RETENTION_DAYS: int = Field(365, env="DATA_RETENTION_DAYS")
    
    # Risk Management Configuration
    VAR_CONFIDENCE_LEVEL: float = Field(0.95, env="VAR_CONFIDENCE_LEVEL")
    MAX_CORRELATION: float = Field(0.7, env="MAX_CORRELATION")
    STOP_LOSS_ATR_MULTIPLIER: float = Field(3.0, env="STOP_LOSS_ATR_MULTIPLIER")
    TAKE_PROFIT_ATR_MULTIPLIER: float = Field(2.0, env="TAKE_PROFIT_ATR_MULTIPLIER")
    
    # Execution Engine Configuration
    MAX_ORDER_SIZE: float = Field(1000.0, env="MAX_ORDER_SIZE")
    ORDER_TIMEOUT: int = Field(30, env="ORDER_TIMEOUT")  # seconds
    SLIPPAGE_TOLERANCE: float = Field(0.001, env="SLIPPAGE_TOLERANCE")  # 0.1%
    
    # Backtesting Configuration
    BACKTEST_START_DATE: str = Field("2020-01-01", env="BACKTEST_START_DATE")
    BACKTEST_END_DATE: str = Field("2024-01-01", env="BACKTEST_END_DATE")
    INITIAL_CAPITAL: float = Field(100000.0, env="INITIAL_CAPITAL")
    COMMISSION_RATE: float = Field(0.001, env="COMMISSION_RATE")  # 0.1%
    
    @validator('TRADING_MODE')
    def validate_trading_mode(cls, v):
        if v not in ['live', 'paper', 'backtest']:
            raise ValueError('TRADING_MODE must be one of: live, paper, backtest')
        return v
    
    @validator('RISK_PERCENTAGE', 'MAX_POSITION_SIZE', 'MAX_DAILY_DRAWDOWN')
    def validate_percentages(cls, v):
        if not 0 < v <= 1:
            raise ValueError('Percentage values must be between 0 and 1')
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of: {valid_levels}')
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load environment variables from file
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        
        return cls()
    
    def get_symbols(self) -> List[str]:
        """Get list of trading symbols."""
        return [self.DEFAULT_SYMBOL]  # Can be extended to support multiple symbols
    
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.TRADING_MODE == 'live' and self.ENABLE_LIVE_TRADING
    
    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled."""
        return self.TRADING_MODE == 'paper' and self.ENABLE_PAPER_TRADING
    
    def is_backtesting(self) -> bool:
        """Check if backtesting mode is enabled."""
        return self.TRADING_MODE == 'backtest'
