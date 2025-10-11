# API Reference

## Configuration API

### Config Class

The main configuration class that manages all system settings.

```python
from src.config import Config

# Load configuration from file
config = Config.from_file("config.env")

# Access configuration values
api_key = config.BINANCE_API_KEY
risk_pct = config.RISK_PERCENTAGE
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `BINANCE_API_KEY` | str | Required | Binance API key |
| `BINANCE_SECRET_KEY` | str | Required | Binance secret key |
| `BINANCE_TESTNET` | bool | True | Use testnet for trading |
| `TIMESCALEDB_URL` | str | Required | TimescaleDB connection URL |
| `REDIS_URL` | str | Required | Redis connection URL |
| `KAFKA_BOOTSTRAP_SERVERS` | str | localhost:9092 | Kafka servers |
| `DEFAULT_SYMBOL` | str | BTCUSDT | Default trading symbol |
| `TRADING_MODE` | str | paper | Trading mode (live/paper/backtest) |
| `RISK_PERCENTAGE` | float | 0.02 | Risk per trade (2%) |
| `MAX_POSITION_SIZE` | float | 0.1 | Maximum position size (10%) |
| `MAX_DAILY_DRAWDOWN` | float | 0.05 | Maximum daily drawdown (5%) |

## Message Queue API

### MessageQueue Class

Central message queue for inter-module communication.

```python
from src.core.message_queue import MessageQueue, MessageType

# Initialize message queue
message_queue = MessageQueue(config)
await message_queue.initialize()

# Publish a message
message = message_queue.create_message(
    MessageType.SIGNAL_GENERATED,
    "ai_core",
    {"symbol": "BTCUSDT", "action": "BUY", "confidence": 0.8}
)
await message_queue.publish("signals.BTCUSDT", message)

# Subscribe to messages
await message_queue.subscribe("signals.BTCUSDT", callback_function)
```

#### Message Types

| Type | Description | Data Fields |
|------|-------------|-------------|
| `MARKET_DATA_RECEIVED` | Market data received | symbol, data |
| `SIGNAL_GENERATED` | Trading signal generated | symbol, action, confidence |
| `ORDER_CREATED` | Order created | order_id, symbol, side, quantity |
| `ORDER_FILLED` | Order filled | order_id, symbol, price, quantity |
| `POSITION_UPDATE` | Position updated | symbol, side, size, pnl |
| `ALERT` | System alert | alert_type, message, severity |

## Data Ingestion API

### DataIngestionModule Class

Handles real-time data collection and feature engineering.

```python
from src.modules.data_ingestion import DataIngestionModule

# Initialize module
data_module = DataIngestionModule(config, message_queue)
await data_module.initialize()

# Start data collection
await data_module.start()
```

#### Methods

- `initialize()`: Initialize the data ingestion module
- `start()`: Start data collection and feature engineering
- `stop()`: Stop data collection
- `_handle_kline_stream(symbol)`: Handle kline data stream
- `_handle_order_book_stream(symbol)`: Handle order book stream
- `_handle_trade_stream(symbol)`: Handle trade stream

## AI Core API

### AICoreModule Class

Multi-model AI system for market analysis and signal generation.

```python
from src.modules.ai_core import AICoreModule

# Initialize AI core
ai_core = AICoreModule(config, message_queue)
await ai_core.initialize()

# Start AI processing
await ai_core.start()
```

#### Components

##### MarketRegimeDetector

```python
from src.modules.ai_core import MarketRegimeDetector

detector = MarketRegimeDetector(config)
await detector.train(features_df)
regime, confidence = await detector.detect_regime(features)
```

##### AlphaModel

```python
from src.modules.ai_core import AlphaModel, TransformerAlphaModel, LightGBMAlphaModel

# Transformer model for breakout patterns
transformer_model = TransformerAlphaModel(config)
await transformer_model.train(features_df, target)
prediction, confidence = await transformer_model.predict(features)

# LightGBM model for microstructure
lgb_model = LightGBMAlphaModel(config)
await lgb_model.train(features_df, target)
prediction, confidence = await lgb_model.predict(features)
```

## Risk Management API

### RiskManagementModule Class

Comprehensive risk management and position sizing.

```python
from src.modules.risk_management import RiskManagementModule

# Initialize risk management
risk_module = RiskManagementModule(config, message_queue)
await risk_module.initialize()

# Start risk monitoring
await risk_module.start()
```

#### Methods

- `_check_risk(symbol, action, position_size, confidence)`: Perform risk check
- `_calculate_volatility_adjusted_size(symbol, base_size, confidence)`: Calculate position size
- `_calculate_correlation_risk(symbol)`: Calculate correlation risk
- `_update_position_from_order(order_data)`: Update position from order
- `_close_all_positions(reason)`: Emergency close all positions

#### RiskCheckResult

```python
@dataclass
class RiskCheckResult:
    approved: bool
    reason: str
    suggested_position_size: float
    risk_level: RiskLevel
    warnings: List[str]
```

## Execution Engine API

### ExecutionEngineModule Class

Order execution with smart routing and latency optimization.

```python
from src.modules.execution_engine import ExecutionEngineModule

# Initialize execution engine
execution_engine = ExecutionEngineModule(config, message_queue)
await execution_engine.initialize()

# Start order processing
await execution_engine.start()
```

#### Order Management

```python
from src.modules.execution_engine import Order, OrderSide, OrderType, OrderStatus

# Create order
order = Order(
    order_id="order_123",
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=0.1,
    price=None,
    status=OrderStatus.NEW
)

# Submit order
await execution_engine._submit_order_to_exchange(order)
```

#### Smart Order Routing

```python
from src.modules.execution_engine import SmartOrderRouter

router = SmartOrderRouter(config)
child_orders = await router.route_order(order, market_data)
```

## Monitoring API

### MonitoringModule Class

Real-time monitoring and dashboard.

```python
from src.modules.monitoring import MonitoringModule

# Initialize monitoring
monitoring = MonitoringModule(config, message_queue)
await monitoring.initialize()

# Start monitoring
await monitoring.start()
```

#### Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
```

## Backtesting API

### BacktestingModule Class

Comprehensive backtesting and optimization framework.

```python
from src.modules.backtesting import BacktestingModule

# Initialize backtesting
backtest = BacktestingModule(config, message_queue)
await backtest.initialize()

# Run single backtest
result = await backtest.run_single_backtest(
    symbol="BTCUSDT",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    strategy_params={"position_size": 0.1}
)

# Run walk-forward optimization
wf_result = await backtest.run_walk_forward_optimization(
    symbol="BTCUSDT",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31),
    parameter_space={
        "position_size": {"type": "float", "low": 0.05, "high": 0.2}
    }
)
```

#### BacktestResult

```python
@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: List[Dict[str, Any]]
    trade_log: List[Dict[str, Any]]
```

## Feature Engineering API

### TechnicalIndicators Class

Advanced technical indicators calculation.

```python
from src.features.technical_indicators import TechnicalIndicators

indicators = TechnicalIndicators()
await indicators.initialize()

# Calculate all indicators
features = await indicators.calculate_all(df)

# Available indicators
print(features.keys())
# ['rsi', 'atr', 'macd', 'bb_upper', 'bb_lower', 'ichimoku_tenkan', ...]
```

### MicrostructureFeatures Class

Order book and trade microstructure analysis.

```python
from src.features.microstructure import MicrostructureFeatures

microstructure = MicrostructureFeatures()
await microstructure.initialize()

# Calculate order book features
ob_features = await microstructure.calculate_order_book_features(order_book_data)

# Calculate trade features
trade_features = await microstructure.calculate_trade_features(trade_data)
```

### NLPFeatures Class

Natural language processing for sentiment analysis.

```python
from src.features.nlp_features import NLPFeatures

nlp = NLPFeatures()
await nlp.initialize()

# Calculate sentiment score
sentiment = await nlp.calculate_sentiment_score("Bitcoin is going up!")

# Extract named entities
entities = await nlp.extract_named_entities("Bitcoin price increased by 5%")

# Calculate topic sentiment
topic_sentiment = await nlp.calculate_topic_sentiment(news_texts)
```

## Database API

### TimescaleDB Class

Time-series database operations.

```python
from src.database.timescale import TimescaleDB

db = TimescaleDB(config)
await db.initialize()

# Store kline data
await db.store_kline_data(
    symbol="BTCUSDT",
    timestamp=time.time(),
    open_price=50000.0,
    high_price=51000.0,
    low_price=49000.0,
    close_price=50500.0,
    volume=1000.0,
    interval="1m"
)

# Get recent klines
klines = await db.get_recent_klines("BTCUSDT", limit=1000)

# Store features
await db.store_features("BTCUSDT", features_dict)

# Get latest features
features = await db.get_latest_features("BTCUSDT")
```

### RedisCache Class

In-memory caching for hot data.

```python
from src.database.redis_cache import RedisCache

cache = RedisCache(config)
await cache.initialize()

# Cache features
await cache.set_latest_features("BTCUSDT", features_dict)

# Get cached features
features = await cache.get_latest_features("BTCUSDT")

# Cache market data
await cache.set_market_data("BTCUSDT", "kline", market_data)

# Set alerts
await cache.set_alert("RISK_BREACH", "Drawdown limit exceeded", "CRITICAL")
```

## Error Handling

### Common Exceptions

```python
from src.exceptions import (
    TradingBotError,
    ConfigurationError,
    DatabaseError,
    APIError,
    RiskManagementError
)

try:
    # Trading bot operation
    pass
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except DatabaseError as e:
    print(f"Database error: {e}")
except APIError as e:
    print(f"API error: {e}")
except RiskManagementError as e:
    print(f"Risk management error: {e}")
except TradingBotError as e:
    print(f"General error: {e}")
```

## Logging

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Get logger
logger = logging.getLogger(__name__)
logger.info("Trading bot started")
```

### Log Levels

- `DEBUG`: Detailed information for debugging
- `INFO`: General information about system operation
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for recoverable errors
- `CRITICAL`: Critical errors that may cause system failure

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_technical_indicators.py

# Run with coverage
pytest --cov=src tests/

# Run async tests
pytest -k async
```

### Test Structure

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
async def mock_config():
    config = Mock()
    config.MAX_POSITION_SIZE = 0.1
    return config

@pytest.mark.asyncio
async def test_feature_calculation(mock_config):
    # Test implementation
    pass
```

## Performance Optimization

### Async Programming

```python
import asyncio

# Use async/await for I/O operations
async def process_data():
    data = await fetch_data()
    processed = await process_data_async(data)
    return processed

# Run multiple operations concurrently
async def parallel_processing():
    tasks = [
        process_symbol("BTCUSDT"),
        process_symbol("ETHUSDT"),
        process_symbol("ADAUSDT")
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Caching Strategies

```python
# Use Redis for hot data caching
await cache.set_latest_features(symbol, features, ttl=300)  # 5 minutes

# Use in-memory caching for frequently accessed data
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_indicator(data_hash):
    # Expensive calculation
    pass
```

This API reference provides comprehensive documentation for all major components of the trading bot system. For more detailed information, refer to the source code and inline documentation.
