# Quick Start Guide

## Prerequisites

Before you begin, ensure you have the following installed:

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** v2.0+
- **Git** for cloning the repository
- **Binance Account** with API keys (for live trading)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/coin-futures-bot.git
cd coin-futures-bot
```

### 2. Set Up Environment

#### Windows

```bash
scripts\setup_environment.bat
```

#### Linux/Mac

```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### 3. Configure API Keys

Edit the `.env` file with your configuration:

```bash
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=true

# Trading Configuration
DEFAULT_SYMBOL=BTCUSDT
TRADING_MODE=paper
RISK_PERCENTAGE=0.02
MAX_POSITION_SIZE=0.1
```

## Running the System

### Start All Services

```bash
docker-compose up -d
```

### Start Only Trading Bot

```bash
docker-compose up trading-bot
```

### View Logs

```bash
# All services
docker-compose logs -f

# Trading bot only
docker-compose logs -f trading-bot
```

## Accessing the Dashboard

Once the system is running, you can access:

- **Trading Dashboard**: <http://localhost:8050>
- **Grafana Monitoring**: <http://localhost:3000> (admin/admin)
- **Prometheus Metrics**: <http://localhost:9090>
- **Jupyter Notebooks**: <http://localhost:8888>

## Trading Modes

### Paper Trading (Default)

The system starts in paper trading mode by default. This allows you to:

- Test strategies without real money
- Monitor performance and system behavior
- Debug and optimize parameters
- Learn the system interface

### Live Trading

⚠️ **WARNING**: Live trading involves real money and significant risk.

To enable live trading:

1. Set `ENABLE_LIVE_TRADING=true` in `.env`
2. Set `BINANCE_TESTNET=false` in `.env`
3. Ensure you have sufficient funds in your Binance account
4. Start with small position sizes

## Configuration

### Basic Configuration

Edit the `.env` file to configure:

```bash
# Trading Parameters
DEFAULT_SYMBOL=BTCUSDT          # Trading symbol
RISK_PERCENTAGE=0.02            # Risk per trade (2%)
MAX_POSITION_SIZE=0.1           # Maximum position size (10%)
MAX_DAILY_DRAWDOWN=0.05         # Maximum daily drawdown (5%)

# AI Model Configuration
MODEL_UPDATE_FREQUENCY=3600     # Model retraining frequency (seconds)
ENABLE_ORDER_BOOK_FEATURES=true # Enable microstructure features
ENABLE_NLP_FEATURES=true        # Enable NLP sentiment analysis
```

### Advanced Configuration

For advanced users, you can modify:

- **Risk Management**: Adjust VaR limits, correlation thresholds
- **AI Models**: Modify model parameters, add new features
- **Execution**: Configure order routing, slippage models
- **Monitoring**: Set up custom alerts and notifications

## Monitoring and Alerts

### Dashboard Features

The trading dashboard provides:

- **Real-time P&L**: Current portfolio value and daily P&L
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate
- **Trade History**: Recent trades with entry/exit points
- **System Health**: CPU, memory, database status
- **Risk Metrics**: VaR, correlation, position exposure

### Setting Up Alerts

Configure alerts in the `.env` file:

```bash
# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# PagerDuty Alerts
PAGERDUTY_API_KEY=your_api_key
```

## Backtesting

### Run a Simple Backtest

```python
from src.modules.backtesting import BacktestingModule
from datetime import datetime, timedelta

# Initialize backtesting module
backtest = BacktestingModule(config, message_queue)
await backtest.initialize()

# Run backtest
result = await backtest.run_single_backtest(
    symbol="BTCUSDT",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    strategy_params={
        "position_size": 0.1,
        "stop_loss": 0.02,
        "take_profit": 0.04
    }
)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

### Walk-Forward Optimization

```python
# Run walk-forward optimization
result = await backtest.run_walk_forward_optimization(
    symbol="BTCUSDT",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31),
    parameter_space={
        "position_size": {"type": "float", "low": 0.05, "high": 0.2},
        "stop_loss": {"type": "float", "low": 0.01, "high": 0.05},
        "take_profit": {"type": "float", "low": 0.02, "high": 0.1}
    }
)
```

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check Docker status
docker-compose ps

# View error logs
docker-compose logs

# Restart services
docker-compose restart
```

#### 2. Database Connection Issues

```bash
# Check TimescaleDB status
docker-compose exec timescaledb pg_isready -U tradingbot

# Reset database
docker-compose down -v
docker-compose up -d
```

#### 3. API Connection Issues

- Verify API keys are correct
- Check network connectivity
- Ensure API keys have trading permissions
- Check rate limits

#### 4. Performance Issues

- Monitor system resources: `docker stats`
- Check database performance
- Review log files for errors
- Adjust configuration parameters

### Getting Help

1. **Check Logs**: Always start by checking the logs
2. **Documentation**: Review the full documentation
3. **Issues**: Create an issue on GitHub
4. **Community**: Join our Discord server

## Best Practices

### Risk Management

1. **Start Small**: Begin with paper trading and small position sizes
2. **Monitor Closely**: Watch the dashboard and logs regularly
3. **Set Limits**: Use appropriate risk limits and stop losses
4. **Diversify**: Don't put all your capital in one strategy

### System Maintenance

1. **Regular Updates**: Keep the system updated
2. **Monitor Performance**: Check metrics and alerts
3. **Backup Data**: Regular database backups
4. **Test Changes**: Always test in paper mode first

### Development

1. **Use Version Control**: Track all changes
2. **Test Thoroughly**: Comprehensive testing before deployment
3. **Document Changes**: Keep documentation updated
4. **Monitor Metrics**: Track performance improvements

## Next Steps

1. **Learn the System**: Spend time understanding the dashboard and features
2. **Paper Trade**: Practice with paper trading for at least a week
3. **Optimize Parameters**: Use backtesting to find optimal settings
4. **Start Small**: Begin live trading with small amounts
5. **Scale Up**: Gradually increase position sizes as you gain confidence

## Support

For additional help and support:

- **Documentation**: [Full Documentation](docs/)
- **GitHub Issues**: [Report Issues](https://github.com/your-username/coin-futures-bot/issues)
- **Discord**: [Join Community](https://discord.gg/your-discord)
- **Email**: <support@tradingbot.com>

Remember: Trading involves significant risk. Never trade with money you can't afford to lose. Always test thoroughly in paper mode before going live.
