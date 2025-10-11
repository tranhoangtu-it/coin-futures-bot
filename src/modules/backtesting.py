"""
Module 6: Backtesting & Optimization Framework
R&D lab to test ideas scientifically and find robust parameter sets.
Implements event-driven simulation, walk-forward optimization, and realistic market simulation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.sklearn

from src.config import Config
from src.core.message_queue import MessageQueue, MessageType
from src.database.timescale import TimescaleDB
from src.features.technical_indicators import TechnicalIndicators
from src.features.microstructure import MicrostructureFeatures


class BacktestEvent(Enum):
    """Backtest event types."""
    MARKET_DATA_RECEIVED = "market_data_received"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    POSITION_UPDATED = "position_updated"


@dataclass
class BacktestEventData:
    """Backtest event data."""
    event_type: BacktestEvent
    timestamp: float
    symbol: str
    data: Dict[str, Any]


@dataclass
class BacktestResult:
    """Backtest result data."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    average_trade_duration: float
    volatility: float
    var_95: float
    cvar_95: float
    equity_curve: List[Dict[str, Any]]
    trade_log: List[Dict[str, Any]]


@dataclass
class WalkForwardResult:
    """Walk-forward optimization result."""
    train_periods: List[Tuple[datetime, datetime]]
    test_periods: List[Tuple[datetime, datetime]]
    results: List[BacktestResult]
    best_params: Dict[str, Any]
    out_of_sample_returns: List[float]


class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Backtest state
        self.current_time = None
        self.portfolio_value = config.INITIAL_CAPITAL
        self.initial_capital = config.INITIAL_CAPITAL
        self.cash = config.INITIAL_CAPITAL
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.equity_curve = []
        
        # Event queue
        self.event_queue = []
        self.event_handlers = {}
        
        # Market data
        self.market_data = {}
        self.features = {}
        
        # Performance tracking
        self.daily_returns = []
        self.trade_returns = []
        
        # Slippage and commission models
        self.commission_rate = config.COMMISSION_RATE
        self.slippage_model = SlippageModel()
        
        # Feature engineering
        self.technical_indicators = TechnicalIndicators()
        self.microstructure_features = MicrostructureFeatures()
    
    async def initialize(self):
        """Initialize the backtest engine."""
        await self.technical_indicators.initialize()
        await self.microstructure_features.initialize()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register event handlers."""
        self.event_handlers = {
            BacktestEvent.MARKET_DATA_RECEIVED: self._handle_market_data,
            BacktestEvent.SIGNAL_GENERATED: self._handle_signal,
            BacktestEvent.ORDER_CREATED: self._handle_order_created,
            BacktestEvent.ORDER_FILLED: self._handle_order_filled,
            BacktestEvent.ORDER_CANCELED: self._handle_order_canceled,
            BacktestEvent.POSITION_UPDATED: self._handle_position_updated
        }
    
    async def run_backtest(self, start_date: datetime, end_date: datetime, 
                          symbol: str, strategy_params: Dict[str, Any]) -> BacktestResult:
        """Run a complete backtest."""
        self.logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Reset state
        self._reset_state()
        
        # Load market data
        await self._load_market_data(symbol, start_date, end_date)
        
        # Generate events
        await self._generate_events(symbol, strategy_params)
        
        # Process events
        await self._process_events()
        
        # Calculate results
        result = await self._calculate_results()
        
        self.logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        return result
    
    def _reset_state(self):
        """Reset backtest state."""
        self.current_time = None
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.trade_returns = []
        self.event_queue = []
    
    async def _load_market_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """Load market data for backtesting."""
        # This would typically load from a database
        # For now, we'll simulate loading data
        self.logger.info(f"Loading market data for {symbol}")
        
        # Generate synthetic data for demonstration
        date_range = pd.date_range(start_date, end_date, freq='1min')
        np.random.seed(42)  # For reproducible results
        
        # Generate price data with trend and volatility
        returns = np.random.normal(0.0001, 0.02, len(date_range))  # 0.01% mean return, 2% volatility
        prices = [100.0]  # Starting price
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create market data
        self.market_data[symbol] = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(date_range))
        })
        
        # Calculate features
        await self._calculate_features(symbol)
    
    async def _calculate_features(self, symbol: str):
        """Calculate technical indicators and features."""
        df = self.market_data[symbol].copy()
        df.set_index('timestamp', inplace=True)
        
        # Calculate technical indicators
        features = await self.technical_indicators.calculate_all(df)
        
        # Store features
        self.features[symbol] = features
    
    async def _generate_events(self, symbol: str, strategy_params: Dict[str, Any]):
        """Generate backtest events."""
        df = self.market_data[symbol]
        
        for idx, row in df.iterrows():
            timestamp = row['timestamp'].timestamp()
            
            # Market data event
            market_data_event = BacktestEventData(
                event_type=BacktestEvent.MARKET_DATA_RECEIVED,
                timestamp=timestamp,
                symbol=symbol,
                data={
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
            )
            self.event_queue.append(market_data_event)
            
            # Generate trading signals (simplified strategy)
            if idx > 20:  # Need some history for indicators
                signal = await self._generate_signal(symbol, timestamp, strategy_params)
                if signal:
                    signal_event = BacktestEventData(
                        event_type=BacktestEvent.SIGNAL_GENERATED,
                        timestamp=timestamp,
                        symbol=symbol,
                        data=signal
                    )
                    self.event_queue.append(signal_event)
        
        # Sort events by timestamp
        self.event_queue.sort(key=lambda x: x.timestamp)
    
    async def _generate_signal(self, symbol: str, timestamp: float, 
                             strategy_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on strategy parameters."""
        # Simple moving average crossover strategy
        df = self.market_data[symbol]
        current_idx = df[df['timestamp'] == pd.to_datetime(timestamp, unit='s')].index[0]
        
        if current_idx < 50:  # Need enough data
            return None
        
        # Calculate moving averages
        short_ma = df['close'].rolling(window=20).mean().iloc[current_idx]
        long_ma = df['close'].rolling(window=50).mean().iloc[current_idx]
        prev_short_ma = df['close'].rolling(window=20).mean().iloc[current_idx - 1]
        prev_long_ma = df['close'].rolling(window=50).mean().iloc[current_idx - 1]
        
        current_price = df['close'].iloc[current_idx]
        
        # Generate signal
        signal = None
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            # Golden cross - buy signal
            signal = {
                'action': 'BUY',
                'price': current_price,
                'quantity': strategy_params.get('position_size', 0.1),
                'confidence': 0.8
            }
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            # Death cross - sell signal
            signal = {
                'action': 'SELL',
                'price': current_price,
                'quantity': strategy_params.get('position_size', 0.1),
                'confidence': 0.8
            }
        
        return signal
    
    async def _process_events(self):
        """Process all events in chronological order."""
        for event in self.event_queue:
            self.current_time = event.timestamp
            
            # Handle event
            handler = self.event_handlers.get(event.event_type)
            if handler:
                await handler(event)
            
            # Update portfolio value
            await self._update_portfolio_value()
            
            # Record equity curve point
            self.equity_curve.append({
                'timestamp': self.current_time,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions_value': sum(pos['quantity'] * pos['current_price'] 
                                     for pos in self.positions.values())
            })
    
    async def _handle_market_data(self, event: BacktestEventData):
        """Handle market data event."""
        symbol = event.symbol
        data = event.data
        
        # Update current price in positions
        for pos in self.positions.values():
            if pos['symbol'] == symbol:
                pos['current_price'] = data['close']
    
    async def _handle_signal(self, event: BacktestEventData):
        """Handle trading signal event."""
        symbol = event.symbol
        signal = event.data
        
        # Create order
        order_id = f"order_{len(self.orders)}_{int(self.current_time)}"
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': signal['action'],
            'quantity': signal['quantity'],
            'price': signal['price'],
            'timestamp': self.current_time,
            'status': 'NEW'
        }
        
        self.orders[order_id] = order
        
        # Create order created event
        order_event = BacktestEventData(
            event_type=BacktestEvent.ORDER_CREATED,
            timestamp=self.current_time,
            symbol=symbol,
            data=order
        )
        
        # Process order immediately (simplified)
        await self._process_order(order)
    
    async def _handle_order_created(self, event: BacktestEventData):
        """Handle order created event."""
        # Orders are processed immediately in this simplified version
        pass
    
    async def _handle_order_filled(self, event: BacktestEventData):
        """Handle order filled event."""
        order_data = event.data
        order_id = order_data['order_id']
        
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'FILLED'
    
    async def _handle_order_canceled(self, event: BacktestEventData):
        """Handle order canceled event."""
        order_data = event.data
        order_id = order_data['order_id']
        
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELED'
    
    async def _handle_position_updated(self, event: BacktestEventData):
        """Handle position updated event."""
        # Position updates are handled in _process_order
        pass
    
    async def _process_order(self, order: Dict[str, Any]):
        """Process an order and update positions."""
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        price = order['price']
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(quantity, price)
        execution_price = price * (1 + slippage)
        
        # Calculate commission
        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate
        
        # Check if we have enough cash for buy orders
        if side == 'BUY' and trade_value + commission > self.cash:
            self.logger.warning(f"Insufficient cash for buy order: {trade_value + commission} > {self.cash}")
            return
        
        # Update cash
        if side == 'BUY':
            self.cash -= (trade_value + commission)
        else:
            self.cash += (trade_value - commission)
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': 0,
                'average_price': 0,
                'current_price': price
            }
        
        position = self.positions[symbol]
        
        if side == 'BUY':
            # Add to position
            if position['quantity'] >= 0:
                # Adding to long position
                total_quantity = position['quantity'] + quantity
                total_value = (position['quantity'] * position['average_price'] + 
                             quantity * execution_price)
                position['average_price'] = total_value / total_quantity if total_quantity > 0 else 0
                position['quantity'] = total_quantity
            else:
                # Covering short position
                if quantity >= abs(position['quantity']):
                    # Complete cover
                    remaining_quantity = quantity - abs(position['quantity'])
                    position['quantity'] = remaining_quantity
                    position['average_price'] = execution_price
                else:
                    # Partial cover
                    position['quantity'] += quantity
        else:
            # Sell order
            if position['quantity'] <= 0:
                # Adding to short position
                total_quantity = position['quantity'] - quantity
                total_value = (abs(position['quantity']) * position['average_price'] + 
                             quantity * execution_price)
                position['average_price'] = total_value / abs(total_quantity) if total_quantity < 0 else 0
                position['quantity'] = total_quantity
            else:
                # Reducing long position
                if quantity >= position['quantity']:
                    # Complete sell
                    remaining_quantity = quantity - position['quantity']
                    position['quantity'] = -remaining_quantity
                    position['average_price'] = execution_price
                else:
                    # Partial sell
                    position['quantity'] -= quantity
        
        # Record trade
        trade = {
            'timestamp': self.current_time,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': execution_price,
            'commission': commission,
            'pnl': 0  # Will be calculated when position is closed
        }
        
        self.trades.append(trade)
        
        # Create position updated event
        position_event = BacktestEventData(
            event_type=BacktestEvent.POSITION_UPDATED,
            timestamp=self.current_time,
            symbol=symbol,
            data=position
        )
        
        # Mark order as filled
        order['status'] = 'FILLED'
        order_event = BacktestEventData(
            event_type=BacktestEvent.ORDER_FILLED,
            timestamp=self.current_time,
            symbol=symbol,
            data=order
        )
    
    async def _update_portfolio_value(self):
        """Update portfolio value."""
        positions_value = 0
        for position in self.positions.values():
            if position['quantity'] != 0:
                positions_value += position['quantity'] * position['current_price']
        
        self.portfolio_value = self.cash + positions_value
    
    async def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        if not self.equity_curve:
            return BacktestResult(
                total_return=0, sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                calmar_ratio=0, win_rate=0, profit_factor=0, total_trades=0,
                average_trade_duration=0, volatility=0, var_95=0, cvar_95=0,
                equity_curve=[], trade_log=[]
            )
        
        # Calculate returns
        equity_values = [point['portfolio_value'] for point in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Basic metrics
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% risk-free rate
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = (np.mean(returns) / np.std(downside_returns) * np.sqrt(252) 
                        if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0)
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Calmar ratio
        calmar_ratio = (np.mean(returns) * 252 / max_drawdown 
                       if max_drawdown > 0 else 0)
        
        # Trade statistics
        trade_pnls = [trade['pnl'] for trade in self.trades if trade['pnl'] != 0]
        if trade_pnls:
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
            profit_factor = (sum(winning_trades) / abs(sum(losing_trades)) 
                           if losing_trades and sum(losing_trades) != 0 else 0)
        else:
            win_rate = 0
            profit_factor = 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0
        
        # Average trade duration
        trade_durations = []
        for i in range(1, len(self.trades)):
            duration = self.trades[i]['timestamp'] - self.trades[i-1]['timestamp']
            trade_durations.append(duration)
        
        average_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            average_trade_duration=average_trade_duration,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            equity_curve=self.equity_curve,
            trade_log=self.trades
        )


class SlippageModel:
    """Slippage model for realistic trade simulation."""
    
    def __init__(self):
        self.base_slippage = 0.0001  # 0.01% base slippage
    
    def calculate_slippage(self, quantity: float, price: float) -> float:
        """Calculate slippage based on order size and price."""
        # Linear slippage model
        size_impact = min(quantity / 1000, 0.01)  # Max 1% slippage for large orders
        return self.base_slippage + size_impact


class WalkForwardOptimizer:
    """Walk-forward optimization engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backtest_engine = BacktestEngine(config)
    
    async def optimize(self, symbol: str, start_date: datetime, end_date: datetime,
                      parameter_space: Dict[str, Any]) -> WalkForwardResult:
        """Run walk-forward optimization."""
        self.logger.info(f"Starting walk-forward optimization for {symbol}")
        
        # Initialize MLflow
        mlflow.set_experiment(f"walk_forward_{symbol}")
        
        # Define time series splits
        splits = self._create_time_splits(start_date, end_date)
        
        train_periods = []
        test_periods = []
        results = []
        
        for train_start, train_end, test_start, test_end in splits:
            train_periods.append((train_start, train_end))
            test_periods.append((test_start, test_end))
            
            # Optimize parameters on training period
            best_params = await self._optimize_parameters(
                symbol, train_start, train_end, parameter_space
            )
            
            # Test on out-of-sample period
            test_result = await self.backtest_engine.run_backtest(
                test_start, test_end, symbol, best_params
            )
            
            results.append(test_result)
            
            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params(best_params)
                mlflow.log_metrics({
                    'total_return': test_result.total_return,
                    'sharpe_ratio': test_result.sharpe_ratio,
                    'max_drawdown': test_result.max_drawdown,
                    'win_rate': test_result.win_rate
                })
        
        # Calculate out-of-sample returns
        out_of_sample_returns = [result.total_return for result in results]
        
        # Find best parameters across all periods
        best_result_idx = np.argmax([result.sharpe_ratio for result in results])
        best_params = {}  # Would be retrieved from optimization
        
        return WalkForwardResult(
            train_periods=train_periods,
            test_periods=test_periods,
            results=results,
            best_params=best_params,
            out_of_sample_returns=out_of_sample_returns
        )
    
    def _create_time_splits(self, start_date: datetime, end_date: datetime) -> List[Tuple]:
        """Create time series splits for walk-forward optimization."""
        total_days = (end_date - start_date).days
        train_days = total_days // 3  # 1/3 for training
        test_days = total_days // 6  # 1/6 for testing
        step_days = test_days  # Step size
        
        splits = []
        current_start = start_date
        
        while current_start + timedelta(days=train_days + test_days) <= end_date:
            train_start = current_start
            train_end = current_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)
            
            splits.append((train_start, train_end, test_start, test_end))
            current_start += timedelta(days=step_days)
        
        return splits
    
    async def _optimize_parameters(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters using Optuna."""
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Run backtest
            result = asyncio.run(self.backtest_engine.run_backtest(
                start_date, end_date, symbol, params
            ))
            
            # Return objective (Sharpe ratio)
            return result.sharpe_ratio
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=50)
        
        return study.best_params


class BacktestingModule:
    """Backtesting and optimization module."""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.backtest_engine = BacktestEngine(config)
        self.walk_forward_optimizer = WalkForwardOptimizer(config)
        
        # Database connections
        self.timescale_db = TimescaleDB(config)
        
        # State
        self.running = False
    
    async def initialize(self):
        """Initialize the backtesting module."""
        self.logger.info("Initializing backtesting module...")
        
        # Initialize database connections
        await self.timescale_db.initialize()
        
        # Initialize backtest engine
        await self.backtest_engine.initialize()
        
        self.logger.info("Backtesting module initialized")
    
    async def start(self):
        """Start the backtesting module."""
        self.logger.info("Starting backtesting module...")
        self.running = True
        
        # Start backtest processing loop
        asyncio.create_task(self._backtest_processing_loop())
        
        self.logger.info("Backtesting module started")
    
    async def stop(self):
        """Stop the backtesting module."""
        self.logger.info("Stopping backtesting module...")
        self.running = False
        
        # Close database connections
        await self.timescale_db.close()
        
        self.logger.info("Backtesting module stopped")
    
    async def _backtest_processing_loop(self):
        """Process backtest requests."""
        while self.running:
            try:
                # Check for backtest requests
                # This would typically listen to a message queue
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in backtest processing loop: {e}")
                await asyncio.sleep(60)
    
    async def run_single_backtest(self, symbol: str, start_date: datetime, 
                                end_date: datetime, strategy_params: Dict[str, Any]) -> BacktestResult:
        """Run a single backtest."""
        return await self.backtest_engine.run_backtest(
            start_date, end_date, symbol, strategy_params
        )
    
    async def run_walk_forward_optimization(self, symbol: str, start_date: datetime,
                                          end_date: datetime, parameter_space: Dict[str, Any]) -> WalkForwardResult:
        """Run walk-forward optimization."""
        return await self.walk_forward_optimizer.optimize(
            symbol, start_date, end_date, parameter_space
        )
