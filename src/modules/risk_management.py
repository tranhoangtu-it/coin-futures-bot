"""
Module 3: Risk & Position Management Engine
Acts as the Chief Risk Officer with absolute veto power over AI signals.
Implements VaR, position sizing, and portfolio-level controls.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from scipy import stats
from binance import AsyncClient

from src.config import Config
from src.core.message_queue import MessageQueue, MessageType
from src.database.timescale import TimescaleDB
from src.database.redis_cache import RedisCache


class RiskLevel(Enum):
    """Risk levels for position sizing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Position information."""
    symbol: str
    side: str  # LONG, SHORT
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: float
    stop_loss: Optional[float]
    take_profit: Optional[float]


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio."""
    portfolio_value: float
    total_exposure: float
    var_1d: float
    var_1d_percent: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_risk: float


@dataclass
class RiskCheckResult:
    """Result of risk check."""
    approved: bool
    reason: str
    suggested_position_size: float
    risk_level: RiskLevel
    warnings: List[str]


class RiskManagementModule:
    """Risk and position management module."""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = logging.getLogger(__name__)
        
        # Binance client
        self.client = None
        
        # Database connections
        self.timescale_db = TimescaleDB(config)
        self.redis_cache = RedisCache(config)
        
        # Risk state
        self.positions = {}
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.max_portfolio_value = 0.0
        self.risk_metrics = None
        
        # Risk limits
        self.max_position_size = config.MAX_POSITION_SIZE
        self.max_daily_drawdown = config.MAX_DAILY_DRAWDOWN
        self.max_correlation = config.MAX_CORRELATION
        self.var_confidence_level = config.VAR_CONFIDENCE_LEVEL
        
        # State
        self.running = False
        self.last_risk_update = 0
    
    async def initialize(self):
        """Initialize the risk management module."""
        self.logger.info("Initializing risk management module...")
        
        # Initialize Binance client
        self.client = await AsyncClient.create(
            api_key=self.config.BINANCE_API_KEY,
            api_secret=self.config.BINANCE_SECRET_KEY,
            testnet=self.config.BINANCE_TESTNET
        )
        
        # Initialize database connections
        await self.timescale_db.initialize()
        await self.redis_cache.initialize()
        
        # Subscribe to signals and order updates
        await self.message_queue.subscribe(
            f"signals.{config.DEFAULT_SYMBOL}",
            self._handle_trading_signal
        )
        await self.message_queue.subscribe(
            "orders",
            self._handle_order_update
        )
        
        # Load existing positions
        await self._load_positions()
        
        self.logger.info("Risk management module initialized")
    
    async def start(self):
        """Start the risk management module."""
        self.logger.info("Starting risk management module...")
        self.running = True
        
        # Start risk monitoring loop
        asyncio.create_task(self._risk_monitoring_loop())
        
        # Start position update loop
        asyncio.create_task(self._position_update_loop())
        
        self.logger.info("Risk management module started")
    
    async def stop(self):
        """Stop the risk management module."""
        self.logger.info("Stopping risk management module...")
        self.running = False
        
        # Close Binance client
        if self.client:
            await self.client.close_connection()
        
        # Close database connections
        await self.timescale_db.close()
        await self.redis_cache.close()
        
        self.logger.info("Risk management module stopped")
    
    async def _handle_trading_signal(self, message):
        """Handle incoming trading signal."""
        try:
            signal_data = message.data
            symbol = signal_data.get('symbol')
            action = signal_data.get('action')
            confidence = signal_data.get('confidence', 0.0)
            position_size = signal_data.get('position_size', 0.0)
            
            # Perform risk check
            risk_result = await self._check_risk(symbol, action, position_size, confidence)
            
            if risk_result.approved:
                # Send approved order to execution engine
                order_data = {
                    'symbol': symbol,
                    'action': action,
                    'position_size': risk_result.suggested_position_size,
                    'confidence': confidence,
                    'risk_level': risk_result.risk_level.value,
                    'stop_loss': signal_data.get('stop_loss'),
                    'take_profit': signal_data.get('take_profit')
                }
                
                await self.message_queue.publish(
                    "risk_approved_orders",
                    self.message_queue.create_message(
                        MessageType.ORDER_CREATED,
                        "risk_management",
                        order_data
                    ),
                    symbol
                )
                
                self.logger.info(f"Approved order for {symbol}: {action} {risk_result.suggested_position_size}")
            else:
                # Log rejected order
                self.logger.warning(f"Rejected order for {symbol}: {risk_result.reason}")
                
                # Send alert if critical
                if risk_result.risk_level == RiskLevel.CRITICAL:
                    await self.message_queue.publish_alert(
                        "RISK_REJECTION",
                        f"Order rejected for {symbol}: {risk_result.reason}",
                        "WARNING"
                    )
            
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {e}")
    
    async def _handle_order_update(self, message):
        """Handle order execution updates."""
        try:
            order_data = message.data
            symbol = order_data.get('symbol')
            order_id = order_data.get('order_id')
            status = order_data.get('status')
            
            if status == "FILLED":
                # Update position
                await self._update_position_from_order(order_data)
                
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    async def _check_risk(self, symbol: str, action: str, position_size: float, 
                         confidence: float) -> RiskCheckResult:
        """Perform comprehensive risk check."""
        warnings = []
        
        # 1. Position size check
        if position_size > self.max_position_size:
            return RiskCheckResult(
                approved=False,
                reason=f"Position size {position_size:.2%} exceeds maximum {self.max_position_size:.2%}",
                suggested_position_size=0.0,
                risk_level=RiskLevel.CRITICAL,
                warnings=[]
            )
        
        # 2. Portfolio value check
        if self.portfolio_value <= 0:
            return RiskCheckResult(
                approved=False,
                reason="Portfolio value is zero or negative",
                suggested_position_size=0.0,
                risk_level=RiskLevel.CRITICAL,
                warnings=[]
            )
        
        # 3. Drawdown check
        if self.risk_metrics and self.risk_metrics.current_drawdown > self.max_daily_drawdown:
            return RiskCheckResult(
                approved=False,
                reason=f"Current drawdown {self.risk_metrics.current_drawdown:.2%} exceeds limit {self.max_daily_drawdown:.2%}",
                suggested_position_size=0.0,
                risk_level=RiskLevel.CRITICAL,
                warnings=[]
            )
        
        # 4. VaR check
        if self.risk_metrics and self.risk_metrics.var_1d_percent > 0.05:  # 5% VaR limit
            warnings.append(f"High VaR: {self.risk_metrics.var_1d_percent:.2%}")
        
        # 5. Correlation check
        if symbol in self.positions:
            correlation = await self._calculate_correlation_risk(symbol)
            if correlation > self.max_correlation:
                warnings.append(f"High correlation risk: {correlation:.2f}")
        
        # 6. Volatility-adjusted position sizing
        suggested_size = await self._calculate_volatility_adjusted_size(symbol, position_size, confidence)
        
        # 7. Determine risk level
        risk_level = self._determine_risk_level(suggested_size, warnings)
        
        # 8. Final approval decision
        approved = (risk_level != RiskLevel.CRITICAL and 
                  suggested_size > 0 and 
                  len([w for w in warnings if "High" in w]) < 3)
        
        return RiskCheckResult(
            approved=approved,
            reason="Risk check passed" if approved else "Risk check failed",
            suggested_position_size=suggested_size,
            risk_level=risk_level,
            warnings=warnings
        )
    
    async def _calculate_volatility_adjusted_size(self, symbol: str, base_size: float, 
                                                confidence: float) -> float:
        """Calculate volatility-adjusted position size."""
        try:
            # Get recent price data
            kline_data = await self.timescale_db.get_recent_klines(symbol, limit=100)
            
            if len(kline_data) < 20:
                return base_size * 0.5  # Reduce size if insufficient data
            
            # Calculate ATR
            prices = [k['close'] for k in kline_data]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(24 * 60)  # Daily volatility
            
            # Adjust size based on volatility
            volatility_multiplier = min(1.0, 0.02 / volatility)  # Target 2% daily volatility
            
            # Adjust based on confidence
            confidence_multiplier = min(1.0, confidence * 1.5)
            
            # Apply risk percentage
            risk_amount = self.config.RISK_PERCENTAGE * self.portfolio_value
            atr = np.mean([abs(k['high'] - k['low']) for k in kline_data[-20:]])
            
            if atr > 0:
                max_size_by_risk = risk_amount / (atr * 2)  # 2 ATR stop loss
                max_size_by_risk = max_size_by_risk / self.portfolio_value  # Normalize
            else:
                max_size_by_risk = base_size
            
            # Final size calculation
            adjusted_size = base_size * volatility_multiplier * confidence_multiplier
            adjusted_size = min(adjusted_size, max_size_by_risk, self.max_position_size)
            
            return max(0.0, adjusted_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility-adjusted size: {e}")
            return base_size * 0.5
    
    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions."""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            # Get price data for all positions
            price_data = {}
            for pos_symbol in self.positions.keys():
                kline_data = await self.timescale_db.get_recent_klines(pos_symbol, limit=100)
                if len(kline_data) >= 20:
                    prices = [k['close'] for k in kline_data]
                    returns = np.diff(np.log(prices))
                    price_data[pos_symbol] = returns
            
            if symbol not in price_data or len(price_data) < 2:
                return 0.0
            
            # Calculate correlations
            max_correlation = 0.0
            for other_symbol, other_returns in price_data.items():
                if other_symbol != symbol and len(other_returns) == len(price_data[symbol]):
                    correlation = np.corrcoef(price_data[symbol], other_returns)[0, 1]
                    if not np.isnan(correlation):
                        max_correlation = max(max_correlation, abs(correlation))
            
            return max_correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def _determine_risk_level(self, position_size: float, warnings: List[str]) -> RiskLevel:
        """Determine risk level based on position size and warnings."""
        high_warnings = len([w for w in warnings if "High" in w])
        
        if position_size == 0 or high_warnings >= 3:
            return RiskLevel.CRITICAL
        elif position_size > self.max_position_size * 0.8 or high_warnings >= 2:
            return RiskLevel.HIGH
        elif position_size > self.max_position_size * 0.5 or high_warnings >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop."""
        while self.running:
            try:
                # Update portfolio value
                await self._update_portfolio_value()
                
                # Calculate risk metrics
                await self._calculate_risk_metrics()
                
                # Check for risk breaches
                await self._check_risk_breaches()
                
                # Update positions
                await self._update_all_positions()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _position_update_loop(self):
        """Position update loop."""
        while self.running:
            try:
                # Update position PnL
                for symbol, position in self.positions.items():
                    await self._update_position_pnl(symbol, position)
                
                # Check stop losses and take profits
                await self._check_stop_losses_and_take_profits()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in position update loop: {e}")
                await asyncio.sleep(30)
    
    async def _update_portfolio_value(self):
        """Update portfolio value from exchange."""
        try:
            if self.config.is_live_trading():
                # Get account info from Binance
                account_info = await self.client.get_account()
                self.portfolio_value = float(account_info['totalWalletBalance'])
            else:
                # For paper trading, calculate from positions
                self.portfolio_value = 100000.0  # Starting capital
                for position in self.positions.values():
                    self.portfolio_value += position.unrealized_pnl
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    async def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics."""
        try:
            # Get historical PnL data
            trades = await self.timescale_db.get_trade_history(
                self.config.DEFAULT_SYMBOL,
                start_time=time.time() - 30 * 24 * 3600,  # Last 30 days
                limit=1000
            )
            
            if len(trades) < 10:
                return
            
            # Calculate returns
            pnl_values = [trade['pnl'] for trade in trades if trade['pnl'] is not None]
            if len(pnl_values) < 10:
                return
            
            returns = np.array(pnl_values)
            
            # Calculate VaR
            var_1d = np.percentile(returns, (1 - self.var_confidence_level) * 100)
            var_1d_percent = abs(var_1d) / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdowns)) / self.portfolio_value if self.portfolio_value > 0 else 0
            current_drawdown = abs(drawdowns[-1]) / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Calculate ratios
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            downside_returns = returns[returns < 0]
            sortino_ratio = (np.mean(returns) / np.std(downside_returns) * np.sqrt(252) 
                           if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0)
            calmar_ratio = (np.mean(returns) * 252 / max_drawdown 
                          if max_drawdown > 0 else 0)
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_portfolio_correlation_risk()
            
            # Calculate total exposure
            total_exposure = sum(abs(pos.size) for pos in self.positions.values())
            
            self.risk_metrics = RiskMetrics(
                portfolio_value=self.portfolio_value,
                total_exposure=total_exposure,
                var_1d=var_1d,
                var_1d_percent=var_1d_percent,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                correlation_risk=correlation_risk
            )
            
            # Cache risk metrics
            await self.redis_cache.set_system_state({
                'risk_metrics': self.risk_metrics.__dict__,
                'portfolio_value': self.portfolio_value,
                'positions_count': len(self.positions)
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
    
    async def _calculate_portfolio_correlation_risk(self) -> float:
        """Calculate portfolio-wide correlation risk."""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            # Get price data for all positions
            price_data = {}
            for symbol in self.positions.keys():
                kline_data = await self.timescale_db.get_recent_klines(symbol, limit=100)
                if len(kline_data) >= 20:
                    prices = [k['close'] for k in kline_data]
                    returns = np.diff(np.log(prices))
                    price_data[symbol] = returns
            
            if len(price_data) < 2:
                return 0.0
            
            # Calculate correlation matrix
            symbols = list(price_data.keys())
            min_length = min(len(returns) for returns in price_data.values())
            
            if min_length < 10:
                return 0.0
            
            # Align all returns to same length
            aligned_returns = {}
            for symbol, returns in price_data.items():
                aligned_returns[symbol] = returns[-min_length:]
            
            # Calculate average correlation
            correlations = []
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    corr = np.corrcoef(aligned_returns[symbol1], aligned_returns[symbol2])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio correlation risk: {e}")
            return 0.0
    
    async def _check_risk_breaches(self):
        """Check for risk limit breaches."""
        if not self.risk_metrics:
            return
        
        # Check drawdown limit
        if self.risk_metrics.current_drawdown > self.max_daily_drawdown:
            await self.message_queue.publish_alert(
                "DRAWDOWN_BREACH",
                f"Drawdown limit breached: {self.risk_metrics.current_drawdown:.2%} > {self.max_daily_drawdown:.2%}",
                "CRITICAL"
            )
            
            # Close all positions
            await self._close_all_positions("Drawdown limit breached")
        
        # Check VaR limit
        if self.risk_metrics.var_1d_percent > 0.05:  # 5% VaR limit
            await self.message_queue.publish_alert(
                "VAR_BREACH",
                f"VaR limit breached: {self.risk_metrics.var_1d_percent:.2%} > 5%",
                "WARNING"
            )
    
    async def _close_all_positions(self, reason: str):
        """Close all positions due to risk breach."""
        self.logger.warning(f"Closing all positions: {reason}")
        
        for symbol, position in self.positions.items():
            if position.size != 0:
                # Send close order
                close_action = "SELL" if position.side == "LONG" else "BUY"
                order_data = {
                    'symbol': symbol,
                    'action': close_action,
                    'position_size': abs(position.size),
                    'reason': reason,
                    'emergency_close': True
                }
                
                await self.message_queue.publish(
                    "emergency_orders",
                    self.message_queue.create_message(
                        MessageType.ORDER_CREATED,
                        "risk_management",
                        order_data
                    ),
                    symbol
                )
    
    async def _load_positions(self):
        """Load existing positions from database."""
        try:
            positions_data = await self.timescale_db.get_positions()
            
            for pos_data in positions_data:
                position = Position(
                    symbol=pos_data['symbol'],
                    side=pos_data['side'],
                    size=pos_data['size'],
                    entry_price=pos_data['entry_price'],
                    current_price=pos_data['current_price'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    entry_time=pos_data['created_at'],
                    stop_loss=None,  # Will be updated from signal
                    take_profit=None
                )
                
                self.positions[position.symbol] = position
            
            self.logger.info(f"Loaded {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
    
    async def _update_position_from_order(self, order_data: Dict[str, Any]):
        """Update position from filled order."""
        try:
            symbol = order_data['symbol']
            side = order_data['action']
            size = order_data['position_size']
            price = order_data['price']
            
            if symbol not in self.positions:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    entry_time=time.time(),
                    stop_loss=order_data.get('stop_loss'),
                    take_profit=order_data.get('take_profit')
                )
            else:
                # Update existing position
                position = self.positions[symbol]
                
                if position.side == side:
                    # Add to position
                    total_size = position.size + size
                    position.entry_price = ((position.entry_price * position.size) + 
                                          (price * size)) / total_size
                    position.size = total_size
                else:
                    # Reduce or close position
                    if size >= position.size:
                        # Close position
                        position.size = 0
                        position.unrealized_pnl = 0.0
                    else:
                        # Reduce position
                        position.size -= size
            
            # Update database
            await self.timescale_db.update_position(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl
            )
            
        except Exception as e:
            self.logger.error(f"Error updating position from order: {e}")
    
    async def _update_all_positions(self):
        """Update all positions with current prices."""
        for symbol, position in self.positions.items():
            await self._update_position_pnl(symbol, position)
    
    async def _update_position_pnl(self, symbol: str, position: Position):
        """Update position PnL with current price."""
        try:
            # Get current price
            kline_data = await self.timescale_db.get_recent_klines(symbol, limit=1)
            if not kline_data:
                return
            
            current_price = kline_data[0]['close']
            position.current_price = current_price
            
            # Calculate unrealized PnL
            if position.side == "LONG":
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.size
            
            # Update database
            await self.timescale_db.update_position(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl
            )
            
        except Exception as e:
            self.logger.error(f"Error updating position PnL for {symbol}: {e}")
    
    async def _check_stop_losses_and_take_profits(self):
        """Check stop losses and take profits."""
        for symbol, position in self.positions.items():
            if position.size == 0 or not position.stop_loss or not position.take_profit:
                continue
            
            current_price = position.current_price
            
            # Check stop loss
            if ((position.side == "LONG" and current_price <= position.stop_loss) or
                (position.side == "SHORT" and current_price >= position.stop_loss)):
                
                # Trigger stop loss
                close_action = "SELL" if position.side == "LONG" else "BUY"
                order_data = {
                    'symbol': symbol,
                    'action': close_action,
                    'position_size': abs(position.size),
                    'reason': 'Stop loss triggered',
                    'price': current_price
                }
                
                await self.message_queue.publish(
                    "stop_loss_orders",
                    self.message_queue.create_message(
                        MessageType.ORDER_CREATED,
                        "risk_management",
                        order_data
                    ),
                    symbol
                )
                
                self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
            
            # Check take profit
            elif ((position.side == "LONG" and current_price >= position.take_profit) or
                  (position.side == "SHORT" and current_price <= position.take_profit)):
                
                # Trigger take profit
                close_action = "SELL" if position.side == "LONG" else "BUY"
                order_data = {
                    'symbol': symbol,
                    'action': close_action,
                    'position_size': abs(position.size),
                    'reason': 'Take profit triggered',
                    'price': current_price
                }
                
                await self.message_queue.publish(
                    "take_profit_orders",
                    self.message_queue.create_message(
                        MessageType.ORDER_CREATED,
                        "risk_management",
                        order_data
                    ),
                    symbol
                )
                
                self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
