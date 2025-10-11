"""
Module 4: Execution Engine
Execute orders accurately, quickly, and intelligently to minimize transaction costs.
Implements smart order routing, latency optimization, and order state management.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

import numpy as np
import aiohttp
from binance import AsyncClient
from binance.exceptions import BinanceAPIException

from src.config import Config
from src.core.message_queue import MessageQueue, MessageType
from src.database.timescale import TimescaleDB
from src.database.redis_cache import RedisCache


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    commission: float
    commission_asset: str
    created_time: float
    updated_time: float
    client_order_id: str
    time_in_force: str = "GTC"


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_orders: int
    filled_orders: int
    canceled_orders: int
    rejected_orders: int
    average_fill_time: float
    average_slippage: float
    total_commission: float
    success_rate: float


class SmartOrderRouter:
    """Smart order routing for optimal execution."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def route_order(self, order: Order, market_data: Dict[str, Any]) -> List[Order]:
        """Route order for optimal execution."""
        try:
            # For large orders, use TWAP/VWAP algorithms
            if order.quantity > self.config.MAX_ORDER_SIZE * 0.1:  # 10% of max order size
                return await self._route_large_order(order, market_data)
            else:
                return await self._route_small_order(order, market_data)
                
        except Exception as e:
            self.logger.error(f"Error routing order: {e}")
            return [order]
    
    async def _route_large_order(self, order: Order, market_data: Dict[str, Any]) -> List[Order]:
        """Route large order using TWAP algorithm."""
        # Split order into smaller chunks over time
        num_chunks = min(10, max(2, int(order.quantity / (self.config.MAX_ORDER_SIZE * 0.05))))
        chunk_size = order.quantity / num_chunks
        time_interval = 60  # 1 minute between chunks
        
        child_orders = []
        for i in range(num_chunks):
            child_order = Order(
                order_id=f"{order.order_id}_chunk_{i}",
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=chunk_size,
                price=order.price,
                stop_price=order.stop_price,
                status=OrderStatus.NEW,
                filled_quantity=0.0,
                remaining_quantity=chunk_size,
                average_price=None,
                commission=0.0,
                commission_asset=order.commission_asset,
                created_time=time.time() + (i * time_interval),
                updated_time=time.time(),
                client_order_id=f"{order.client_order_id}_chunk_{i}",
                time_in_force=order.time_in_force
            )
            child_orders.append(child_order)
        
        return child_orders
    
    async def _route_small_order(self, order: Order, market_data: Dict[str, Any]) -> List[Order]:
        """Route small order for immediate execution."""
        # For small orders, try to place at favorable price
        if order.order_type == OrderType.LIMIT and order.price:
            # Adjust price slightly to improve fill probability
            order_book = market_data.get('order_book', {})
            if order_book:
                best_bid = order_book.get('bids', [[0, 0]])[0][0]
                best_ask = order_book.get('asks', [[0, 0]])[0][0]
                
                if order.side == OrderSide.BUY and best_ask > 0:
                    # Place buy order slightly below best ask
                    order.price = min(order.price, best_ask * 0.9999)
                elif order.side == OrderSide.SELL and best_bid > 0:
                    # Place sell order slightly above best bid
                    order.price = max(order.price, best_bid * 1.0001)
        
        return [order]


class OrderStateManager:
    """Manages order state and lifecycle."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.orders = {}  # order_id -> Order
        self.order_timeouts = {}  # order_id -> timeout_task
    
    def add_order(self, order: Order):
        """Add order to state manager."""
        self.orders[order.order_id] = order
        
        # Set timeout for order
        if order.order_type in [OrderType.MARKET, OrderType.LIMIT]:
            timeout_task = asyncio.create_task(
                self._handle_order_timeout(order.order_id)
            )
            self.order_timeouts[order.order_id] = timeout_task
    
    def update_order(self, order_id: str, updates: Dict[str, Any]):
        """Update order state."""
        if order_id in self.orders:
            order = self.orders[order_id]
            for key, value in updates.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            order.updated_time = time.time()
    
    def remove_order(self, order_id: str):
        """Remove order from state manager."""
        if order_id in self.orders:
            del self.orders[order_id]
        
        if order_id in self.order_timeouts:
            self.order_timeouts[order_id].cancel()
            del self.order_timeouts[order_id]
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        active_statuses = [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
        return [order for order in self.orders.values() if order.status in active_statuses]
    
    async def _handle_order_timeout(self, order_id: str):
        """Handle order timeout."""
        await asyncio.sleep(self.config.ORDER_TIMEOUT)
        
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                self.logger.warning(f"Order {order_id} timed out")
                order.status = OrderStatus.EXPIRED
                order.updated_time = time.time()


class ExecutionEngineModule:
    """Execution engine for order processing."""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = logging.getLogger(__name__)
        
        # Binance client
        self.client = None
        
        # Components
        self.order_router = SmartOrderRouter(config)
        self.state_manager = OrderStateManager(config)
        
        # Database connections
        self.timescale_db = TimescaleDB(config)
        self.redis_cache = RedisCache(config)
        
        # Execution metrics
        self.metrics = ExecutionMetrics(
            total_orders=0,
            filled_orders=0,
            canceled_orders=0,
            rejected_orders=0,
            average_fill_time=0.0,
            average_slippage=0.0,
            total_commission=0.0,
            success_rate=0.0
        )
        
        # State
        self.running = False
        self.order_counter = 0
    
    async def initialize(self):
        """Initialize the execution engine."""
        self.logger.info("Initializing execution engine...")
        
        # Initialize Binance client
        self.client = await AsyncClient.create(
            api_key=self.config.BINANCE_API_KEY,
            api_secret=self.config.BINANCE_SECRET_KEY,
            testnet=self.config.BINANCE_TESTNET
        )
        
        # Initialize database connections
        await self.timescale_db.initialize()
        await self.redis_cache.initialize()
        
        # Subscribe to order messages
        await self.message_queue.subscribe(
            "risk_approved_orders",
            self._handle_approved_order
        )
        await self.message_queue.subscribe(
            "emergency_orders",
            self._handle_emergency_order
        )
        await self.message_queue.subscribe(
            "stop_loss_orders",
            self._handle_stop_loss_order
        )
        await self.message_queue.subscribe(
            "take_profit_orders",
            self._handle_take_profit_order
        )
        
        self.logger.info("Execution engine initialized")
    
    async def start(self):
        """Start the execution engine."""
        self.logger.info("Starting execution engine...")
        self.running = True
        
        # Start order monitoring loop
        asyncio.create_task(self._order_monitoring_loop())
        
        # Start reconciliation loop
        asyncio.create_task(self._reconciliation_loop())
        
        # Start metrics update loop
        asyncio.create_task(self._metrics_update_loop())
        
        self.logger.info("Execution engine started")
    
    async def stop(self):
        """Stop the execution engine."""
        self.logger.info("Stopping execution engine...")
        self.running = False
        
        # Cancel all active orders
        active_orders = self.state_manager.get_active_orders()
        for order in active_orders:
            await self._cancel_order(order)
        
        # Close Binance client
        if self.client:
            await self.client.close_connection()
        
        # Close database connections
        await self.timescale_db.close()
        await self.redis_cache.close()
        
        self.logger.info("Execution engine stopped")
    
    async def _handle_approved_order(self, message):
        """Handle approved order from risk management."""
        try:
            order_data = message.data
            symbol = order_data['symbol']
            action = order_data['action']
            position_size = order_data['position_size']
            confidence = order_data.get('confidence', 0.0)
            stop_loss = order_data.get('stop_loss')
            take_profit = order_data.get('take_profit')
            
            # Create order
            order = await self._create_order(
                symbol=symbol,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.MARKET,  # Start with market order
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Execute order
            await self._execute_order(order)
            
        except Exception as e:
            self.logger.error(f"Error handling approved order: {e}")
    
    async def _handle_emergency_order(self, message):
        """Handle emergency order (e.g., risk breach)."""
        try:
            order_data = message.data
            symbol = order_data['symbol']
            action = order_data['action']
            position_size = order_data['position_size']
            
            # Create market order for immediate execution
            order = await self._create_order(
                symbol=symbol,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.MARKET
            )
            
            # Execute immediately
            await self._execute_order(order)
            
            self.logger.warning(f"Emergency order executed: {symbol} {action} {position_size}")
            
        except Exception as e:
            self.logger.error(f"Error handling emergency order: {e}")
    
    async def _handle_stop_loss_order(self, message):
        """Handle stop loss order."""
        try:
            order_data = message.data
            symbol = order_data['symbol']
            action = order_data['action']
            position_size = order_data['position_size']
            price = order_data.get('price')
            
            # Create stop loss order
            order = await self._create_order(
                symbol=symbol,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.STOP_LOSS,
                price=price
            )
            
            await self._execute_order(order)
            
        except Exception as e:
            self.logger.error(f"Error handling stop loss order: {e}")
    
    async def _handle_take_profit_order(self, message):
        """Handle take profit order."""
        try:
            order_data = message.data
            symbol = order_data['symbol']
            action = order_data['action']
            position_size = order_data['position_size']
            price = order_data.get('price')
            
            # Create take profit order
            order = await self._create_order(
                symbol=symbol,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.TAKE_PROFIT,
                price=price
            )
            
            await self._execute_order(order)
            
        except Exception as e:
            self.logger.error(f"Error handling take profit order: {e}")
    
    async def _create_order(self, symbol: str, side: OrderSide, quantity: float,
                           order_type: OrderType, price: Optional[float] = None,
                           stop_loss: Optional[float] = None, 
                           take_profit: Optional[float] = None) -> Order:
        """Create a new order."""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}_{int(time.time())}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_loss,
            status=OrderStatus.NEW,
            filled_quantity=0.0,
            remaining_quantity=quantity,
            average_price=None,
            commission=0.0,
            commission_asset="USDT",
            created_time=time.time(),
            updated_time=time.time(),
            client_order_id=order_id
        )
        
        # Add to state manager
        self.state_manager.add_order(order)
        
        return order
    
    async def _execute_order(self, order: Order):
        """Execute order on exchange."""
        try:
            # Get current market data
            market_data = await self.redis_cache.get_market_data(order.symbol, "kline")
            if not market_data:
                # Fallback to exchange data
                market_data = await self._get_market_data_from_exchange(order.symbol)
            
            # Route order for optimal execution
            child_orders = await self.order_router.route_order(order, market_data)
            
            # Execute child orders
            for child_order in child_orders:
                await self._submit_order_to_exchange(child_order)
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            self.metrics.rejected_orders += 1
    
    async def _submit_order_to_exchange(self, order: Order):
        """Submit order to Binance exchange."""
        try:
            if self.config.is_paper_trading():
                # Simulate order execution for paper trading
                await self._simulate_order_execution(order)
            else:
                # Submit real order to exchange
                await self._submit_real_order(order)
                
        except Exception as e:
            self.logger.error(f"Error submitting order to exchange: {e}")
            order.status = OrderStatus.REJECTED
            self.metrics.rejected_orders += 1
    
    async def _simulate_order_execution(self, order: Order):
        """Simulate order execution for paper trading."""
        try:
            # Get current price
            ticker = await self.client.get_symbol_ticker(symbol=order.symbol)
            current_price = float(ticker['price'])
            
            # Simulate execution with some slippage
            slippage = np.random.uniform(-0.001, 0.001)  # ±0.1% slippage
            execution_price = current_price * (1 + slippage)
            
            # Simulate fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.remaining_quantity = 0.0
            order.average_price = execution_price
            order.commission = order.quantity * execution_price * 0.001  # 0.1% commission
            order.updated_time = time.time()
            
            # Update metrics
            self.metrics.filled_orders += 1
            self.metrics.total_commission += order.commission
            
            # Publish order filled event
            await self._publish_order_filled(order)
            
            # Store trade in database
            await self.timescale_db.store_trade(
                symbol=order.symbol,
                timestamp=order.updated_time,
                order_id=order.order_id,
                side=order.side.value,
                price=execution_price,
                quantity=order.quantity,
                commission=order.commission,
                commission_asset=order.commission_asset
            )
            
        except Exception as e:
            self.logger.error(f"Error simulating order execution: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _submit_real_order(self, order: Order):
        """Submit real order to Binance exchange."""
        try:
            # Prepare order parameters
            order_params = {
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'newClientOrderId': order.client_order_id
            }
            
            if order.price:
                order_params['price'] = order.price
            if order.stop_price:
                order_params['stopPrice'] = order.stop_price
            if order.time_in_force:
                order_params['timeInForce'] = order.time_in_force
            
            # Submit order
            result = await self.client.create_order(**order_params)
            
            # Update order with exchange response
            order.order_id = result['orderId']
            order.status = OrderStatus(result['status'])
            order.updated_time = time.time()
            
            # If immediately filled, update fill information
            if order.status == OrderStatus.FILLED:
                order.filled_quantity = float(result['executedQty'])
                order.remaining_quantity = float(result['origQty']) - order.filled_quantity
                order.average_price = float(result.get('cummulativeQuoteQty', 0)) / order.filled_quantity if order.filled_quantity > 0 else None
                order.commission = sum(float(fill['commission']) for fill in result.get('fills', []))
                
                # Update metrics
                self.metrics.filled_orders += 1
                self.metrics.total_commission += order.commission
                
                # Publish order filled event
                await self._publish_order_filled(order)
                
                # Store trade in database
                await self.timescale_db.store_trade(
                    symbol=order.symbol,
                    timestamp=order.updated_time,
                    order_id=order.order_id,
                    side=order.side.value,
                    price=order.average_price or 0,
                    quantity=order.filled_quantity,
                    commission=order.commission,
                    commission_asset=order.commission_asset
                )
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error: {e}")
            order.status = OrderStatus.REJECTED
            self.metrics.rejected_orders += 1
        except Exception as e:
            self.logger.error(f"Error submitting real order: {e}")
            order.status = OrderStatus.REJECTED
            self.metrics.rejected_orders += 1
    
    async def _publish_order_filled(self, order: Order):
        """Publish order filled event."""
        try:
            order_data = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.filled_quantity,
                'price': order.average_price,
                'commission': order.commission,
                'commission_asset': order.commission_asset,
                'timestamp': order.updated_time
            }
            
            await self.message_queue.publish_order_update(order.order_id, order_data)
            
        except Exception as e:
            self.logger.error(f"Error publishing order filled event: {e}")
    
    async def _cancel_order(self, order: Order):
        """Cancel an order."""
        try:
            if not self.config.is_paper_trading():
                await self.client.cancel_order(
                    symbol=order.symbol,
                    orderId=order.order_id
                )
            
            order.status = OrderStatus.CANCELED
            order.updated_time = time.time()
            self.metrics.canceled_orders += 1
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order.order_id}: {e}")
    
    async def _get_market_data_from_exchange(self, symbol: str) -> Dict[str, Any]:
        """Get market data from exchange."""
        try:
            # Get 24hr ticker
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            
            # Get order book
            order_book = await self.client.get_order_book(symbol=symbol, limit=20)
            
            return {
                'price': float(ticker['price']),
                'order_book': {
                    'bids': [[float(bid[0]), float(bid[1])] for bid in order_book['bids']],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in order_book['asks']]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data from exchange: {e}")
            return {}
    
    async def _order_monitoring_loop(self):
        """Monitor order status and handle updates."""
        while self.running:
            try:
                active_orders = self.state_manager.get_active_orders()
                
                for order in active_orders:
                    if not self.config.is_paper_trading():
                        await self._check_order_status(order)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_order_status(self, order: Order):
        """Check order status on exchange."""
        try:
            result = await self.client.get_order(
                symbol=order.symbol,
                orderId=order.order_id
            )
            
            # Update order status
            old_status = order.status
            order.status = OrderStatus(result['status'])
            order.filled_quantity = float(result['executedQty'])
            order.remaining_quantity = float(result['origQty']) - order.filled_quantity
            
            if result.get('cummulativeQuoteQty'):
                order.average_price = float(result['cummulativeQuoteQty']) / order.filled_quantity if order.filled_quantity > 0 else None
            
            # Calculate commission from fills
            if result.get('fills'):
                order.commission = sum(float(fill['commission']) for fill in result['fills'])
            
            order.updated_time = time.time()
            
            # If order was filled, publish event
            if old_status != OrderStatus.FILLED and order.status == OrderStatus.FILLED:
                self.metrics.filled_orders += 1
                self.metrics.total_commission += order.commission
                await self._publish_order_filled(order)
                
                # Store trade in database
                await self.timescale_db.store_trade(
                    symbol=order.symbol,
                    timestamp=order.updated_time,
                    order_id=order.order_id,
                    side=order.side.value,
                    price=order.average_price or 0,
                    quantity=order.filled_quantity,
                    commission=order.commission,
                    commission_asset=order.commission_asset
                )
            
        except Exception as e:
            self.logger.error(f"Error checking order status for {order.order_id}: {e}")
    
    async def _reconciliation_loop(self):
        """Reconcile internal state with exchange state."""
        while self.running:
            try:
                # Run reconciliation every 30 seconds
                await asyncio.sleep(30)
                
                if not self.config.is_paper_trading():
                    await self._reconcile_with_exchange()
                
            except Exception as e:
                self.logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(60)
    
    async def _reconcile_with_exchange(self):
        """Reconcile internal state with exchange."""
        try:
            # Get all open orders from exchange
            open_orders = await self.client.get_open_orders()
            exchange_order_ids = {order['orderId'] for order in open_orders}
            
            # Check internal orders
            active_orders = self.state_manager.get_active_orders()
            internal_order_ids = {order.order_id for order in active_orders}
            
            # Find orders that exist internally but not on exchange
            missing_orders = internal_order_ids - exchange_order_ids
            for order_id in missing_orders:
                order = self.state_manager.get_order(order_id)
                if order:
                    order.status = OrderStatus.CANCELED
                    self.logger.warning(f"Order {order_id} not found on exchange, marking as canceled")
            
        except Exception as e:
            self.logger.error(f"Error reconciling with exchange: {e}")
    
    async def _metrics_update_loop(self):
        """Update execution metrics."""
        while self.running:
            try:
                # Calculate metrics
                total_orders = len(self.state_manager.orders)
                self.metrics.total_orders = total_orders
                
                if total_orders > 0:
                    self.metrics.success_rate = self.metrics.filled_orders / total_orders
                
                # Calculate average fill time
                filled_orders = [order for order in self.state_manager.orders.values() 
                               if order.status == OrderStatus.FILLED]
                
                if filled_orders:
                    fill_times = [order.updated_time - order.created_time for order in filled_orders]
                    self.metrics.average_fill_time = np.mean(fill_times)
                
                # Cache metrics
                await self.redis_cache.set_system_state({
                    'execution_metrics': self.metrics.__dict__,
                    'active_orders': len(self.state_manager.get_active_orders()),
                    'total_orders': total_orders
                })
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
