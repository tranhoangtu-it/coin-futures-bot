"""
Module 1: Data Ingestion & Feature Engineering
Responsible for collecting, validating, cleaning, storing, and transforming raw data.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import websockets
import aiohttp
import pandas as pd
import numpy as np
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from src.config import Config
from src.core.message_queue import MessageQueue, MessageType
from src.features.technical_indicators import TechnicalIndicators
from src.features.microstructure import MicrostructureFeatures
from src.features.nlp_features import NLPFeatures
from src.database.timescale import TimescaleDB
from src.database.redis_cache import RedisCache


@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: float
    data_type: str  # kline, order_book, trade
    data: Dict[str, Any]


class DataIngestionModule:
    """Data ingestion and feature engineering module."""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = logging.getLogger(__name__)
        
        # Binance client
        self.client = None
        self.socket_manager = None
        
        # Feature engineering
        self.technical_indicators = TechnicalIndicators()
        self.microstructure_features = MicrostructureFeatures()
        self.nlp_features = NLPFeatures() if config.ENABLE_NLP_FEATURES else None
        
        # Database connections
        self.timescale_db = TimescaleDB(config)
        self.redis_cache = RedisCache(config)
        
        # Data buffers
        self.kline_buffer = {}
        self.order_book_buffer = {}
        self.trade_buffer = {}
        
        # WebSocket connections
        self.websocket_tasks = []
        self.running = False
        
    async def initialize(self):
        """Initialize the data ingestion module."""
        self.logger.info("Initializing data ingestion module...")
        
        # Initialize Binance client
        self.client = await AsyncClient.create(
            api_key=self.config.BINANCE_API_KEY,
            api_secret=self.config.BINANCE_SECRET_KEY,
            testnet=self.config.BINANCE_TESTNET
        )
        self.socket_manager = BinanceSocketManager(self.client)
        
        # Initialize databases
        await self.timescale_db.initialize()
        await self.redis_cache.initialize()
        
        # Initialize feature engineering
        await self.technical_indicators.initialize()
        await self.microstructure_features.initialize()
        if self.nlp_features:
            await self.nlp_features.initialize()
        
        self.logger.info("Data ingestion module initialized")
    
    async def start(self):
        """Start data ingestion."""
        self.logger.info("Starting data ingestion...")
        self.running = True
        
        # Start WebSocket connections for each symbol
        for symbol in self.config.get_symbols():
            await self._start_websocket_connections(symbol)
        
        # Start historical data backfill
        asyncio.create_task(self._backfill_historical_data())
        
        # Start feature engineering pipeline
        asyncio.create_task(self._feature_engineering_pipeline())
        
        self.logger.info("Data ingestion started")
    
    async def stop(self):
        """Stop data ingestion."""
        self.logger.info("Stopping data ingestion...")
        self.running = False
        
        # Cancel all WebSocket tasks
        for task in self.websocket_tasks:
            task.cancel()
        
        # Close Binance client
        if self.client:
            await self.client.close_connection()
        
        # Close database connections
        await self.timescale_db.close()
        await self.redis_cache.close()
        
        self.logger.info("Data ingestion stopped")
    
    async def _start_websocket_connections(self, symbol: str):
        """Start WebSocket connections for a symbol."""
        # Kline data (OHLCV)
        kline_task = asyncio.create_task(
            self._handle_kline_stream(symbol)
        )
        self.websocket_tasks.append(kline_task)
        
        # Order book data
        if self.config.ENABLE_ORDER_BOOK_FEATURES:
            order_book_task = asyncio.create_task(
                self._handle_order_book_stream(symbol)
            )
            self.websocket_tasks.append(order_book_task)
        
        # Trade data
        trade_task = asyncio.create_task(
            self._handle_trade_stream(symbol)
        )
        self.websocket_tasks.append(trade_task)
    
    async def _handle_kline_stream(self, symbol: str):
        """Handle kline data stream."""
        try:
            async with self.socket_manager.kline_socket(symbol, interval='1m') as stream:
                async for msg in stream:
                    if not self.running:
                        break
                    
                    try:
                        kline_data = msg['k']
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=kline_data['t'] / 1000,
                            data_type='kline',
                            data={
                                'open': float(kline_data['o']),
                                'high': float(kline_data['h']),
                                'low': float(kline_data['l']),
                                'close': float(kline_data['c']),
                                'volume': float(kline_data['v']),
                                'interval': kline_data['i'],
                                'is_closed': kline_data['x']
                            }
                        )
                        
                        # Store in buffer
                        if symbol not in self.kline_buffer:
                            self.kline_buffer[symbol] = []
                        self.kline_buffer[symbol].append(market_data)
                        
                        # Keep only last 1000 records
                        if len(self.kline_buffer[symbol]) > 1000:
                            self.kline_buffer[symbol] = self.kline_buffer[symbol][-1000:]
                        
                        # Publish to message queue
                        await self.message_queue.publish_market_data(symbol, market_data.data)
                        
                        # Store in database
                        await self._store_kline_data(market_data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing kline data for {symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in kline stream for {symbol}: {e}")
            if self.running:
                # Reconnect after delay
                await asyncio.sleep(self.config.WEBSOCKET_RECONNECT_DELAY)
                asyncio.create_task(self._handle_kline_stream(symbol))
    
    async def _handle_order_book_stream(self, symbol: str):
        """Handle order book data stream."""
        try:
            async with self.socket_manager.depth_socket(symbol) as stream:
                async for msg in stream:
                    if not self.running:
                        break
                    
                    try:
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=time.time(),
                            data_type='order_book',
                            data={
                                'bids': [[float(bid[0]), float(bid[1])] for bid in msg['bids'][:20]],
                                'asks': [[float(ask[0]), float(ask[1])] for ask in msg['asks'][:20]],
                                'last_update_id': msg['lastUpdateId']
                            }
                        )
                        
                        # Store in buffer
                        if symbol not in self.order_book_buffer:
                            self.order_book_buffer[symbol] = []
                        self.order_book_buffer[symbol].append(market_data)
                        
                        # Keep only last 100 records
                        if len(self.order_book_buffer[symbol]) > 100:
                            self.order_book_buffer[symbol] = self.order_book_buffer[symbol][-100:]
                        
                        # Publish to message queue
                        await self.message_queue.publish_market_data(symbol, market_data.data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing order book data for {symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in order book stream for {symbol}: {e}")
            if self.running:
                await asyncio.sleep(self.config.WEBSOCKET_RECONNECT_DELAY)
                asyncio.create_task(self._handle_order_book_stream(symbol))
    
    async def _handle_trade_stream(self, symbol: str):
        """Handle trade data stream."""
        try:
            async with self.socket_manager.aggtrade_socket(symbol) as stream:
                async for msg in stream:
                    if not self.running:
                        break
                    
                    try:
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=msg['T'] / 1000,
                            data_type='trade',
                            data={
                                'price': float(msg['p']),
                                'quantity': float(msg['q']),
                                'is_buyer_maker': msg['m'],
                                'trade_id': msg['a']
                            }
                        )
                        
                        # Store in buffer
                        if symbol not in self.trade_buffer:
                            self.trade_buffer[symbol] = []
                        self.trade_buffer[symbol].append(market_data)
                        
                        # Keep only last 1000 records
                        if len(self.trade_buffer[symbol]) > 1000:
                            self.trade_buffer[symbol] = self.trade_buffer[symbol][-1000:]
                        
                        # Publish to message queue
                        await self.message_queue.publish_market_data(symbol, market_data.data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing trade data for {symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in trade stream for {symbol}: {e}")
            if self.running:
                await asyncio.sleep(self.config.WEBSOCKET_RECONNECT_DELAY)
                asyncio.create_task(self._handle_trade_stream(symbol))
    
    async def _backfill_historical_data(self):
        """Backfill historical data for all symbols."""
        self.logger.info("Starting historical data backfill...")
        
        for symbol in self.config.get_symbols():
            try:
                await self._backfill_symbol_data(symbol)
            except Exception as e:
                self.logger.error(f"Error backfilling data for {symbol}: {e}")
    
    async def _backfill_symbol_data(self, symbol: str):
        """Backfill historical data for a specific symbol."""
        # Get the last stored timestamp
        last_timestamp = await self.timescale_db.get_last_kline_timestamp(symbol)
        
        if last_timestamp:
            start_date = datetime.fromtimestamp(last_timestamp)
        else:
            start_date = datetime.now() - timedelta(days=30)
        
        end_date = datetime.now()
        
        # Backfill in chunks to avoid rate limits
        current_date = start_date
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=1), end_date)
            
            try:
                # Fetch kline data
                klines = await self.client.get_historical_klines(
                    symbol,
                    '1m',
                    start_str=current_date.strftime('%d %b %Y %H:%M:%S'),
                    end_str=chunk_end.strftime('%d %b %Y %H:%M:%S')
                )
                
                # Store kline data
                for kline in klines:
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=kline[0] / 1000,
                        data_type='kline',
                        data={
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'interval': '1m',
                            'is_closed': True
                        }
                    )
                    await self._store_kline_data(market_data)
                
                self.logger.info(f"Backfilled data for {symbol} from {current_date} to {chunk_end}")
                current_date = chunk_end
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except BinanceAPIException as e:
                if e.code == -1003:  # Rate limit exceeded
                    self.logger.warning("Rate limit exceeded, waiting...")
                    await asyncio.sleep(60)
                else:
                    raise
    
    async def _store_kline_data(self, market_data: MarketData):
        """Store kline data in TimescaleDB."""
        await self.timescale_db.store_kline_data(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            open_price=market_data.data['open'],
            high_price=market_data.data['high'],
            low_price=market_data.data['low'],
            close_price=market_data.data['close'],
            volume=market_data.data['volume'],
            interval=market_data.data['interval']
        )
    
    async def _feature_engineering_pipeline(self):
        """Run feature engineering pipeline."""
        while self.running:
            try:
                for symbol in self.config.get_symbols():
                    await self._generate_features(symbol)
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in feature engineering pipeline: {e}")
                await asyncio.sleep(60)
    
    async def _generate_features(self, symbol: str):
        """Generate features for a symbol."""
        try:
            # Get recent kline data
            kline_data = await self.timescale_db.get_recent_klines(symbol, limit=1000)
            
            if len(kline_data) < 50:  # Need minimum data for indicators
                return
            
            df = pd.DataFrame(kline_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Technical indicators
            features = await self.technical_indicators.calculate_all(df)
            
            # Microstructure features
            if symbol in self.order_book_buffer and len(self.order_book_buffer[symbol]) > 0:
                ob_features = await self.microstructure_features.calculate_order_book_features(
                    self.order_book_buffer[symbol][-100:]
                )
                features.update(ob_features)
            
            if symbol in self.trade_buffer and len(self.trade_buffer[symbol]) > 0:
                trade_features = await self.microstructure_features.calculate_trade_features(
                    self.trade_buffer[symbol][-1000:]
                )
                features.update(trade_features)
            
            # Store features
            await self.timescale_db.store_features(symbol, features)
            
            # Cache latest features
            await self.redis_cache.set_latest_features(symbol, features)
            
        except Exception as e:
            self.logger.error(f"Error generating features for {symbol}: {e}")
