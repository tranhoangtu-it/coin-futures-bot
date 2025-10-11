"""
TimescaleDB integration for time-series data storage.
Handles kline data, features, and trade history storage.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import asyncpg
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.config import Config


class TimescaleDB:
    """TimescaleDB client for time-series data storage."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize TimescaleDB connection and create tables."""
        self.logger.info("Initializing TimescaleDB...")
        
        try:
            # Create async engine
            self.async_engine = create_async_engine(
                self.config.TIMESCALEDB_URL.replace('postgresql://', 'postgresql+asyncpg://'),
                echo=False,
                pool_size=10,
                max_overflow=20
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            await self._create_tables()
            
            self.initialized = True
            self.logger.info("TimescaleDB initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TimescaleDB: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
        self.logger.info("TimescaleDB connections closed")
    
    async def _create_tables(self):
        """Create required tables and hypertables."""
        async with self.async_engine.begin() as conn:
            # Create kline data table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS kline_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open_price DECIMAL(20, 8) NOT NULL,
                    high_price DECIMAL(20, 8) NOT NULL,
                    low_price DECIMAL(20, 8) NOT NULL,
                    close_price DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(20, 8) NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for kline_data
            await conn.execute(text("""
                SELECT create_hypertable('kline_data', 'timestamp', 
                    if_not_exists => TRUE);
            """))
            
            # Create features table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    feature_name VARCHAR(100) NOT NULL,
                    feature_value DECIMAL(20, 8) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for features
            await conn.execute(text("""
                SELECT create_hypertable('features', 'timestamp', 
                    if_not_exists => TRUE);
            """))
            
            # Create trades table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    order_id VARCHAR(100) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    quantity DECIMAL(20, 8) NOT NULL,
                    commission DECIMAL(20, 8) NOT NULL,
                    commission_asset VARCHAR(10) NOT NULL,
                    pnl DECIMAL(20, 8),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for trades
            await conn.execute(text("""
                SELECT create_hypertable('trades', 'timestamp', 
                    if_not_exists => TRUE);
            """))
            
            # Create positions table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS positions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    size DECIMAL(20, 8) NOT NULL,
                    entry_price DECIMAL(20, 8) NOT NULL,
                    current_price DECIMAL(20, 8) NOT NULL,
                    unrealized_pnl DECIMAL(20, 8) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create indexes for better performance
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_kline_symbol_timestamp 
                ON kline_data (symbol, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp 
                ON features (symbol, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
                ON trades (symbol, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_positions_symbol 
                ON positions (symbol);
            """))
    
    async def store_kline_data(self, symbol: str, timestamp: float, open_price: float,
                              high_price: float, low_price: float, close_price: float,
                              volume: float, interval: str):
        """Store kline data."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            async with self.session_factory() as session:
                await session.execute(text("""
                    INSERT INTO kline_data (symbol, timestamp, open_price, high_price, 
                                          low_price, close_price, volume, interval)
                    VALUES (:symbol, :timestamp, :open_price, :high_price, 
                           :low_price, :close_price, :volume, :interval)
                    ON CONFLICT (symbol, timestamp, interval) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume
                """), {
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'open_price': open_price,
                    'high_price': high_price,
                    'low_price': low_price,
                    'close_price': close_price,
                    'volume': volume,
                    'interval': interval
                })
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing kline data: {e}")
            raise
    
    async def get_recent_klines(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get recent kline data."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("""
                    SELECT timestamp, open_price, high_price, low_price, 
                           close_price, volume, interval
                    FROM kline_data
                    WHERE symbol = :symbol
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """), {
                    'symbol': symbol,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [
                    {
                        'timestamp': row.timestamp.timestamp(),
                        'open': float(row.open_price),
                        'high': float(row.high_price),
                        'low': float(row.low_price),
                        'close': float(row.close_price),
                        'volume': float(row.volume),
                        'interval': row.interval
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting recent klines: {e}")
            raise
    
    async def get_last_kline_timestamp(self, symbol: str) -> Optional[float]:
        """Get the last kline timestamp for a symbol."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("""
                    SELECT MAX(timestamp) as last_timestamp
                    FROM kline_data
                    WHERE symbol = :symbol
                """), {'symbol': symbol})
                
                row = result.fetchone()
                if row and row.last_timestamp:
                    return row.last_timestamp.timestamp()
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting last kline timestamp: {e}")
            raise
    
    async def store_features(self, symbol: str, features: Dict[str, float]):
        """Store features for a symbol."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            timestamp = datetime.now()
            
            async with self.session_factory() as session:
                for feature_name, feature_value in features.items():
                    await session.execute(text("""
                        INSERT INTO features (symbol, timestamp, feature_name, feature_value)
                        VALUES (:symbol, :timestamp, :feature_name, :feature_value)
                        ON CONFLICT (symbol, timestamp, feature_name) DO UPDATE SET
                            feature_value = EXCLUDED.feature_value
                    """), {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'feature_name': feature_name,
                        'feature_value': feature_value
                    })
                
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing features: {e}")
            raise
    
    async def get_latest_features(self, symbol: str) -> Dict[str, float]:
        """Get the latest features for a symbol."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("""
                    SELECT feature_name, feature_value
                    FROM features
                    WHERE symbol = :symbol
                    AND timestamp = (
                        SELECT MAX(timestamp)
                        FROM features
                        WHERE symbol = :symbol
                    )
                """), {'symbol': symbol})
                
                rows = result.fetchall()
                return {row.feature_name: float(row.feature_value) for row in rows}
                
        except Exception as e:
            self.logger.error(f"Error getting latest features: {e}")
            raise
    
    async def store_trade(self, symbol: str, timestamp: float, order_id: str,
                         side: str, price: float, quantity: float, commission: float,
                         commission_asset: str, pnl: Optional[float] = None):
        """Store trade data."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            async with self.session_factory() as session:
                await session.execute(text("""
                    INSERT INTO trades (symbol, timestamp, order_id, side, price, 
                                      quantity, commission, commission_asset, pnl)
                    VALUES (:symbol, :timestamp, :order_id, :side, :price, 
                           :quantity, :commission, :commission_asset, :pnl)
                """), {
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'order_id': order_id,
                    'side': side,
                    'price': price,
                    'quantity': quantity,
                    'commission': commission,
                    'commission_asset': commission_asset,
                    'pnl': pnl
                })
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing trade: {e}")
            raise
    
    async def get_trade_history(self, symbol: str, start_time: Optional[float] = None,
                               end_time: Optional[float] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get trade history."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            query = """
                SELECT timestamp, order_id, side, price, quantity, commission, 
                       commission_asset, pnl
                FROM trades
                WHERE symbol = :symbol
            """
            params = {'symbol': symbol}
            
            if start_time:
                query += " AND timestamp >= :start_time"
                params['start_time'] = datetime.fromtimestamp(start_time)
            
            if end_time:
                query += " AND timestamp <= :end_time"
                params['end_time'] = datetime.fromtimestamp(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT :limit"
            params['limit'] = limit
            
            async with self.session_factory() as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                return [
                    {
                        'timestamp': row.timestamp.timestamp(),
                        'order_id': row.order_id,
                        'side': row.side,
                        'price': float(row.price),
                        'quantity': float(row.quantity),
                        'commission': float(row.commission),
                        'commission_asset': row.commission_asset,
                        'pnl': float(row.pnl) if row.pnl else None
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            raise
    
    async def update_position(self, symbol: str, side: str, size: float,
                             entry_price: float, current_price: float, unrealized_pnl: float):
        """Update position data."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            async with self.session_factory() as session:
                await session.execute(text("""
                    INSERT INTO positions (symbol, side, size, entry_price, current_price, unrealized_pnl)
                    VALUES (:symbol, :side, :size, :entry_price, :current_price, :unrealized_pnl)
                    ON CONFLICT (symbol) DO UPDATE SET
                        side = EXCLUDED.side,
                        size = EXCLUDED.size,
                        entry_price = EXCLUDED.entry_price,
                        current_price = EXCLUDED.current_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        updated_at = NOW()
                """), {
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl
                })
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        if not self.initialized:
            raise RuntimeError("TimescaleDB not initialized")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("""
                    SELECT symbol, side, size, entry_price, current_price, unrealized_pnl,
                           created_at, updated_at
                    FROM positions
                    ORDER BY symbol
                """))
                
                rows = result.fetchall()
                return [
                    {
                        'symbol': row.symbol,
                        'side': row.side,
                        'size': float(row.size),
                        'entry_price': float(row.entry_price),
                        'current_price': float(row.current_price),
                        'unrealized_pnl': float(row.unrealized_pnl),
                        'created_at': row.created_at.timestamp(),
                        'updated_at': row.updated_at.timestamp()
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            raise
