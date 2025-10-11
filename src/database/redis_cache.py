"""
Redis cache for hot data storage and fast access.
Handles real-time features, market data, and system state caching.
"""

import json
import logging
from typing import Dict, Any, Optional, List
import redis.asyncio as redis
from datetime import timedelta

from src.config import Config


class RedisCache:
    """Redis cache client for hot data storage."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        self.logger.info("Initializing Redis cache...")
        
        try:
            self.redis_client = redis.from_url(
                self.config.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.initialized = True
            self.logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("Redis cache connection closed")
    
    async def set_latest_features(self, symbol: str, features: Dict[str, float]):
        """Cache latest features for a symbol."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"features:{symbol}:latest"
            await self.redis_client.setex(
                key,
                timedelta(minutes=5),  # Cache for 5 minutes
                json.dumps(features)
            )
        except Exception as e:
            self.logger.error(f"Error caching features for {symbol}: {e}")
    
    async def get_latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest features from cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"features:{symbol}:latest"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting features for {symbol}: {e}")
            return None
    
    async def set_market_data(self, symbol: str, data_type: str, data: Dict[str, Any]):
        """Cache market data."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"market_data:{symbol}:{data_type}"
            await self.redis_client.setex(
                key,
                timedelta(seconds=30),  # Cache for 30 seconds
                json.dumps(data)
            )
        except Exception as e:
            self.logger.error(f"Error caching market data for {symbol}: {e}")
    
    async def get_market_data(self, symbol: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Get market data from cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"market_data:{symbol}:{data_type}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def set_order_book(self, symbol: str, order_book: Dict[str, Any]):
        """Cache order book data."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"order_book:{symbol}"
            await self.redis_client.setex(
                key,
                timedelta(seconds=10),  # Cache for 10 seconds
                json.dumps(order_book)
            )
        except Exception as e:
            self.logger.error(f"Error caching order book for {symbol}: {e}")
    
    async def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get order book from cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"order_book:{symbol}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    async def set_trading_signal(self, symbol: str, signal: Dict[str, Any]):
        """Cache trading signal."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"signal:{symbol}:latest"
            await self.redis_client.setex(
                key,
                timedelta(minutes=1),  # Cache for 1 minute
                json.dumps(signal)
            )
        except Exception as e:
            self.logger.error(f"Error caching signal for {symbol}: {e}")
    
    async def get_trading_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get trading signal from cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"signal:{symbol}:latest"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting signal for {symbol}: {e}")
            return None
    
    async def set_position(self, symbol: str, position: Dict[str, Any]):
        """Cache position data."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"position:{symbol}"
            await self.redis_client.setex(
                key,
                timedelta(minutes=5),  # Cache for 5 minutes
                json.dumps(position)
            )
        except Exception as e:
            self.logger.error(f"Error caching position for {symbol}: {e}")
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position from cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = f"position:{symbol}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return None
    
    async def set_system_state(self, state: Dict[str, Any]):
        """Cache system state."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = "system:state"
            await self.redis_client.setex(
                key,
                timedelta(seconds=30),  # Cache for 30 seconds
                json.dumps(state)
            )
        except Exception as e:
            self.logger.error(f"Error caching system state: {e}")
    
    async def get_system_state(self) -> Optional[Dict[str, Any]]:
        """Get system state from cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            key = "system:state"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting system state: {e}")
            return None
    
    async def push_to_stream(self, stream_name: str, data: Dict[str, Any]):
        """Push data to a Redis stream."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            await self.redis_client.xadd(stream_name, data)
        except Exception as e:
            self.logger.error(f"Error pushing to stream {stream_name}: {e}")
    
    async def get_stream_data(self, stream_name: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get data from a Redis stream."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            data = await self.redis_client.xrevrange(stream_name, count=count)
            return [{"id": item[0], "fields": item[1]} for item in data]
        except Exception as e:
            self.logger.error(f"Error getting stream data from {stream_name}: {e}")
            return []
    
    async def set_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """Set an alert in cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            alert_data = {
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": json.dumps({"$date": {"$numberLong": str(int(time.time() * 1000))}})
            }
            
            # Store in alerts list
            await self.redis_client.lpush("alerts", json.dumps(alert_data))
            
            # Keep only last 100 alerts
            await self.redis_client.ltrim("alerts", 0, 99)
            
        except Exception as e:
            self.logger.error(f"Error setting alert: {e}")
    
    async def get_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts from cache."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            alerts = await self.redis_client.lrange("alerts", 0, count - 1)
            return [json.loads(alert) for alert in alerts]
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    async def increment_counter(self, key: str, increment: int = 1) -> int:
        """Increment a counter in Redis."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            return await self.redis_client.incrby(key, increment)
        except Exception as e:
            self.logger.error(f"Error incrementing counter {key}: {e}")
            return 0
    
    async def set_expiring_key(self, key: str, value: Any, ttl_seconds: int):
        """Set a key with expiration."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            await self.redis_client.setex(key, ttl_seconds, json.dumps(value))
        except Exception as e:
            self.logger.error(f"Error setting expiring key {key}: {e}")
    
    async def get_expiring_key(self, key: str) -> Optional[Any]:
        """Get a value from an expiring key."""
        if not self.initialized:
            raise RuntimeError("Redis cache not initialized")
        
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting expiring key {key}: {e}")
            return None
