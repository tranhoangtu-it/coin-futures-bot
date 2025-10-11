"""
Central message queue system for inter-module communication.
Uses Kafka as the message broker with Avro schemas for data consistency.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import faust
from faust import Record

from src.config import Config


class MessageType(Enum):
    """Message types for the trading system."""
    # Market Data
    MARKET_DATA_RECEIVED = "market_data_received"
    KLINE_DATA = "kline_data"
    ORDER_BOOK_UPDATE = "order_book_update"
    TRADE_DATA = "trade_data"
    
    # AI Signals
    SIGNAL_GENERATED = "signal_generated"
    MARKET_REGIME_DETECTED = "market_regime_detected"
    ALPHA_PREDICTION = "alpha_prediction"
    
    # Risk Management
    RISK_CHECK_REQUEST = "risk_check_request"
    RISK_CHECK_RESPONSE = "risk_check_response"
    POSITION_UPDATE = "position_update"
    
    # Execution
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"
    
    # System
    SYSTEM_HEALTH_CHECK = "system_health_check"
    ALERT = "alert"
    SHUTDOWN = "shutdown"


@dataclass
class BaseMessage(Record):
    """Base message class for all system messages."""
    message_id: str
    timestamp: float
    message_type: str
    source_module: str
    data: Dict[str, Any]


class MessageQueue:
    """Central message queue for inter-module communication."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.app = None
        self.producer = None
        self.consumers = {}
        self.subscribers = {}
        self.running = False
        
    async def initialize(self):
        """Initialize the message queue system."""
        self.logger.info("Initializing message queue...")
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(','),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            retries=3,
            retry_backoff_ms=100,
            request_timeout_ms=30000,
        )
        
        # Initialize Faust app for stream processing
        self.app = faust.App(
            'trading_bot',
            broker=f'kafka://{self.config.KAFKA_BOOTSTRAP_SERVERS}',
            value_serializer='json',
        )
        
        self.logger.info("Message queue initialized successfully")
    
    async def start(self):
        """Start the message queue system."""
        self.logger.info("Starting message queue...")
        self.running = True
        
        # Start Faust app
        if self.app:
            await self.app.start()
        
        self.logger.info("Message queue started")
    
    async def stop(self):
        """Stop the message queue system."""
        self.logger.info("Stopping message queue...")
        self.running = False
        
        # Close all consumers
        for consumer in self.consumers.values():
            consumer.close()
        
        # Close producer
        if self.producer:
            self.producer.close()
        
        # Stop Faust app
        if self.app:
            await self.app.stop()
        
        self.logger.info("Message queue stopped")
    
    async def publish(self, topic: str, message: BaseMessage, key: Optional[str] = None):
        """Publish a message to a topic."""
        try:
            future = self.producer.send(
                topic,
                value=message.asdict(),
                key=key
            )
            await asyncio.get_event_loop().run_in_executor(
                None, future.get, timeout=10
            )
            self.logger.debug(f"Published message to {topic}: {message.message_type}")
        except KafkaError as e:
            self.logger.error(f"Failed to publish message to {topic}: {e}")
            raise
    
    async def subscribe(self, topic: str, callback: Callable[[BaseMessage], None]):
        """Subscribe to a topic with a callback function."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        
        self.subscribers[topic].append(callback)
        
        # Create consumer if it doesn't exist
        if topic not in self.consumers:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id=self.config.KAFKA_GROUP_ID,
                auto_offset_reset='latest',
                enable_auto_commit=True,
            )
            self.consumers[topic] = consumer
            
            # Start consumer task
            asyncio.create_task(self._consume_messages(topic, consumer))
        
        self.logger.info(f"Subscribed to topic: {topic}")
    
    async def _consume_messages(self, topic: str, consumer: KafkaConsumer):
        """Consume messages from a topic."""
        try:
            for message in consumer:
                if not self.running:
                    break
                
                try:
                    # Parse message
                    msg_data = message.value
                    base_message = BaseMessage(
                        message_id=msg_data.get('message_id', ''),
                        timestamp=msg_data.get('timestamp', 0),
                        message_type=msg_data.get('message_type', ''),
                        source_module=msg_data.get('source_module', ''),
                        data=msg_data.get('data', {})
                    )
                    
                    # Call all subscribers
                    if topic in self.subscribers:
                        for callback in self.subscribers[topic]:
                            try:
                                await callback(base_message)
                            except Exception as e:
                                self.logger.error(f"Error in message callback: {e}")
                
                except Exception as e:
                    self.logger.error(f"Error processing message from {topic}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error consuming messages from {topic}: {e}")
    
    def create_message(self, message_type: MessageType, source_module: str, 
                      data: Dict[str, Any]) -> BaseMessage:
        """Create a new message."""
        import uuid
        import time
        
        return BaseMessage(
            message_id=str(uuid.uuid4()),
            timestamp=time.time(),
            message_type=message_type.value,
            source_module=source_module,
            data=data
        )
    
    # Convenience methods for common message types
    async def publish_market_data(self, symbol: str, data: Dict[str, Any]):
        """Publish market data message."""
        message = self.create_message(
            MessageType.MARKET_DATA_RECEIVED,
            "data_ingestion",
            {"symbol": symbol, **data}
        )
        await self.publish(f"market_data.{symbol}", message, symbol)
    
    async def publish_signal(self, symbol: str, signal_data: Dict[str, Any]):
        """Publish trading signal message."""
        message = self.create_message(
            MessageType.SIGNAL_GENERATED,
            "ai_core",
            {"symbol": symbol, **signal_data}
        )
        await self.publish(f"signals.{symbol}", message, symbol)
    
    async def publish_order_update(self, order_id: str, order_data: Dict[str, Any]):
        """Publish order update message."""
        message = self.create_message(
            MessageType.ORDER_FILLED,
            "execution_engine",
            {"order_id": order_id, **order_data}
        )
        await self.publish("orders", message, order_id)
    
    async def publish_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """Publish system alert message."""
        alert_message = self.create_message(
            MessageType.ALERT,
            "monitoring",
            {
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": time.time()
            }
        )
        await self.publish("alerts", alert_message)
