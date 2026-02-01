"""
WebSocket Manager for Binance Futures.

Implements async WebSocket connection with:
- Automatic reconnection with exponential backoff
- Heartbeat/ping-pong mechanism
- Event-driven message handling
- Multiple stream subscription

Follows @python-pro skill patterns for async programming.
"""

import asyncio
import random
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any

import orjson
import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosed, WebSocketException

from .config import Settings, get_settings


class ConnectionState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


class WebSocketManager:
    """
    Manages WebSocket connections to Binance Futures.

    Features:
    - Automatic reconnection with exponential backoff
    - Heartbeat mechanism for connection health monitoring
    - Support for multiple stream subscriptions
    - Async message handling with callbacks

    Example:
        ```python
        manager = WebSocketManager()
        manager.on_message(handle_depth_update)
        await manager.subscribe(["btcusdt@depth@100ms", "ethusdt@depth@100ms"])
        await manager.start()
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize WebSocket manager.

        Args:
            settings: Application settings. If None, uses default settings.
        """
        self._settings = settings or get_settings()
        self._ws_settings = self._settings.websocket
        self._binance_settings = self._settings.binance

        # Connection state
        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_count = 0
        self._running = False

        # Subscriptions and handlers
        self._streams: list[str] = []
        self._message_handlers: list[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = []

        # Tasks
        self._receive_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

        # Locks
        self._connect_lock = asyncio.Lock()

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._state == ConnectionState.CONNECTED and self._websocket is not None

    def on_message(
        self, handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]]
    ) -> None:
        """
        Register a message handler.

        Args:
            handler: Async function to handle incoming messages.
        """
        self._message_handlers.append(handler)
        logger.debug(f"Registered message handler: {handler.__name__}")

    async def subscribe(self, streams: list[str]) -> None:
        """
        Subscribe to WebSocket streams.

        Args:
            streams: List of stream names (e.g., ["btcusdt@depth@100ms"]).
        """
        self._streams.extend(streams)
        logger.info(f"Subscribed to streams: {streams}")

        # If already connected, send subscription request
        if self.is_connected and self._websocket:
            await self._send_subscription()

    async def unsubscribe(self, streams: list[str]) -> None:
        """
        Unsubscribe from WebSocket streams.

        Args:
            streams: List of stream names to unsubscribe.
        """
        for stream in streams:
            if stream in self._streams:
                self._streams.remove(stream)
        logger.info(f"Unsubscribed from streams: {streams}")

        # If connected, send unsubscription request
        if self.is_connected and self._websocket:
            await self._send_unsubscription(streams)

    async def start(self) -> None:
        """Start the WebSocket manager."""
        if self._running:
            logger.warning("WebSocket manager already running")
            return

        self._running = True
        logger.info("Starting WebSocket manager")

        try:
            await self._connect()
            await self._run()
        except asyncio.CancelledError:
            logger.info("WebSocket manager cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the WebSocket manager gracefully."""
        logger.info("Stopping WebSocket manager")
        self._running = False
        self._state = ConnectionState.CLOSED

        # Cancel tasks
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        logger.info("WebSocket manager stopped")

    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        async with self._connect_lock:
            if self.is_connected:
                return

            self._state = ConnectionState.CONNECTING
            ws_url = self._build_ws_url()

            try:
                logger.info(f"Connecting to {ws_url}")
                self._websocket = await websockets.connect(
                    ws_url,
                    ping_interval=None,  # We handle our own heartbeat
                    ping_timeout=None,
                    close_timeout=10,
                )
                self._state = ConnectionState.CONNECTED
                self._reconnect_count = 0
                logger.info("WebSocket connected successfully")

                # Send subscription if streams are configured
                if self._streams:
                    await self._send_subscription()

            except (WebSocketException, OSError) as e:
                self._state = ConnectionState.DISCONNECTED
                logger.error(f"Failed to connect: {e}")
                raise

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if not self._running:
            return

        self._state = ConnectionState.RECONNECTING

        while self._running and self._reconnect_count < self._ws_settings.reconnect_max_attempts:
            self._reconnect_count += 1

            # Calculate delay with exponential backoff and jitter
            delay = min(
                self._ws_settings.reconnect_delay_base * (2 ** (self._reconnect_count - 1)),
                self._ws_settings.reconnect_delay_max,
            )
            # Add jitter (±25%)
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter

            logger.info(
                f"Reconnecting in {delay:.2f}s (attempt {self._reconnect_count}/"
                f"{self._ws_settings.reconnect_max_attempts})"
            )
            await asyncio.sleep(delay)

            try:
                await self._connect()
                return
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")

        logger.error("Max reconnection attempts reached")
        self._state = ConnectionState.CLOSED
        self._running = False

    async def _run(self) -> None:
        """Main run loop."""
        while self._running:
            if not self.is_connected:
                await self._reconnect()
                if not self._running:
                    break

            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            try:
                await self._receive_loop()
            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                self._state = ConnectionState.DISCONNECTED
                self._websocket = None
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self._state = ConnectionState.DISCONNECTED
                self._websocket = None
            finally:
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass

    async def _receive_loop(self) -> None:
        """Receive and process messages."""
        if not self._websocket:
            return

        async for message in self._websocket:
            if not self._running:
                break

            try:
                data = orjson.loads(message)
                await self._dispatch_message(data)
            except orjson.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _dispatch_message(self, data: dict[str, Any]) -> None:
        """Dispatch message to all registered handlers."""
        for handler in self._message_handlers:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Handler {handler.__name__} error: {e}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while self._running and self.is_connected:
            await asyncio.sleep(self._ws_settings.heartbeat_interval)

            if not self._websocket:
                break

            try:
                # Send ping and wait for pong
                pong = await self._websocket.ping()
                await asyncio.wait_for(pong, timeout=self._ws_settings.heartbeat_timeout)
                logger.debug("Heartbeat OK")
            except asyncio.TimeoutError:
                logger.warning("Heartbeat timeout - connection may be stale")
                await self._websocket.close()
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def _send_subscription(self) -> None:
        """Send subscription request to Binance."""
        if not self._websocket or not self._streams:
            return

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": self._streams,
            "id": 1,
        }
        await self._websocket.send(orjson.dumps(subscribe_msg).decode())
        logger.debug(f"Sent subscription request: {self._streams}")

    async def _send_unsubscription(self, streams: list[str]) -> None:
        """Send unsubscription request to Binance."""
        if not self._websocket or not streams:
            return

        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": 2,
        }
        await self._websocket.send(orjson.dumps(unsubscribe_msg).decode())
        logger.debug(f"Sent unsubscription request: {streams}")

    def _build_ws_url(self) -> str:
        """Build WebSocket URL with combined streams."""
        base_url = self._binance_settings.ws_url

        if self._streams:
            streams_path = "/".join(self._streams)
            return f"{base_url}/stream?streams={streams_path}"

        return f"{base_url}/ws"
