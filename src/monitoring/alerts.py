"""
Alerting and notification system.

Provides alerts for:
- Risk limit breaches
- Trade executions
- System errors
- Performance anomalies

Supports Telegram and Discord notifications.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
from loguru import logger


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert message."""

    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: dict[str, Any] | None = None


class AlertManager:
    """
    Alert and notification manager.

    Features:
    - Multi-channel notifications (Telegram, Discord, Console)
    - Alert throttling to prevent spam
    - Alert history tracking

    Example:
        ```python
        alerts = AlertManager(
            telegram_token="xxx",
            telegram_chat_id="xxx",
        )

        await alerts.send_alert(
            level=AlertLevel.WARNING,
            title="Daily Loss Limit",
            message="Daily loss limit reached: -5%",
        )
        ```
    """

    def __init__(
        self,
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
        discord_webhook: str | None = None,
        throttle_seconds: int = 60,
    ) -> None:
        """
        Initialize alert manager.

        Args:
            telegram_token: Telegram bot token.
            telegram_chat_id: Telegram chat ID.
            discord_webhook: Discord webhook URL.
            throttle_seconds: Minimum seconds between same alerts.
        """
        self._telegram_token = telegram_token
        self._telegram_chat_id = telegram_chat_id
        self._discord_webhook = discord_webhook
        self._throttle_seconds = throttle_seconds

        self._session: aiohttp.ClientSession | None = None
        self._alert_history: list[Alert] = []
        self._last_alerts: dict[str, datetime] = {}  # For throttling

    async def initialize(self) -> None:
        """Initialize HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Send an alert through all configured channels.

        Args:
            level: Alert severity level.
            title: Alert title.
            message: Alert message body.
            data: Optional data to include.
        """
        # Check throttling
        alert_key = f"{level.value}:{title}"
        if alert_key in self._last_alerts:
            elapsed = (datetime.utcnow() - self._last_alerts[alert_key]).total_seconds()
            if elapsed < self._throttle_seconds:
                logger.debug(f"Alert throttled: {title}")
                return

        self._last_alerts[alert_key] = datetime.utcnow()

        # Create alert
        alert = Alert(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            data=data,
        )
        self._alert_history.append(alert)

        # Log to console
        self._log_alert(alert)

        # Send to channels
        if self._telegram_token and self._telegram_chat_id:
            await self._send_telegram(alert)

        if self._discord_webhook:
            await self._send_discord(alert)

    def _log_alert(self, alert: Alert) -> None:
        """Log alert to console."""
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)

        log_func(f"[{alert.level.value.upper()}] {alert.title}: {alert.message}")

    async def _send_telegram(self, alert: Alert) -> None:
        """Send alert to Telegram."""
        if self._session is None:
            await self.initialize()

        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }.get(alert.level, "📢")

        text = f"{emoji} *{alert.title}*\n\n{alert.message}"

        if alert.data:
            text += "\n\n```\n"
            for key, value in alert.data.items():
                text += f"{key}: {value}\n"
            text += "```"

        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {
            "chat_id": self._telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Telegram error: {await response.text()}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    async def _send_discord(self, alert: Alert) -> None:
        """Send alert to Discord."""
        if self._session is None:
            await self.initialize()

        color = {
            AlertLevel.INFO: 0x3498DB,  # Blue
            AlertLevel.WARNING: 0xF1C40F,  # Yellow
            AlertLevel.ERROR: 0xE74C3C,  # Red
            AlertLevel.CRITICAL: 0x9B59B6,  # Purple
        }.get(alert.level, 0x95A5A6)

        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": color,
            "timestamp": alert.timestamp.isoformat(),
        }

        if alert.data:
            embed["fields"] = [
                {"name": key, "value": str(value), "inline": True}
                for key, value in alert.data.items()
            ]

        payload = {"embeds": [embed]}

        try:
            async with self._session.post(self._discord_webhook, json=payload) as response:
                if response.status not in [200, 204]:
                    logger.error(f"Discord error: {await response.text()}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    # Convenience methods for common alerts
    async def trade_executed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> None:
        """Send trade execution alert."""
        await self.send_alert(
            level=AlertLevel.INFO,
            title="Trade Executed",
            message=f"{side} {quantity} {symbol} @ {price}",
            data={"symbol": symbol, "side": side, "quantity": quantity, "price": price},
        )

    async def position_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
    ) -> None:
        """Send position closed alert."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        emoji = "🟢" if pnl >= 0 else "🔴"

        await self.send_alert(
            level=level,
            title=f"Position Closed {emoji}",
            message=f"{symbol}: {pnl:+.2f} ({pnl_pct:+.2%})",
            data={"symbol": symbol, "pnl": pnl, "pnl_pct": pnl_pct},
        )

    async def risk_limit_warning(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
    ) -> None:
        """Send risk limit warning."""
        await self.send_alert(
            level=AlertLevel.WARNING,
            title="Risk Limit Warning",
            message=f"{limit_type}: {current_value:.2%} / {limit_value:.2%}",
            data={"type": limit_type, "current": current_value, "limit": limit_value},
        )

    async def risk_limit_breached(
        self,
        limit_type: str,
        current_value: float,
    ) -> None:
        """Send risk limit breach alert."""
        await self.send_alert(
            level=AlertLevel.CRITICAL,
            title="⛔ Risk Limit Breached",
            message=f"{limit_type} limit exceeded: {current_value:.2%}. Trading halted.",
            data={"type": limit_type, "value": current_value},
        )

    async def system_error(
        self,
        component: str,
        error: str,
    ) -> None:
        """Send system error alert."""
        await self.send_alert(
            level=AlertLevel.ERROR,
            title="System Error",
            message=f"Error in {component}",
            data={"component": component, "error": error},
        )

    def get_recent_alerts(
        self,
        limit: int = 50,
        level: AlertLevel | None = None,
    ) -> list[Alert]:
        """Get recent alerts."""
        alerts = self._alert_history[-limit:]
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts
