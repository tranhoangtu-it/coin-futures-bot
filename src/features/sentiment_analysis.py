"""
Sentiment Analysis for crypto markets.

Uses VADER for social media sentiment analysis.
Integrates with CryptoPanic API for news sentiment.

Follows @quant-analyst skill patterns.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """
    Sentiment analyzer for cryptocurrency markets.

    Features:
    - VADER sentiment analysis for text
    - CryptoPanic API integration for crypto news
    - Rolling sentiment scores
    - Sentiment signals for trading filters

    Example:
        ```python
        analyzer = SentimentAnalyzer(cryptopanic_api_key="xxx")

        # Analyze text
        score = analyzer.analyze_text("Bitcoin to the moon! 🚀")

        # Get news sentiment
        await analyzer.initialize()
        news_sentiment = await analyzer.get_news_sentiment("BTC")
        ```
    """

    def __init__(
        self,
        cryptopanic_api_key: str | None = None,
        sentiment_window_hours: int = 1,
    ) -> None:
        """
        Initialize sentiment analyzer.

        Args:
            cryptopanic_api_key: API key for CryptoPanic (optional).
            sentiment_window_hours: Window for rolling sentiment.
        """
        self._vader = SentimentIntensityAnalyzer()
        self._cryptopanic_key = cryptopanic_api_key
        self._window_hours = sentiment_window_hours

        # Cache for news sentiment
        self._news_cache: dict[str, list[dict[str, Any]]] = {}
        self._cache_timestamp: datetime | None = None

        # HTTP session
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        """Initialize async resources."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close async resources."""
        if self._session:
            await self._session.close()
            self._session = None

    def analyze_text(self, text: str) -> dict[str, float]:
        """
        Analyze sentiment of text using VADER.

        VADER is specifically tuned for social media text and
        understands emojis, slang, and intensity modifiers.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with neg, neu, pos, compound scores.
        """
        return self._vader.polarity_scores(text)

    def get_compound_score(self, text: str) -> float:
        """
        Get compound sentiment score for text.

        Compound score is normalized between -1 (negative) and +1 (positive).

        Args:
            text: Text to analyze.

        Returns:
            Compound score (-1 to 1).
        """
        return self.analyze_text(text)["compound"]

    async def get_news_sentiment(
        self,
        currency: str = "BTC",
        limit: int = 50,
    ) -> dict[str, float]:
        """
        Get aggregated sentiment from CryptoPanic news.

        Args:
            currency: Currency code (e.g., "BTC", "ETH").
            limit: Maximum news items to fetch.

        Returns:
            Dictionary with average sentiment scores.
        """
        if not self._cryptopanic_key:
            logger.warning("CryptoPanic API key not configured")
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0}

        if self._session is None:
            await self.initialize()

        try:
            news_items = await self._fetch_cryptopanic_news(currency, limit)

            if not news_items:
                return {"compound": 0.0, "positive": 0.0, "negative": 0.0}

            # Analyze each headline
            scores = []
            for item in news_items:
                title = item.get("title", "")
                score = self.get_compound_score(title)
                scores.append(score)

            avg_compound = sum(scores) / len(scores)

            # Count positive/negative
            positive = sum(1 for s in scores if s > 0.05)
            negative = sum(1 for s in scores if s < -0.05)
            total = len(scores)

            return {
                "compound": avg_compound,
                "positive_ratio": positive / total if total > 0 else 0.5,
                "negative_ratio": negative / total if total > 0 else 0.5,
                "sample_size": total,
            }

        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0}

    async def _fetch_cryptopanic_news(
        self,
        currency: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch news from CryptoPanic API."""
        if self._session is None:
            return []

        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self._cryptopanic_key,
            "currencies": currency,
            "kind": "news",
            "filter": "hot",
            "public": "true",
        }

        try:
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"CryptoPanic API error: {response.status}")
                    return []

                data = await response.json()
                results = data.get("results", [])[:limit]

                # Cache results
                self._news_cache[currency] = results
                self._cache_timestamp = datetime.utcnow()

                return results

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching news: {e}")
            return []

    def analyze_batch(self, texts: list[str]) -> list[float]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of compound scores.
        """
        return [self.get_compound_score(text) for text in texts]

    def get_sentiment_signal(
        self,
        compound_score: float,
        positive_threshold: float = 0.2,
        negative_threshold: float = -0.2,
    ) -> int:
        """
        Convert sentiment score to trading signal.

        Args:
            compound_score: Compound sentiment score.
            positive_threshold: Threshold for positive signal.
            negative_threshold: Threshold for negative signal.

        Returns:
            1 (bullish), -1 (bearish), or 0 (neutral).
        """
        if compound_score >= positive_threshold:
            return 1
        elif compound_score <= negative_threshold:
            return -1
        return 0

    def should_filter_trade(
        self,
        trade_signal: int,
        sentiment_score: float,
        filter_threshold: float = 0.1,
    ) -> bool:
        """
        Check if trade should be filtered based on sentiment.

        Filters trades that go against strong sentiment.

        Args:
            trade_signal: Trade signal (1=long, -1=short).
            sentiment_score: Current sentiment score.
            filter_threshold: Minimum sentiment for filtering.

        Returns:
            True if trade should be filtered out.
        """
        if trade_signal == 1 and sentiment_score < -filter_threshold:
            # Block long when sentiment is negative
            return True
        if trade_signal == -1 and sentiment_score > filter_threshold:
            # Block short when sentiment is positive
            return True
        return False
