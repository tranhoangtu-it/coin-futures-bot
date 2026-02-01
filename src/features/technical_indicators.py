"""
Technical Indicators for trading signals.

Implements common technical analysis indicators:
- RSI, MACD, Bollinger Bands
- ATR for volatility
- Moving averages (SMA, EMA)

Follows @quant-analyst skill patterns.
"""

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalIndicators:
    """
    Calculator for technical analysis indicators.

    Provides vectorized implementations optimized for pandas.

    Example:
        ```python
        ti = TechnicalIndicators()

        # Calculate all indicators
        features = ti.calculate_all(df, close_col="close")

        # Individual indicators
        rsi = ti.rsi(df["close"])
        atr = ti.atr(df["high"], df["low"], df["close"])
        ```
    """

    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

        Args:
            close: Close prices.
            period: Lookback period.

        Returns:
            RSI values (0-100).
        """
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def macd(
        self,
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            close: Close prices.
            fast_period: Fast EMA period.
            slow_period: Slow EMA period.
            signal_period: Signal line EMA period.

        Returns:
            Tuple of (MACD line, Signal line, Histogram).
        """
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            close: Close prices.
            period: Moving average period.
            std_dev: Standard deviation multiplier.

        Returns:
            Tuple of (Upper band, Middle band, Lower band).
        """
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average True Range.

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = EMA of True Range

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            period: ATR period.

        Returns:
            ATR values.
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()

        return atr

    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            series: Input series.
            period: Window size.

        Returns:
            SMA values.
        """
        return series.rolling(period).mean()

    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            series: Input series.
            period: EMA period.

        Returns:
            EMA values.
        """
        return series.ewm(span=period, adjust=False).mean()

    def stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            k_period: %K period.
            d_period: %D smoothing period.

        Returns:
            Tuple of (%K, %D).
        """
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(d_period).mean()

        return k, d

    def adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average Directional Index.

        Measures trend strength (0-100).

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            period: ADX period.

        Returns:
            ADX values.
        """
        # True Range
        atr = self.atr(high, low, close, period)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smoothed DI
        plus_di = 100 * plus_dm.ewm(alpha=1 / period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / period).mean() / atr

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period).mean()

        return adx

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.

        Args:
            close: Close prices.
            volume: Trading volume.

        Returns:
            OBV values.
        """
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        return obv

    def calculate_all(
        self,
        df: pd.DataFrame,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators and return as new DataFrame.

        Args:
            df: OHLCV DataFrame.
            open_col: Open price column name.
            high_col: High price column name.
            low_col: Low price column name.
            close_col: Close price column name.
            volume_col: Volume column name.

        Returns:
            DataFrame with all technical indicators.
        """
        result = pd.DataFrame(index=df.index)

        close = df[close_col]
        high = df[high_col]
        low = df[low_col]

        # RSI
        result["rsi_14"] = self.rsi(close, 14)
        result["rsi_7"] = self.rsi(close, 7)

        # MACD
        macd, signal, hist = self.macd(close)
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self.bollinger_bands(close)
        result["bb_upper"] = bb_upper
        result["bb_mid"] = bb_mid
        result["bb_lower"] = bb_lower
        result["bb_width"] = (bb_upper - bb_lower) / bb_mid
        result["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower)

        # ATR
        result["atr_14"] = self.atr(high, low, close, 14)
        result["atr_pct"] = result["atr_14"] / close * 100

        # Moving Averages
        result["sma_20"] = self.sma(close, 20)
        result["sma_50"] = self.sma(close, 50)
        result["ema_12"] = self.ema(close, 12)
        result["ema_26"] = self.ema(close, 26)

        # Price relative to MAs
        result["price_sma20_ratio"] = close / result["sma_20"]
        result["price_sma50_ratio"] = close / result["sma_50"]

        # Stochastic
        stoch_k, stoch_d = self.stochastic(high, low, close)
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d

        # ADX
        result["adx"] = self.adx(high, low, close)

        # OBV (if volume available)
        if volume_col in df.columns:
            result["obv"] = self.obv(close, df[volume_col])
            result["obv_sma"] = self.sma(result["obv"], 20)

        # Returns
        result["returns_1"] = close.pct_change(1)
        result["returns_5"] = close.pct_change(5)
        result["returns_10"] = close.pct_change(10)

        # Volatility
        result["volatility_20"] = result["returns_1"].rolling(20).std() * np.sqrt(365)

        return result
