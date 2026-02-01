"""
Fractional Differencing for stationarity with memory preservation.

Implements the fractional differencing operator to make price series stationary
while retaining long-term memory. Uses the ADF test to find optimal d.

Based on Advances in Financial Machine Learning by Marcos López de Prado.
Follows @quant-analyst skill patterns.
"""

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller


class FractionalDifferencing:
    """
    Fractional differencing transformer.

    Applies fractional differencing to make time series stationary while
    preserving memory. Finds optimal d using ADF test.

    Theory:
    - Standard differencing (d=1) removes all memory
    - Fractional differencing (0 < d < 1) preserves partial memory
    - Uses binomial expansion weights that decay geometrically

    Example:
        ```python
        fd = FractionalDifferencing()
        
        # Find optimal d
        optimal_d = fd.find_optimal_d(prices)
        
        # Apply transformation
        stationary_series = fd.transform(prices, d=optimal_d)
        ```
    """

    def __init__(
        self,
        adf_threshold: float = 0.05,
        weight_threshold: float = 1e-5,
        max_lags: int = 100,
    ) -> None:
        """
        Initialize fractional differencing transformer.

        Args:
            adf_threshold: p-value threshold for ADF test (stationarity).
            weight_threshold: Minimum weight to include in computation.
            max_lags: Maximum number of lags to consider.
        """
        self.adf_threshold = adf_threshold
        self.weight_threshold = weight_threshold
        self.max_lags = max_lags
        self._optimal_d: float | None = None

    @property
    def optimal_d(self) -> float | None:
        """Get the last computed optimal d value."""
        return self._optimal_d

    def _get_weights(self, d: float, size: int) -> np.ndarray:
        """
        Compute fractional differencing weights using binomial expansion.

        The weights are computed as:
        w_k = (-1)^k * C(d, k) = w_{k-1} * (k - d - 1) / k

        Args:
            d: Fractional differencing order.
            size: Number of weights to compute.

        Returns:
            Array of weights.
        """
        weights = [1.0]
        for k in range(1, size):
            w = weights[-1] * (k - d - 1) / k
            if abs(w) < self.weight_threshold:
                break
            weights.append(w)
        return np.array(weights[::-1])  # Reverse for convolution

    def _get_weights_ffd(self, d: float, threshold: float = 1e-5) -> np.ndarray:
        """
        Compute Fixed-Width Window Fractional Differencing weights.

        This version cuts off weights below threshold regardless of series length.

        Args:
            d: Fractional differencing order.
            threshold: Minimum weight threshold.

        Returns:
            Array of weights.
        """
        weights = [1.0]
        k = 1
        while True:
            w = weights[-1] * (k - d - 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1
            if k > self.max_lags:
                break
        return np.array(weights[::-1])

    def transform(
        self,
        series: pd.Series | np.ndarray,
        d: float,
        fixed_width: bool = True,
    ) -> pd.Series:
        """
        Apply fractional differencing to series.

        Args:
            series: Input time series (prices).
            d: Fractional differencing order (0 to 1).
            fixed_width: Use fixed-width window (FFD) method.

        Returns:
            Fractionally differenced series.
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        if fixed_width:
            weights = self._get_weights_ffd(d, self.weight_threshold)
        else:
            weights = self._get_weights(d, len(series))

        width = len(weights)

        # Apply convolution
        result = pd.Series(index=series.index, dtype=float)

        for i in range(width - 1, len(series)):
            window = series.iloc[i - width + 1 : i + 1].values
            result.iloc[i] = np.dot(weights, window)

        return result.dropna()

    def find_optimal_d(
        self,
        series: pd.Series | np.ndarray,
        d_range: tuple[float, float] = (0.0, 1.0),
        step: float = 0.05,
    ) -> float:
        """
        Find minimum d that achieves stationarity.

        Uses binary search to find the smallest d where ADF test
        rejects the null hypothesis of a unit root.

        Args:
            series: Input time series.
            d_range: Range of d values to search.
            step: Step size for initial search.

        Returns:
            Optimal d value.
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        logger.info(f"Finding optimal d in range {d_range}")

        # Initial grid search
        best_d = d_range[1]

        for d in np.arange(d_range[0], d_range[1] + step, step):
            if d == 0:
                continue

            try:
                diff_series = self.transform(series, d)
                if len(diff_series) < 10:
                    continue

                # Run ADF test
                adf_stat, p_value, *_ = adfuller(diff_series.dropna(), maxlag=1)

                if p_value < self.adf_threshold:
                    best_d = d
                    logger.info(f"d={d:.3f}: ADF p-value={p_value:.4f} - Stationary")
                    break
                else:
                    logger.debug(f"d={d:.3f}: ADF p-value={p_value:.4f} - Not stationary")

            except Exception as e:
                logger.warning(f"Error at d={d}: {e}")
                continue

        self._optimal_d = best_d
        logger.info(f"Optimal d found: {best_d:.3f}")
        return best_d

    def fit_transform(
        self,
        series: pd.Series | np.ndarray,
        d_range: tuple[float, float] = (0.0, 1.0),
    ) -> pd.Series:
        """
        Find optimal d and apply transformation.

        Args:
            series: Input time series.
            d_range: Range of d values to search.

        Returns:
            Fractionally differenced series.
        """
        optimal_d = self.find_optimal_d(series, d_range)
        return self.transform(series, optimal_d)

    @staticmethod
    def get_memory_ratio(d: float) -> float:
        """
        Estimate memory retention ratio for given d.

        Higher d = less memory retained.

        Args:
            d: Fractional differencing order.

        Returns:
            Estimated memory ratio (0-1).
        """
        return 1.0 - d
