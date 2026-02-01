"""
Walk-Forward Validation for robust strategy testing.

Implements anchored and rolling walk-forward analysis to test
strategy robustness with proper train/test splits.

Based on Advances in Financial Machine Learning.
Follows @backtesting-frameworks skill patterns.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""

    fold_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_samples: int
    test_samples: int
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""

    n_folds: int
    fold_results: list[FoldResult]
    avg_train_metrics: dict[str, float]
    avg_test_metrics: dict[str, float]
    std_test_metrics: dict[str, float]
    overall_stability: float  # Metric consistency across folds


class WalkForwardValidator:
    """
    Walk-Forward Validation framework.

    Features:
    - Anchored walk-forward (expanding training window)
    - Rolling walk-forward (fixed training window)
    - Purged k-fold (gap between train/test)
    - Metric collection across folds

    Example:
        ```python
        validator = WalkForwardValidator(
            n_folds=5,
            train_ratio=0.8,
            purge_gap=10,  # 10 bars gap
        )

        result = validator.validate(
            data=df,
            train_func=train_model,
            test_func=evaluate_model,
        )
        ```
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.8,
        purge_gap: int = 10,
        embargo_gap: int = 5,
        rolling: bool = False,
    ) -> None:
        """
        Initialize walk-forward validator.

        Args:
            n_folds: Number of walk-forward folds.
            train_ratio: Ratio of data for training in each fold.
            purge_gap: Number of samples to exclude between train/test (look-ahead prevention).
            embargo_gap: Additional samples to exclude at train end.
            rolling: Use rolling window (True) or expanding window (False).
        """
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.rolling = rolling

    def get_fold_indices(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            data: Input DataFrame.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n_samples = len(data)
        folds = []

        # Calculate fold size
        fold_size = n_samples // self.n_folds

        for i in range(self.n_folds):
            # Test window
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            if self.rolling:
                # Rolling: fixed training window before test
                train_size = int(fold_size * self.train_ratio / (1 - self.train_ratio))
                train_start = max(0, test_start - train_size - self.purge_gap - self.embargo_gap)
                train_end = test_start - self.purge_gap
            else:
                # Anchored: expanding training window
                train_start = 0
                train_end = test_start - self.purge_gap

            if train_end <= train_start:
                continue

            # Apply embargo
            train_end = train_end - self.embargo_gap

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            folds.append((train_indices, test_indices))

        return folds

    def validate(
        self,
        data: pd.DataFrame,
        train_func: Callable[[pd.DataFrame], Any],
        test_func: Callable[[Any, pd.DataFrame], dict[str, float]],
        train_metrics_func: Callable[[Any, pd.DataFrame], dict[str, float]] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: Full dataset.
            train_func: Function to train model, takes train data, returns model.
            test_func: Function to evaluate model, takes model and test data, returns metrics dict.
            train_metrics_func: Optional function to get train metrics.

        Returns:
            WalkForwardResult with all fold results.
        """
        folds = self.get_fold_indices(data)
        fold_results: list[FoldResult] = []

        logger.info(f"Starting walk-forward validation with {len(folds)} folds")

        for fold_num, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"Fold {fold_num + 1}/{len(folds)}")

            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Train model
            model = train_func(train_data)

            # Get metrics
            if train_metrics_func:
                train_metrics = train_metrics_func(model, train_data)
            else:
                train_metrics = {}

            test_metrics = test_func(model, test_data)

            # Record fold result
            result = FoldResult(
                fold_number=fold_num + 1,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            fold_results.append(result)

            logger.info(f"Fold {fold_num + 1} test metrics: {test_metrics}")

        # Aggregate results
        return self._aggregate_results(fold_results)

    def _aggregate_results(
        self,
        fold_results: list[FoldResult],
    ) -> WalkForwardResult:
        """Aggregate results across folds."""
        if not fold_results:
            return WalkForwardResult(
                n_folds=0,
                fold_results=[],
                avg_train_metrics={},
                avg_test_metrics={},
                std_test_metrics={},
                overall_stability=0.0,
            )

        # Collect all metric keys
        test_metric_keys = set()
        train_metric_keys = set()
        for fold in fold_results:
            test_metric_keys.update(fold.test_metrics.keys())
            train_metric_keys.update(fold.train_metrics.keys())

        # Calculate averages and stds
        avg_test_metrics = {}
        std_test_metrics = {}
        avg_train_metrics = {}

        for key in test_metric_keys:
            values = [f.test_metrics.get(key, 0) for f in fold_results]
            avg_test_metrics[key] = np.mean(values)
            std_test_metrics[key] = np.std(values)

        for key in train_metric_keys:
            values = [f.train_metrics.get(key, 0) for f in fold_results]
            avg_train_metrics[key] = np.mean(values)

        # Calculate stability (coefficient of variation for key metrics)
        stability_scores = []
        for key in ["sharpe", "accuracy", "profit_factor"]:
            if key in avg_test_metrics and avg_test_metrics[key] != 0:
                cv = std_test_metrics[key] / abs(avg_test_metrics[key])
                stability_scores.append(1 - min(cv, 1))  # Higher is more stable

        overall_stability = np.mean(stability_scores) if stability_scores else 0.0

        return WalkForwardResult(
            n_folds=len(fold_results),
            fold_results=fold_results,
            avg_train_metrics=avg_train_metrics,
            avg_test_metrics=avg_test_metrics,
            std_test_metrics=std_test_metrics,
            overall_stability=overall_stability,
        )

    def validate_with_retraining(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        strategy_func: Callable[[Any, pd.DataFrame], pd.Series],
        train_func: Callable[[pd.DataFrame, pd.Series], Any],
        metrics_func: Callable[[pd.Series, pd.Series], dict[str, float]],
    ) -> WalkForwardResult:
        """
        Walk-forward with model retraining at each fold.

        Args:
            data: Feature data.
            signals: True signals/labels.
            strategy_func: Function to generate predictions from model.
            train_func: Function to train model on data and signals.
            metrics_func: Function to calculate metrics from predictions and actuals.

        Returns:
            WalkForwardResult.
        """
        folds = self.get_fold_indices(data)
        fold_results: list[FoldResult] = []

        for fold_num, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"Fold {fold_num + 1}/{len(folds)}: retraining model")

            # Split
            train_data = data.iloc[train_idx]
            train_signals = signals.iloc[train_idx]
            test_data = data.iloc[test_idx]
            test_signals = signals.iloc[test_idx]

            # Train
            model = train_func(train_data, train_signals)

            # Predict
            train_preds = strategy_func(model, train_data)
            test_preds = strategy_func(model, test_data)

            # Metrics
            train_metrics = metrics_func(train_preds, train_signals)
            test_metrics = metrics_func(test_preds, test_signals)

            result = FoldResult(
                fold_number=fold_num + 1,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            fold_results.append(result)

        return self._aggregate_results(fold_results)
