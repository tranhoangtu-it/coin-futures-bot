"""
MLflow setup for experiment tracking and model registry.

Provides integration with MLflow for:
- Experiment tracking
- Model versioning
- Artifact storage

Follows @ml-engineer skill patterns.
"""

from pathlib import Path
from typing import Any

import mlflow
from loguru import logger

from src.infrastructure.config import Settings, get_settings


class MLflowManager:
    """
    MLflow integration manager.

    Features:
    - Local experiment tracking
    - Model logging and loading
    - Artifact storage
    - Metric logging

    Example:
        ```python
        mlflow_mgr = MLflowManager()
        mlflow_mgr.setup()

        with mlflow_mgr.start_run("training_experiment"):
            mlflow_mgr.log_params({"learning_rate": 0.01})
            mlflow_mgr.log_metrics({"accuracy": 0.95})
            mlflow_mgr.log_model(model, "ensemble_model")
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize MLflow manager.

        Args:
            settings: Application settings.
        """
        self._settings = settings or get_settings()
        self._mlflow_settings = self._settings.mlflow
        self._active_run: mlflow.ActiveRun | None = None

    def setup(self) -> None:
        """Configure MLflow with local storage."""
        # Set tracking URI
        mlflow.set_tracking_uri(self._mlflow_settings.tracking_uri)

        # Create artifact directory
        artifact_path = Path(self._mlflow_settings.artifact_location)
        artifact_path.mkdir(parents=True, exist_ok=True)

        # Set experiment
        mlflow.set_experiment(self._mlflow_settings.experiment_name)

        logger.info(
            f"MLflow configured: tracking_uri={self._mlflow_settings.tracking_uri}, "
            f"experiment={self._mlflow_settings.experiment_name}"
        )

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run.
            tags: Tags to attach to the run.

        Returns:
            ActiveRun context manager.
        """
        self._active_run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name or 'unnamed'}")
        return self._active_run

    def end_run(self) -> None:
        """End the current run."""
        if self._active_run:
            mlflow.end_run()
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        mlflow.log_params(params)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str | None = None,
    ) -> None:
        """
        Log a model to MLflow.

        Args:
            model: Model object to log.
            artifact_path: Path within artifacts.
            registered_model_name: Optional name for model registry.
        """
        # Detect model type and use appropriate logger
        model_type = type(model).__name__

        if "XGBClassifier" in model_type or "XGBoost" in model_type:
            mlflow.xgboost.log_model(model, artifact_path)
        elif hasattr(model, "state_dict"):  # PyTorch
            mlflow.pytorch.log_model(model, artifact_path)
        else:
            # Generic pickle-based logging
            mlflow.sklearn.log_model(model, artifact_path)

        if registered_model_name:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            mlflow.register_model(model_uri, registered_model_name)

        logger.info(f"Logged model to {artifact_path}")

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLflow.

        Args:
            model_uri: URI of the model (runs:/ or models:/).

        Returns:
            Loaded model.
        """
        return mlflow.pyfunc.load_model(model_uri)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a file as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact."""
        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib figure."""
        mlflow.log_figure(figure, artifact_file)

    def get_run_id(self) -> str | None:
        """Get current run ID."""
        if mlflow.active_run():
            return mlflow.active_run().info.run_id
        return None

    def get_experiment_id(self) -> str | None:
        """Get current experiment ID."""
        experiment = mlflow.get_experiment_by_name(self._mlflow_settings.experiment_name)
        return experiment.experiment_id if experiment else None

    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for runs in the experiment.

        Args:
            filter_string: MLflow filter string.
            max_results: Maximum results to return.
            order_by: Columns to order by.

        Returns:
            List of run info dictionaries.
        """
        experiment_id = self.get_experiment_id()
        if not experiment_id:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
        )

        return runs.to_dict("records") if not runs.empty else []

    def get_best_run(self, metric: str = "sharpe", ascending: bool = False) -> dict[str, Any] | None:
        """
        Get the best run by a metric.

        Args:
            metric: Metric to sort by.
            ascending: Sort ascending if True.

        Returns:
            Best run info or None.
        """
        order = "ASC" if ascending else "DESC"
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        return runs[0] if runs else None
