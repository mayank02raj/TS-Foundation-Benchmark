# ts_benchmark/evaluator.py

"""Evaluation harness: runs all models, computes metrics, builds report."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ts_benchmark.baselines import ARIMABaseline, LSTMBaseline, XGBoostBaseline
from ts_benchmark.datasets import load_dataset
from ts_benchmark.exceptions import EvaluationError, ModelError
from ts_benchmark.foundation import ChronosModel
from ts_benchmark.models import (
    BenchmarkReport,
    ForecastResult,
    MetricResult,
    TimeSeriesData,
)

logger = logging.getLogger(__name__)

# registry of available models
MODEL_REGISTRY: dict[str, type] = {
    "arima": ARIMABaseline,
    "lstm": LSTMBaseline,
    "xgboost": XGBoostBaseline,
    "chronos": ChronosModel,
}


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Skips zero values to avoid division by zero."""
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_crps(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Continuous Ranked Probability Score (simplified for point forecasts).

    For a proper CRPS you need a predictive distribution. With point
    forecasts, CRPS reduces to MAE. This is a placeholder that returns
    MAE; when probabilistic forecasts are available (Chronos samples),
    we'd use the full integral.
    """
    return compute_mae(actual, predicted)


class Evaluator:
    """Runs the benchmark: trains models, generates forecasts, computes metrics."""

    def __init__(self, methods: list[str] | None = None) -> None:
        if methods:
            invalid = [m for m in methods if m not in MODEL_REGISTRY]
            if invalid:
                raise EvaluationError(f"Unknown methods: {invalid}. Available: {list(MODEL_REGISTRY.keys())}")
            self._methods = methods
        else:
            self._methods = list(MODEL_REGISTRY.keys())

    def run(
        self,
        dataset_name: str,
        horizon: int = 24,
        data_length: int = 2000,
    ) -> BenchmarkReport:
        """Run the full benchmark on a dataset.

        Args:
            dataset_name: Name of dataset to load.
            horizon: Forecast horizon (steps ahead).
            data_length: Length of synthetic data to generate.

        Returns:
            BenchmarkReport with per-model metrics.
        """
        data = load_dataset(dataset_name, length=data_length)
        train, test = data.split(horizon)

        logger.info(
            f"Benchmark: dataset={dataset_name}, train={len(train)}, "
            f"test={len(test)}, horizon={horizon}, methods={self._methods}"
        )

        report = BenchmarkReport(
            dataset_name=dataset_name,
            horizon=horizon,
        )

        for method_name in self._methods:
            logger.info(f"Running {method_name}...")
            try:
                result = self._run_single(method_name, train, test, horizon)
                metrics = self._compute_metrics(result)
                report.results.append(metrics)
                logger.info(
                    f"  {method_name}: MAE={metrics.mae:.4f}, "
                    f"RMSE={metrics.rmse:.4f}, MAPE={metrics.mape:.2f}%"
                )
            except ModelError as e:
                logger.warning(f"  {method_name} failed: {e}")
            except Exception as e:
                logger.error(f"  {method_name} error: {e}")

        # few-shot learning curves (only for foundation models)
        for method_name in self._methods:
            if method_name in ("chronos",):
                curves = self._compute_few_shot_curve(method_name, train, test, horizon)
                if curves:
                    report.few_shot_curves[method_name] = curves

        return report

    def _run_single(
        self,
        method_name: str,
        train: np.ndarray,
        test: np.ndarray,
        horizon: int,
    ) -> ForecastResult:
        """Run a single model."""
        cls = MODEL_REGISTRY[method_name]
        model = cls()

        if method_name in ("chronos",):
            result = model.forecast(train, horizon)
        else:
            result = model.forecast(train, horizon)

        # attach actuals
        result.actuals = test[:horizon]
        # trim predictions to match
        if len(result.predictions) > horizon:
            result.predictions = result.predictions[:horizon]
        elif len(result.predictions) < horizon:
            # pad with last value if model produced fewer steps
            pad_len = horizon - len(result.predictions)
            result.predictions = np.concatenate([
                result.predictions,
                np.full(pad_len, result.predictions[-1]),
            ])

        return result

    def _compute_metrics(self, result: ForecastResult) -> MetricResult:
        """Compute all metrics for a single forecast."""
        actual = result.actuals
        predicted = result.predictions

        if len(actual) == 0 or len(predicted) == 0:
            raise EvaluationError("Empty actuals or predictions")

        # ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        return MetricResult(
            model_name=result.model_name,
            mae=compute_mae(actual, predicted),
            rmse=compute_rmse(actual, predicted),
            mape=compute_mape(actual, predicted),
            crps=compute_crps(actual, predicted),
            train_time=result.train_time_seconds,
            inference_time=result.inference_time_seconds,
            memory_mb=result.peak_memory_mb,
        )

    def _compute_few_shot_curve(
        self,
        method_name: str,
        train: np.ndarray,
        test: np.ndarray,
        horizon: int,
    ) -> list[tuple[int, float]]:
        """Compute MAE at different context lengths for foundation models.

        Shows how much domain data the model needs before it starts
        outperforming zero-shot.
        """
        context_lengths = [24, 48, 96, 168, 336, 720]
        curve: list[tuple[int, float]] = []

        for ctx_len in context_lengths:
            if ctx_len >= len(train):
                break

            try:
                cls = MODEL_REGISTRY[method_name]
                model = cls()
                result = model.forecast(train, horizon, context_length=ctx_len)
                result.actuals = test[:horizon]

                min_len = min(len(result.actuals), len(result.predictions))
                mae = compute_mae(result.actuals[:min_len], result.predictions[:min_len])
                curve.append((ctx_len, round(mae, 4)))
            except Exception as e:
                logger.debug(f"Few-shot at ctx={ctx_len} failed: {e}")

        return curve
