# ts_benchmark/models.py

"""Data models for benchmark configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimeSeriesData:
    """A single time series for benchmarking."""
    name: str
    values: np.ndarray
    timestamps: np.ndarray | None = None
    frequency: str = "H"  # hourly default

    @property
    def length(self) -> int:
        return len(self.values)

    def split(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Split into train and test by forecast horizon."""
        return self.values[:-horizon], self.values[-horizon:]


@dataclass
class ForecastResult:
    """Predictions from a single model on a single series."""
    model_name: str
    predictions: np.ndarray
    actuals: np.ndarray
    train_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def horizon(self) -> int:
        return len(self.predictions)


@dataclass
class MetricResult:
    """Computed metrics for a single forecast."""
    model_name: str
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    crps: float = 0.0  # continuous ranked probability score
    train_time: float = 0.0
    inference_time: float = 0.0
    memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "crps": round(self.crps, 4),
            "train_time_s": round(self.train_time, 2),
            "inference_time_s": round(self.inference_time, 4),
            "memory_mb": round(self.memory_mb, 1),
        }


@dataclass
class BenchmarkReport:
    """Full benchmark results across all models and datasets."""
    dataset_name: str
    horizon: int
    results: list[MetricResult] = field(default_factory=list)
    few_shot_curves: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "horizon": self.horizon,
            "results": [r.to_dict() for r in self.results],
            "few_shot_curves": self.few_shot_curves,
        }

    def best_model(self, metric: str = "mae") -> str:
        if not self.results:
            return "N/A"
        return min(self.results, key=lambda r: getattr(r, metric, float("inf"))).model_name
