# ts_benchmark/datasets.py

"""Dataset loaders for benchmarking.

Generates synthetic datasets for testing. In production you'd
load real datasets (ETTh1, Electricity, Traffic, etc.) but
synthetic data lets anyone run the benchmark without downloading.
"""

from __future__ import annotations

import logging

import numpy as np

from ts_benchmark.exceptions import DatasetError
from ts_benchmark.models import TimeSeriesData

logger = logging.getLogger(__name__)

AVAILABLE_DATASETS = ["synthetic_trend", "synthetic_seasonal", "synthetic_complex"]


def load_dataset(name: str, length: int = 2000) -> TimeSeriesData:
    """Load or generate a dataset by name."""
    generators = {
        "synthetic_trend": _generate_trend,
        "synthetic_seasonal": _generate_seasonal,
        "synthetic_complex": _generate_complex,
    }

    if name not in generators:
        raise DatasetError(
            f"Unknown dataset: {name}. Available: {', '.join(generators.keys())}"
        )

    logger.info(f"Generating dataset: {name} (length={length})")
    return generators[name](length)


def _generate_trend(length: int) -> TimeSeriesData:
    """Linear trend with noise. Tests basic extrapolation."""
    rng = np.random.default_rng(42)
    t = np.arange(length, dtype=np.float64)
    values = 50.0 + 0.05 * t + rng.normal(0, 2.0, length)
    return TimeSeriesData(name="synthetic_trend", values=values, frequency="H")


def _generate_seasonal(length: int) -> TimeSeriesData:
    """Strong seasonal pattern (daily cycle) with noise."""
    rng = np.random.default_rng(42)
    t = np.arange(length, dtype=np.float64)
    daily_cycle = 10.0 * np.sin(2 * np.pi * t / 24)
    values = 100.0 + daily_cycle + rng.normal(0, 3.0, length)
    return TimeSeriesData(name="synthetic_seasonal", values=values, frequency="H")


def _generate_complex(length: int) -> TimeSeriesData:
    """Trend + multiple seasonalities + level shifts + noise.
    This is the hard one -- mimics real-world electricity/traffic data."""
    rng = np.random.default_rng(42)
    t = np.arange(length, dtype=np.float64)

    trend = 0.02 * t
    daily = 15.0 * np.sin(2 * np.pi * t / 24)
    weekly = 8.0 * np.sin(2 * np.pi * t / 168)

    # level shift at 60% of the series
    shift_point = int(length * 0.6)
    level_shift = np.zeros(length)
    level_shift[shift_point:] = 20.0

    noise = rng.normal(0, 5.0, length)

    # occasional outliers (simulate anomalies)
    outlier_mask = rng.random(length) < 0.01
    outliers = outlier_mask.astype(float) * rng.normal(0, 30.0, length)

    values = 200.0 + trend + daily + weekly + level_shift + noise + outliers
    return TimeSeriesData(name="synthetic_complex", values=values, frequency="H")
