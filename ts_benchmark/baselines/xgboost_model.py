# ts_benchmark/baselines/xgboost_model.py

"""XGBoost baseline with engineered lag features.

Converts the time series into a tabular supervised problem using
lagged values, rolling statistics, and time-of-day features.
This is how most Kaggle-style time series solutions work and
it's often surprisingly competitive with deep learning methods.
"""

from __future__ import annotations

import logging
import time
import tracemalloc

import numpy as np

from ts_benchmark.exceptions import ModelError
from ts_benchmark.models import ForecastResult

logger = logging.getLogger(__name__)


class XGBoostBaseline:
    """XGBoost forecaster with lag feature engineering."""

    name = "XGBoost"

    def __init__(
        self,
        lags: list[int] | None = None,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self._lags = lags or [1, 2, 3, 6, 12, 24, 48, 168]
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._lr = learning_rate
        self._model = None

    def forecast(self, train: np.ndarray, horizon: int) -> ForecastResult:
        """Build lag features, train XGBoost, forecast autoregressively."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ModelError("xgboost required. pip install xgboost")

        tracemalloc.start()
        train_start = time.perf_counter()

        max_lag = max(self._lags)
        if len(train) <= max_lag + 10:
            raise ModelError(f"Training data too short ({len(train)}) for max lag {max_lag}")

        X, y = self._build_features(train)

        model = XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._lr,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        self._model = model

        train_time = time.perf_counter() - train_start

        # autoregressive forecast
        infer_start = time.perf_counter()
        extended = list(train)
        predictions: list[float] = []

        for _ in range(horizon):
            feat = self._compute_single_features(np.array(extended))
            pred = float(model.predict(feat.reshape(1, -1))[0])
            predictions.append(pred)
            extended.append(pred)

        infer_time = time.perf_counter() - infer_start

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logger.info(f"XGBoost trained in {train_time:.2f}s, {len(X)} samples, {X.shape[1]} features")

        return ForecastResult(
            model_name=self.name,
            predictions=np.array(predictions),
            actuals=np.array([]),
            train_time_seconds=train_time,
            inference_time_seconds=infer_time,
            peak_memory_mb=peak_mem / (1024 * 1024),
        )

    def _build_features(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build tabular features from the full training series."""
        max_lag = max(self._lags)
        n_samples = len(series) - max_lag

        X_rows: list[np.ndarray] = []
        y_vals: list[float] = []

        for i in range(max_lag, len(series)):
            feat = self._compute_single_features(series[:i + 1])
            X_rows.append(feat)
            y_vals.append(series[i])

        return np.array(X_rows), np.array(y_vals)

    def _compute_single_features(self, series: np.ndarray) -> np.ndarray:
        """Compute features for the last position in the series."""
        features: list[float] = []

        # lag values
        for lag in self._lags:
            if lag <= len(series):
                features.append(float(series[-lag]))
            else:
                features.append(0.0)

        # rolling statistics
        for window in [6, 12, 24]:
            if window <= len(series):
                window_data = series[-window:]
                features.append(float(np.mean(window_data)))
                features.append(float(np.std(window_data)))
                features.append(float(np.max(window_data) - np.min(window_data)))
            else:
                features.extend([0.0, 0.0, 0.0])

        # position in day (assuming hourly data)
        hour = len(series) % 24
        features.append(float(np.sin(2 * np.pi * hour / 24)))
        features.append(float(np.cos(2 * np.pi * hour / 24)))

        # position in week
        day_of_week = (len(series) // 24) % 7
        features.append(float(np.sin(2 * np.pi * day_of_week / 7)))
        features.append(float(np.cos(2 * np.pi * day_of_week / 7)))

        return np.array(features, dtype=np.float64)
