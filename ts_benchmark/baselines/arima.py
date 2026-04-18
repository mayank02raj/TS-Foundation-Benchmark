# ts_benchmark/baselines/arima.py

"""ARIMA baseline using statsmodels.

Auto-selects order (p, d, q) via AIC minimization over a grid.
Not as sophisticated as auto_arima from pmdarima, but avoids
the dependency and the grid search is fast enough for benchmarking.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
import warnings
from itertools import product

import numpy as np

from ts_benchmark.exceptions import ModelError
from ts_benchmark.models import ForecastResult

logger = logging.getLogger(__name__)

# search grid for auto-selection
_P_RANGE = range(0, 4)
_D_RANGE = range(0, 2)
_Q_RANGE = range(0, 4)


class ARIMABaseline:
    """ARIMA forecaster with automatic order selection."""

    name = "ARIMA"

    def __init__(self, max_order: int = 3) -> None:
        self._max_order = max_order
        self._order: tuple[int, int, int] | None = None
        self._model_fit = None

    def forecast(self, train: np.ndarray, horizon: int) -> ForecastResult:
        """Fit ARIMA and forecast.

        Tries a grid of (p, d, q) orders and picks the one with
        lowest AIC. Suppresses convergence warnings since we're
        testing many configurations and some won't converge.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ModelError("statsmodels required for ARIMA. pip install statsmodels")

        tracemalloc.start()
        train_start = time.perf_counter()

        best_aic = float("inf")
        best_order = (1, 1, 1)
        best_fit = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for p, d, q in product(_P_RANGE, _D_RANGE, _Q_RANGE):
                if p == 0 and q == 0:
                    continue
                if p + d + q > self._max_order * 2:
                    continue

                try:
                    model = ARIMA(train, order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        best_fit = fit
                except Exception:
                    continue

        if best_fit is None:
            # fallback to simple (1,1,1)
            logger.warning("No ARIMA order converged, falling back to (1,1,1)")
            try:
                model = ARIMA(train, order=(1, 1, 1))
                best_fit = model.fit()
                best_order = (1, 1, 1)
            except Exception as e:
                raise ModelError(f"ARIMA fitting failed completely: {e}") from e

        self._order = best_order
        self._model_fit = best_fit

        train_time = time.perf_counter() - train_start

        # forecast
        infer_start = time.perf_counter()
        predictions = best_fit.forecast(steps=horizon)
        infer_time = time.perf_counter() - infer_start

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logger.info(f"ARIMA order={best_order}, AIC={best_aic:.1f}, train={train_time:.2f}s")

        return ForecastResult(
            model_name=self.name,
            predictions=np.array(predictions),
            actuals=np.array([]),  # filled by evaluator
            train_time_seconds=train_time,
            inference_time_seconds=infer_time,
            peak_memory_mb=peak_mem / (1024 * 1024),
        )
