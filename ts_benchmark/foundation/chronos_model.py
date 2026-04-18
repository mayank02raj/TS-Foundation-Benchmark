# ts_benchmark/foundation/chronos_model.py

"""Wrapper for Amazon Chronos time-series foundation model.

Chronos is a family of pretrained models for probabilistic time series
forecasting. It tokenizes time series into bins and uses a transformer
architecture trained on a large corpus of public time series data.

Falls back to a naive seasonal model if the chronos package isn't
installed, so the benchmark can still run without GPU dependencies.
"""

from __future__ import annotations

import logging
import time
import tracemalloc

import numpy as np

from ts_benchmark.exceptions import ModelError
from ts_benchmark.models import ForecastResult

logger = logging.getLogger(__name__)


class ChronosModel:
    """Chronos foundation model wrapper.

    Supports three modes:
      - zero_shot: no training on your data at all
      - few_shot: provide a small context window
      - (fine-tuning requires the full chronos training pipeline,
         which is out of scope for this wrapper)
    """

    name = "Chronos"

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pipeline = None
        self._using_fallback = False

    def _load_model(self) -> None:
        """Lazy-load the model. Falls back to naive if chronos isn't installed."""
        if self._pipeline is not None:
            return

        try:
            from chronos import ChronosPipeline
            import torch

            self._pipeline = ChronosPipeline.from_pretrained(
                self._model_name,
                device_map=self._device,
                torch_dtype=torch.float32,
            )
            logger.info(f"Loaded Chronos model: {self._model_name}")

        except ImportError:
            logger.warning(
                "chronos-forecasting not installed. Using seasonal naive fallback. "
                "Install with: pip install chronos-forecasting"
            )
            self._using_fallback = True

        except Exception as e:
            logger.warning(f"Failed to load Chronos ({e}). Using fallback.")
            self._using_fallback = True

    def forecast(
        self,
        train: np.ndarray,
        horizon: int,
        context_length: int | None = None,
    ) -> ForecastResult:
        """Generate forecast using Chronos or fallback.

        Args:
            train: Training time series.
            horizon: Number of steps to forecast.
            context_length: How much history to provide as context.
                None = use all available data (zero-shot uses full series).
        """
        self._load_model()

        tracemalloc.start()
        train_start = time.perf_counter()

        # select context window
        if context_length is not None and context_length < len(train):
            context = train[-context_length:]
        else:
            context = train

        if self._using_fallback:
            predictions = self._naive_seasonal_forecast(context, horizon)
            train_time = 0.0  # no training for naive
        else:
            predictions = self._chronos_forecast(context, horizon)
            train_time = 0.0  # zero-shot = no training

        infer_time = time.perf_counter() - train_start - train_time

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        model_name = self.name if not self._using_fallback else "SeasonalNaive"

        return ForecastResult(
            model_name=model_name,
            predictions=predictions,
            actuals=np.array([]),
            train_time_seconds=train_time,
            inference_time_seconds=infer_time,
            peak_memory_mb=peak_mem / (1024 * 1024),
        )

    def _chronos_forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Run actual Chronos inference."""
        import torch

        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            forecast = self._pipeline.predict(
                context_tensor,
                prediction_length=horizon,
                num_samples=20,  # for probabilistic forecast
            )

        # take median of samples as point forecast
        median_forecast = np.median(forecast[0].numpy(), axis=0)
        return median_forecast

    def _naive_seasonal_forecast(
        self, context: np.ndarray, horizon: int, period: int = 24,
    ) -> np.ndarray:
        """Seasonal naive: repeat the last `period` values.

        This is the standard baseline in the Monash forecasting benchmark
        and surprisingly hard to beat on strongly seasonal data.
        """
        if len(context) < period:
            # not enough data for seasonal, fall back to last-value repeat
            return np.full(horizon, context[-1])

        last_season = context[-period:]
        repeats = (horizon // period) + 1
        tiled = np.tile(last_season, repeats)
        return tiled[:horizon]
