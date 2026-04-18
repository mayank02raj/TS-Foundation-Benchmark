# tests/test_benchmark.py

"""Tests for the time-series benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from ts_benchmark.datasets import load_dataset, AVAILABLE_DATASETS
from ts_benchmark.evaluator import (
    Evaluator,
    compute_mae,
    compute_mape,
    compute_rmse,
)
from ts_benchmark.exceptions import DatasetError, EvaluationError
from ts_benchmark.models import (
    BenchmarkReport,
    ForecastResult,
    MetricResult,
    TimeSeriesData,
)


# -- Datasets --

class TestDatasets:
    @pytest.mark.parametrize("name", AVAILABLE_DATASETS)
    def test_load_all_datasets(self, name):
        data = load_dataset(name, length=500)
        assert data.length == 500
        assert data.name == name
        assert not np.any(np.isnan(data.values))

    def test_unknown_dataset_raises(self):
        with pytest.raises(DatasetError):
            load_dataset("nonexistent")

    def test_split(self):
        data = load_dataset("synthetic_trend", length=200)
        train, test = data.split(horizon=24)
        assert len(train) == 176
        assert len(test) == 24

    def test_complex_has_level_shift(self):
        data = load_dataset("synthetic_complex", length=1000)
        first_half = np.mean(data.values[:400])
        second_half = np.mean(data.values[700:])
        # level shift should make second half higher
        assert second_half > first_half + 10

    def test_seasonal_has_periodicity(self):
        data = load_dataset("synthetic_seasonal", length=240)
        # check that there's a clear 24-hour cycle
        # autocorrelation at lag 24 should be high
        values = data.values - np.mean(data.values)
        if len(values) > 48:
            autocorr = np.correlate(values[:48], values[:48], mode="full")
            mid = len(autocorr) // 2
            # lag 24 should have a peak
            assert autocorr[mid + 24] > autocorr[mid + 12]


# -- Metrics --

class TestMetrics:
    def test_mae_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert compute_mae(actual, actual) == 0.0

    def test_mae_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert compute_mae(actual, predicted) == 1.0

    def test_rmse_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert abs(compute_rmse(actual, predicted) - 1.0) < 0.001

    def test_rmse_greater_than_mae(self):
        rng = np.random.default_rng(42)
        actual = rng.standard_normal(100)
        predicted = actual + rng.standard_normal(100) * 0.5
        assert compute_rmse(actual, predicted) >= compute_mae(actual, predicted)

    def test_mape_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert compute_mape(actual, actual) == 0.0

    def test_mape_with_zeros(self):
        actual = np.array([0.0, 1.0, 2.0])
        predicted = np.array([0.5, 1.5, 2.5])
        # should skip the zero value
        result = compute_mape(actual, predicted)
        assert result > 0


# -- Models --

class TestModels:
    def test_time_series_split(self):
        ts = TimeSeriesData(name="test", values=np.arange(100, dtype=float))
        train, test = ts.split(horizon=10)
        assert len(train) == 90
        assert len(test) == 10

    def test_forecast_result_horizon(self):
        result = ForecastResult(
            model_name="test",
            predictions=np.zeros(24),
            actuals=np.zeros(24),
        )
        assert result.horizon == 24

    def test_metric_result_to_dict(self):
        m = MetricResult(model_name="ARIMA", mae=1.5, rmse=2.0, mape=5.0)
        d = m.to_dict()
        assert d["model"] == "ARIMA"
        assert d["mae"] == 1.5

    def test_benchmark_report_best_model(self):
        report = BenchmarkReport(dataset_name="test", horizon=24)
        report.results.append(MetricResult(model_name="A", mae=2.0))
        report.results.append(MetricResult(model_name="B", mae=1.0))
        report.results.append(MetricResult(model_name="C", mae=3.0))
        assert report.best_model("mae") == "B"


# -- Baselines (unit tests that don't require heavy deps) --

class TestXGBoostBaseline:
    def test_forecast_shape(self):
        from ts_benchmark.baselines.xgboost_model import XGBoostBaseline

        rng = np.random.default_rng(42)
        train = 100.0 + np.cumsum(rng.standard_normal(500))
        horizon = 24

        model = XGBoostBaseline(n_estimators=10)
        result = model.forecast(train, horizon)

        assert len(result.predictions) == horizon
        assert result.train_time_seconds > 0
        assert result.model_name == "XGBoost"

    def test_xgboost_not_all_same(self):
        """Predictions shouldn't be a flat line."""
        from ts_benchmark.baselines.xgboost_model import XGBoostBaseline

        rng = np.random.default_rng(42)
        t = np.arange(500, dtype=float)
        train = 50.0 + 10.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, 500)

        model = XGBoostBaseline(n_estimators=50)
        result = model.forecast(train, 24)

        assert np.std(result.predictions) > 0.1


class TestARIMABaseline:
    def test_forecast_shape(self):
        from ts_benchmark.baselines.arima import ARIMABaseline

        rng = np.random.default_rng(42)
        train = 100.0 + np.cumsum(rng.standard_normal(200))

        model = ARIMABaseline(max_order=2)
        result = model.forecast(train, 12)

        assert len(result.predictions) == 12
        assert result.model_name == "ARIMA"


# -- Foundation Model (fallback mode) --

class TestChronosModel:
    def test_fallback_forecast(self):
        """Should work without chronos installed (uses seasonal naive)."""
        from ts_benchmark.foundation.chronos_model import ChronosModel

        rng = np.random.default_rng(42)
        t = np.arange(200, dtype=float)
        train = 100.0 + 10.0 * np.sin(2 * np.pi * t / 24)

        model = ChronosModel()
        result = model.forecast(train, 24)

        assert len(result.predictions) == 24
        # seasonal naive should roughly repeat the last 24 values
        assert np.corrcoef(result.predictions, train[-24:])[0, 1] > 0.9


# -- Evaluator --

class TestEvaluator:
    def test_unknown_method_raises(self):
        with pytest.raises(EvaluationError):
            Evaluator(methods=["nonexistent"])

    def test_run_xgboost_only(self):
        evaluator = Evaluator(methods=["xgboost"])
        report = evaluator.run("synthetic_trend", horizon=12, data_length=500)

        assert len(report.results) == 1
        assert report.results[0].model_name == "XGBoost"
        assert report.results[0].mae > 0

    def test_report_to_dict(self):
        evaluator = Evaluator(methods=["xgboost"])
        report = evaluator.run("synthetic_trend", horizon=12, data_length=500)
        d = report.to_dict()
        assert "dataset" in d
        assert "results" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
