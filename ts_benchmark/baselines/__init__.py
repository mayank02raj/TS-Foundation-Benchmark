# ts_benchmark/baselines/__init__.py

"""Classical forecasting baselines."""

from ts_benchmark.baselines.arima import ARIMABaseline
from ts_benchmark.baselines.lstm import LSTMBaseline
from ts_benchmark.baselines.xgboost_model import XGBoostBaseline

__all__ = ["ARIMABaseline", "LSTMBaseline", "XGBoostBaseline"]
