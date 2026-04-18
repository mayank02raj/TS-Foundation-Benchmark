# ts_benchmark/baselines/lstm.py

"""LSTM baseline using PyTorch.

Simple single-layer LSTM with sliding window input. Not state-of-the-art
but a fair representative of how most teams use LSTMs for time series
in practice: minimal architecture, train for a fixed number of epochs,
hope for the best.
"""

from __future__ import annotations

import logging
import time
import tracemalloc

import numpy as np

from ts_benchmark.exceptions import ModelError
from ts_benchmark.models import ForecastResult

logger = logging.getLogger(__name__)


class LSTMBaseline:
    """Single-layer LSTM forecaster."""

    name = "LSTM"

    def __init__(
        self,
        lookback: int = 48,
        hidden_size: int = 64,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
    ) -> None:
        self._lookback = lookback
        self._hidden_size = hidden_size
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size

    def forecast(self, train: np.ndarray, horizon: int) -> ForecastResult:
        """Train LSTM and produce multi-step forecast."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ModelError("PyTorch required for LSTM. pip install torch")

        tracemalloc.start()
        train_start = time.perf_counter()

        # normalize
        mean = float(np.mean(train))
        std = float(np.std(train)) + 1e-8
        normalized = (train - mean) / std

        # create sliding window sequences
        X, y = self._create_sequences(normalized)
        if len(X) < self._batch_size:
            raise ModelError(f"Not enough data for LSTM: {len(X)} sequences (need >= {self._batch_size})")

        X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (N, lookback, 1)
        y_tensor = torch.FloatTensor(y)

        # train/val split (last 20% for validation)
        val_size = max(int(len(X_tensor) * 0.2), 1)
        train_size = len(X_tensor) - val_size

        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]

        # model
        model = _LSTMModel(input_size=1, hidden_size=self._hidden_size, output_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        # training with early stopping
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = 10  # stop after 10 epochs without improvement

        for epoch in range(self._epochs):
            # training
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).squeeze(-1)
                val_loss = float(loss_fn(val_pred, y_val))

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
                    break

        # restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        train_time = time.perf_counter() - train_start

        # multi-step forecast (autoregressive)
        infer_start = time.perf_counter()
        model.eval()
        predictions: list[float] = []
        current_input = torch.FloatTensor(normalized[-self._lookback:]).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            for _ in range(horizon):
                pred = model(current_input)
                pred_val = float(pred[0, -1, 0])
                predictions.append(pred_val)
                # shift window
                new_step = torch.FloatTensor([[[pred_val]]])
                current_input = torch.cat([current_input[:, 1:, :], new_step], dim=1)

        infer_time = time.perf_counter() - infer_start

        # denormalize
        predictions_arr = np.array(predictions) * std + mean

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logger.info(f"LSTM trained {self._epochs} epochs in {train_time:.2f}s")

        return ForecastResult(
            model_name=self.name,
            predictions=predictions_arr,
            actuals=np.array([]),
            train_time_seconds=train_time,
            inference_time_seconds=infer_time,
            peak_memory_mb=peak_mem / (1024 * 1024),
        )

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for supervised learning."""
        X, y = [], []
        for i in range(self._lookback, len(data)):
            X.append(data[i - self._lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)


class _LSTMModel:
    """Minimal LSTM model. Implemented as a class wrapping nn.Module
    to keep the import scoped."""

    def __new__(cls, input_size: int, hidden_size: int, output_size: int):
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out)

        return LSTMNet()
