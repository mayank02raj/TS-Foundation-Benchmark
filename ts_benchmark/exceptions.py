# ts_benchmark/exceptions.py

from __future__ import annotations

class BenchmarkError(Exception):
    """Base exception."""

class DatasetError(BenchmarkError):
    """Dataset loading or validation failed."""

class ModelError(BenchmarkError):
    """Model training or prediction failed."""

class EvaluationError(BenchmarkError):
    """Metric computation failed."""
