# ts_benchmark/reporter.py

"""Report generation for benchmark results."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from ts_benchmark.models import BenchmarkReport

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, output_dir: str = "output") -> None:
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json(self, report: BenchmarkReport, filename: str | None = None) -> str:
        if not filename:
            filename = f"benchmark_{report.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self._output_dir, filename)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")
        return path

    def print_table(self, report: BenchmarkReport) -> None:
        """Print a formatted comparison table to stdout."""
        print(f"\n  Benchmark: {report.dataset_name} (horizon={report.horizon})")
        print("  " + "=" * 85)
        print(f"  {'Model':<18} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Train(s)':>10} {'Infer(s)':>10} {'Mem(MB)':>9}")
        print("  " + "-" * 85)

        sorted_results = sorted(report.results, key=lambda r: r.mae)
        best_mae = sorted_results[0].mae if sorted_results else 0

        for r in sorted_results:
            marker = " *" if r.mae == best_mae else "  "
            print(
                f"{marker}{r.model_name:<16} {r.mae:>8.4f} {r.rmse:>8.4f} "
                f"{r.mape:>8.2f} {r.train_time:>10.2f} {r.inference_time:>10.4f} "
                f"{r.memory_mb:>9.1f}"
            )

        print("  " + "=" * 85)
        print(f"  Best model (by MAE): {report.best_model('mae')}")

        if report.few_shot_curves:
            print("\n  Few-shot learning curves (MAE by context length):")
            for model_name, curve in report.few_shot_curves.items():
                print(f"\n  {model_name}:")
                for ctx_len, mae in curve:
                    bar = "#" * int(mae * 2)
                    print(f"    ctx={ctx_len:>4}: MAE={mae:.4f} {bar}")

        print()
