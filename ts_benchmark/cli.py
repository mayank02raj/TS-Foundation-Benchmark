# ts_benchmark/cli.py

"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ts_benchmark import __version__
from ts_benchmark.datasets import AVAILABLE_DATASETS
from ts_benchmark.evaluator import MODEL_REGISTRY, Evaluator
from ts_benchmark.reporter import Reporter

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time-Series Foundation Model Benchmark")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the benchmark")
    run_parser.add_argument(
        "--dataset", "-d", default="synthetic_complex",
        help=f"Dataset name. Available: {', '.join(AVAILABLE_DATASETS)}",
    )
    run_parser.add_argument("--horizon", type=int, default=24)
    run_parser.add_argument("--length", type=int, default=2000, help="Data length for synthetic datasets")
    run_parser.add_argument(
        "--methods", "-m", default=None,
        help=f"Comma-separated methods. Available: {', '.join(MODEL_REGISTRY.keys())}",
    )
    run_parser.add_argument("--output-dir", "-o", default="output")
    run_parser.add_argument("--json", action="store_true", help="Output as JSON only")

    report_parser = sub.add_parser("report", help="Display results from a previous run")
    report_parser.add_argument("--results", "-r", required=True, help="Path to results JSON")

    return parser


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "report":
        return _cmd_report(args)
    else:
        build_parser().print_help()
        return 0


def _cmd_run(args) -> int:
    methods = args.methods.split(",") if args.methods else None

    evaluator = Evaluator(methods=methods)
    report = evaluator.run(
        dataset_name=args.dataset,
        horizon=args.horizon,
        data_length=args.length,
    )

    reporter = Reporter(output_dir=args.output_dir)
    json_path = reporter.save_json(report)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        reporter.print_table(report)
        print(f"  Results saved to: {json_path}\n")

    return 0


def _cmd_report(args) -> int:
    from ts_benchmark.models import BenchmarkReport, MetricResult

    with open(args.results) as f:
        data = json.load(f)

    report = BenchmarkReport(
        dataset_name=data["dataset"],
        horizon=data["horizon"],
        results=[MetricResult(**r) for r in data.get("results", [])],
        few_shot_curves=data.get("few_shot_curves", {}),
    )

    reporter = Reporter()
    reporter.print_table(report)
    return 0
