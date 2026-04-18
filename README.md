# Time-Series Foundation Model Fine-Tuning Benchmark

A benchmarking harness that compares classical forecasting methods against zero-shot and fine-tuned time-series foundation models. Evaluates point accuracy, probabilistic calibration, computational cost, and few-shot learning curves across multiple datasets.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## The Question

Time-series foundation models (Chronos, TimesFM, Moirai) promise zero-shot forecasting across domains. But when should you fine-tune them on your data? When is classical ARIMA or XGBoost still better? How much domain data do you need before fine-tuning pays off?

This benchmark answers those questions empirically.

## What It Evaluates

- **Classical baselines**: ARIMA, Prophet, LSTM, XGBoost
- **Foundation models**: Chronos (Amazon), with an extensible interface for others
- **Three modes per foundation model**: zero-shot, few-shot (10/50/100 samples), full fine-tuning
- **Four metric categories**: point accuracy (MAE, RMSE, MAPE), probabilistic calibration (CRPS), computational cost (train/inference time, memory), few-shot learning curves

## Quick Start

```bash
git clone https://github.com/yourusername/ts-foundation-benchmark.git
cd ts-foundation-benchmark
pip install -e ".[dev]"

# run benchmark on a built-in dataset
python -m ts_benchmark run --dataset electricity --horizon 24

# compare specific methods
python -m ts_benchmark run --dataset etth1 --methods arima,xgboost,chronos

# generate report
python -m ts_benchmark report --results output/results.json
```

## Project Structure

```
├── ts_benchmark/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── exceptions.py
│   ├── models.py
│   ├── datasets.py           # Dataset loaders
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── arima.py
│   │   ├── lstm.py
│   │   └── xgboost_model.py
│   ├── foundation/
│   │   ├── __init__.py
│   │   └── chronos_model.py
│   ├── evaluator.py           # Metrics and evaluation harness
│   └── reporter.py            # Report generation
├── tests/
├── pyproject.toml
├── Dockerfile
└── .github/workflows/ci.yml
```

## License

MIT
