"""Qlib intraday compatibility patch for the crypto fork.

Qlib's ``PortfolioMetrics._cal_benchmark`` asks
``get_higher_eq_freq_feature(..., freq=<backtest freq>)``.  On a 240min/60min
provider the first read can raise, and Qlib then retries with ``1min``, which a
240min provider cannot resample.  The whole baseline backtest aborts even though
the benchmark is optional for our custom long/short simulator.

This module is imported automatically by Python when this directory is on
``PYTHONPATH`` and makes an unavailable benchmark non-fatal.  It is only put on
``PYTHONPATH`` for crypto intraday qrun processes (see factor_runner.py).
"""

from __future__ import annotations

import sys

try:
    from qlib.backtest.report import PortfolioMetrics

    _original_cal_benchmark = PortfolioMetrics._cal_benchmark

    @staticmethod
    def _cal_benchmark_without_1min_fallback(benchmark_config, freq):
        try:
            return _original_cal_benchmark(benchmark_config, freq)
        except (ValueError, KeyError) as error:
            print(
                f"[crypto-qlib-patch] benchmark unavailable at freq={freq}: {error}",
                file=sys.stderr,
                flush=True,
            )
            return None

    PortfolioMetrics._cal_benchmark = _cal_benchmark_without_1min_fallback
except Exception as error:  # pragma: no cover - fail open, never break qrun
    print(f"[crypto-qlib-patch] patch not applied: {error}", file=sys.stderr, flush=True)
