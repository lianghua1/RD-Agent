"""Offline tests for the three-mode portfolio simulator.

The simulator is copied into every Qlib experiment workspace, so it must stay
dependency-light and must not import the RD-Agent package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "rdagent/scenarios/qlib/experiment/factor_template/long_short_backtest.py"
)
SPEC = importlib.util.spec_from_file_location("long_short_backtest", MODULE_PATH)
assert SPEC and SPEC.loader
LSB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LSB)


def _cfg(mode: str, **overrides) -> dict:
    cfg = {
        "mode": mode,
        "top_k": 1,
        "bottom_k": 1,
        "n_drop": 1,
        "risk": 1.0,
        "net": 0.0,
        "short_borrow_rate": 0.0,
        "long_threshold": 0.0,
        "short_threshold": 0.0,
        "hold_thresh": 1,
        "rebalance_days": 1,
        "ann_scaler": 365.0,
        "cost_rate": 0.0,
        "ic_mode": "cross",
    }
    cfg.update(overrides)
    return cfg


def _market(n_days: int = 40):
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    alpha = np.tile(np.array([[1.0, 0.0, -1.0]]), (n_days, 1))
    forward = np.tile(np.array([[0.01, 0.0, -0.01]]), (n_days, 1))
    return dates, alpha, forward


def test_long_short_holds_both_legs():
    dates, alpha, forward = _market()
    out = LSB.simulate(alpha, forward, dates, _cfg("long_short"))
    assert out["avg_long_weight"] > 0
    assert out["avg_short_weight"] > 0
    metrics, report = LSB.run_from_recorder(
        SimpleNamespace(load_object=lambda name: _recorder_object(name)),
        _cfg("long_short"),
    )
    assert metrics["long_short.annualized_return"] > 0
    assert metrics["long_leg.annualized_return"] > 0
    assert metrics["short_leg.annualized_return"] > 0
    assert {"account", "bench", "long", "short"}.issubset(report.columns)


def test_short_only_has_no_long_leg_and_cash_benchmark():
    dates, alpha, forward = _market()
    out = LSB.simulate(alpha, forward, dates, _cfg("short_only"))
    assert out["avg_long_weight"] == 0
    assert out["avg_short_weight"] > 0
    assert np.allclose(out["benchmark_returns"], 0.0)
    assert out["strategy_returns"][-1] > 0


def test_long_only_matches_top_leg():
    dates, alpha, forward = _market()
    out = LSB.simulate(alpha, forward, dates, _cfg("long_only"))
    assert out["avg_long_weight"] > 0
    assert out["avg_short_weight"] == 0
    assert out["strategy_returns"][-1] > 0


def test_short_borrow_reduces_short_only_return():
    dates, alpha, forward = _market()
    free = LSB.simulate(alpha, forward, dates, _cfg("short_only"))
    charged = LSB.simulate(alpha, forward, dates, _cfg("short_only", short_borrow_rate=0.5))
    assert charged["strategy_returns"].sum() < free["strategy_returns"].sum()
    assert charged["short_carry_cost_sum"] > 0


def test_funding_history_replaces_borrow_rate():
    dates, alpha, forward = _market()
    cfg = _cfg("short_only", short_borrow_rate=0.5)
    base = LSB.simulate(alpha, forward, dates, cfg)
    funding = np.full(forward.shape, 0.0001)
    charged = LSB.simulate(alpha, forward, dates, cfg, funding_rates=funding)
    assert base["funding_source"] == "borrow"
    assert charged["funding_source"] == "history"
    assert charged["funding_pnl_sum"] > 0
    assert charged["strategy_returns"].sum() > base["strategy_returns"].sum()


def test_time_mode_short_only_uses_negative_threshold():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    alpha = np.full((10, 1), -2.0)
    forward = np.full((10, 1), -0.01)
    out = LSB.simulate(alpha, forward, dates, _cfg("short_only", ic_mode="time"))
    assert out["strategy_returns"][-1] > 0
    assert out["avg_long_weight"] == 0
    assert np.allclose(out["benchmark_returns"], 0.0)


def test_long_short_legs_do_not_overlap():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    alpha = np.tile(np.array([[1.0, 0.5, -0.5, -1.0]]), (5, 1))
    forward = np.zeros_like(alpha)
    cfg = _cfg("long_short", top_k=2, bottom_k=2)
    out = LSB.simulate(alpha, forward, dates, cfg)
    assert out["avg_long_weight"] > 0
    assert out["avg_short_weight"] > 0


def _recorder_object(name: str):
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    instruments = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    if name == "pred.pkl":
        values = np.tile([1.0, 0.0, -1.0], 40)
        return pd.DataFrame({"score": values}, index=index)
    if name == "label.pkl":
        values = np.tile([0.01, 0.0, -0.01], 40)
        return pd.DataFrame({"LABEL0": values}, index=index)
    raise KeyError(name)
