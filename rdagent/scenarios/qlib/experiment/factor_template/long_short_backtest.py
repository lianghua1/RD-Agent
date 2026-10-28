"""Three-mode portfolio simulator used after the Qlib training run.

Qlib's own backtester is long-only (``TopkDropoutStrategy`` / ``Position``), so
the long-short and short-only portfolios are simulated here from the recorder's
``pred.pkl`` and ``label.pkl``.  ``label.pkl`` is the raw forward return, which
makes the timing easy to reason about: a target chosen at bar ``t`` earns
``label[t]``; there is no extra shift and therefore no look-ahead.

Modes
-----
long_only
    TopK long leg.  Kept for regression tests and as a reference.
short_only
    BottomK short leg, no long exposure.
long_short
    TopK long + BottomK short, with optional net exposure.  This is the
    first-class mode; ``net_exposure=0`` is market-neutral.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

MODES = ("long_only", "short_only", "long_short")


def _env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or str(value).strip() == "" else str(value).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_text(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_text(name, str(default)))
    except ValueError:
        return default


def config_from_env(n_stocks: int) -> dict:
    mode = _env_text("FACTOR_POSITION_MODE", "long_only").lower().replace("-", "_")
    if mode not in MODES:
        mode = "long_only"
    top_k = max(0, _env_int("FACTOR_TOP_K", 50))
    bottom_k = max(0, _env_int("FACTOR_BOTTOM_K", top_k))
    risk = min(1.0, max(0.0, _env_float("FACTOR_RISK_DEGREE", 1.0)))
    net = min(risk, max(-risk, _env_float("FACTOR_NET_EXPOSURE", 0.0)))
    return {
        "mode": mode,
        "top_k": top_k,
        "bottom_k": bottom_k,
        "n_drop": max(1, _env_int("FACTOR_N_DROP", 5)),
        "risk": risk,
        "net": net,
        "short_borrow_rate": max(0.0, _env_float("FACTOR_SHORT_BORROW_RATE", 0.0)),
        "long_threshold": max(0.0, _env_float("FACTOR_LONG_THRESHOLD", 0.0)),
        "short_threshold": max(0.0, _env_float("FACTOR_SHORT_THRESHOLD", 0.0)),
        "hold_thresh": max(1, _env_int("FACTOR_HOLD_THRESH", 1)),
        "rebalance_days": max(1, _env_int("FACTOR_REBALANCE_DAYS", 1)),
        "ann_scaler": max(1.0, _env_float("FACTOR_ANN_SCALER", 365.0)),
        "cost_rate": max(0.0, _env_float("FACTOR_COST_RATE", 0.0005)),
        "ic_mode": _env_text("FACTOR_IC_MODE", "cross" if n_stocks >= 2 else "time"),
    }


def _dropout_choose(order: np.ndarray, current: np.ndarray, side: int, k: int, n_drop: int) -> list[int]:
    """Keep most of the current leg and replace up to ``n_drop`` names."""

    if k <= 0 or order.size == 0:
        return []
    k = int(min(k, order.size))
    drop = int(min(max(1, n_drop), k))
    if side > 0:
        held = [int(i) for i in order if current[i] > 0]
    else:
        held = [int(i) for i in order if current[i] < 0]
    keep = held[: max(0, k - drop)]
    keep_set = set(keep)
    fill = [int(i) for i in order if int(i) not in keep_set][: k - len(keep)]
    return keep + fill


def _exposures(cfg: dict, n_stocks: int) -> tuple[float, float]:
    mode = cfg["mode"]
    gross = cfg["risk"]
    if mode == "long_only":
        return gross, 0.0
    if mode == "short_only":
        return 0.0, gross
    net = cfg["net"] if n_stocks >= 2 else 0.0
    return (gross + net) / 2.0, (gross - net) / 2.0


def _target_weights(
    alpha: np.ndarray,
    current: np.ndarray,
    cfg: dict,
    long_held_days: np.ndarray,
    short_held_days: np.ndarray,
) -> np.ndarray:
    """Cross-sectional target weights for all three modes."""

    mode = cfg["mode"]
    n_stocks = alpha.size
    long_exposure, short_exposure = _exposures(cfg, n_stocks)
    long_k = cfg["top_k"] if mode in ("long_only", "long_short") else 0
    short_k = cfg["bottom_k"] if mode in ("short_only", "long_short") else 0
    if n_stocks >= 2 and mode == "long_short":
        long_k = min(long_k, n_stocks - 1)
        short_k = min(short_k, n_stocks - long_k)
    long_k = min(long_k, n_stocks)
    short_k = min(short_k, n_stocks)

    weights = np.zeros(n_stocks, dtype=float)
    valid = np.flatnonzero(np.isfinite(alpha))
    if valid.size == 0:
        if cfg["hold_thresh"] > 1:
            keep_long = [int(i) for i in np.flatnonzero((current > 0) & (long_held_days < cfg["hold_thresh"]))][
                :long_k
            ]
            keep_short = [int(i) for i in np.flatnonzero((current < 0) & (short_held_days < cfg["hold_thresh"]))][
                :short_k
            ]
            if keep_long and long_exposure > 0:
                weights[keep_long] = long_exposure / len(keep_long)
            if keep_short and short_exposure > 0:
                weights[keep_short] = -short_exposure / len(keep_short)
        return weights

    long_order = valid[np.argsort(-alpha[valid], kind="stable")]
    short_order = valid[np.argsort(alpha[valid], kind="stable")]

    if cfg["hold_thresh"] > 1:
        long_protected = [
            int(i) for i in long_order if current[i] > 0 and long_held_days[i] < cfg["hold_thresh"]
        ]
        short_protected = [
            int(i) for i in short_order if current[i] < 0 and short_held_days[i] < cfg["hold_thresh"]
        ]
        long_protected += [
            int(i)
            for i in np.flatnonzero((current > 0) & (long_held_days < cfg["hold_thresh"]))
            if int(i) not in long_protected
        ]
        short_protected += [
            int(i)
            for i in np.flatnonzero((current < 0) & (short_held_days < cfg["hold_thresh"]))
            if int(i) not in short_protected
        ]
        long_keep = long_protected[:long_k]
        short_keep = short_protected[:short_k]
        long_keep_set = set(long_keep)
        short_keep_set = set(short_keep)
        long_protected_set = set(long_protected)
        short_protected_set = set(short_protected)
        long_fill = [
            int(i)
            for i in long_order
            if int(i) not in long_keep_set and int(i) not in short_protected_set
        ][: max(0, long_k - len(long_keep))]
        long_chosen = long_keep + long_fill
        long_set = set(long_chosen)
        short_fill = [
            int(i)
            for i in short_order
            if int(i) not in short_keep_set
            and int(i) not in long_protected_set
            and int(i) not in long_set
        ][: max(0, short_k - len(short_keep))]
        short_chosen = short_keep + short_fill
    else:
        long_chosen = _dropout_choose(long_order, current, +1, long_k, cfg["n_drop"])
        long_set = set(long_chosen)
        short_candidates = np.asarray([int(i) for i in short_order if int(i) not in long_set], dtype=int)
        short_chosen = _dropout_choose(short_candidates, current, -1, short_k, cfg["n_drop"])

    if long_chosen and long_exposure > 0:
        weights[long_chosen] = long_exposure / len(long_chosen)
    if short_chosen and short_exposure > 0:
        weights[short_chosen] = -short_exposure / len(short_chosen)
    return weights


def _time_target(
    signal: float,
    current: float,
    cfg: dict,
    long_held_days: int,
    short_held_days: int,
) -> float:
    if cfg["mode"] == "long_only":
        long_exposure, short_exposure = cfg["risk"], 0.0
    elif cfg["mode"] == "short_only":
        long_exposure, short_exposure = 0.0, cfg["risk"]
    else:
        long_exposure, short_exposure = cfg["risk"], cfg["risk"]
    if signal > cfg["long_threshold"]:
        target = long_exposure
    elif signal < -cfg["short_threshold"]:
        target = -short_exposure
    else:
        target = 0.0
    if cfg["hold_thresh"] > 1:
        if current > 0 and long_held_days < cfg["hold_thresh"] and target <= 0:
            target = current
        elif current < 0 and short_held_days < cfg["hold_thresh"] and target >= 0:
            target = current
    return target


def _risk_analysis(returns: np.ndarray, periods_per_year: float, risk_free_rate: float = 0.0) -> dict:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"annualized_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    excess = values - risk_free_rate / periods_per_year
    mean = float(excess.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    cumulative = float(np.prod(1.0 + values) - 1.0)
    annual = (1.0 + cumulative) ** (periods_per_year / values.size) - 1.0 if values.size > 0 else 0.0
    if cumulative <= -1.0:
        annual = -1.0
    sharpe = mean / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    equity = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(equity)
    drawdown = float((equity / np.maximum(peak, 1e-12) - 1.0).min())
    return {
        "annualized_return": float(annual),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown),
    }


def simulate(
    alpha: np.ndarray,
    forward_returns: np.ndarray,
    dates,
    cfg: dict | None = None,
    funding_rates=None,
) -> dict:
    """Run the daily portfolio simulation.

    ``alpha`` and ``forward_returns`` are ``(n_days, n_instruments)`` matrices.
    ``forward_returns[t]`` is the return earned by a position opened at ``t``.
    """

    alpha = np.asarray(alpha, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    if alpha.shape != forward_returns.shape:
        raise ValueError("alpha and forward_returns must have the same shape")
    n_days, n_stocks = alpha.shape
    cfg = cfg or config_from_env(n_stocks)
    mode = cfg["mode"]
    ic_mode = str(cfg.get("ic_mode") or ("cross" if n_stocks >= 2 else "time"))
    funding_arr = None
    if funding_rates is not None:
        try:
            candidate = np.asarray(funding_rates, dtype=float)
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate.shape == forward_returns.shape:
            funding_arr = candidate
    funding_source = "history" if funding_arr is not None else ("borrow" if cfg["short_borrow_rate"] > 0 else "none")

    weights = np.zeros(n_stocks, dtype=float)
    held_days = np.zeros(n_stocks, dtype=np.int32)
    long_held_days = np.zeros(n_stocks, dtype=np.int32)
    short_held_days = np.zeros(n_stocks, dtype=np.int32)
    strategy_returns = np.zeros(n_days, dtype=float)
    long_contrib = np.zeros(n_days, dtype=float)
    short_contrib = np.zeros(n_days, dtype=float)
    turnover_costs = np.zeros(n_days, dtype=float)
    borrow_costs = np.zeros(n_days, dtype=float)
    funding_pnls = np.zeros(n_days, dtype=float)
    long_weight_sum = 0.0
    short_weight_sum = 0.0
    exposure_bars = 0

    rebalance = np.zeros(n_days, dtype=bool)
    if n_days:
        rebalance[0] = True
        period = max(1, int(cfg["rebalance_days"]))
        for t in range(1, n_days):
            rebalance[t] = t % period == 0

    for t in range(n_days):
        if rebalance[t]:
            if n_stocks >= 2 and ic_mode == "cross":
                target = _target_weights(alpha[t], weights, cfg, long_held_days, short_held_days)
            else:
                signal = float(alpha[t, 0]) if np.isfinite(alpha[t, 0]) else 0.0
                target = np.zeros(n_stocks, dtype=float)
                target[0] = _time_target(
                    signal,
                    float(weights[0]),
                    cfg,
                    int(long_held_days[0]),
                    int(short_held_days[0]),
                )
            cost = cfg["cost_rate"] * float(np.abs(target - weights).sum())
            turnover_costs[t] = cost
            strategy_returns[t] -= cost
            if mode in ("long_only", "long_short", "short_only"):
                next_long = np.zeros_like(long_held_days)
                next_short = np.zeros_like(short_held_days)
                for index in range(n_stocks):
                    if target[index] > 0:
                        next_long[index] = long_held_days[index] + 1 if weights[index] > 0 else 1
                    elif target[index] < 0:
                        next_short[index] = short_held_days[index] + 1 if weights[index] < 0 else 1
                long_held_days = next_long
                short_held_days = next_short
                next_held = np.zeros_like(held_days)
                for index in range(n_stocks):
                    if target[index] > 0:
                        next_held[index] = held_days[index] + 1 if weights[index] > 0 else 1
                held_days = next_held
            weights = target

        row = forward_returns[t]
        if not np.isfinite(row).any():
            continue
        realized = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        long_weights = np.maximum(weights, 0.0)
        short_weights = np.minimum(weights, 0.0)
        long_piece = float(np.dot(long_weights, realized))
        short_piece = float(np.dot(short_weights, realized))
        funding_pnl = 0.0
        if funding_arr is not None:
            funding_row = np.nan_to_num(funding_arr[t], nan=0.0, posinf=0.0, neginf=0.0)
            long_funding = -float(np.dot(funding_row, long_weights))
            short_funding = -float(np.dot(funding_row, short_weights))
            funding_pnl = long_funding + short_funding
            funding_pnls[t] = funding_pnl
            long_contrib[t] = long_piece + long_funding
            short_contrib[t] = short_piece + short_funding
            strategy_returns[t] += long_piece + short_piece + funding_pnl
        else:
            borrow = abs(float(short_weights.sum())) * cfg["short_borrow_rate"] / cfg["ann_scaler"]
            borrow_costs[t] = borrow
            long_contrib[t] = long_piece
            short_contrib[t] = short_piece - borrow
            strategy_returns[t] += long_piece + short_piece - borrow
        long_weight_sum += float(np.abs(long_weights).sum())
        short_weight_sum += float(np.abs(short_weights).sum())
        exposure_bars += 1

    if mode == "short_only":
        benchmark_returns = np.zeros(n_days, dtype=float)
    else:
        with np.errstate(invalid="ignore"):
            benchmark_returns = np.nanmean(forward_returns, axis=1)
        benchmark_returns = np.nan_to_num(benchmark_returns, nan=0.0, posinf=0.0, neginf=0.0)

    strategy_equity = np.cumprod(1.0 + strategy_returns)
    benchmark_equity = np.cumprod(1.0 + benchmark_returns)
    return {
        "dates": pd.DatetimeIndex(dates),
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "long_contrib": long_contrib,
        "short_contrib": short_contrib,
        "turnover_costs": turnover_costs,
        "borrow_costs": borrow_costs,
        "turnover_cost_mean": float(turnover_costs.mean()) if n_days else 0.0,
        "short_carry_cost_sum": float(borrow_costs.sum()),
        "avg_long_weight": float(long_weight_sum / max(1, exposure_bars)),
        "avg_short_weight": float(short_weight_sum / max(1, exposure_bars)),
        "funding_pnl_sum": float(funding_pnls.sum()),
        "funding_source": funding_source,
    }


def _extract_series(value) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            raise ValueError("empty recorder frame")
        return value.iloc[:, 0]
    return pd.Series(value)


def load_funding_from_qlib(recorder, freq: str = "day"):
    """Load the funding-rate sum for the recorder's test universe.

    ``freq`` is the Qlib frequency (``day`` / ``240min``).  Returns ``None``
    when the dataset has no ``$funding_rate_sum`` field or Qlib cannot serve
    it; the simulator then falls back to the configured borrow rate instead of
    fabricating a funding series.
    """

    from qlib.data import D

    prediction = recorder.load_object("pred.pkl")
    if prediction is None:
        return None
    index = prediction.index
    if not isinstance(index, pd.MultiIndex):
        return None
    if index.names[0] == "instrument":
        index = index.swaplevel(0, 1)
    dates = index.get_level_values(0)
    instruments = index.get_level_values(1).unique().tolist()
    if not instruments:
        return None
    frame = D.features(
        instruments,
        ["$funding_rate_sum"],
        start_time=pd.Timestamp(dates.min()),
        end_time=pd.Timestamp(dates.max()),
        freq=freq,
    )
    if frame is None or frame.empty or frame.dropna(how="all").empty:
        return None
    return frame


def _align_funding(funding, index, columns):
    if funding is None:
        return None
    if isinstance(funding, pd.DataFrame):
        if funding.empty:
            return None
        if isinstance(funding.index, pd.MultiIndex):
            series = funding.iloc[:, 0]
            if series.index.names[0] != "datetime":
                series = series.swaplevel(0, 1)
            matrix = series.unstack("instrument")
        else:
            matrix = funding
    else:
        series = pd.Series(funding)
        if not isinstance(series.index, pd.MultiIndex):
            return None
        if series.index.names[0] != "datetime":
            series = series.swaplevel(0, 1)
        matrix = series.unstack("instrument")
    matrix = matrix.reindex(index=index, columns=columns)
    return matrix.fillna(0.0)


def load_pred_label(recorder) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a recorder's prediction and raw forward label as date x instrument matrices."""

    prediction = _extract_series(recorder.load_object("pred.pkl"))
    label = _extract_series(recorder.load_object("label.pkl"))
    frame = pd.concat(
        [prediction.rename("pred"), label.rename("label")],
        axis=1,
        join="inner",
    ).dropna()
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("pred.pkl/label.pkl must have a (datetime, instrument) MultiIndex")
    names = list(frame.index.names)
    if names[-1] != "instrument" and names[0] == "instrument":
        frame = frame.swaplevel(0, 1)
    frame = frame.sort_index()
    if frame.index.names[-1] == "instrument":
        alpha = frame["pred"].unstack("instrument")
        forward = frame["label"].unstack("instrument")
    else:
        alpha = frame["pred"].unstack(level=-1)
        forward = frame["label"].unstack(level=-1)
    alpha, forward = alpha.align(forward, join="inner")
    return alpha, forward


def run_from_recorder(recorder, cfg: dict | None = None, funding=None) -> tuple[dict, pd.DataFrame]:
    """Compute custom metrics and the ``ret.parquet`` frame for the recorder."""

    alpha, forward = load_pred_label(recorder)
    cfg = cfg or config_from_env(alpha.shape[1])
    funding_matrix = _align_funding(funding, alpha.index, alpha.columns)
    sim = simulate(
        alpha.values,
        forward.values,
        alpha.index,
        cfg,
        funding_rates=funding_matrix.to_numpy() if funding_matrix is not None else None,
    )
    periods_per_year = float(cfg["ann_scaler"])
    strategy = _risk_analysis(sim["strategy_returns"], periods_per_year)
    long_leg = _risk_analysis(sim["long_contrib"], periods_per_year)
    short_leg = _risk_analysis(sim["short_contrib"], periods_per_year)
    mode = cfg["mode"]

    metrics: dict[str, float] = {
        "strategy.annualized_return": strategy["annualized_return"],
        "strategy.sharpe": strategy["sharpe"],
        "strategy.max_drawdown": strategy["max_drawdown"],
        "long_leg.annualized_return": long_leg["annualized_return"],
        "long_leg.sharpe": long_leg["sharpe"],
        "short_leg.annualized_return": short_leg["annualized_return"],
        "short_leg.sharpe": short_leg["sharpe"],
        "turnover_cost_mean": sim["turnover_cost_mean"],
        "short_carry_cost_sum": sim["short_carry_cost_sum"],
        "funding_pnl_sum": sim["funding_pnl_sum"],
        "avg_long_weight": sim["avg_long_weight"],
        "avg_short_weight": sim["avg_short_weight"],
    }
    if mode == "long_short":
        metrics["long_short.annualized_return"] = strategy["annualized_return"]
        metrics["long_short.sharpe"] = strategy["sharpe"]
        metrics["long_short.max_drawdown"] = strategy["max_drawdown"]
    elif mode == "short_only":
        metrics["short_only.annualized_return"] = strategy["annualized_return"]
        metrics["short_only.sharpe"] = strategy["sharpe"]
        metrics["short_only.max_drawdown"] = strategy["max_drawdown"]

    report = pd.DataFrame(
        {
            "account": sim["strategy_equity"],
            "bench": sim["benchmark_returns"],
            "long": np.cumprod(1.0 + sim["long_contrib"]),
            "short": np.cumprod(1.0 + sim["short_contrib"]),
        },
        index=sim["dates"],
    )
    return metrics, report
