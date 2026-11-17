"""Position-mode and portfolio-config plumbing shared by the Qlib runners.

The Qlib factor loop was historically long-only: the templates hard-code
``TopkDropoutStrategy`` and the feedback only looks at long-leg metrics.  This
module keeps the new three-mode switch (``long_only`` / ``short_only`` /
``long_short``) in one place so the factor runner and the workspace backtest
script cannot drift apart.
"""

from __future__ import annotations

import os

VALID_POSITION_MODES = ("long_only", "short_only", "long_short")


def _text(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or str(value).strip() == "" else str(value).strip()


def _number(name: str, default: float) -> float:
    try:
        return float(_text(name, str(default)))
    except ValueError:
        return default


def position_mode() -> str:
    """Return the validated position mode (default ``long_only``)."""

    mode = _text("FACTOR_POSITION_MODE", "long_only").lower().replace("-", "_")
    return mode if mode in VALID_POSITION_MODES else "long_only"


def factor_backtest_env(current: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment consumed by ``long_short_backtest.py``.

    ``current`` is the env already assembled for the Qlib run (topk/n_drop and
    crypto costs).  Explicit ``FACTOR_*`` variables win; otherwise the existing
    Qlib values are reused so the internal simulator and Qlib agree.
    """

    current = current or {}
    mode = position_mode()

    def pick(name: str, fallback: str) -> str:
        value = current.get(name)
        if value is None or str(value).strip() == "":
            return fallback
        return str(value)

    top_k = _text("FACTOR_TOP_K", pick("topk", "50"))
    bottom_k = _text("FACTOR_BOTTOM_K", top_k if mode in ("short_only", "long_short") else "0")
    n_drop = _text("FACTOR_N_DROP", pick("n_drop", "5"))
    risk = _text("FACTOR_RISK_DEGREE", "1.0")
    net = _text("FACTOR_NET_EXPOSURE", "0.0")
    borrow = _text("FACTOR_SHORT_BORROW_RATE", pick("short_borrow_rate", "0.0"))
    long_threshold = _text("FACTOR_LONG_THRESHOLD", "0.0")
    short_threshold = _text("FACTOR_SHORT_THRESHOLD", "0.0")
    hold = _text("FACTOR_HOLD_THRESH", "1")
    rebalance = _text("FACTOR_REBALANCE_DAYS", pick("FACTOR_REBALANCE_DAYS", "1"))
    crypto_mode = os.environ.get("CRYPTO_MODE", "").strip().lower() == "true"
    if crypto_mode and os.environ.get("CRYPTO_FREQ", "1d").strip().lower() == "4h":
        default_ann = "2190"
    else:
        default_ann = "365" if crypto_mode else "252"
    ann_scaler = _text("FACTOR_ANN_SCALER", pick("ann_scaler", default_ann))
    # A one-way fee applied to each unit of turnover.  Crypto uses the same
    # taker fee on both sides; A-share templates have asymmetric costs, so the
    # conservative (sell-side) value is the fallback.
    fallback_cost = pick("close_cost", "0.0005")
    cost = _text("FACTOR_COST_RATE", fallback_cost)

    return {
        "FACTOR_POSITION_MODE": mode,
        "FACTOR_TOP_K": top_k,
        "FACTOR_BOTTOM_K": bottom_k,
        "FACTOR_N_DROP": n_drop,
        "FACTOR_RISK_DEGREE": risk,
        "FACTOR_NET_EXPOSURE": net,
        "FACTOR_SHORT_BORROW_RATE": borrow,
        "FACTOR_LONG_THRESHOLD": long_threshold,
        "FACTOR_SHORT_THRESHOLD": short_threshold,
        "FACTOR_HOLD_THRESH": hold,
        "FACTOR_REBALANCE_DAYS": rebalance,
        "FACTOR_ANN_SCALER": ann_scaler,
        "FACTOR_COST_RATE": cost,
    }
