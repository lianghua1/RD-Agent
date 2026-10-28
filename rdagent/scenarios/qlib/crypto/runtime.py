"""Glue between the crypto data settings and the Qlib runners.

Keeps the runner edits to a single ``env_to_use.update(...)`` call so the factor
and model runners cannot drift apart, and so nothing changes when
``CRYPTO_MODE=false``.
"""

from __future__ import annotations

from rdagent.scenarios.qlib.crypto.conf import CRYPTO_SETTINGS


def crypto_qlib_pythonpath(current: str = "./") -> str:
    """Prepend the intraday qlib compatibility patch to ``PYTHONPATH``.

    ``sitecustomize.py`` in the patch directory is imported automatically by the
    qrun subprocess and makes an unavailable benchmark non-fatal instead of
    falling back to a 1min read that a 240min provider cannot serve.
    """
    from pathlib import Path

    patch_dir = Path(__file__).resolve().parent / "qlib_patch"
    return f"{current}:{patch_dir}" if current else str(patch_dir)


def crypto_runner_env() -> dict[str, str]:
    """Jinja2 variables injected into the Qlib backtest configs for crypto."""

    return CRYPTO_SETTINGS.runner_env()


def crypto_prompt_args() -> dict[str, str]:
    """Jinja2 variables for the scenario ``experiment_setting`` table.

    Mirrors the A-share ``{train,valid,test}_*`` variables, plus labels that make
    the prompt describe a crypto universe instead of CSI300.
    """

    return {
        "market_name": "Crypto (multi-coin)",
        "factor_library": "CryptoAlpha (OHLCV)",
        **CRYPTO_SETTINGS.resolve_segments(),
    }


__all__ = ["crypto_runner_env", "crypto_prompt_args"]
