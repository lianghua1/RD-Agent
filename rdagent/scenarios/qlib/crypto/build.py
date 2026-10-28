"""Build a crypto dataset for the Qlib scenarios.

Usage::

    # Full pipeline: download -> H5 snapshots -> Qlib binary dataset
    python -m rdagent.scenarios.qlib.crypto.build --market usdm --freq 1d \\
        --universe-size 50 --start 2022-01-01

    # Re-generate the Qlib binary files from an existing H5 snapshot
    python -m rdagent.scenarios.qlib.crypto.build --provider-only

Everything is also configurable through ``CRYPTO_*`` environment variables; CLI
flags win over environment values.
"""

from __future__ import annotations

import json
from typing import Any

import fire
import pandas as pd

from rdagent.scenarios.qlib.crypto.conf import CryptoDataSettings
from rdagent.scenarios.qlib.crypto.dataset import H5_KEY, build_h5
from rdagent.scenarios.qlib.crypto.qlib_bin import write_qlib_bin


def _settings(**overrides: Any) -> CryptoDataSettings:
    clean = {key: value for key, value in overrides.items() if value is not None}
    settings = CryptoDataSettings(**clean)
    if not settings.mode:
        # The build helper is an explicit crypto action, so enable it here even
        # if the caller forgot CRYPTO_MODE=true. The loop itself still respects
        # the environment switch.
        settings = CryptoDataSettings(mode=True, **clean)
    return settings


def build_provider(settings: CryptoDataSettings) -> dict[str, Any]:
    """Write the Qlib binary dataset from the canonical H5 snapshot."""

    all_h5 = settings.data_dir / "daily_pv_all.h5"
    if not all_h5.exists():
        raise FileNotFoundError(
            f"{all_h5} not found. Run the full build first (without --provider-only)."
        )
    panel = pd.read_hdf(all_h5, key=H5_KEY)
    return write_qlib_bin(panel, settings.provider_uri, freq=settings.qlib_freq)


def main(
    market: str | None = None,
    freq: str | None = None,
    exchange: str | None = None,
    quote: str | None = None,
    universe_mode: str | None = None,
    universe_size: int | None = None,
    symbols: str | None = None,
    start: str | None = None,
    end: str | None = None,
    dataset_name: str | None = None,
    include_derivatives: bool | None = None,
    derivatives_period: str | None = None,
    provider_only: bool = False,
) -> None:
    """Build a crypto dataset (download -> H5 -> Qlib binary)."""

    settings = _settings(
        market=market,
        freq=freq,
        exchange=exchange,
        quote=quote,
        universe_mode=universe_mode,
        universe_size=universe_size,
        symbols=symbols,
        start=start,
        end=end,
        dataset_name=dataset_name,
        include_derivatives=include_derivatives,
        derivatives_period=derivatives_period,
    )

    if settings.freq not in {"1d", "4h"}:
        raise SystemExit(
            f"Unsupported frequency {settings.freq!r}; use 1d or 4h. "
            "4h bars are stored as Qlib's 240min calendar."
        )

    if provider_only:
        summary: dict[str, Any] = {"h5": None, "provider": build_provider(settings)}
    else:
        h5_summary = build_h5(settings)
        provider_summary = build_provider(settings)
        summary = {"h5": h5_summary, "provider": provider_summary}

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    fire.Fire(main)
