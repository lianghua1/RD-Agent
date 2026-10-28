"""Validate a generated crypto dataset before wiring the R&D loop to it.

Runs against Qlib only (no LLM, no Docker), so it is the cheapest way to catch
data-layer mistakes: calendar alignment, instrument coverage, cross-sectional
shape, and whether factors actually have dispersion to correlate against.

Usage::

    python -m rdagent.scenarios.qlib.crypto.smoke
    python -m rdagent.scenarios.qlib.crypto.smoke --provider-uri ~/.qlib/qlib_data/crypto_usdm_1d
"""

from __future__ import annotations

import fire
import pandas as pd

from rdagent.scenarios.qlib.crypto.conf import CryptoDataSettings


def main(
    provider_uri: str | None = None,
    start: str | None = None,
    end: str | None = None,
    min_instruments: int = 5,
) -> None:
    settings = CryptoDataSettings()
    provider = provider_uri or settings.provider_uri
    region = settings.runner_env()["region"]

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider, region=region)
    instruments = D.instruments(market="all")
    frame = D.features(instruments, ["$close"], freq="day", start_time=start, end_time=end)
    if frame is None or frame.empty:
        raise SystemExit(f"Qlib returned no data for provider_uri={provider}")

    close = frame["$close"].unstack(level="instrument").sort_index()

    # 1) structural checks
    non_null = close.notna()
    instruments_per_day = non_null.sum(axis=1)
    print("=== dataset ===")
    print(f"provider_uri        : {provider}")
    print(f"calendar days       : {len(close)}")
    print(f"instruments         : {close.shape[1]}")
    print(f"date range          : {close.index.min().date()} -> {close.index.max().date()}")
    print(f"instruments/day     : median={instruments_per_day.median():.0f} min={instruments_per_day.min():.0f}")

    # 2) cross-sectional sanity: 1-day reversal vs forward 1-day return
    factor = close.shift(1) / close - 1.0
    label = close.shift(-1) / close - 1.0
    usable = non_null.astype(int).sum(axis=1)
    # Spearman = Pearson on per-date ranks. Computed this way to stay compatible
    # with pandas versions that lack DataFrame.corrwith(method=...).
    ic = factor.rank(axis=1).corrwith(label.rank(axis=1), axis=1)
    ic = ic[usable >= min_instruments].dropna()

    print("\n=== cross-sectional Rank IC (1d reversal, demo) ===")
    if ic.empty:
        print("no date had enough instruments to compute an IC")
    else:
        print(f"days with IC        : {len(ic)}")
        print(f"mean Rank IC        : {ic.mean():.4f}")
        print(f"std  Rank IC        : {ic.std():.4f}")

    if instruments_per_day.median() < min_instruments:
        raise SystemExit(
            f"Median instruments per day ({instruments_per_day.median():.0f}) is below "
            f"min_instruments={min_instruments}. Cross-sectional factors need a wider universe."
        )
    print("\nOK: dataset loads and is cross-sectionally usable.")


if __name__ == "__main__":
    fire.Fire(main)
