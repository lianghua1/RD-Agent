from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.crypto.conf import CryptoDataSettings
from rdagent.scenarios.qlib.crypto.exchange import ExchangeAdapter, build_adapter
from rdagent.scenarios.qlib.crypto.universe import resolve_universe

PV_FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
DERIVATIVE_COLUMNS = {
    "funding_rate": "$funding_rate",
    "funding_rate_last": "$funding_rate_last",
    "funding_rate_sum": "$funding_rate_sum",
    "open_interest": "$open_interest",
    "open_interest_value": "$open_interest_value",
    "global_account_ls_ratio": "$global_account_ls_ratio",
    "top_account_ls_ratio": "$top_account_ls_ratio",
    "top_position_ls_ratio": "$top_position_ls_ratio",
    "taker_buy_sell_ratio": "$taker_buy_sell_ratio",
    "taker_buy_vol": "$taker_buy_vol",
    "taker_sell_vol": "$taker_sell_vol",
}
DERIVATIVE_FIELDS = list(DERIVATIVE_COLUMNS.values())
H5_KEY = "data"


def fetch_frames(
    settings: CryptoDataSettings,
    adapter: ExchangeAdapter,
    symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download OHLCV for every symbol.

    Returns ``(frames, failures)``. The dropped symbols are surfaced to the
    caller instead of being silently discarded, because silent drops turn data
    gaps into survivorship bias.
    """

    frames: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    end = settings.resolved_end
    for index, symbol in enumerate(symbols, start=1):
        try:
            frame = adapter.klines(symbol, settings.start, end)
        except Exception as error:  # noqa: BLE001 - one bad symbol must not kill the build
            logger.warning(f"[{index}/{len(symbols)}] {symbol}: download failed ({error}); skipping")
            failures.append(symbol)
            continue
        if frame is None or frame.empty:
            logger.warning(f"[{index}/{len(symbols)}] {symbol}: no bars in range; skipping")
            failures.append(symbol)
            continue
        frames[symbol] = frame
        logger.info(
            f"[{index}/{len(symbols)}] {symbol}: {len(frame)} bars "
            f"({frame.index.min().date()} -> {frame.index.max().date()})"
        )
    if not frames:
        raise RuntimeError("No market data was downloaded; check the exchange, market and date range.")
    if failures:
        logger.warning(
            f"{len(failures)}/{len(symbols)} symbols were dropped during download: {failures}. "
            "Missing symbols bias cross-sectional statistics; re-run or use an explicit universe."
        )
    return frames, failures


def fetch_derivatives(
    settings: CryptoDataSettings,
    adapter: ExchangeAdapter,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Download optional daily derivatives fields for the requested symbols.

    Derivatives are best-effort: a symbol or endpoint failure is logged and the
    OHLCV panel still builds.  Binance's public futures-data endpoints only
    expose a short recent window for OI / long-short ratios, so old datasets may
    legitimately have no rows there; never substitute zero for missing data.
    """

    if not settings.include_derivatives:
        return {}
    if settings.market.lower() != "usdm":
        logger.warning("Derivative fields are only available for market=usdm; skipping.")
        return {}
    result: dict[str, pd.DataFrame] = {}
    end = settings.resolved_end
    for index, symbol in enumerate(symbols, start=1):
        try:
            frame = adapter.derivatives(symbol, settings.start, end)
        except Exception as error:  # noqa: BLE001 - one bad symbol must not kill the build
            logger.warning(f"[{index}/{len(symbols)}] {symbol}: derivatives unavailable ({error}); skipping")
            continue
        if frame is None or frame.empty:
            logger.warning(f"[{index}/{len(symbols)}] {symbol}: no derivative rows in range")
            continue
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        frame.index = frame.index.normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        result[symbol.upper()] = frame
        logger.info(
            f"[{index}/{len(symbols)}] {symbol}: {len(frame)} derivative days "
            f"({frame.index.min().date()} -> {frame.index.max().date()})"
        )
    if not result:
        logger.warning("No derivative data was downloaded; continuing with OHLCV only.")
    return result


def build_panel(
    frames: dict[str, pd.DataFrame],
    derivatives: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Stack per-symbol frames into a ``(datetime, instrument)`` MultiIndex panel."""

    derivatives = derivatives or {}
    parts: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        part = pd.DataFrame(
            {
                "$open": frame["open"].astype("float64"),
                "$close": frame["close"].astype("float64"),
                "$high": frame["high"].astype("float64"),
                "$low": frame["low"].astype("float64"),
                "$volume": frame["volume"].astype("float64"),
                # Crypto has no ex-rights/dividend adjustment; factor 1.0 is a no-op.
                "$factor": 1.0,
            }
        )
        extra = derivatives.get(symbol.upper())
        if extra is not None and not extra.empty:
            aligned = extra.reindex(frame.index).ffill()
            for column in aligned.columns:
                field = DERIVATIVE_COLUMNS.get(str(column))
                if field is not None:
                    part[field] = pd.to_numeric(aligned[column], errors="coerce")
        part.index = pd.MultiIndex.from_arrays(
            [frame.index, [symbol.upper()] * len(frame)],
            names=["datetime", "instrument"],
        )
        parts.append(part)
    panel = pd.concat(parts).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    return panel


def write_panel_h5(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_hdf(path, key=H5_KEY)


def build_debug_panel(panel: pd.DataFrame, settings: CryptoDataSettings) -> pd.DataFrame:
    # Keep the most liquid instruments, not merely the first ones in index
    # order.  Rank by traded value: raw base-asset volume lets high-supply,
    # low-price alts crowd out BTC/ETH in the debug universe.
    liquidity = (
        (panel["$volume"] * panel["$close"]).groupby(level="instrument").sum().sort_values(ascending=False)
    )
    instruments = liquidity.index[: settings.debug_symbol_count].tolist()
    subset = panel.loc[panel.index.get_level_values("instrument").isin(instruments)]
    dates = sorted(subset.index.get_level_values("datetime").unique())
    if dates and settings.debug_days > 0:
        cutoff = dates[-min(settings.debug_days, len(dates))]
        subset = subset.loc[subset.index.get_level_values("datetime") >= cutoff]
    # Keep only columns that carry data in the reduced universe; otherwise the
    # coder prompt advertises symbols that are not present in the rows.
    return subset.dropna(axis=1, how="all")


def crypto_data_readme(settings: CryptoDataSettings, panel: pd.DataFrame) -> str:
    instruments = list(dict.fromkeys(panel.index.get_level_values("instrument")))
    dates = sorted(panel.index.get_level_values("datetime").unique())
    extra_fields = [column for column in panel.columns if column in DERIVATIVE_FIELDS]
    descriptions = {
        "$funding_rate": "Daily mean perpetual funding rate.",
        "$funding_rate_last": "Last funding rate settled on the UTC day.",
        "$funding_rate_sum": "Sum of all funding settlements on the UTC day.",
        "$open_interest": "Daily open interest (base asset).",
        "$open_interest_value": "Daily open interest notional (quote asset).",
        "$global_account_ls_ratio": "Global long/short account ratio.",
        "$top_account_ls_ratio": "Top-trader long/short account ratio.",
        "$top_position_ls_ratio": "Top-trader long/short position ratio.",
        "$taker_buy_sell_ratio": "Taker buy/sell volume ratio.",
        "$taker_buy_vol": "Taker buy volume.",
        "$taker_sell_vol": "Taker sell volume.",
    }
    extra_rows = "".join(f"| `{field}` | {descriptions.get(field, 'Derivative field.')} |\n" for field in extra_fields)
    derivative_note = (
        "\nThe derivatives fields are point-in-time daily values. Missing rows mean the "
        "exchange did not provide the metric for that symbol/day; they are not zero.\n"
        if extra_fields
        else ""
    )
    single_note = (
        "\n**Single-instrument time-series mode.** This dataset has one instrument. "
        "Factors are time-series timing signals for it, not cross-sectional rankings; "
        "windows are measured in bars and the backtest uses the time-series simulator.\n"
        if len(instruments) == 1
        else ""
    )
    return f"""# Crypto market data ({settings.market} / {settings.freq})

Read the file with:

```python
import pandas as pd
df = pd.read_hdf("daily_pv.h5", key="data")
```

The HDF5 key is always `data`.

## Structure

- Index: MultiIndex `(datetime, instrument)`.
- `instrument` is a trading pair symbol (e.g. `{", ".join(instruments[:3])}`), not a stock code.
- Coverage: {len(instruments)} instruments, {dates[0].date()} to {dates[-1].date()}.
{single_note}

## Columns

| Column | Description |
| ------ | ----------- |
| `$open`  | Open price of the bar. |
| `$close` | Close price of the bar. |
| `$high`  | High price of the bar. |
| `$low`   | Low price of the bar. |
| `$volume`| Base-asset volume. |
| `$factor`| Always `1.0`. Crypto has no ex-rights or dividend adjustment. |
{extra_rows}
## Things to know

- This is a cryptocurrency market: 7x24 continuous trading, no weekends or holidays.
- There is no fundamental data (no shares outstanding, no industry, no financial statements).
  Factors are derived from price, volume, and any listed derivatives fields.
- Timestamps are continuous calendar days and already aligned across instruments.
{derivative_note}"""


def build_h5(settings: CryptoDataSettings, adapter: ExchangeAdapter | None = None) -> dict[str, object]:
    """Download the universe and materialise the H5 snapshots.

    Writes:

    - canonical copies under ``<data_root>/<dataset>/`` (``daily_pv_all.h5`` / ``daily_pv_debug.h5``)
    - the snapshots the CoSTEER coder reads, at ``FACTOR_COSTEER_SETTINGS.data_folder``
      and ``...data_folder_debug`` (``daily_pv.h5`` + ``README.md``)
    """

    if not settings.mode:
        raise RuntimeError("CRYPTO_MODE is not enabled; refusing to build crypto data.")

    adapter = adapter or build_adapter(settings)
    universe = resolve_universe(settings, adapter)
    logger.info(f"Universe: {len(universe)} symbols ({settings.universe_mode})")

    frames, missed_symbols = fetch_frames(settings, adapter, universe)
    derivatives = fetch_derivatives(settings, adapter, universe)
    panel = build_panel(frames, derivatives)
    debug_panel = build_debug_panel(panel, settings)
    readme = crypto_data_readme(settings, panel)

    all_path = settings.data_dir / "daily_pv_all.h5"
    debug_path = settings.data_dir / "daily_pv_debug.h5"
    write_panel_h5(panel, all_path)
    write_panel_h5(debug_panel, debug_path)
    (settings.data_dir / "README.md").write_text(readme, encoding="utf-8")

    # CoSTEER reads from its own folders; keep them in sync with the snapshot.
    for panel_for_folder, folder in (
        (panel, FACTOR_COSTEER_SETTINGS.data_folder),
        (debug_panel, FACTOR_COSTEER_SETTINGS.data_folder_debug),
    ):
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        write_panel_h5(panel_for_folder, target / "daily_pv.h5")
        shutil.copyfile(settings.data_dir / "README.md", target / "README.md")

    summary = {
        "dataset": settings.resolved_dataset_name,
        "market": settings.market,
        "freq": settings.freq,
        "instruments": sorted(panel.index.get_level_values("instrument").unique().tolist()),
        "requested_symbols": len(universe),
        "missed_symbols": sorted(missed_symbols),
        "rows": int(len(panel)),
        "derivatives": [column for column in panel.columns if column in DERIVATIVE_FIELDS],
        "start": str(panel.index.get_level_values("datetime").min().date()),
        "end": str(panel.index.get_level_values("datetime").max().date()),
        "all_h5": str(all_path),
        "debug_h5": str(debug_path),
        "data_dir": str(settings.data_dir),
        "provider_uri": settings.provider_uri,
    }
    return summary


__all__ = [
    "PV_FIELDS",
    "fetch_frames",
    "build_panel",
    "build_debug_panel",
    "write_panel_h5",
    "crypto_data_readme",
    "build_h5",
]
