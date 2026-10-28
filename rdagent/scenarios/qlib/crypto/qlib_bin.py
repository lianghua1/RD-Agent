from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rdagent.log import rdagent_logger as logger

# Qlib stores the first element of every ``*.bin`` file as the index of the
# instrument's first valid value inside the global calendar, followed by the
# float32 values themselves.  This mirrors ``scripts/dump_bin.py`` in the Qlib
# repository: the header is essential, and writing 0 unconditionally (a common
# mistake) corrupts every instrument whose history does not start on the very
# first calendar day.
FEATURE_FILES = {
    "$open": "open",
    "$close": "close",
    "$high": "high",
    "$low": "low",
    "$volume": "volume",
    "$factor": "factor",
    # Optional derivatives fields (only present when CRYPTO_INCLUDE_DERIVATIVES=true
    # or when selected Data Zoo sources provide them).
    "$funding_rate": "funding_rate",
    "$funding_rate_last": "funding_rate_last",
    "$funding_rate_sum": "funding_rate_sum",
    "$open_interest": "open_interest",
    "$open_interest_value": "open_interest_value",
    "$global_account_ls_ratio": "global_account_ls_ratio",
    "$top_account_ls_ratio": "top_account_ls_ratio",
    "$top_position_ls_ratio": "top_position_ls_ratio",
    "$taker_buy_sell_ratio": "taker_buy_sell_ratio",
    "$taker_buy_vol": "taker_buy_vol",
    "$taker_sell_vol": "taker_sell_vol",
}


def write_qlib_bin(panel: pd.DataFrame, out_dir: str | Path, freq: str = "day") -> dict[str, object]:
    """Materialise ``panel`` as a Qlib native dataset.

    ``panel`` must be indexed by ``(datetime, instrument)`` and carry the
    ``$open/$close/$high/$low/$volume/$factor`` columns.  Writes::

        <out_dir>/calendars/<freq>.txt
        <out_dir>/instruments/all.txt
        <out_dir>/features/<symbol>/<field>.<freq>.bin
    """

    output = Path(out_dir)
    calendar = pd.DatetimeIndex(sorted(panel.index.get_level_values("datetime").unique()))
    if len(calendar) == 0:
        raise RuntimeError("Cannot write Qlib data from an empty panel.")

    calendars_dir = output / "calendars"
    features_dir = output / "features"
    instruments_dir = output / "instruments"
    for folder in (calendars_dir, features_dir, instruments_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # 1) calendar.  Intraday calendars must keep the time-of-day; writing six
    # identical YYYY-MM-DD lines corrupts the calendar index mapping.
    stamp_format = "%Y-%m-%d" if freq == "day" else "%Y-%m-%d %H:%M:%S"
    with (calendars_dir / f"{freq}.txt").open("w", encoding="utf-8") as handle:
        for date in calendar:
            handle.write(date.strftime(stamp_format) + "\n")

    calendar_index = {date: position for position, date in enumerate(calendar)}
    instruments: list[tuple[str, str, str]] = []

    # 2) per-instrument feature files
    for symbol, frame in panel.groupby(level="instrument"):
        code = str(symbol)
        frame = frame.droplevel("instrument").sort_index()
        if frame.empty:
            continue
        start, end = frame.index.min(), frame.index.max()
        start_index = calendar_index[start]
        end_index = calendar_index[end]
        # Qlib aligns each instrument to the calendar slice spanning its own
        # history, so the array starts at the instrument's first bar.
        window = calendar[start_index : end_index + 1]

        symbol_dir = features_dir / code.lower()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for column, field in FEATURE_FILES.items():
            if column not in frame.columns:
                continue
            series = frame[column].reindex(window)
            values = series.to_numpy(dtype="float64")
            payload = np.hstack([start_index, values]).astype("<f")
            payload.tofile(str((symbol_dir / f"{field}.{freq}.bin").resolve()))

        instruments.append((code.upper(), start.strftime(stamp_format), end.strftime(stamp_format)))

    # 3) instruments file (tab separated, no header)
    instruments.sort(key=lambda item: item[0])
    with (instruments_dir / "all.txt").open("w", encoding="utf-8") as handle:
        for code, start, end in instruments:
            handle.write(f"{code}\t{start}\t{end}\n")

    meta = {
        "freq": freq,
        "calendar_days": len(calendar),
        "start": calendar[0].strftime("%Y-%m-%d"),
        "end": calendar[-1].strftime("%Y-%m-%d"),
        "instruments": len(instruments),
        "output": str(output.resolve()),
    }
    (output / "crypto_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        f"Qlib data written to {output.resolve()} "
        f"({meta['instruments']} instruments, {meta['calendar_days']} days)"
    )
    return meta


__all__ = ["write_qlib_bin", "FEATURE_FILES"]
