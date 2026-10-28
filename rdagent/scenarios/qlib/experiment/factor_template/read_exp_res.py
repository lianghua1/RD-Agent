import os
from pathlib import Path

import pandas as pd
import qlib

qlib.init()

from qlib.workflow import R

# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

experiments = R.list_experiments()

# Iterate through each experiment to find the latest recorder.  The workspace
# removes its local mlruns store before qrun, so the only recorder here is the
# one produced by this run.
experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info["end_time"]
            try:
                if end_time is not None:
                    if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                        latest_recorder = recorder
                else:
                    print(f"Warning: Recorder {recorder_id} has no valid end time")
            except Exception as e:
                print(f"Error: {e}")

output_path = Path(__file__).resolve().parent / "qlib_res.csv"
if latest_recorder is None:
    print("No recorders found")
    raise SystemExit("No recorders found")

print(f"Latest recorder: {latest_recorder}")
metrics = pd.Series(latest_recorder.list_metrics())

mode = os.environ.get("FACTOR_POSITION_MODE", "long_only").strip().lower().replace("-", "_")
report_freq = os.environ.get("CRYPTO_REPORT_FREQ", "1day").strip() or "1day"
qlib_freq = os.environ.get("CRYPTO_QLIB_FREQ", "day").strip() or "day"
try:
    n_instruments = int(os.environ.get("FACTOR_N_INSTRUMENTS", "2") or "2")
except ValueError:
    n_instruments = 2

report = None
report_path = Path("portfolio_analysis") / f"report_normal_{report_freq}.pkl"
try:
    report = latest_recorder.load_object(str(report_path))
except Exception as error:
    print(f"Portfolio report {report_path} unavailable: {error}")

custom = n_instruments < 2 or mode in {"short_only", "long_short"} or report is None
if custom:
    from long_short_backtest import load_funding_from_qlib, run_from_recorder

    funding = None
    try:
        funding = load_funding_from_qlib(latest_recorder, freq=qlib_freq)
    except Exception as error:
        print(f"Funding series unavailable; falling back to the borrow rate: {error}")
    extra_metrics, custom_report = run_from_recorder(latest_recorder, funding=funding)
    metrics = pd.concat([metrics, pd.Series(extra_metrics)])
    # Qlib's PortAnaRecord is long-only.  Keeping its
    # <freq>.excess_return_* numbers next to a short/multi-direction or
    # single-instrument result would mislead both the feedback prompt and the
    # UI, so drop only those portfolio metrics.
    metrics = metrics[~metrics.index.astype(str).str.startswith(f"{report_freq}.excess_return")]
    if n_instruments < 2 and mode == "long_only":
        # Give the feedback prompt a stable long-only name for the
        # time-series simulator's headline metrics.
        for source, target in (
            ("strategy.annualized_return", "long_only.annualized_return"),
            ("strategy.sharpe", "long_only.sharpe"),
            ("strategy.max_drawdown", "long_only.max_drawdown"),
        ):
            if source in metrics.index:
                metrics[target] = metrics[source]
    report = custom_report

metrics.to_csv(output_path)
print(f"Output has been saved to {output_path}")
if report is not None:
    report.to_parquet("ret.parquet")
