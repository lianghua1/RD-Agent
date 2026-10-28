"""Offline tests for the cryptocurrency scenario (data layer + integration guards).

No network, no LLM, no Qlib runtime is required: the exchange is faked, the
Qlib binary writer is exercised on a tiny in-memory panel, and the two
integration fixes (empty CoSTEER knowledge, secret-redacted logging) are unit
tested directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.qlib.crypto.conf import CryptoDataSettings
from rdagent.scenarios.qlib.crypto.dataset import build_debug_panel, build_panel, crypto_data_readme, fetch_derivatives
from rdagent.scenarios.qlib.crypto.exchange import BinanceAdapter, _retry_after_seconds
from rdagent.scenarios.qlib.crypto.qlib_bin import write_qlib_bin
from rdagent.scenarios.qlib.crypto.universe import resolve_universe

pytestmark = pytest.mark.offline


class FakeAdapter:
    """Minimal exchange standing in for :class:`BinanceAdapter`."""

    name = "fake"


    def list_symbols(self):
        return list(self._listed)

    def quote_volumes(self):
        return dict(self._volumes)

    def quote_volumes_at(self, start, lookback_days):
        return dict(self._window)

    def klines(self, symbol, start, end):
        raise NotImplementedError

    def derivatives(self, symbol, start, end):
        return self._derivatives.get(symbol.upper(), pd.DataFrame())

    def __init__(self, listed=None, volumes=None, window=None, derivatives=None):
        self._listed = list(listed or [])
        self._volumes = dict(volumes or {})
        self._window = dict(window or {})
        self._derivatives = dict(derivatives or {})


def _settings(tmp_path, **kwargs):
    base = dict(
        mode=True,
        market="usdm",
        start="2022-01-01",
        end="2023-01-01",
        data_root=str(tmp_path / "data"),
        bin_root=str(tmp_path / "bin"),
        dataset_name="test",
    )
    base.update(kwargs)
    return CryptoDataSettings(**base)


# --------------------------------------------------------------------- config


def test_label_expression_is_forward_return(tmp_path):
    settings = _settings(tmp_path, label_horizon=3)
    assert settings.label_expression() == "Ref($close, -3)/$close - 1"


def test_resolved_end_is_deterministic(tmp_path, monkeypatch):
    settings = _settings(tmp_path, end="")
    first = settings.resolved_end
    # Freeze the clock: the cached value on the existing instance must not move.
    monkeypatch.setattr("rdagent.scenarios.qlib.crypto.conf.datetime", _FrozenDatetime)
    assert settings.resolved_end == first
    # A fresh instance does see the frozen clock, proving the patch is active.
    assert _settings(tmp_path, end="").resolved_end == _FrozenDatetime._TODAY


class _FrozenDatetime:
    _TODAY = "2099-12-31"

    @staticmethod
    def now(tz=None):
        return _FrozenNow()


class _FrozenNow:
    def strftime(self, fmt):
        return _FrozenDatetime._TODAY


def test_resolve_segments_explicit_wins(tmp_path):
    explicit = dict(
        train_start="2022-01-01",
        train_end="2022-06-01",
        valid_start="2022-06-02",
        valid_end="2022-09-01",
        test_start="2022-09-02",
        test_end="2022-12-01",
    )
    settings = _settings(tmp_path, **explicit)
    assert settings.resolve_segments() == explicit


def test_resolve_segments_derived(tmp_path):
    settings = _settings(tmp_path)
    segments = settings.resolve_segments()
    assert segments["train_start"] == "2022-01-01"
    assert segments["train_start"] < segments["train_end"] < segments["valid_end"] < segments["test_end"]


def test_resolve_segments_too_short(tmp_path):
    settings = _settings(tmp_path, start="2022-01-01", end="2022-01-10")
    with pytest.raises(ValueError):
        settings.resolve_segments()


def test_resolve_benchmark_prefers_btcusdt(tmp_path):
    settings = _settings(tmp_path)
    instruments = Path(settings.provider_uri) / "instruments"
    instruments.mkdir(parents=True, exist_ok=True)
    (instruments / "all.txt").write_text("ETHUSDT\t2022-01-01\t2023-01-01\nBTCUSDT\t2022-01-01\t2023-01-01\n")
    assert settings.resolve_benchmark() == "BTCUSDT"


# ------------------------------------------------------------------- universe


def test_universe_top_n_excludes_stables_and_leveraged(tmp_path):
    adapter = FakeAdapter(
        listed=["BTCUSDT", "ETHUSDT", "USDCUSDT", "BTCUPUSDT", "DOGEUSDT"],
        volumes={"BTCUSDT": 100, "ETHUSDT": 90, "USDCUSDT": 80, "BTCUPUSDT": 70, "DOGEUSDT": 60},
    )
    settings = _settings(tmp_path, universe_mode="top_n", universe_size=3)
    assert resolve_universe(settings, adapter) == ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]


def test_universe_point_in_time_uses_start_window_not_today(tmp_path):
    adapter = FakeAdapter(
        listed=["BTCUSDT", "ETHUSDT", "LUNAUSDT"],
        # Today LUNA is illiquid, but at the start of the backtest it dominated.
        volumes={"BTCUSDT": 100, "ETHUSDT": 90, "LUNAUSDT": 1},
        window={"BTCUSDT": 10, "LUNAUSDT": 1000},
    )
    settings = _settings(tmp_path, universe_mode="point_in_time", universe_size=2)
    assert resolve_universe(settings, adapter) == ["LUNAUSDT", "BTCUSDT"]


def test_universe_explicit_drops_unlisted(tmp_path):
    adapter = FakeAdapter(listed=["BTCUSDT", "ETHUSDT"])
    settings = _settings(tmp_path, universe_mode="explicit", symbols="btcusdt,ETHUSDT,ZZZUSDT")
    assert resolve_universe(settings, adapter) == ["BTCUSDT", "ETHUSDT"]


def test_universe_unknown_mode(tmp_path):
    settings = _settings(tmp_path, universe_mode="nonsense")
    with pytest.raises(ValueError):
        resolve_universe(settings, FakeAdapter())


# --------------------------------------------------------------------- dataset


def _frame(rows):
    index = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[1] + 1 for r in rows],
            "low": [r[1] - 1 for r in rows],
            "close": [r[1] for r in rows],
            "volume": [r[2] for r in rows],
            "quote_volume": [r[2] * r[1] for r in rows],
        },
        index=index,
    )


def test_build_panel_multiindex_and_dedup():
    frames = {
        "AAAUSDT": _frame([("2022-01-01", 10, 1), ("2022-01-02", 11, 2), ("2022-01-02", 11.5, 3)]),
        "BBBUSDT": _frame([("2022-01-02", 20, 5)]),
    }
    panel = build_panel(frames)
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert (panel["$factor"] == 1.0).all()
    # The duplicate (2022-01-02, AAAUSDT) must keep the last row (close 11.5).
    assert panel.loc[(pd.Timestamp("2022-01-02"), "AAAUSDT"), "$close"] == 11.5
    assert not panel.index.duplicated().any()


def _derivative_frame():
    index = pd.to_datetime(["2022-01-01", "2022-01-03"])
    return pd.DataFrame(
        {
            "funding_rate": [0.0001, 0.0003],
            "funding_rate_last": [0.0002, 0.0004],
            "funding_rate_sum": [0.0006, 0.0009],
            "open_interest": [100.0, 120.0],
            "global_account_ls_ratio": [1.2, 1.4],
        },
        index=index,
    )


def test_build_panel_merges_derivatives_and_forwards_missing_days():
    frames = {
        "AAAUSDT": _frame(
            [("2022-01-01", 10, 1), ("2022-01-02", 11, 2), ("2022-01-03", 12, 3)]
        )
    }
    panel = build_panel(frames, {"AAAUSDT": _derivative_frame()})
    assert panel.loc[(pd.Timestamp("2022-01-01"), "AAAUSDT"), "$funding_rate"] == 0.0001
    # Missing 01-02 is carried forward from 01-01, never filled with zero.
    assert panel.loc[(pd.Timestamp("2022-01-02"), "AAAUSDT"), "$funding_rate"] == 0.0001
    assert panel.loc[(pd.Timestamp("2022-01-03"), "AAAUSDT"), "$open_interest"] == 120.0
    assert panel.loc[(pd.Timestamp("2022-01-03"), "AAAUSDT"), "$global_account_ls_ratio"] == 1.4


def test_fetch_derivatives_skips_spot(tmp_path):
    settings = _settings(tmp_path, market="spot", include_derivatives=True)
    adapter = FakeAdapter(derivatives={"BTCUSDT": _derivative_frame()})
    assert fetch_derivatives(settings, adapter, ["BTCUSDT"]) == {}


def test_binance_futures_data_keeps_values_aligned_with_timestamps(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    adapter = BinanceAdapter(settings)

    def fake_get(path, params):
        return [
            {"symbol": "BTCUSDT", "sumOpenInterest": "10.5", "sumOpenInterestValue": "1000", "timestamp": 1640995200000},
            {"symbol": "BTCUSDT", "sumOpenInterest": "11.5", "sumOpenInterestValue": "1100", "timestamp": 1641081600000},
        ]

    monkeypatch.setattr(adapter, "_get", fake_get)
    frame = adapter._futures_data(
        "/futures/data/openInterestHist",
        "BTCUSDT",
        "2022-01-01",
        "2022-01-03",
        {"sumOpenInterest": "open_interest", "sumOpenInterestValue": "open_interest_value"},
    )
    assert list(frame["open_interest"]) == [10.5, 11.5]
    assert list(frame["open_interest_value"]) == [1000.0, 1100.0]
    assert frame.index[0] == pd.Timestamp("2022-01-01")


def test_crypto_data_readme_lists_derivatives():
    frames = {"AAAUSDT": _frame([("2022-01-01", 10, 1), ("2022-01-02", 11, 2)])}
    panel = build_panel(frames, {"AAAUSDT": _derivative_frame()})
    readme = crypto_data_readme(_settings(Path("/tmp")), panel)
    assert "$funding_rate" in readme
    assert "$global_account_ls_ratio" in readme
    assert "$funding_rate_sum" in readme
    assert "not zero" in readme


def test_build_debug_panel_keeps_most_liquid():
    frames = {
        "LOWUSDT": _frame([("2022-01-01", 1, 1), ("2022-01-02", 1, 1)]),
        "HIGHUSDT": _frame([("2022-01-01", 1, 100), ("2022-01-02", 1, 100)]),
    }
    panel = build_panel(frames)
    settings = CryptoDataSettings(debug_symbol_count=1, debug_days=0)
    debug = build_debug_panel(panel, settings)
    assert sorted(debug.index.get_level_values("instrument").unique()) == ["HIGHUSDT"]


# ------------------------------------------------------------------ qlib bins


def test_write_qlib_bin_header_uses_instrument_start_index(tmp_path):
    frames = {
        "AAAUSDT": _frame([("2022-01-01", 10, 1), ("2022-01-02", 11, 2)]),
        "BBBUSDT": _frame([("2022-01-02", 20, 5), ("2022-01-03", 21, 6)]),
    }
    panel = build_panel(frames)
    out = tmp_path / "provider"
    write_qlib_bin(panel, out, freq="day")

    calendar = (out / "calendars" / "day.txt").read_text().splitlines()
    assert calendar == ["2022-01-01", "2022-01-02", "2022-01-03"]

    # AAA starts on the first calendar day -> header 0; BBB starts a day later -> 1.
    aaa = np.fromfile(out / "features" / "aaausdt" / "close.day.bin", dtype="<f")
    bbb = np.fromfile(out / "features" / "bbbusdt" / "close.day.bin", dtype="<f")
    assert aaa[0] == 0.0
    assert bbb[0] == 1.0

    # Optional derivative columns become qlib .bin fields with the same header
    # contract as OHLCV; only instruments that carry them get the file.
    panel = build_panel(
        {"AAAUSDT": _frame([("2022-01-01", 10, 1), ("2022-01-02", 11, 2)])},
        {"AAAUSDT": _derivative_frame()},
    )
    out2 = tmp_path / "provider2"
    write_qlib_bin(panel, out2, freq="day")
    funding = np.fromfile(out2 / "features" / "aaausdt" / "funding_rate.day.bin", dtype="<f")
    assert funding[0] == 0.0
    assert funding[1] == 0.0001


# ------------------------------------------------------- integration guards


def test_empty_queried_knowledge_shape():
    from rdagent.components.coder.factor_coder.evolving_strategy import _empty_queried_knowledge

    class _Task:
        def __init__(self, i):
            self.i = i

        def get_task_information(self):
            return f"task-{self.i}"

    class _Evo:
        sub_tasks = [_Task(1), _Task(2)]

    knowledge = _empty_queried_knowledge(_Evo())
    assert knowledge.task_to_former_failed_traces["task-1"] == ([], None)
    assert knowledge.task_to_similar_task_successful_knowledge["task-2"] == []
    assert knowledge.task_to_similar_error_successful_knowledge["task-1"] == []
    # Fresh dicts per instance (the data container uses mutable default args).
    other = _empty_queried_knowledge(_Evo())
    assert knowledge.task_to_former_failed_traces is not other.task_to_former_failed_traces


def test_litellm_settings_are_redacted():
    from rdagent.oai.backend.litellm import _redacted_settings

    class _Settings:
        def model_dump(self):
            return {
                "openai_api_key": "secret",
                "embedding_endpoint_key": "x",
                "chat_model": "m",
                "empty_key": "",
                "none_key": None,
            }

    redacted = _redacted_settings(_Settings())
    assert redacted["openai_api_key"] == "***redacted***"
    assert redacted["embedding_endpoint_key"] == "***redacted***"
    assert redacted["chat_model"] == "m"
    assert redacted["empty_key"] == "" and redacted["none_key"] is None


def test_retry_after_parsing():
    assert _retry_after_seconds("12") == 12
    assert _retry_after_seconds(None) is None
    assert _retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") is None
