from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import PrivateAttr
from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


def crypto_enabled() -> bool:
    """Single source of truth for the ``CRYPTO_MODE`` switch.

    All crypto behaviour is gated behind this so that ``CRYPTO_MODE=false``
    (the default) keeps the original A-share scenario byte-for-byte intact.
    """

    return os.environ.get("CRYPTO_MODE", "").strip().lower() == "true"


#: Injected into scenario backgrounds so the model reasons about a crypto market
#: instead of the default A-share market. Kept factual: everything here is true
#: of the generated dataset.
CRYPTO_MARKET_NOTE = (
    "====== Cryptocurrency Market ======\n"
    "- Cross-sectional setting: the universe contains many trading pairs (e.g. BTCUSDT, ETHUSDT). "
    "On each date the factors rank the whole universe; a factor predicts which pairs will outperform.\n"
    "- The market trades 7x24: no weekends, no holidays, and no price limits.\n"
    "- Base fields are OHLCV: $open, $close, $high, $low and $volume, plus $factor which is always "
    "1.0 (crypto has no ex-rights or dividend adjustment). Selected Data Zoo derivatives may add "
    "extra fields; always check the data README / source-data description for the exact columns. "
    "There is no fundamental data (no shares outstanding, no industry, no financial statements) "
    "and no $vwap column unless a selected source provides it.\n"
    "- Trading costs are charged on both the open and the close, so turnover directly reduces returns.\n"
    "- Terminology: refer to instruments as trading pairs/symbols (e.g. BTCUSDT), never as stocks, "
    "A-shares, or equities. A-share rules (T+1, price limits, IPO lockups, suspended stocks) do not apply.\n"
)

#: Appended only when the dataset was built with ``CRYPTO_INCLUDE_DERIVATIVES=true``.
CRYPTO_DERIVATIVES_NOTE = (
    "====== Cryptocurrency Derivatives Fields ======\n"
    "- The dataset also carries point-in-time derivatives fields where available: "
    "$funding_rate, $funding_rate_last, $funding_rate_sum, $open_interest, $open_interest_value, "
    "$global_account_ls_ratio, $top_account_ls_ratio, $top_position_ls_ratio, "
    "$taker_buy_sell_ratio, $taker_buy_vol and $taker_sell_vol.\n"
    "- These are daily values (funding rates are averaged per UTC day; the last rate is kept "
    "separately). They are useful for short-side / crowding / overvaluation hypotheses, but "
    "coverage differs by exchange and history: missing values are NaN, not zero.\n"
)

#: Appended when the provider has exactly one instrument.  The default crypto
#: scenario is cross-sectional; a one-symbol dataset must be evaluated as a
#: time-series timing problem instead.
CRYPTO_SINGLE_ASSET_NOTE = (
    "====== Single-Asset Time-Series Mode ======\n"
    "- The selected dataset contains exactly one instrument (for example BTCUSDT). "
    "This is not a cross-sectional universe: factors are time-series signals for that "
    "instrument and are evaluated on forward bars.\n"
    "- Do not require cross-sectional ranking, cross-sectional IC, or a full multi-coin "
    "universe. Use the custom time-series simulator; windows are measured in bars (4h "
    "bars for a 4h dataset).\n"
)


class CryptoDataSettings(ExtendedBaseSettings):
    """Configuration for the cryptocurrency data layer.

    Every field maps to an environment variable with the ``CRYPTO_`` prefix
    (e.g. ``market`` -> ``CRYPTO_MARKET``).  Defaults target Binance USDT-M
    perpetual futures, daily bars, point-in-time top-50 by quote volume.
    """

    model_config = SettingsConfigDict(env_prefix="CRYPTO_", protected_namespaces=())

    # -- master switch ------------------------------------------------------
    mode: bool = False
    """``CRYPTO_MODE``. False keeps the original A-share behaviour."""

    # -- data source --------------------------------------------------------
    exchange: str = "binance"
    """Exchange adapter to use. Only ``binance`` is implemented for now."""

    market: str = "usdm"
    """``usdm`` (USDT-M perpetual futures) or ``spot``."""

    base_url: str = ""
    """Override the exchange REST base URL.

    Useful when the main API host is unreachable: Binance's public market-data
    mirror ``https://data-api.binance.vision`` serves the same spot endpoints.
    Empty -> the built-in default for the selected exchange/market.
    """

    symbol_source: str = "exchange_info"
    """How to enumerate tradable symbols: ``exchange_info`` or ``ticker``.

    ``exchange_info`` is the accurate source (contract type, listing status) but
    can be a 10MB+ download on mirrors. ``ticker`` derives the universe from the
    24h volume list, which is smaller and usually reachable.
    """

    quote: str = "USDT"
    """Quote asset used to filter and rank the universe."""

    freq: str = "1d"
    """Bar frequency. v1 supports ``1d`` only (Qlib native ``day`` calendar)."""

    include_derivatives: bool = False
    """``CRYPTO_INCLUDE_DERIVATIVES``. Fetch funding rate / OI / long-short / taker data.

    Off by default so existing OHLCV-only datasets stay byte-for-byte compatible.
    Only supported for ``market=usdm``; spot builds ignore it with a warning.
    """

    derivatives_period: str = "1d"
    """Aggregation period for Binance futures-data endpoints (``1d`` for daily)."""

    # -- universe -----------------------------------------------------------
    universe_mode: str = "point_in_time"
    """``point_in_time`` (rank by volume at the start of the requested range,
    the default), ``top_n`` (rank by today's 24h quote volume, look-ahead
    survivorship bias: only use it for live/current snapshots) or ``explicit``."""

    universe_size: int = 50
    """How many symbols to keep when ranking the universe."""

    rank_lookback_days: int = 30
    """Window (calendar days from ``start``) used to rank the universe in
    ``universe_mode=point_in_time``."""

    enable_knowledge: bool = False
    """Whether the CoSTEER knowledge base (RAG) is used for factor coding.

    The knowledge base embeds past tasks via an embeddings endpoint. Off by
    default because chat-only OpenAI-compatible gateways have no /embeddings
    route; set ``CRYPTO_ENABLE_KNOWLEDGE=true`` once an embedding model is
    configured.
    """

    symbols: str = ""
    """Comma separated symbol list, used when ``universe_mode=explicit``."""

    # -- date range ---------------------------------------------------------
    start: str = "2022-01-01"
    """Inclusive start date (UTC), ``YYYY-MM-DD``."""

    end: str = ""
    """Exclusive end date (UTC). Empty means "now"."""

    # Train/valid/test overrides. All six must be set to take effect; otherwise
    # the segments are derived from the built dataset (60% / 20% / 20%).
    train_start: str = ""
    train_end: str = ""
    valid_start: str = ""
    valid_end: str = ""
    test_start: str = ""
    test_end: str = ""

    # -- output locations ---------------------------------------------------
    data_root: str = "git_ignore_folder/crypto_data"
    """Root folder for generated H5 snapshots (one sub-folder per dataset)."""

    bin_root: str = "~/.qlib/qlib_data"
    """Root folder for generated Qlib binary datasets (never ``cn_data``)."""

    dataset_name: str = ""
    """Dataset id. Empty -> ``crypto_<market>_<freq>``."""

    debug_symbol_count: int = 10
    """How many instruments the debug H5 snapshot keeps."""

    debug_days: int = 365
    """How many trailing calendar days the debug H5 snapshot keeps."""

    # -- backtest parameters ------------------------------------------------
    ann_scaler: int = 365
    """Annualisation factor. Crypto trades 7x24, so natural days."""

    label_horizon: int = 1
    """Forward horizon (in bars) for the label."""

    topk: int = 10
    """Portfolio size for ``TopkDropoutStrategy``."""

    n_drop: int = 2
    """Positions rotated out each rebalance."""

    account: float = 10_000.0
    """Initial backtest cash."""

    benchmark: str = ""
    """Backtest benchmark instrument. Empty -> first instrument of the universe."""

    open_cost: float = 0.0005
    """Buy cost (taker fee), fraction of notional."""

    close_cost: float = 0.0005
    """Sell cost (taker fee), fraction of notional."""

    min_cost: float = 0.0
    """Minimum commission per trade."""

    short_borrow_rate: float = 0.10
    """Annualised borrow cost on short notional (fraction of notional).

    ``long_short_backtest.py`` charges this when derivatives funding history is
    unavailable (``include_derivatives`` off). Set ``CRYPTO_SHORT_BORROW_RATE=0``
    to reproduce the old zero-carry behaviour.
    """

    trade_unit: float = 1.0
    """Tradable unit. Must be 1 for crypto; Qlib defaults to 100 (A-share lots)."""

    # -- LightGBM hyperparameters -------------------------------------------
    # The A-share values baked into the qlib templates are tuned for a panel of
    # thousands of stocks. Applied to a ~10-coin crypto universe their L1/L2
    # regularisation is so strong that LightGBM builds a single tree and
    # predicts a constant, which makes the cross-sectional IC undefined
    # (numpy: "an input array is constant"). These defaults are deliberately
    # lighter; increase them once the universe is much wider.
    lgb_learning_rate: float = 0.05
    lgb_max_depth: int = 6
    lgb_num_leaves: int = 31
    lgb_colsample: float = 0.8
    lgb_subsample: float = 0.8
    lgb_lambda_l1: float = 0.0
    lgb_lambda_l2: float = 1.0
    lgb_num_threads: int = 20

    # -- downloader ---------------------------------------------------------
    request_sleep: float = 0.2
    """Seconds to sleep between HTTP requests (rate-limit politeness)."""

    max_retries: int = 5
    """Retries per request before giving up."""

    page_limit: int = 1000
    """Rows per kline page (exchange maximum)."""

    # -- derived ------------------------------------------------------------
    @property
    def resolved_dataset_name(self) -> str:
        return self.dataset_name or f"crypto_{self.market.lower()}_{self.freq}"

    @property
    def data_dir(self) -> Path:
        return Path(self.data_root).expanduser() / self.resolved_dataset_name

    @property
    def provider_uri(self) -> str:
        return str((Path(self.bin_root).expanduser() / self.resolved_dataset_name).resolve())

    @property
    def qlib_freq(self) -> str:
        """Qlib's frequency token.

        Qlib has no literal ``4h``; intraday bars are expressed as ``240min``.
        """

        if self.freq == "1d":
            return "day"
        if self.freq == "4h":
            return "240min"
        return self.freq

    @property
    def report_freq(self) -> str:
        """Prefix Qlib uses for portfolio report files and metrics."""

        return "1day" if self.qlib_freq == "day" else self.qlib_freq

    @property
    def effective_ann_scaler(self) -> int:
        """Bars per year for risk annualisation (365 days or 6x365 4H bars)."""

        if self.freq == "4h" and self.ann_scaler == 365:
            return 2190
        return self.ann_scaler

    @property
    def explicit_symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    #: Cached value of :attr:`resolved_end`. Memoised so a long build never
    #: splits its date range across midnight (keeps a run reproducible).
    _resolved_end_cache: str | None = PrivateAttr(default=None)

    @property
    def resolved_end(self) -> str:
        if self.end:
            return self.end
        if self._resolved_end_cache is None:
            self._resolved_end_cache = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._resolved_end_cache

    def label_expression(self) -> str:
        """Forward return label with no A-share T+1 shift.

        ``Ref($close, -h) / $close - 1`` predicts the next ``h`` bars directly,
        which matches a market where you can trade at the close.
        """

        return f"Ref($close, -{self.label_horizon})/$close - 1"

    def _dataset_range(self) -> tuple[str, str]:
        """Date range of the built dataset, preferring its metadata."""

        meta_file = Path(self.provider_uri) / "crypto_meta.json"
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            return str(meta["start"]), str(meta["end"])
        except (OSError, KeyError, ValueError):
            return self.start, self.resolved_end

    def resolve_segments(self) -> dict[str, str]:
        """Train / valid / test splits for the Qlib templates.

        Explicit ``CRYPTO_TRAIN_*`` / ``CRYPTO_VALID_*`` / ``CRYPTO_TEST_*``
        overrides win when all six are set. Otherwise the dataset range is split
        60% / 20% / 20%, leaving ``label_horizon`` days of margin at the end so
        forward labels are not truncated.
        """

        explicit = {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }
        if all(explicit.values()):
            return explicit

        start_str, end_str = self._dataset_range()
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        span = (end - start).days
        if span < 30:
            raise ValueError(
                f"Dataset range {start_str}..{end_str} is too short ({span} days) to split."
            )
        train_end = start + timedelta(days=int(span * 0.6))
        valid_start = train_end + timedelta(days=1)
        valid_end = start + timedelta(days=int(span * 0.8))
        test_start = valid_end + timedelta(days=1)
        test_end = end - timedelta(days=max(1, self.label_horizon))
        fmt = "%Y-%m-%d"
        return {
            "train_start": start.strftime(fmt),
            "train_end": train_end.strftime(fmt),
            "valid_start": valid_start.strftime(fmt),
            "valid_end": valid_end.strftime(fmt),
            "test_start": test_start.strftime(fmt),
            "test_end": test_end.strftime(fmt),
        }

    def resolve_benchmark(self) -> str:
        """Backtest benchmark instrument.

        Prefers ``CRYPTO_BENCHMARK``. Otherwise falls back to ``BTCUSDT`` when it
        is in the dataset, else the first instrument. Never returns an empty
        string when a dataset exists, because Qlib needs a real benchmark series.
        """

        if self.benchmark:
            return self.benchmark.upper()
        instruments_file = Path(self.provider_uri) / "instruments" / "all.txt"
        try:
            codes = [
                line.split("\t")[0].strip()
                for line in instruments_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            codes = []
        if not codes:
            return ""
        return "BTCUSDT" if "BTCUSDT" in codes else codes[0]

    def runner_env(self) -> dict[str, str]:
        """Extra Jinja2 variables injected into the Qlib backtest templates.

        Kept in one place so the factor and model runners cannot drift apart.
        """

        return {
            **self.resolve_segments(),
            "provider_uri": self.provider_uri,
            "region": "cn",  # Qlib only exposes cn/us; our own calendar drives the dates.
            "market": "all",
            "benchmark": self.resolve_benchmark(),
            "label_expression": self.label_expression(),
            "freq": self.freq,
            "qlib_freq": self.qlib_freq,
            "report_freq": self.report_freq,
            "CRYPTO_FREQ": self.freq,
            "CRYPTO_QLIB_FREQ": self.qlib_freq,
            "CRYPTO_REPORT_FREQ": self.report_freq,
            "FACTOR_REBALANCE_DAYS": "6" if self.freq == "4h" else "1",
            "ann_scaler": str(self.effective_ann_scaler),
            "limit_threshold": "1.0",  # crypto has no price limit
            "open_cost": str(self.open_cost),
            "close_cost": str(self.close_cost),
            "min_cost": str(self.min_cost),
            "short_borrow_rate": str(self.short_borrow_rate),
            "trade_unit": str(self.trade_unit),
            "topk": str(self.topk),
            "n_drop": str(self.n_drop),
            "account": str(int(self.account)),
            "lgb_learning_rate": str(self.lgb_learning_rate),
            "lgb_max_depth": str(self.lgb_max_depth),
            "lgb_num_leaves": str(self.lgb_num_leaves),
            "lgb_colsample": str(self.lgb_colsample),
            "lgb_subsample": str(self.lgb_subsample),
            "lgb_lambda_l1": str(self.lgb_lambda_l1),
            "lgb_lambda_l2": str(self.lgb_lambda_l2),
            "lgb_num_threads": str(self.lgb_num_threads),
            # `run_env` entries are exported as real environment variables to the
            # backtest subprocess (see LocalEnv._run). mlflow >= 3 refuses the
            # filesystem tracking store that qlib's qrun relies on unless this is
            # set. Scoped to the crypto path so the A-share behaviour is untouched.
            "MLFLOW_ALLOW_FILE_STORE": "true",
        }


CRYPTO_SETTINGS = CryptoDataSettings()

__all__ = ["CryptoDataSettings", "CRYPTO_SETTINGS", "crypto_enabled"]
