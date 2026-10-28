from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Protocol

import pandas as pd

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.crypto.conf import CryptoDataSettings

_KLINES_PATH = {
    "usdm": "/fapi/v1/klines",
    "spot": "/api/v3/klines",
}
_EXCHANGE_INFO_PATH = {
    "usdm": "/fapi/v1/exchangeInfo",
    "spot": "/api/v3/exchangeInfo",
}
_TICKER_PATH = {
    "usdm": "/fapi/v1/ticker/24hr",
    "spot": "/api/v3/ticker/24hr",
}
_BASE_URL = {
    ("binance", "usdm"): "https://fapi.binance.com",
    ("binance", "spot"): "https://api.binance.com",
}

_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
]

_HEADERS = {
    "User-Agent": "rdagent-crypto/1.0",
    "Accept": "application/json",
}


class ExchangeError(RuntimeError):
    """Raised when the exchange returns an unrecoverable error."""


class ExchangeAdapter(Protocol):
    """Minimal surface the data pipeline needs from an exchange."""

    name: str

    def list_symbols(self) -> list[str]:
        """All currently tradable symbols for the configured quote asset."""

    def quote_volumes(self) -> dict[str, float]:
        """Rolling 24h quote volume per symbol, used to rank the universe."""

    def quote_volumes_at(self, start: str, lookback_days: int) -> dict[str, float]:
        """Average quote volume over the ``lookback_days`` window from ``start``.

        Used for point-in-time universe ranking so the selection does not depend
        on today's liquidity. Symbols without data in the window are omitted.
        """

    def klines(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """OHLCV bars for ``symbol`` between ``start`` (inclusive) and ``end`` (exclusive).

        Returns a frame indexed by tz-naive UTC date with columns
        ``open, high, low, close, volume, quote_volume``.
        """

    def derivatives(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Optional daily derivatives fields for ``symbol``.

        Implementations may return an empty frame; the dataset layer treats
        derivative data as best-effort and never fabricates missing rows.
        """


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (seconds form only); HTTP-date -> None."""

    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class BinanceAdapter:
    """Binance public-data adapter (no API key required).

    Uses only the stdlib so the data layer adds no new dependency.
    """

    name = "binance"

    def __init__(self, settings: CryptoDataSettings) -> None:
        key = (settings.exchange.lower(), settings.market.lower())
        if key not in _BASE_URL:
            raise ExchangeError(f"Unsupported exchange/market combination: {key}")
        self.settings = settings
        self.market = settings.market.lower()
        self.base_url = settings.base_url.rstrip("/") or _BASE_URL[key]
        self.quote = settings.quote.upper()

    # ---------------------------------------------------------------- HTTP
    def _get(self, path: str, params: dict[str, object] | None = None) -> object:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        attempt = 0
        while True:
            attempt += 1
            try:
                request = urllib.request.Request(url, headers=_HEADERS)
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                retry_after = _retry_after_seconds(error.headers.get("Retry-After") if error.headers else None)
                retryable = error.code in (418, 429) or 500 <= error.code < 600
                if not retryable or attempt > self.settings.max_retries:
                    body = error.read().decode("utf-8", errors="replace")[:200]
                    raise ExchangeError(f"HTTP {error.code} for {url}: {body}") from error
                delay = retry_after if retry_after is not None else min(2 ** attempt, 30)
                logger.warning(f"Binance HTTP {error.code}, retrying in {delay:.1f}s (attempt {attempt})")
                time.sleep(delay)
                continue
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt > self.settings.max_retries:
                    raise ExchangeError(f"Network error for {url}: {error}") from error
                delay = min(2 ** attempt, 30)
                logger.warning(f"Network error ({error}), retrying in {delay:.1f}s (attempt {attempt})")
                time.sleep(delay)
                continue
            # Decode outside the HTTP try: a mirror or WAF returning an HTML
            # error page must be reported/retried, not escape as a bare
            # ``JSONDecodeError`` (the original code left it unhandled).
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                if attempt > self.settings.max_retries:
                    snippet = raw[:200].decode("utf-8", errors="replace")
                    raise ExchangeError(f"Non-JSON response from {url}: {snippet}") from error
                delay = min(2 ** attempt, 30)
                logger.warning(f"Non-JSON response from {url} ({error}), retrying in {delay:.1f}s (attempt {attempt})")
                time.sleep(delay)
                continue
            # Politeness delay only on success; retries have their own backoff.
            if self.settings.request_sleep > 0:
                time.sleep(self.settings.request_sleep)
            return payload

    # ----------------------------------------------------------- universe
    def list_symbols(self) -> list[str]:
        # exchangeInfo is large (10MB+) on some mirrors. When it is unavailable
        # (or explicitly disabled) we return an empty list and the universe
        # resolver derives candidates from the 24h ticker instead.
        if self.settings.symbol_source.strip().lower() == "ticker":
            return []
        try:
            info = self._get(_EXCHANGE_INFO_PATH[self.market])
        except ExchangeError as error:
            logger.warning(f"exchangeInfo unavailable ({error}); ranking the universe from 24h tickers instead")
            return []
        symbols = info.get("symbols", []) if isinstance(info, dict) else []
        result: list[str] = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "TRADING":
                continue
            if item.get("quoteAsset", "").upper() != self.quote:
                continue
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            if self.market == "usdm":
                # Only perpetual contracts; exclude quarterly/delivery contracts.
                if item.get("contractType") != "PERPETUAL":
                    continue
            else:
                if item.get("isSpotTradingAllowed") is False:
                    continue
                # Exclude leveraged tokens (BTCUPUSDT, BTCDOWNUSDT, ...).
                if symbol.endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")):
                    continue
            result.append(symbol)
        return sorted(set(result))

    def quote_volumes(self) -> dict[str, float]:
        tickers = self._get(_TICKER_PATH[self.market])
        if not isinstance(tickers, list):
            raise ExchangeError("Unexpected ticker payload from Binance")
        volumes: dict[str, float] = {}
        for item in tickers:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            try:
                volumes[symbol] = float(item.get("quoteVolume", 0.0))
            except (TypeError, ValueError):
                continue
        return volumes

    def quote_volumes_at(self, start: str, lookback_days: int) -> dict[str, float]:
        """Ranking volumes as of ``start`` (point-in-time, no look-ahead).

        Downloads a short window per candidate and averages its ``quote_volume``.
        Coins that were not yet tradable at ``start`` return no bars and are
        dropped. Note that the exchange only reports *currently* listed symbols,
        so coins delisted before the build are still missed (residual
        survivorship); use ``CRYPTO_UNIVERSE_MODE=explicit`` for a curated,
        delisting-free research universe.
        """

        candidates = self.list_symbols()
        if not candidates:
            # exchangeInfo unavailable: fall back to the 24h ticker symbol list,
            # restricted to the configured quote asset.
            candidates = [s for s in self.quote_volumes() if s.endswith(self.quote)]
        window_end = (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=max(1, lookback_days))).strftime(
            "%Y-%m-%d"
        )
        volumes: dict[str, float] = {}
        for symbol in candidates:
            try:
                frame = self.klines(symbol, start, window_end)
            except ExchangeError as error:
                logger.warning(f"{symbol}: point-in-time window unavailable ({error}); skipping")
                continue
            if frame is not None and not frame.empty and "quote_volume" in frame:
                volumes[symbol] = float(frame["quote_volume"].mean())
        return volumes

    # ------------------------------------------------------------- klines
    def klines(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        _interval_ms(self.settings.freq)  # reject unsupported frequencies early
        start_ms = _to_ms(start)
        # Binance's endTime is inclusive; our `end` is exclusive, so subtract 1ms.
        end_ms = _to_ms(end) - 1
        if end_ms <= start_ms:
            raise ExchangeError(f"Empty date range: {start} .. {end}")
        rows: list[list[object]] = []
        cursor = start_ms
        while cursor < end_ms:
            page = self._get(
                _KLINES_PATH[self.market],
                {
                    "symbol": symbol,
                    "interval": self.settings.freq,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": self.settings.page_limit,
                },
            )
            if not isinstance(page, list) or not page:
                break
            rows.extend(page)
            if len(page) < self.settings.page_limit:
                break
            last_close = int(page[-1][6])
            if last_close <= cursor:
                break
            cursor = last_close + 1
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])

        frame = pd.DataFrame([row[:9] for row in rows], columns=_KLINE_COLUMNS)
        frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        timestamps = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None)
        if self.settings.freq == "1d":
            frame["datetime"] = timestamps.dt.normalize()
        else:
            # Intraday bars must keep their time-of-day; normalizing to midnight
            # collapsed all six 4H bars into one and silently produced a daily
            # dataset.
            frame["datetime"] = timestamps.dt.floor(self.settings.freq)
        for column in ("open", "high", "low", "close", "volume", "quote_volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        return frame.set_index("datetime")[["open", "high", "low", "close", "volume", "quote_volume"]]


    # --------------------------------------------------------- derivatives
    def _futures_data(
        self,
        path: str,
        symbol: str,
        start: str,
        end: str,
        value_columns: dict[str, str],
        limit: int = 500,
    ) -> pd.DataFrame:
        """Page through a Binance futures-data endpoint and return daily rows.

        The public futures-data endpoints only expose a limited history (about
        the last 30 days for OI / long-short ratios), so old requests can return
        HTTP 400.  Callers must treat an empty frame as "not available", never
        as zero.
        """

        start_ms = _to_ms(start)
        end_ms = _to_ms(end) - 1
        if end_ms <= start_ms:
            return pd.DataFrame()
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            page = self._get(
                path,
                {
                    "symbol": symbol,
                    "period": self.settings.derivatives_period,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": limit,
                },
            )
            if not isinstance(page, list) or not page:
                break
            rows.extend(item for item in page if isinstance(item, dict))
            if len(page) < limit:
                break
            try:
                last = int(page[-1].get("timestamp"))
            except (TypeError, ValueError, AttributeError):
                break
            if last < cursor:
                break
            cursor = last + 1
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        if "timestamp" not in frame:
            return pd.DataFrame()
        frame["datetime"] = (
            pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        )
        frame = frame.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
        output = pd.DataFrame(index=frame["datetime"])
        for source, target in value_columns.items():
            if source in frame:
                output[target] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
        output = output.dropna(how="all")
        return output.sort_index()

    def funding_rate(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Daily funding-rate summary (mean and last settlement)."""

        start_ms = _to_ms(start)
        end_ms = _to_ms(end) - 1
        if end_ms <= start_ms:
            return pd.DataFrame()
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            page = self._get(
                "/fapi/v1/fundingRate",
                {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            )
            if not isinstance(page, list) or not page:
                break
            rows.extend(item for item in page if isinstance(item, dict))
            if len(page) < 1000:
                break
            try:
                last = int(page[-1].get("fundingTime"))
            except (TypeError, ValueError, AttributeError):
                break
            if last < cursor:
                break
            cursor = last + 1
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["datetime"] = (
            pd.to_datetime(frame["fundingTime"], unit="ms", utc=True, errors="coerce")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        frame["funding_rate"] = pd.to_numeric(frame.get("fundingRate"), errors="coerce")
        frame = frame.dropna(subset=["datetime", "funding_rate"])
        if frame.empty:
            return pd.DataFrame()
        grouped = frame.groupby("datetime", sort=True)["funding_rate"]
        return pd.DataFrame(
            {
                "funding_rate": grouped.mean(),
                "funding_rate_last": grouped.last(),
                "funding_rate_sum": grouped.sum(),
            }
        )

    def derivatives(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Best-effort daily derivatives panel for one USDT-M perpetual."""

        if self.market != "usdm":
            return pd.DataFrame()
        pieces: list[pd.DataFrame] = []
        try:
            funding = self.funding_rate(symbol, start, end)
            if not funding.empty:
                pieces.append(funding)
        except ExchangeError as error:
            logger.warning(f"{symbol}: funding-rate history unavailable ({error})")
        # Binance's public futures-data endpoints reject startTime values older
        # than ~30 days.  Clip metrics to the recent window (funding still has
        # full history) instead of issuing 400s for every historical symbol.
        recent_start = (datetime.now(timezone.utc) - timedelta(days=29)).strftime("%Y-%m-%d")
        metric_start = max(start, recent_start)
        if metric_start < end:
            specs = (
                (
                    "/futures/data/openInterestHist",
                    {"sumOpenInterest": "open_interest", "sumOpenInterestValue": "open_interest_value"},
                ),
                ("/futures/data/globalLongShortAccountRatio", {"longShortRatio": "global_account_ls_ratio"}),
                ("/futures/data/topLongShortAccountRatio", {"longShortRatio": "top_account_ls_ratio"}),
                ("/futures/data/topLongShortPositionRatio", {"longShortRatio": "top_position_ls_ratio"}),
                (
                    "/futures/data/takerlongshortRatio",
                    {"buySellRatio": "taker_buy_sell_ratio", "buyVol": "taker_buy_vol", "sellVol": "taker_sell_vol"},
                ),
            )
            if metric_start != start:
                logger.info(f"{symbol}: OI / long-short metrics clipped to {metric_start}..{end} (30-day API limit)")
            for path, columns in specs:
                try:
                    piece = self._futures_data(path, symbol, metric_start, end, columns)
                except ExchangeError as error:
                    logger.warning(f"{symbol}: {path} unavailable ({error})")
                    continue
                if not piece.empty:
                    pieces.append(piece)
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, axis=1).sort_index()


def _interval_ms(freq: str) -> int:
    table = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    if freq not in table:
        raise ExchangeError(f"Unsupported frequency: {freq}")
    return table[freq]


def build_adapter(settings: CryptoDataSettings) -> ExchangeAdapter:
    exchange = settings.exchange.lower()
    if exchange == "binance":
        return BinanceAdapter(settings)
    raise ExchangeError(f"Unsupported exchange: {exchange}")


__all__ = ["ExchangeAdapter", "ExchangeError", "BinanceAdapter", "build_adapter"]
