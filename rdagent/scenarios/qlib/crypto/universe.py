from __future__ import annotations

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.crypto.conf import CryptoDataSettings
from rdagent.scenarios.qlib.crypto.exchange import ExchangeAdapter

#: Bases that are stablecoins or fiat. Ranking by quote volume otherwise puts
#: pairs like USDCUSDT / USDTTRY at the top, which carry no factor signal.
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USD1", "XUSD", "USDE", "PYUSD",
    "EUR", "GBP", "TRY", "BRL", "IDRT", "BIDR", "IDR", "UAH", "RUB", "JPY", "ZAR",
    "NGN", "ARS", "COP", "PLN", "RON", "CZK", "AEUR", "BKRW", "VAI", "BVND",
}

_LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def _is_tradable_pair(symbol: str, quote: str) -> bool:
    if not symbol.endswith(quote):
        return False
    base = symbol[: -len(quote)]
    if not base:
        return False
    if base.upper() in EXCLUDED_BASES:
        return False
    if symbol.endswith(_LEVERAGED_SUFFIXES):
        return False
    return True


def resolve_universe(settings: CryptoDataSettings, adapter: ExchangeAdapter) -> list[str]:
    """Resolve the tradable coin universe for the configured dataset.

    Three modes:

    - ``explicit``: trust the user's ``CRYPTO_SYMBOLS`` list.
    - ``top_n``: rank every tradable ``<quote>`` pair by rolling 24h quote
      volume and keep the largest ``CRYPTO_UNIVERSE_SIZE``. Uses *today's*
      liquidity, so backtests over a past period suffer survivorship bias.
    - ``point_in_time``: rank by average quote volume over the first
      ``CRYPTO_RANK_LOOKBACK_DAYS`` days of the requested range. This removes
      the "only liquid today" and "listed later" biases, but the exchange still
      only reports currently-listed symbols, so coins delisted before the build
      remain missing. For rigorous research use ``explicit`` with a curated,
      point-in-time symbol list.

    ``exchangeInfo`` (used to filter by contract type / trading status) can be
    unavailable on some hosts, so the resolver treats it as best-effort: when it
    returns nothing the ranking is derived from the 24h ticker symbols, which
    still yields a sane universe.
    """

    mode = settings.universe_mode.strip().lower()
    listed = set(adapter.list_symbols())  # best-effort; may be empty
    quote = settings.quote.upper()

    if mode == "explicit":
        requested = settings.explicit_symbols
        if not requested:
            raise ValueError("CRYPTO_UNIVERSE_MODE=explicit requires CRYPTO_SYMBOLS to be set.")
        kept = [symbol for symbol in requested if not listed or symbol in listed]
        missing = sorted(set(requested) - set(kept))
        if missing:
            logger.warning(f"Dropping {len(missing)} symbols not listed by the exchange: {missing}")
        if not kept:
            raise ValueError("None of the requested symbols are listed by the exchange.")
        return kept

    if mode not in ("top_n", "point_in_time"):
        raise ValueError(
            f"Unknown CRYPTO_UNIVERSE_MODE: {settings.universe_mode} (use 'top_n', 'point_in_time' or 'explicit')"
        )

    if mode == "point_in_time":
        volumes = adapter.quote_volumes_at(settings.start, settings.rank_lookback_days)
    else:
        logger.warning(
            "CRYPTO_UNIVERSE_MODE=top_n ranks by today's liquidity, so backtests over a "
            "past range suffer look-ahead survivorship bias; use 'point_in_time' (default) "
            "or an explicit curated universe for research."
        )
        volumes = adapter.quote_volumes()

    candidates = [symbol for symbol in volumes if _is_tradable_pair(symbol, quote)]
    if listed:
        candidates = [symbol for symbol in candidates if symbol in listed]
    ranked = sorted(candidates, key=lambda symbol: volumes[symbol], reverse=True)
    size = max(1, settings.universe_size)
    universe = ranked[:size]
    if not universe:
        raise RuntimeError("Could not build a universe from the exchange tickers.")
    logger.info(f"Resolved top-{len(universe)} universe by {mode} quote volume: {universe}")
    return universe


__all__ = ["resolve_universe", "EXCLUDED_BASES"]
