"""
Multi-exchange adapter using ccxt.
Supports Binance, Bybit, OKX with rate limiting and caching.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import ccxt.async_support as ccxt

from src.config import Config


@dataclass
class OHLCV:
    """Single OHLCV candle."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class RateLimiter:
    """Simple rate limiter per exchange."""
    min_interval: float = 0.1  # seconds between requests
    _last_request: float = field(default=0.0, repr=False)

    async def wait(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()


class ExchangeAdapter:
    """
    Unified multi-exchange adapter.
    Fetches OHLCV data from Binance, Bybit, or OKX via ccxt.
    """

    EXCHANGE_CLASSES = {
        "binance": ccxt.binance,
        "bybit": ccxt.bybit,
        "okx": ccxt.okx,
    }

    def __init__(self, config: Config):
        self.config = config
        self._exchanges: dict[str, ccxt.Exchange] = {}
        self._limiters: dict[str, RateLimiter] = {}
        self._init_exchanges()

    def _init_exchanges(self):
        """Initialize enabled exchanges from config."""
        for name, settings in self.config.exchanges.items():
            if not settings.get("enabled", False):
                continue
            cls = self.EXCHANGE_CLASSES.get(name)
            if not cls:
                continue

            opts: dict[str, Any] = {"enableRateLimit": True}
            if settings.get("testnet"):
                opts["sandbox"] = True

            # Load API keys from env
            key_env = name.upper()
            api_key = getattr(self.config._env, f"get", lambda k, d="": d)(f"{key_env}_API_KEY", "")
            api_secret = getattr(self.config._env, f"get", lambda k, d="": d)(f"{key_env}_API_SECRET", "")
            if api_key:
                opts["apiKey"] = api_key
            if api_secret:
                opts["secret"] = api_secret

            self._exchanges[name] = cls(opts)
            self._limiters[name] = RateLimiter(min_interval=0.2)

    @property
    def primary(self) -> str:
        """Return the first enabled exchange name."""
        for name in self._exchanges:
            return name
        return "binance"

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        exchange: str | None = None,
    ) -> list[OHLCV]:
        """
        Fetch OHLCV candles from an exchange.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"
            timeframe: Candle interval, e.g. "1h", "5m", "4h"
            limit: Number of candles to fetch
            exchange: Specific exchange name, or None for primary

        Returns:
            List of OHLCV candles, oldest first
        """
        ex_name = exchange or self.primary
        ex = self._exchanges.get(ex_name)
        if not ex:
            raise ValueError(f"Exchange {ex_name} not initialized")

        limiter = self._limiters[ex_name]
        await limiter.wait()

        try:
            raw = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [
                OHLCV(
                    timestamp=int(c[0]),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                )
                for c in raw
            ]
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching {symbol} from {ex_name}: {e}") from e
        except ccxt.ExchangeError as e:
            raise ValueError(f"Exchange error fetching {symbol} from {ex_name}: {e}") from e

    async def fetch_ticker(self, symbol: str, exchange: str | None = None) -> dict:
        """Fetch current ticker for a symbol."""
        ex_name = exchange or self.primary
        ex = self._exchanges.get(ex_name)
        if not ex:
            raise ValueError(f"Exchange {ex_name} not initialized")

        await self._limiters[ex_name].wait()
        return await ex.fetch_ticker(symbol)

    async def close(self):
        """Close all exchange connections."""
        for ex in self._exchanges.values():
            await ex.close()
        self._exchanges.clear()

    def ohlcv_to_dicts(self, candles: list[OHLCV]) -> list[dict]:
        """Convert OHLCV objects to dicts for DataFrame construction."""
        return [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]