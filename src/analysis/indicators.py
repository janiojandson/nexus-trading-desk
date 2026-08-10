"""
Technical indicator calculations.
Pure Python implementations — no external TA library dependency for core indicators.
All functions operate on numpy arrays where index 0 is the oldest value.
"""

import numpy as np
from typing import NamedTuple


class IndicatorSnapshot(NamedTuple):
    """Complete indicator state at a single bar."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    ema_9: float
    ema_20: float
    ema_50: float
    ema_200: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr: float
    adx: float
    stoch_k: float
    stoch_d: float
    vwap: float


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    cumsum = np.cumsum(data)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    result[period - 1:] = cumsum[period - 1:] / period
    return result


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (Wilder's smoothing)."""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    k = 2.0 / (period + 1)
    # Seed with SMA
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result


def rma(data: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Relative Moving Average (used in RSI/ATR)."""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    result[period - 1] = np.mean(data[:period])
    alpha = 1.0 / period
    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder's method)."""
    result = np.full_like(close, np.nan, dtype=float)
    if len(close) < period + 1:
        return result

    changes = np.diff(close)
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)

    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)

    for i in range(period, len(avg_gain)):
        g = avg_gain[i]
        l = avg_loss[i]
        if np.isnan(g) or np.isnan(l):
            continue
        if l == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + g / l)

    return result


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def calculate_macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    # Signal line is EMA of MACD
    valid = ~np.isnan(macd_line)
    if not np.any(valid):
        empty = np.full_like(close, np.nan)
        return empty, empty.copy(), empty.copy()

    first_valid = np.argmax(valid)
    signal_line = np.full_like(close, np.nan)
    if first_valid < len(close):
        k = 2.0 / (signal + 1)
        signal_line[first_valid] = macd_line[first_valid]
        for i in range(first_valid + 1, len(close)):
            if not np.isnan(macd_line[i]):
                signal_line[i] = macd_line[i] * k + signal_line[i - 1] * (1 - k)

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def calculate_bollinger(
    close: np.ndarray, period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper, middle, lower Bollinger Bands."""
    middle = sma(close, period)
    std = np.full_like(close, np.nan)
    if len(close) < period:
        return std.copy(), middle, std.copy()

    for i in range(period - 1, len(close)):
        window = close[i - period + 1: i + 1]
        std[i] = np.std(window, ddof=0)

    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def calculate_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range (Wilder's method)."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]  # First bar has no prev close
    return rma(tr, period)


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

def calculate_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average Directional Index."""
    result = np.full_like(close, np.nan, dtype=float)
    if len(close) < period * 2:
        return result

    # True Range
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]

    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_vals = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / np.where(atr_vals > 0, atr_vals, 1)
    minus_di = 100 * rma(minus_dm, period) / np.where(atr_vals > 0, atr_vals, 1)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    result = rma(dx, period)

    return result


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

def calculate_stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    k_period: int = 14, d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator (%K and %D)."""
    k_line = np.full_like(close, np.nan, dtype=float)
    if len(close) < k_period:
        return k_line, k_line.copy()

    for i in range(k_period - 1, len(close)):
        window_high = high[i - k_period + 1: i + 1]
        window_low = low[i - k_period + 1: i + 1]
        hh = np.max(window_high)
        ll = np.min(window_low)
        if hh != ll:
            k_line[i] = 100 * (close[i] - ll) / (hh - ll)
        else:
            k_line[i] = 50.0

    d_line = sma(k_line, d_period)
    return k_line, d_line


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def calculate_vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Volume Weighted Average Price (cumulative session)."""
    typical = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical * volume)
    cum_vol = np.cumsum(volume)
    return np.where(cum_vol > 0, cum_tp_vol / cum_vol, typical)


# ---------------------------------------------------------------------------
# Compute All Indicators
# ---------------------------------------------------------------------------

def compute_indicators(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all indicators and return as a dict of arrays."""
    rsi_vals = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close)
    ema_9 = ema(close, 9)
    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    ema_200 = ema(close, 200)
    bb_upper, bb_middle, bb_lower = calculate_bollinger(close)
    atr_vals = calculate_atr(high, low, close, 14)
    adx_vals = calculate_adx(high, low, close, 14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    vwap_vals = calculate_vwap(high, low, close, volume)

    return {
        "rsi": rsi_vals,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
        "ema_9": ema_9,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "atr": atr_vals,
        "adx": adx_vals,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "vwap": vwap_vals,
    }


def snapshot_at(indicators: dict[str, np.ndarray], index: int) -> IndicatorSnapshot:
    """Extract indicator values at a specific bar index."""
    def _val(arr, idx, default=0.0):
        v = arr[idx] if idx < len(arr) else default
        return float(v) if not np.isnan(v) else default

    return IndicatorSnapshot(
        rsi=_val(indicators["rsi"], index, 50.0),
        macd=_val(indicators["macd"], index),
        macd_signal=_val(indicators["macd_signal"], index),
        macd_histogram=_val(indicators["macd_histogram"], index),
        ema_9=_val(indicators["ema_9"], index),
        ema_20=_val(indicators["ema_20"], index),
        ema_50=_val(indicators["ema_50"], index),
        ema_200=_val(indicators["ema_200"], index),
        bb_upper=_val(indicators["bb_upper"], index),
        bb_middle=_val(indicators["bb_middle"], index),
        bb_lower=_val(indicators["bb_lower"], index),
        atr=_val(indicators["atr"], index),
        adx=_val(indicators["adx"], index, 25.0),
        stoch_k=_val(indicators["stoch_k"], index, 50.0),
        stoch_d=_val(indicators["stoch_d"], index, 50.0),
        vwap=_val(indicators["vwap"], index),
    )