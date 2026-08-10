"""
Candlestick pattern recognition.
Every detector returns a reliability score (0-1) rather than a bare boolean.
Contextual reliability adjusts for location (at key level vs. mid-range).

Ported from QuantDesk's analysis/patterns.ts.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.analysis.smc import Direction


class PatternName(str, Enum):
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"


@dataclass
class CandlestickPattern:
    """Detected candlestick pattern with reliability score."""
    index: int
    name: PatternName
    direction: Direction
    reliability: float  # 0-1, how well the geometry matches + context
    reasons: list[str]


def _anatomy(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray):
    """Compute per-candle anatomy metrics."""
    body = np.abs(close - open_)
    range_ = high - low
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    body_ratio = np.where(range_ > 0, body / range_, 0)
    bullish = close > open_
    bearish = close < open_
    return body, range_, upper_wick, lower_wick, body_ratio, bullish, bearish


def _prior_trend(close: np.ndarray, index: int, lookback: int = 8) -> Direction:
    """Determine prior trend direction."""
    start = max(0, index - lookback)
    if start >= index:
        return Direction.NEUTRAL
    change = (close[index - 1] - close[start]) / close[start] * 100
    if change > 0.4:
        return Direction.BULLISH
    if change < -0.4:
        return Direction.BEARISH
    return Direction.NEUTRAL


def detect_patterns(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> list[CandlestickPattern]:
    """Detect candlestick patterns on the most recent bars."""
    patterns: list[CandlestickPattern] = []
    n = len(close)
    if n < 3:
        return patterns

    body, range_, upper_wick, lower_wick, body_ratio, bullish, bearish = _anatomy(open_, high, low, close)

    # Only check last 20 bars — older patterns are not actionable
    start = max(2, n - 20)

    for i in range(start, n):
        # Skip zero-range candles
        if range_[i] <= 0:
            continue

        prior = _prior_trend(close, i)

        # --- Engulfing ---
        if i >= 1 and range_[i] > 0 and range_[i - 1] > 0:
            # Bullish engulfing
            if (bearish[i - 1] and bullish[i] and
                close[i] > open_[i - 1] and open_[i] < close[i - 1] and
                body_ratio[i] > 0.5):
                reliability = min(1.0, body_ratio[i] * 1.2)
                if prior == Direction.BEARISH:
                    reliability = min(1.0, reliability * 1.3)
                patterns.append(CandlestickPattern(
                    index=i, name=PatternName.BULLISH_ENGULFING,
                    direction=Direction.BULLISH, reliability=reliability,
                    reasons=["body-engulfs-previous", "prior-downtrend" if prior == Direction.BEARISH else ""],
                ))

            # Bearish engulfing
            if (bullish[i - 1] and bearish[i] and
                close[i] < open_[i - 1] and open_[i] > close[i - 1] and
                body_ratio[i] > 0.5):
                reliability = min(1.0, body_ratio[i] * 1.2)
                if prior == Direction.BULLISH:
                    reliability = min(1.0, reliability * 1.3)
                patterns.append(CandlestickPattern(
                    index=i, name=PatternName.BEARISH_ENGULFING,
                    direction=Direction.BEARISH, reliability=reliability,
                    reasons=["body-engulfs-previous", "prior-uptrend" if prior == Direction.BULLISH else ""],
                ))

        # --- Hammer ---
        if (lower_wick[i] > body[i] * 2 and
            upper_wick[i] < body[i] * 0.5 and
            body_ratio[i] < 0.4 and
            prior in (Direction.BEARISH, Direction.NEUTRAL)):
            reliability = min(1.0, lower_wick[i] / (body[i] + 1) / 3)
            patterns.append(CandlestickPattern(
                index=i, name=PatternName.HAMMER,
                direction=Direction.BULLISH, reliability=reliability,
                reasons=["long-lower-wick", "small-body", "prior-downtrend"],
            ))

        # --- Shooting Star ---
        if (upper_wick[i] > body[i] * 2 and
            lower_wick[i] < body[i] * 0.5 and
            body_ratio[i] < 0.4 and
            prior in (Direction.BULLISH, Direction.NEUTRAL)):
            reliability = min(1.0, upper_wick[i] / (body[i] + 1) / 3)
            patterns.append(CandlestickPattern(
                index=i, name=PatternName.SHOOTING_STAR,
                direction=Direction.BEARISH, reliability=reliability,
                reasons=["long-upper-wick", "small-body", "prior-uptrend"],
            ))

        # --- Doji ---
        if body_ratio[i] < 0.1 and range_[i] > 0:
            reliability = 1.0 - body_ratio[i] * 10
            patterns.append(CandlestickPattern(
                index=i, name=PatternName.DOJI,
                direction=Direction.NEUTRAL, reliability=reliability,
                reasons=["body-ratio-below-10pct"],
            ))

        # --- Morning/Evening Star (3-candle) ---
        if i >= 2:
            # Morning star: bearish -> small body -> bullish
            if (bearish[i - 2] and body_ratio[i - 1] < 0.3 and bullish[i] and
                close[i] > (open_[i - 2] + close[i - 2]) / 2):
                reliability = 0.7
                if prior == Direction.BEARISH:
                    reliability = 0.9
                patterns.append(CandlestickPattern(
                    index=i, name=PatternName.MORNING_STAR,
                    direction=Direction.BULLISH, reliability=reliability,
                    reasons=["3-candle-reversal", "prior-downtrend"],
                ))

            # Evening star: bullish -> small body -> bearish
            if (bullish[i - 2] and body_ratio[i - 1] < 0.3 and bearish[i] and
                close[i] < (open_[i - 2] + close[i - 2]) / 2):
                reliability = 0.7
                if prior == Direction.BULLISH:
                    reliability = 0.9
                patterns.append(CandlestickPattern(
                    index=i, name=PatternName.EVENING_STAR,
                    direction=Direction.BEARISH, reliability=reliability,
                    reasons=["3-candle-reversal", "prior-uptrend"],
                ))

    return patterns