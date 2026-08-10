"""
Market regime classifier.
Classifies market state as Trend, Volatile, or Range using
ADX, ATR, and Bollinger Band width.

Inspired by TradeClaw's HMM regime classifier with Viterbi decoding.
This implementation uses a simpler structural approach for the prototype.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.analysis.indicators import calculate_adx, calculate_atr, calculate_bollinger


class Regime(str, Enum):
    TREND = "trend"
    VOLATILE = "volatile"
    RANGE = "range"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    """Regime classification result."""
    regime: Regime
    confidence: float  # 0-1
    features: dict  # Raw feature values used for classification
    allowed_directions: list[str]  # "BUY", "SELL", or both


# Regime allocation rules (from TradeClaw)
REGIME_ALLOCATION_RULES = {
    Regime.TREND: {
        "allowed_directions": ["BUY", "SELL"],
        "position_scale": 1.0,
        "description": "Trending market — follow momentum",
    },
    Regime.VOLATILE: {
        "allowed_directions": ["BUY", "SELL"],
        "position_scale": 0.5,
        "description": "Volatile market — reduce size, mean-revert",
    },
    Regime.RANGE: {
        "allowed_directions": ["BUY", "SELL"],
        "position_scale": 0.75,
        "description": "Ranging market — fade extremes",
    },
    Regime.UNKNOWN: {
        "allowed_directions": ["BUY", "SELL"],
        "position_scale": 0.25,
        "description": "Unknown regime — minimal exposure",
    },
}


class RegimeClassifier:
    """
    Structural regime classifier.

    Uses ADX for trend strength, ATR percentile for volatility,
    and Bollinger Band width for range detection.
    """

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        atr_volatility_percentile: float = 80.0,
        bb_squeeze_threshold: float = 0.02,
        lookback: int = 60,
    ):
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.atr_volatility_percentile = atr_volatility_percentile
        self.bb_squeeze_threshold = bb_squeeze_threshold
        self.lookback = lookback

    def classify(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> RegimeResult:
        """
        Classify the current market regime.

        Args:
            high, low, close: Price arrays

        Returns:
            RegimeResult with regime, confidence, features, and allowed directions
        """
        n = len(close)
        if n < 30:
            return RegimeResult(
                regime=Regime.UNKNOWN,
                confidence=0.0,
                features={},
                allowed_directions=["BUY", "SELL"],
            )

        # Compute features
        adx_vals = calculate_adx(high, low, close, 14)
        atr_vals = calculate_atr(high, low, close, 14)
        bb_upper, bb_middle, bb_lower = calculate_bollinger(close, 20, 2.0)

        # Current values
        adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 25.0
        atr = atr_vals[-1] if not np.isnan(atr_vals[-1]) else 0.0
        price = close[-1]

        # ATR percentile
        recent_atr = atr_vals[-self.lookback:]
        valid_atr = recent_atr[~np.isnan(recent_atr)]
        atr_percentile = 0.0
        if len(valid_atr) > 0 and atr > 0:
            atr_percentile = np.sum(valid_atr <= atr) / len(valid_atr) * 100

        # BB width
        bb_width = 0.0
        if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]) and bb_middle[-1] > 0:
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]

        # EMA trend direction
        from src.analysis.indicators import ema
        ema_20 = ema(close, 20)
        ema_50 = ema(close, 50)
        ema_trend = "neutral"
        if not np.isnan(ema_20[-1]) and not np.isnan(ema_50[-1]):
            if price > ema_20[-1] and ema_20[-1] > ema_50[-1]:
                ema_trend = "uptrend"
            elif price < ema_20[-1] and ema_20[-1] < ema_50[-1]:
                ema_trend = "downtrend"

        features = {
            "adx": float(adx),
            "atr_percentile": float(atr_percentile),
            "bb_width": float(bb_width),
            "ema_trend": ema_trend,
        }

        # Classification logic
        regime = Regime.UNKNOWN
        confidence = 0.5

        if adx > self.adx_trend_threshold:
            regime = Regime.TREND
            confidence = min(1.0, (adx - self.adx_trend_threshold) / 20.0 + 0.5)
        elif adx < self.adx_range_threshold and bb_width < self.bb_squeeze_threshold:
            regime = Regime.RANGE
            confidence = min(1.0, (self.adx_range_threshold - adx) / 15.0 + 0.5)
        elif atr_percentile > self.atr_volatility_percentile:
            regime = Regime.VOLATILE
            confidence = min(1.0, (atr_percentile - self.atr_volatility_percentile) / 15.0 + 0.5)
        elif adx < self.adx_range_threshold:
            regime = Regime.RANGE
            confidence = 0.6
        else:
            # Between range and trend — default to trend with low confidence
            regime = Regime.TREND
            confidence = 0.4

        rules = REGIME_ALLOCATION_RULES[regime]

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            features=features,
            allowed_directions=rules["allowed_directions"],
        )