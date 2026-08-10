"""
Strategy presets — 5 swappable entry strategies.

1. Classic — RSI + MACD + EMA + Stochastic + BB (TradeClaw baseline)
2. HMM Top-3 — Regime-classified with HMM
3. Regime-Aware — Classic gated by regime classifier
4. VWAP+EMA+BB — Intraday mean-reversion
5. Full-Risk Pipeline — All signals + circuit breaker + drawdown tracker

Ported from TradeClaw's packages/strategies/src/entry/.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from src.analysis.indicators import IndicatorSnapshot


class StrategyId(str, Enum):
    CLASSIC = "classic"
    HMM_TOP3 = "hmm-top3"
    REGIME_AWARE = "regime-aware"
    VWAP_EMA_BB = "vwap-ema-bb"
    FULL_RISK = "full-risk"


@dataclass
class EntrySignal:
    """A generated entry signal."""
    bar_index: int
    direction: str  # "BUY" or "SELL"
    price: float
    score: float  # 0-100
    confidence: float  # 0-1
    reasons: list[str]
    strategy_id: StrategyId


class EntryModule(Protocol):
    """Protocol for strategy entry modules."""
    id: StrategyId

    def generate_signals(
        self,
        indicators: dict[str, np.ndarray],
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        ctx: dict | None = None,
    ) -> list[EntrySignal]:
        ...


# ---------------------------------------------------------------------------
# Classic Entry — RSI + MACD + EMA + Stochastic + BB
# ---------------------------------------------------------------------------

CLASSIC_WEIGHTS = {
    "RSI_OVERSOLD": 20, "RSI_OVERBOUGHT": 20,
    "MACD_BULLISH": 25, "MACD_BEARISH": 25,
    "EMA_TREND_UP": 20, "EMA_TREND_DOWN": 20,
    "STOCH_OVERSOLD": 15, "STOCH_OVERBOUGHT": 15,
    "BB_LOWER_TOUCH": 10, "BB_UPPER_TOUCH": 10,
}

SIGNAL_THRESHOLD = 25
MIN_CONFIDENCE = 0.58


def classic_entry(
    indicators: dict[str, np.ndarray],
    close: np.ndarray,
    high: np.ndarray = None,
    low: np.ndarray = None,
    ctx: dict | None = None,
) -> list[EntrySignal]:
    """
    Classic entry: baseline scoring using RSI + MACD + EMA + Stochastic + BB.
    Reproduces TradeClaw's pre-regime-filter signal generator behavior.
    """
    signals: list[EntrySignal] = []
    n = len(close)

    rsi = indicators.get("rsi", np.full(n, 50.0))
    macd_hist = indicators.get("macd_histogram", np.zeros(n))
    ema_20 = indicators.get("ema_20", close.copy())
    ema_50 = indicators.get("ema_50", close.copy())
    stoch_k = indicators.get("stoch_k", np.full(n, 50.0))
    stoch_d = indicators.get("stoch_d", np.full(n, 50.0))
    bb_upper = indicators.get("bb_upper", np.full(n, np.nan))
    bb_lower = indicators.get("bb_lower", np.full(n, np.nan))

    for i in range(1, n):
        buy_score = 0.0
        sell_score = 0.0
        reasons: list[str] = []

        r = rsi[i] if not np.isnan(rsi[i]) else 50.0
        h = macd_hist[i] if not np.isnan(macd_hist[i]) else 0.0
        prev_h = macd_hist[i - 1] if not np.isnan(macd_hist[i - 1]) else 0.0
        e20 = ema_20[i] if not np.isnan(ema_20[i]) else close[i]
        e50 = ema_50[i] if not np.isnan(ema_50[i]) else close[i]
        sk = stoch_k[i] if not np.isnan(stoch_k[i]) else 50.0
        sd = stoch_d[i] if not np.isnan(stoch_d[i]) else 50.0
        bbu = bb_upper[i] if not np.isnan(bb_upper[i]) else 0.0
        bbl = bb_lower[i] if not np.isnan(bb_lower[i]) else 0.0
        price = close[i]

        # RSI
        if r < 30:
            buy_score += CLASSIC_WEIGHTS["RSI_OVERSOLD"]
            reasons.append("rsi-oversold")
        elif r < 40:
            buy_score += CLASSIC_WEIGHTS["RSI_OVERSOLD"] * 0.5
            reasons.append("rsi-near-oversold")
        elif r > 70:
            sell_score += CLASSIC_WEIGHTS["RSI_OVERBOUGHT"]
            reasons.append("rsi-overbought")
        elif r > 60:
            sell_score += CLASSIC_WEIGHTS["RSI_OVERBOUGHT"] * 0.5
            reasons.append("rsi-near-overbought")

        # MACD histogram
        if h > 0:
            if prev_h <= 0:
                buy_score += CLASSIC_WEIGHTS["MACD_BULLISH"]
                reasons.append("macd-bullish-crossover")
            else:
                buy_score += CLASSIC_WEIGHTS["MACD_BULLISH"] * 0.5
                reasons.append("macd-bullish")
        elif h < 0:
            if prev_h >= 0:
                sell_score += CLASSIC_WEIGHTS["MACD_BEARISH"]
                reasons.append("macd-bearish-crossover")
            else:
                sell_score += CLASSIC_WEIGHTS["MACD_BEARISH"] * 0.5
                reasons.append("macd-bearish")

        # EMA trend
        if price > e20 and e20 > e50:
            buy_score += CLASSIC_WEIGHTS["EMA_TREND_UP"]
            reasons.append("ema-uptrend")
        elif price < e20 and e20 < e50:
            sell_score += CLASSIC_WEIGHTS["EMA_TREND_DOWN"]
            reasons.append("ema-downtrend")

        # Stochastic
        if sk < 20 and sd < 20:
            buy_score += CLASSIC_WEIGHTS["STOCH_OVERSOLD"]
            reasons.append("stoch-oversold")
        elif sk > 80 and sd > 80:
            sell_score += CLASSIC_WEIGHTS["STOCH_OVERBOUGHT"]
            reasons.append("stoch-overbought")

        # Bollinger Bands
        if bbl > 0 and price <= bbl:
            buy_score += CLASSIC_WEIGHTS["BB_LOWER_TOUCH"]
            reasons.append("bb-lower-touch")
        if bbu > 0 and price >= bbu:
            sell_score += CLASSIC_WEIGHTS["BB_UPPER_TOUCH"]
            reasons.append("bb-upper-touch")

        # Generate signal if threshold met
        if buy_score >= SIGNAL_THRESHOLD:
            confidence = min(1.0, buy_score / 100.0)
            if confidence >= MIN_CONFIDENCE:
                signals.append(EntrySignal(
                    bar_index=i, direction="BUY", price=price,
                    score=buy_score, confidence=confidence,
                    reasons=reasons, strategy_id=StrategyId.CLASSIC,
                ))

        if sell_score >= SIGNAL_THRESHOLD:
            confidence = min(1.0, sell_score / 100.0)
            if confidence >= MIN_CONFIDENCE:
                signals.append(EntrySignal(
                    bar_index=i, direction="SELL", price=price,
                    score=sell_score, confidence=confidence,
                    reasons=reasons, strategy_id=StrategyId.CLASSIC,
                ))

    return signals


# ---------------------------------------------------------------------------
# Regime-Aware Entry — Classic gated by regime classifier
# ---------------------------------------------------------------------------

def regime_aware_entry(
    indicators: dict[str, np.ndarray],
    close: np.ndarray,
    high: np.ndarray = None,
    low: np.ndarray = None,
    ctx: dict | None = None,
) -> list[EntrySignal]:
    """
    Regime-aware entry: classic signals gated by regime classifier.
    Signals whose direction is not allowed by the current regime are filtered out.
    """
    raw = classic_entry(indicators, close, high, low, ctx)
    if not raw or not ctx or "regime" not in ctx:
        return raw

    regime_result = ctx["regime"]
    return [
        sig for sig in raw
        if sig.direction in regime_result.allowed_directions
    ]


# ---------------------------------------------------------------------------
# VWAP+EMA+BB Entry — Intraday mean-reversion
# ---------------------------------------------------------------------------

def vwap_ema_bb_entry(
    indicators: dict[str, np.ndarray],
    close: np.ndarray,
    high: np.ndarray = None,
    low: np.ndarray = None,
    ctx: dict | None = None,
) -> list[EntrySignal]:
    """VWAP + EMA + BB mean-reversion entry."""
    signals: list[EntrySignal] = []
    n = len(close)

    ema_20 = indicators.get("ema_20", close.copy())
    bb_upper = indicators.get("bb_upper", np.full(n, np.nan))
    bb_lower = indicators.get("bb_lower", np.full(n, np.nan))
    vwap = indicators.get("vwap", close.copy())

    for i in range(1, n):
        price = close[i]
        e20 = ema_20[i] if not np.isnan(ema_20[i]) else price
        bbl = bb_lower[i] if not np.isnan(bb_lower[i]) else 0
        bbu = bb_upper[i] if not np.isnan(bb_upper[i]) else 0
        vw = vwap[i] if not np.isnan(vwap[i]) else price

        # Buy: price below VWAP and BB lower, bouncing
        if bbl > 0 and price <= bbl and price < vw:
            signals.append(EntrySignal(
                bar_index=i, direction="BUY", price=price,
                score=60.0, confidence=0.6,
                reasons=["vwap-below", "bb-lower-touch"],
                strategy_id=StrategyId.VWAP_EMA_BB,
            ))

        # Sell: price above VWAP and BB upper
        if bbu > 0 and price >= bbu and price > vw:
            signals.append(EntrySignal(
                bar_index=i, direction="SELL", price=price,
                score=60.0, confidence=0.6,
                reasons=["vwap-above", "bb-upper-touch"],
                strategy_id=StrategyId.VWAP_EMA_BB,
            ))

    return signals


# ---------------------------------------------------------------------------
# Strategy Router
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    StrategyId.CLASSIC: classic_entry,
    StrategyId.REGIME_AWARE: regime_aware_entry,
    StrategyId.VWAP_EMA_BB: vwap_ema_bb_entry,
    # HMM_TOP3 and FULL_RISK use classic as base with additional filters
    StrategyId.HMM_TOP3: classic_entry,
    StrategyId.FULL_RISK: classic_entry,
}


def get_strategy(preset: str) -> callable:
    """Get the entry function for a strategy preset."""
    try:
        sid = StrategyId(preset)
    except ValueError:
        sid = StrategyId.REGIME_AWARE
    return STRATEGY_MAP.get(sid, classic_entry)