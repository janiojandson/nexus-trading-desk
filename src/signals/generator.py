"""
Signal generation engine.

Orchestrates the full analysis pipeline:
1. Compute indicators
2. Run SMC/ICT analysis
3. Detect candlestick patterns
4. Classify regime
5. Run strategy preset
6. Score confluence
7. Apply risk filters
8. Emit or reject signal

Ported from QuantDesk's analysis/signal.ts — "The default answer is WAIT."
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.analysis.indicators import compute_indicators, snapshot_at
from src.analysis.smc import SmcDetector, Direction
from src.analysis.patterns import detect_patterns
from src.analysis.regime import RegimeClassifier, RegimeResult
from src.analysis.confluence import compute_confluence, ConfluenceBreakdown
from src.signals.presets import get_strategy, EntrySignal, StrategyId


@dataclass
class Signal:
    """A complete trading signal."""
    timestamp: int
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry: float
    stop_loss: float
    take_profits: list[float]  # [1:1, 1:2, 1:3] R:R targets
    confidence: float  # 0-100
    regime: str
    reasons: list[str]
    wait_reason: str | None  # Why signal was rejected (if WAIT)
    confluence: ConfluenceBreakdown | None
    preset: str


class SignalGenerator:
    """
    Signal generation engine.

    Everything is deterministic. Entry, stop, targets, risk-reward,
    confidence and probability are all computed from the analysis.
    The default answer is WAIT.
    """

    def __init__(
        self,
        confluence_threshold: int = 65,
        min_confidence: float = 58.0,
        tp_r_multiple: float = 2.0,
        sl_atr_multiple: float = 1.5,
        preset: str = "regime-aware",
        swing_strength: int = 3,
    ):
        self.confluence_threshold = confluence_threshold
        self.min_confidence = min_confidence
        self.tp_r_multiple = tp_r_multiple
        self.sl_atr_multiple = sl_atr_multiple
        self.preset = preset
        self.smc_detector = SmcDetector(swing_strength=swing_strength)
        self.regime_classifier = RegimeClassifier()
        self.strategy_fn = get_strategy(preset)

    def generate(
        self,
        symbol: str,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        higher_tf_trend: str = "neutral",
    ) -> list[Signal]:
        """
        Generate trading signals for a symbol.

        Returns a list of Signal objects. Each signal is either actionable
        (wait_reason=None) or a WAIT with the rejection reason.
        """
        n = len(close)
        if n < 50:
            return []

        # 1. Compute indicators
        indicators = compute_indicators(open_, high, low, close, volume)

        # 2. SMC/ICT analysis
        atr_arr = indicators["atr"]
        smc = self.smc_detector.analyze(open_, high, low, close, atr_arr)

        # 3. Detect patterns
        patterns = detect_patterns(open_, high, low, close)

        # 4. Classify regime
        regime = self.regime_classifier.classify(high, low, close)

        # 5. Run strategy preset
        ctx = {"regime": regime, "symbol": symbol}
        entry_signals = self.strategy_fn(indicators, close, high, low, ctx)

        # 6. Process each entry signal through confluence + risk
        signals: list[Signal] = []
        now_ms = int(time.time() * 1000)

        for entry in entry_signals[-5:]:  # Only process last 5 signals
            i = entry.bar_index
            direction = Direction.BULLISH if entry.direction == "BUY" else Direction.BEARISH

            # Get indicator snapshot at this bar
            snap = snapshot_at(indicators, i)

            # 6a. Compute confluence
            confluence = compute_confluence(
                direction=direction,
                smc=smc,
                patterns=patterns,
                indicators=snap,
                regime=regime,
                higher_tf_trend=higher_tf_trend,
            )

            # 6b. Check confluence threshold
            if confluence.total < self.confluence_threshold:
                signals.append(Signal(
                    timestamp=now_ms,
                    symbol=symbol,
                    direction=entry.direction,
                    entry=entry.price,
                    stop_loss=0,
                    take_profits=[],
                    confidence=confluence.total,
                    regime=regime.regime.value,
                    reasons=entry.reasons,
                    wait_reason=f"confluence-below-threshold ({confluence.total:.0f} < {self.confluence_threshold})",
                    confluence=confluence,
                    preset=self.preset,
                ))
                continue

            # 6c. Compute stop loss and take profits
            atr_val = snap.atr if snap.atr > 0 else (high[i] - low[i]) if i < len(high) else 1.0
            risk = atr_val * self.sl_atr_multiple

            if entry.direction == "BUY":
                stop_loss = entry.price - risk
                tp1 = entry.price + risk * 1.0
                tp2 = entry.price + risk * 2.0
                tp3 = entry.price + risk * self.tp_r_multiple
            else:
                stop_loss = entry.price + risk
                tp1 = entry.price - risk * 1.0
                tp2 = entry.price - risk * 2.0
                tp3 = entry.price - risk * self.tp_r_multiple

            # 6d. Check minimum confidence
            signal_confidence = confluence.total
            if signal_confidence < self.min_confidence:
                signals.append(Signal(
                    timestamp=now_ms,
                    symbol=symbol,
                    direction=entry.direction,
                    entry=entry.price,
                    stop_loss=stop_loss,
                    take_profits=[tp1, tp2, tp3],
                    confidence=signal_confidence,
                    regime=regime.regime.value,
                    reasons=entry.reasons,
                    wait_reason=f"confidence-below-minimum ({signal_confidence:.0f} < {self.min_confidence})",
                    confluence=confluence,
                    preset=self.preset,
                ))
                continue

            # 6e. Emit actionable signal
            signals.append(Signal(
                timestamp=now_ms,
                symbol=symbol,
                direction=entry.direction,
                entry=entry.price,
                stop_loss=stop_loss,
                take_profits=[tp1, tp2, tp3],
                confidence=signal_confidence,
                regime=regime.regime.value,
                reasons=entry.reasons + confluence.top_factors,
                wait_reason=None,
                confluence=confluence,
                preset=self.preset,
            ))

        return signals