"""
Confluence scorer — weighted scoring across analysis categories.

Combines SMC, Pattern, Indicator, Multi-Timeframe, and Regime signals
into a single confluence score (0-100).

Ported from QuantDesk's analysis/confluence.ts.
"""

from dataclasses import dataclass

from src.analysis.smc import SmcAnalysis, Direction, TrendDirection
from src.analysis.patterns import CandlestickPattern, PatternName
from src.analysis.regime import RegimeResult, Regime
from src.analysis.indicators import IndicatorSnapshot


@dataclass
class ConfluenceBreakdown:
    """Detailed confluence score breakdown."""
    smc_score: float  # 0-30
    pattern_score: float  # 0-20
    indicator_score: float  # 0-25
    mtf_score: float  # 0-15
    regime_score: float  # 0-10
    total: float  # 0-100
    top_factors: list[str]


# Weights
SMC_WEIGHT = 30.0
PATTERN_WEIGHT = 20.0
INDICATOR_WEIGHT = 25.0
MTF_WEIGHT = 15.0
REGIME_WEIGHT = 10.0


def score_smc(smc: SmcAnalysis, direction: Direction) -> tuple[float, list[str]]:
    """Score SMC confluence for a given direction."""
    score = 0.0
    factors: list[str] = []

    # Order block alignment
    for ob in smc.order_blocks[-3:]:
        if ob.direction == direction:
            score += 8.0
            factors.append(f"ob-{ob.direction.value}")
            break

    # FVG alignment
    for fvg in smc.fvgs[-3:]:
        if fvg.direction == direction and not fvg.filled:
            score += 6.0
            factors.append(f"fvg-{fvg.direction.value}")
            break

    # BOS alignment
    for brk in smc.breaks[-2:]:
        if (brk.direction == Direction.BULLISH and direction == Direction.BULLISH) or \
           (brk.direction == Direction.BEARISH and direction == Direction.BEARISH):
            score += 8.0
            factors.append(f"bos-{brk.direction.value}")
            break

    # Liquidity sweep (opposite direction = fuel for our direction)
    for sweep in smc.sweeps[-2:]:
        if sweep.is_high and direction == Direction.BULLISH:
            score += 5.0
            factors.append("liquidity-sweep-high")
            break
        if not sweep.is_high and direction == Direction.BEARISH:
            score += 5.0
            factors.append("liquidity-sweep-low")
            break

    # Trend alignment
    if smc.trend == TrendDirection.UPTREND and direction == Direction.BULLISH:
        score += 3.0
        factors.append("trend-uptrend")
    elif smc.trend == TrendDirection.DOWNTREND and direction == Direction.BEARISH:
        score += 3.0
        factors.append("trend-downtrend")

    return min(score, SMC_WEIGHT), factors


def score_patterns(
    patterns: list[CandlestickPattern], direction: Direction
) -> tuple[float, list[str]]:
    """Score pattern confluence."""
    score = 0.0
    factors: list[str] = []

    for p in patterns[-3:]:
        if p.direction == direction or p.direction == Direction.NEUTRAL:
            score += p.reliability * 10.0
            factors.append(f"{p.name.value}(r={p.reliability:.2f})")

    return min(score, PATTERN_WEIGHT), factors


def score_indicators(
    indicators: IndicatorSnapshot, direction: Direction
) -> tuple[float, list[str]]:
    """Score indicator confluence."""
    score = 0.0
    factors: list[str] = []

    # RSI
    if direction == Direction.BULLISH:
        if indicators.rsi < 30:
            score += 6.0
            factors.append("rsi-oversold")
        elif indicators.rsi < 40:
            score += 3.0
            factors.append("rsi-near-oversold")
    else:
        if indicators.rsi > 70:
            score += 6.0
            factors.append("rsi-overbought")
        elif indicators.rsi > 60:
            score += 3.0
            factors.append("rsi-near-overbought")

    # MACD
    if direction == Direction.BULLISH and indicators.macd_histogram > 0:
        score += 5.0
        factors.append("macd-bullish")
    elif direction == Direction.BEARISH and indicators.macd_histogram < 0:
        score += 5.0
        factors.append("macd-bearish")

    # EMA trend
    if direction == Direction.BULLISH:
        if indicators.ema_20 > indicators.ema_50:
            score += 5.0
            factors.append("ema-uptrend")
    else:
        if indicators.ema_20 < indicators.ema_50:
            score += 5.0
            factors.append("ema-downtrend")

    # Bollinger Bands
    if direction == Direction.BULLISH and indicators.rsi < 50:
        if indicators.bb_lower > 0 and indicators.rsi < 40:
            score += 4.0
            factors.append("bb-lower-touch")
    elif direction == Direction.BEARISH and indicators.rsi > 50:
        if indicators.bb_upper > 0 and indicators.rsi > 60:
            score += 4.0
            factors.append("bb-upper-touch")

    # Stochastic
    if direction == Direction.BULLISH and indicators.stoch_k < 20:
        score += 3.0
        factors.append("stoch-oversold")
    elif direction == Direction.BEARISH and indicators.stoch_k > 80:
        score += 3.0
        factors.append("stoch-overbought")

    # ADX
    if indicators.adx > 25:
        score += 2.0
        factors.append("adx-trending")

    return min(score, INDICATOR_WEIGHT), factors


def score_mtf(
    higher_tf_trend: str, direction: Direction
) -> tuple[float, list[str]]:
    """Score multi-timeframe confluence."""
    score = 0.0
    factors: list[str] = []

    if higher_tf_trend == "uptrend" and direction == Direction.BULLISH:
        score = MTF_WEIGHT
        factors.append("htf-uptrend-aligned")
    elif higher_tf_trend == "downtrend" and direction == Direction.BEARISH:
        score = MTF_WEIGHT
        factors.append("htf-downtrend-aligned")
    elif higher_tf_trend == "neutral":
        score = MTF_WEIGHT * 0.3
        factors.append("htf-neutral")
    else:
        score = 0.0
        factors.append("htf-misaligned")

    return score, factors


def score_regime(
    regime: RegimeResult, direction: Direction
) -> tuple[float, list[str]]:
    """Score regime alignment."""
    score = 0.0
    factors: list[str] = []

    if direction.value.upper() in regime.allowed_directions:
        score = REGIME_WEIGHT * regime.confidence
        factors.append(f"regime-{regime.regime.value}-aligned")
    else:
        score = 0.0
        factors.append(f"regime-{regime.regime.value}-blocked")

    return score, factors


def compute_confluence(
    direction: Direction,
    smc: SmcAnalysis,
    patterns: list[CandlestickPattern],
    indicators: IndicatorSnapshot,
    regime: RegimeResult,
    higher_tf_trend: str = "neutral",
) -> ConfluenceBreakdown:
    """
    Compute full confluence score for a given direction.

    Returns a ConfluenceBreakdown with category scores and total (0-100).
    """
    smc_score, smc_factors = score_smc(smc, direction)
    pattern_score, pattern_factors = score_patterns(patterns, direction)
    indicator_score, indicator_factors = score_indicators(indicators, direction)
    mtf_score, mtf_factors = score_mtf(higher_tf_trend, direction)
    regime_score, regime_factors = score_regime(regime, direction)

    total = smc_score + pattern_score + indicator_score + mtf_score + regime_score
    all_factors = smc_factors + pattern_factors + indicator_factors + mtf_factors + regime_factors

    return ConfluenceBreakdown(
        smc_score=smc_score,
        pattern_score=pattern_score,
        indicator_score=indicator_score,
        mtf_score=mtf_score,
        regime_score=regime_score,
        total=total,
        top_factors=all_factors[:5],  # Top 5 factors
    )