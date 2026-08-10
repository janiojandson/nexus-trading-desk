"""
Smart Money Concepts / ICT analysis module.

Implements structural reading of price action:
- Swing Points (fractal pivots with configurable strength)
- Break of Structure (BOS) — close-based, not wick
- Order Blocks — last opposing candle before impulsive move
- Fair Value Gaps (FVG) — 3-candle gaps
- Liquidity Pools — clusters of equal highs/lows
- Liquidity Sweeps — wick beyond pool without close confirmation
- Supply/Demand Zones

Ported from QuantDesk's analysis/smc.ts with Python adaptations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TrendDirection(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    NEUTRAL = "neutral"


@dataclass
class SwingPoint:
    """A confirmed swing pivot."""
    index: int
    price: float
    is_high: bool  # True = swing high, False = swing low
    label: str  # HH, LH, HL, LL


@dataclass
class OrderBlock:
    """Institutional order block."""
    index: int
    high: float
    low: float
    direction: Direction  # Bullish OB = last bearish candle before up move
    strength: float  # Move size relative to ATR


@dataclass
class FairValueGap:
    """3-candle gap where body extremes don't overlap."""
    index: int  # Middle candle index
    top: float
    bottom: float
    direction: Direction
    filled: bool = False


@dataclass
class LiquidityPool:
    """Cluster of equal highs or lows."""
    level: float
    is_high: bool  # True = equal highs, False = equal lows
    count: int  # Number of touches
    indices: list[int] = field(default_factory=list)


@dataclass
class LiquiditySweep:
    """Wick beyond a liquidity pool without close confirmation."""
    index: int
    pool_level: float
    is_high: bool
    sweep_high: float
    close: float  # Close returned below/above the pool


@dataclass
class StructureBreak:
    """Break of structure event."""
    index: int
    direction: Direction  # Bullish BOS = close above swing high
    swing_point: SwingPoint
    break_price: float


@dataclass
class SupplyDemandZone:
    """Supply or demand zone."""
    index: int
    high: float
    low: float
    direction: Direction  # Demand = bullish, Supply = bearish
    strength: float


@dataclass
class SmcAnalysis:
    """Complete SMC analysis result."""
    swings: list[SwingPoint]
    order_blocks: list[OrderBlock]
    fvgs: list[FairValueGap]
    liquidity_pools: list[LiquidityPool]
    sweeps: list[LiquiditySweep]
    breaks: list[StructureBreak]
    zones: list[SupplyDemandZone]
    trend: TrendDirection


class SmcDetector:
    """
    SMC/ICT pattern detector.

    Two key design decisions (from QuantDesk):
    1. Swings are CONFIRMED, never provisional — a pivot requires `strength`
       bars on both sides, preventing repaint.
    2. Breaks require a CLOSE, not a wick — a wick piercing a swing high
       and closing back below is a liquidity sweep, not a BOS.
    """

    def __init__(self, swing_strength: int = 3, max_per_kind: int = 12):
        self.swing_strength = swing_strength
        self.max_per_kind = max_per_kind

    def analyze(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray | None = None,
    ) -> SmcAnalysis:
        """Run full SMC analysis on OHLC data."""
        swings = self._detect_swings(high, low, close)
        breaks = self._detect_breaks(swings, close)
        order_blocks = self._detect_order_blocks(open_, high, low, close, atr)
        fvgs = self._detect_fvgs(open_, close)
        pools = self._detect_liquidity_pools(high, low)
        sweeps = self._detect_sweeps(pools, high, low, close)
        zones = self._detect_supply_demand(open_, high, low, close, atr)
        trend = self._determine_trend(swings, breaks)

        return SmcAnalysis(
            swings=swings[-self.max_per_kind:],
            order_blocks=order_blocks[-self.max_per_kind:],
            fvgs=fvgs[-self.max_per_kind:],
            liquidity_pools=pools[-self.max_per_kind:],
            sweeps=sweeps[-self.max_per_kind:],
            breaks=breaks[-self.max_per_kind:],
            zones=zones[-self.max_per_kind:],
            trend=trend,
        )

    def _detect_swings(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> list[SwingPoint]:
        """Detect confirmed swing pivots with fractal method."""
        s = self.swing_strength
        swings: list[SwingPoint] = []
        if len(close) < s * 2 + 1:
            return swings

        prev_high_price: float | None = None
        prev_low_price: float | None = None

        for i in range(s, len(close) - s):
            # Check swing high
            is_swing_high = True
            for j in range(i - s, i + s + 1):
                if j != i and high[j] >= high[i]:
                    is_swing_high = False
                    break

            if is_swing_high:
                label = "HH" if prev_high_price and high[i] > prev_high_price else "LH"
                swings.append(SwingPoint(index=i, price=high[i], is_high=True, label=label))
                prev_high_price = high[i]

            # Check swing low
            is_swing_low = True
            for j in range(i - s, i + s + 1):
                if j != i and low[j] <= low[i]:
                    is_swing_low = False
                    break

            if is_swing_low:
                label = "HL" if prev_low_price and low[i] > prev_low_price else "LL"
                swings.append(SwingPoint(index=i, price=low[i], is_high=False, label=label))
                prev_low_price = low[i]

        return sorted(swings, key=lambda s: s.index)

    def _detect_breaks(
        self, swings: list[SwingPoint], close: np.ndarray
    ) -> list[StructureBreak]:
        """Detect Break of Structure (BOS) — close-based only."""
        breaks: list[StructureBreak] = []
        swing_highs = [s for s in swings if s.is_high]
        swing_lows = [s for s in swings if not s.is_high]

        for sw in swing_highs:
            for i in range(sw.index + 1, len(close)):
                if close[i] > sw.price:
                    breaks.append(StructureBreak(
                        index=i,
                        direction=Direction.BULLISH,
                        swing_point=sw,
                        break_price=close[i],
                    ))
                    break

        for sw in swing_lows:
            for i in range(sw.index + 1, len(close)):
                if close[i] < sw.price:
                    breaks.append(StructureBreak(
                        index=i,
                        direction=Direction.BEARISH,
                        swing_point=sw,
                        break_price=close[i],
                    ))
                    break

        return sorted(breaks, key=lambda b: b.index)

    def _detect_order_blocks(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray | None,
    ) -> list[OrderBlock]:
        """Detect order blocks — last opposing candle before impulsive move."""
        obs: list[OrderBlock] = []
        if len(close) < 3:
            return obs

        for i in range(1, len(close) - 1):
            # Bullish OB: bearish candle followed by strong bullish move
            if close[i] < open_[i] and close[i + 1] > open_[i + 1]:
                move_size = close[i + 1] - low[i]
                avg_atr = atr[i] if atr is not None and i < len(atr) and not np.isnan(atr[i]) else (high[i] - low[i])
                strength = move_size / avg_atr if avg_atr > 0 else 0
                if strength > 1.5:  # Impulsive move
                    obs.append(OrderBlock(
                        index=i,
                        high=open_[i],
                        low=close[i],
                        direction=Direction.BULLISH,
                        strength=strength,
                    ))

            # Bearish OB: bullish candle followed by strong bearish move
            if close[i] > open_[i] and close[i + 1] < open_[i + 1]:
                move_size = high[i] - close[i + 1]
                avg_atr = atr[i] if atr is not None and i < len(atr) and not np.isnan(atr[i]) else (high[i] - low[i])
                strength = move_size / avg_atr if avg_atr > 0 else 0
                if strength > 1.5:
                    obs.append(OrderBlock(
                        index=i,
                        high=close[i],
                        low=open_[i],
                        direction=Direction.BEARISH,
                        strength=strength,
                    ))

        return obs

    def _detect_fvgs(
        self, open_: np.ndarray, close: np.ndarray
    ) -> list[FairValueGap]:
        """Detect Fair Value Gaps — 3-candle body gaps."""
        fvgs: list[FairValueGap] = []
        if len(close) < 3:
            return fvgs

        for i in range(1, len(close) - 1):
            # Bullish FVG: candle[i-1] high < candle[i+1] low (gap up)
            if close[i + 1] > open_[i + 1] and close[i - 1] < open_[i - 1]:
                gap_bottom = max(close[i - 1], open_[i - 1])
                gap_top = min(close[i + 1], open_[i + 1])
                if gap_top > gap_bottom:
                    fvgs.append(FairValueGap(
                        index=i,
                        top=gap_top,
                        bottom=gap_bottom,
                        direction=Direction.BULLISH,
                    ))

            # Bearish FVG: candle[i-1] low > candle[i+1] high (gap down)
            if close[i + 1] < open_[i + 1] and close[i - 1] > open_[i - 1]:
                gap_top = min(close[i - 1], open_[i - 1])
                gap_bottom = max(close[i + 1], open_[i + 1])
                if gap_top > gap_bottom:
                    fvgs.append(FairValueGap(
                        index=i,
                        top=gap_top,
                        bottom=gap_bottom,
                        direction=Direction.BEARISH,
                    ))

        return fvgs

    def _detect_liquidity_pools(
        self, high: np.ndarray, low: np.ndarray, tolerance: float = 0.001
    ) -> list[LiquidityPool]:
        """Detect clusters of equal highs or lows."""
        pools: list[LiquidityPool] = []

        # Group highs
        high_groups: dict[int, list[int]] = {}
        for i in range(len(high)):
            rounded = round(high[i] / (high[i] * tolerance + 1))
            high_groups.setdefault(rounded, []).append(i)

        for _, indices in high_groups.items():
            if len(indices) >= 2:
                level = np.mean([high[i] for i in indices])
                pools.append(LiquidityPool(level=level, is_high=True, count=len(indices), indices=indices))

        # Group lows
        low_groups: dict[int, list[int]] = {}
        for i in range(len(low)):
            rounded = round(low[i] / (low[i] * tolerance + 1))
            low_groups.setdefault(rounded, []).append(i)

        for _, indices in low_groups.items():
            if len(indices) >= 2:
                level = np.mean([low[i] for i in indices])
                pools.append(LiquidityPool(level=level, is_high=False, count=len(indices), indices=indices))

        return sorted(pools, key=lambda p: p.count, reverse=True)

    def _detect_sweeps(
        self, pools: list[LiquidityPool], high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> list[LiquiditySweep]:
        """Detect liquidity sweeps — wick beyond pool without close confirmation."""
        sweeps: list[LiquiditySweep] = []

        for pool in pools:
            for idx in range(len(close)):
                if pool.is_high:
                    # Wick above pool but close below
                    if high[idx] > pool.level and close[idx] < pool.level:
                        sweeps.append(LiquiditySweep(
                            index=idx,
                            pool_level=pool.level,
                            is_high=True,
                            sweep_high=high[idx],
                            close=close[idx],
                        ))
                else:
                    # Wick below pool but close above
                    if low[idx] < pool.level and close[idx] > pool.level:
                        sweeps.append(LiquiditySweep(
                            index=idx,
                            pool_level=pool.level,
                            is_high=False,
                            sweep_high=low[idx],
                            close=close[idx],
                        ))

        return sweeps

    def _detect_supply_demand(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray | None,
    ) -> list[SupplyDemandZone]:
        """Detect supply and demand zones."""
        zones: list[SupplyDemandZone] = []
        if len(close) < 5:
            return zones

        for i in range(2, len(close) - 2):
            avg_atr = atr[i] if atr is not None and i < len(atr) and not np.isnan(atr[i]) else (high[i] - low[i])
            if avg_atr <= 0:
                continue

            # Demand zone: consolidation then bullish impulse
            range_size = max(high[i-2:i+1]) - min(low[i-2:i+1])
            impulse = close[i + 1] - min(low[i-2:i+1])
            if range_size < avg_atr * 0.5 and impulse > avg_atr * 1.5:
                zones.append(SupplyDemandZone(
                    index=i,
                    high=max(high[i-2:i+1]),
                    low=min(low[i-2:i+1]),
                    direction=Direction.BULLISH,
                    strength=impulse / avg_atr,
                ))

            # Supply zone: consolidation then bearish impulse
            impulse = max(high[i-2:i+1]) - close[i + 1]
            if range_size < avg_atr * 0.5 and impulse > avg_atr * 1.5:
                zones.append(SupplyDemandZone(
                    index=i,
                    high=max(high[i-2:i+1]),
                    low=min(low[i-2:i+1]),
                    direction=Direction.BEARISH,
                    strength=impulse / avg_atr,
                ))

        return zones

    def _determine_trend(
        self, swings: list[SwingPoint], breaks: list[StructureBreak]
    ) -> TrendDirection:
        """Determine current trend from swing structure and BOS."""
        if not breaks:
            return TrendDirection.NEUTRAL

        recent_breaks = breaks[-3:]
        bullish_breaks = sum(1 for b in recent_breaks if b.direction == Direction.BULLISH)
        bearish_breaks = sum(1 for b in recent_breaks if b.direction == Direction.BEARISH)

        if bullish_breaks > bearish_breaks:
            return TrendDirection.UPTREND
        elif bearish_breaks > bullish_breaks:
            return TrendDirection.DOWNTREND
        return TrendDirection.NEUTRAL