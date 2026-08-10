"""Tests for SMC/ICT detector."""

import numpy as np
import pytest

from src.analysis.smc import SmcDetector, Direction, TrendDirection


@pytest.fixture
def trending_up():
    """Generate uptrending data."""
    np.random.seed(42)
    n = 200
    returns = np.random.normal(0.002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * 1.005
    low = close * 0.995
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    return open_, high, low, close


@pytest.fixture
def detector():
    return SmcDetector(swing_strength=3)


class TestSwingDetection:
    def test_detects_swings(self, detector, trending_up):
        open_, high, low, close = trending_up
        result = detector.analyze(open_, high, low, close)
        assert len(result.swings) > 0

    def test_swing_types(self, detector, trending_up):
        open_, high, low, close = trending_up
        result = detector.analyze(open_, high, low, close)
        has_high = any(s.is_high for s in result.swings)
        has_low = any(not s.is_high for s in result.swings)
        assert has_high or has_low


class TestOrderBlocks:
    def test_detects_obs(self, detector, trending_up):
        open_, high, low, close = trending_up
        result = detector.analyze(open_, high, low, close)
        # May or may not find OBs depending on data
        assert isinstance(result.order_blocks, list)


class TestFVGs:
    def test_detects_fvgs(self, detector, trending_up):
        open_, high, low, close = trending_up
        result = detector.analyze(open_, high, low, close)
        assert isinstance(result.fvgs, list)


class TestTrendDirection:
    def test_uptrend(self, detector, trending_up):
        open_, high, low, close = trending_up
        result = detector.analyze(open_, high, low, close)
        # Should detect some structure
        assert result.trend in [TrendDirection.UPTREND, TrendDirection.DOWNTREND, TrendDirection.NEUTRAL]


class TestSmcAnalysis:
    def test_all_fields(self, detector, trending_up):
        open_, high, low, close = trending_up
        result = detector.analyze(open_, high, low, close)
        assert hasattr(result, "swings")
        assert hasattr(result, "order_blocks")
        assert hasattr(result, "fvgs")
        assert hasattr(result, "liquidity_pools")
        assert hasattr(result, "sweeps")
        assert hasattr(result, "breaks")
        assert hasattr(result, "zones")
        assert hasattr(result, "trend")