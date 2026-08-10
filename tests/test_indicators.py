"""Tests for technical indicators."""

import numpy as np
import pytest

from src.analysis.indicators import (
    sma, ema, rma, calculate_rsi, calculate_macd,
    calculate_bollinger, calculate_atr, calculate_stochastic,
    compute_indicators, snapshot_at,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    returns = np.random.normal(0.0001, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    volume = np.random.uniform(100, 1000, n)
    return open_, high, low, close, volume


class TestSMA:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(data, 3)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[4] == pytest.approx(4.0)

    def test_short_data(self):
        data = np.array([1.0, 2.0])
        result = sma(data, 5)
        assert all(np.isnan(result))


class TestEMA:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema(data, 3)
        assert not np.isnan(result[2])
        assert result[-1] > result[0]

    def test_short_data(self):
        data = np.array([1.0])
        result = ema(data, 5)
        assert all(np.isnan(result))


class TestRSI:
    def test_range(self, sample_data):
        _, _, _, close, _ = sample_data
        result = calculate_rsi(close, 14)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_short_data(self):
        close = np.array([1.0, 2.0])
        result = calculate_rsi(close, 14)
        assert all(np.isnan(result))


class TestMACD:
    def test_output_shapes(self, sample_data):
        _, _, _, close, _ = sample_data
        macd, signal, hist = calculate_macd(close)
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(hist) == len(close)


class TestBollinger:
    def test_bands(self, sample_data):
        _, _, _, close, _ = sample_data
        upper, middle, lower = calculate_bollinger(close, 20, 2.0)
        # Upper should be above middle, lower below
        for i in range(20, len(close)):
            if not np.isnan(upper[i]):
                assert upper[i] >= middle[i]
                assert lower[i] <= middle[i]


class TestATR:
    def test_positive(self, sample_data):
        _, high, low, close, _ = sample_data
        result = calculate_atr(high, low, close, 14)
        valid = result[~np.isnan(result)]
        assert all(v > 0 for v in valid)


class TestComputeIndicators:
    def test_all_keys(self, sample_data):
        open_, high, low, close, volume = sample_data
        result = compute_indicators(open_, high, low, close, volume)
        expected_keys = [
            "rsi", "macd", "macd_signal", "macd_histogram",
            "ema_9", "ema_20", "ema_50", "ema_200",
            "bb_upper", "bb_middle", "bb_lower",
            "atr", "adx", "stoch_k", "stoch_d", "vwap",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_snapshot(self, sample_data):
        open_, high, low, close, volume = sample_data
        indicators = compute_indicators(open_, high, low, close, volume)
        snap = snapshot_at(indicators, 100)
        assert 0 <= snap.rsi <= 100
        assert snap.atr > 0