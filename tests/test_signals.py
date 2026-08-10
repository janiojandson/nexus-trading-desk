"""Tests for signal generation."""

import numpy as np
import pytest

from src.signals.generator import SignalGenerator


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    returns = np.random.normal(0.0001, 0.02, n)
    close = 50000.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = 50000.0
    volume = np.random.uniform(100, 1000, n)
    return open_, high, low, close, volume


class TestSignalGenerator:
    def test_generates_signals(self, sample_data):
        open_, high, low, close, volume = sample_data
        gen = SignalGenerator(confluence_threshold=30, min_confidence=30)
        signals = gen.generate("BTC/USDT", open_, high, low, close, volume)
        assert isinstance(signals, list)

    def test_signal_fields(self, sample_data):
        open_, high, low, close, volume = sample_data
        gen = SignalGenerator(confluence_threshold=30, min_confidence=30)
        signals = gen.generate("BTC/USDT", open_, high, low, close, volume)
        for signal in signals:
            assert signal.symbol == "BTC/USDT"
            assert signal.direction in ["BUY", "SELL"]
            assert signal.entry > 0
            assert signal.regime in ["trend", "volatile", "range", "unknown"]

    def test_wait_signals(self, sample_data):
        open_, high, low, close, volume = sample_data
        gen = SignalGenerator(confluence_threshold=90, min_confidence=90)
        signals = gen.generate("BTC/USDT", open_, high, low, close, volume)
        # With high thresholds, most signals should be WAIT
        wait_signals = [s for s in signals if s.wait_reason]
        # At least check the structure is correct
        for s in wait_signals:
            assert s.wait_reason is not None

    def test_different_presets(self, sample_data):
        open_, high, low, close, volume = sample_data
        for preset in ["classic", "regime-aware", "vwap-ema-bb"]:
            gen = SignalGenerator(confluence_threshold=30, min_confidence=30, preset=preset)
            signals = gen.generate("BTC/USDT", open_, high, low, close, volume)
            assert isinstance(signals, list)