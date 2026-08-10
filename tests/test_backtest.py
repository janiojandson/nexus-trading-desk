"""Tests for backtesting engine."""

import numpy as np
import pytest

from src.execution.backtest import BacktestEngine
from src.signals.presets import StrategyId


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    returns = np.random.normal(0.0001, 0.02, n)
    close = 50000.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = 50000.0
    volume = np.random.uniform(100, 1000, n)
    return open_, high, low, close, volume


class TestBacktestEngine:
    def test_run_classic(self, sample_data):
        open_, high, low, close, volume = sample_data
        engine = BacktestEngine(initial_balance=10000)
        result = engine.run(StrategyId.CLASSIC, open_, high, low, close, volume)
        assert result.strategy_id == StrategyId.CLASSIC
        assert result.total_trades >= 0
        assert result.start_balance == 10000

    def test_run_regime_aware(self, sample_data):
        open_, high, low, close, volume = sample_data
        engine = BacktestEngine(initial_balance=10000)
        result = engine.run(StrategyId.REGIME_AWARE, open_, high, low, close, volume)
        assert result.strategy_id == StrategyId.REGIME_AWARE

    def test_compare_strategies(self, sample_data):
        open_, high, low, close, volume = sample_data
        engine = BacktestEngine(initial_balance=10000)
        results = engine.compare_strategies(open_, high, low, close, volume)
        assert len(results) >= 1
        for name, result in results.items():
            assert result.total_trades >= 0

    def test_short_data(self):
        close = np.array([1.0, 2.0, 3.0])
        open_ = close.copy()
        high = close * 1.01
        low = close * 0.99
        volume = np.ones(3)
        engine = BacktestEngine()
        result = engine.run(StrategyId.CLASSIC, open_, high, low, close, volume)
        assert result.reason == "no-data"

    def test_metrics_range(self, sample_data):
        open_, high, low, close, volume = sample_data
        engine = BacktestEngine(initial_balance=10000)
        result = engine.run(StrategyId.CLASSIC, open_, high, low, close, volume)
        if result.total_trades > 0:
            assert 0 <= result.win_rate <= 100
            assert result.max_drawdown >= 0