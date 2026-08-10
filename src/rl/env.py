"""
Gymnasium trading environment for RL agent training.

State space: [RSI, MACD_histogram, EMA_trend, ATR, BB_position, regime, P&L]
Action space: [hold, buy, sell, close]
Reward: Risk-adjusted return (Sharpe-like)
"""

import numpy as np
from typing import Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

from src.analysis.indicators import compute_indicators, snapshot_at
from src.analysis.regime import RegimeClassifier


if GYMNASIUM_AVAILABLE:

    class TradingEnv(gym.Env):
        """
        Custom Gymnasium environment for crypto trading.

        The agent observes market indicators and decides whether to
        hold, buy, sell, or close positions.
        """

        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            open_: np.ndarray,
            high: np.ndarray,
            low: np.ndarray,
            close: np.ndarray,
            volume: np.ndarray,
            initial_balance: float = 10000,
            render_mode: str | None = None,
        ):
            super().__init__()

            self.open_ = open_
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume
            self.initial_balance = initial_balance
            self.render_mode = render_mode

            # State: 7 features
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )

            # Actions: 0=hold, 1=buy, 2=sell, 3=close
            self.action_space = spaces.Discrete(4)

            # Pre-compute indicators
            self.indicators = compute_indicators(open_, high, low, close, volume)
            self.regime_classifier = RegimeClassifier()

            self._reset_state()

        def _reset_state(self):
            """Reset internal state."""
            self.current_step = 50  # Start after warmup
            self.balance = self.initial_balance
            self.position = None  # None or {"direction": "long"/"short", "entry": float}
            self.peak_equity = self.initial_balance
            self.total_pnl = 0.0
            self.trades = 0
            self.wins = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._reset_state()
            return self._get_obs(), {}

        def step(self, action):
            """Execute one step in the environment."""
            price = self.close[self.current_step]
            prev_equity = self.balance

            # Execute action
            if action == 1 and self.position is None:  # Buy
                self.position = {"direction": "long", "entry": price}
            elif action == 2 and self.position is None:  # Sell
                self.position = {"direction": "short", "entry": price}
            elif action == 3 and self.position is not None:  # Close
                if self.position["direction"] == "long":
                    pnl = price - self.position["entry"]
                else:
                    pnl = self.position["entry"] - price
                self.balance += pnl
                self.total_pnl += pnl
                self.trades += 1
                if pnl > 0:
                    self.wins += 1
                self.position = None

            # Update equity
            unrealized = 0.0
            if self.position is not None:
                if self.position["direction"] == "long":
                    unrealized = price - self.position["entry"]
                else:
                    unrealized = self.position["entry"] - price

            equity = self.balance + unrealized
            if equity > self.peak_equity:
                self.peak_equity = equity

            # Reward: risk-adjusted return
            reward = (equity - prev_equity) / self.initial_balance * 100
            # Penalize drawdown
            dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
            reward -= dd * 10

            # Advance step
            self.current_step += 1
            terminated = self.current_step >= len(self.close) - 1
            truncated = False

            # Additional termination: bankruptcy
            if equity <= 0:
                terminated = True
                reward = -100

            return self._get_obs(), reward, terminated, truncated, {}

        def _get_obs(self) -> np.ndarray:
            """Get current observation."""
            i = min(self.current_step, len(self.close) - 1)
            snap = snapshot_at(self.indicators, i)

            # Regime
            regime_result = self.regime_classifier.classify(
                self.high[:i+1], self.low[:i+1], self.close[:i+1]
            )
            regime_map = {"trend": 1.0, "volatile": 0.5, "range": 0.0, "unknown": -0.5}
            regime_val = regime_map.get(regime_result.regime.value, 0.0)

            # P&L
            equity = self.balance
            if self.position is not None:
                price = self.close[i]
                if self.position["direction"] == "long":
                    equity += price - self.position["entry"]
                else:
                    equity += self.position["entry"] - price
            pnl_norm = (equity - self.initial_balance) / self.initial_balance

            # BB position (0-1)
            bb_range = snap.bb_upper - snap.bb_lower
            bb_pos = (self.close[i] - snap.bb_lower) / bb_range if bb_range > 0 else 0.5

            obs = np.array([
                snap.rsi / 100.0,  # Normalize to 0-1
                snap.macd_histogram,
                1.0 if snap.ema_20 > snap.ema_50 else -1.0,  # EMA trend
                snap.atr / self.close[i] if self.close[i] > 0 else 0,  # Normalized ATR
                bb_pos,
                regime_val,
                pnl_norm,
            ], dtype=np.float32)

            return obs

        def render(self):
            if self.render_mode == "human":
                equity = self.balance
                if self.position:
                    equity += self.total_pnl
                print(f"Step {self.current_step} | Equity: ${equity:.2f} | Trades: {self.trades} | Win Rate: {self.wins/max(self.trades,1)*100:.1f}%")

else:
    class TradingEnv:
        """Stub when gymnasium is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium is required for RL training. Install with: pip install gymnasium")