"""
PPO agent trainer for strategy optimization.

Uses Stable-Baselines3 PPO to train a trading agent
on the custom Gymnasium environment.
"""

import os
from pathlib import Path

import numpy as np

from src.rl.env import TradingEnv, GYMNASIUM_AVAILABLE


class RLAgent:
    """
    RL agent trainer using PPO.

    Trains on historical data and saves the best model
    for use in live signal generation.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        total_timesteps: int = 100_000,
        model_dir: str = "models",
    ):
        self.learning_rate = learning_rate
        self.total_timesteps = total_timesteps
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def train(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> dict:
        """
        Train the PPO agent on historical data.

        Returns training metrics.
        """
        if not GYMNASIUM_AVAILABLE:
            return {"error": "gymnasium not installed"}

        try:
            from stable_baselines3 import PPO
        except ImportError:
            return {"error": "stable-baselines3 not installed"}

        # Create environment
        env = TradingEnv(open_, high, low, close, volume)

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            verbose=1,
        )

        # Train
        self.model.learn(total_timesteps=self.total_timesteps)

        # Save model
        model_path = self.model_dir / "ppo_trading_model"
        self.model.save(str(model_path))

        return {
            "status": "trained",
            "total_timesteps": self.total_timesteps,
            "model_path": str(model_path),
        }

    def load_model(self, path: str | None = None):
        """Load a trained model."""
        if not GYMNASIUM_AVAILABLE:
            return

        from stable_baselines3 import PPO

        model_path = path or str(self.model_dir / "ppo_trading_model")
        if Path(model_path + ".zip").exists():
            self.model = PPO.load(model_path)

    def predict(self, observation: np.ndarray) -> int:
        """
        Get action prediction from the trained model.

        Args:
            observation: Current state observation

        Returns:
            Action index (0=hold, 1=buy, 2=sell, 3=close)
        """
        if self.model is None:
            return 0  # Default to hold

        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)