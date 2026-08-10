"""Configuration loader for NexusTradingDesk."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration, merging YAML file + env vars."""

    def __init__(self, config_path: str | None = None):
        self._yaml = self._load_yaml(config_path)
        self._env = os.environ

    # ---- YAML loader ----
    @staticmethod
    def _load_yaml(path: str | None) -> dict[str, Any]:
        default = Path(__file__).parent.parent / "config.yaml"
        p = Path(path) if path else default
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
        return {}

    # ---- Exchange config ----
    @property
    def exchanges(self) -> dict:
        return self._yaml.get("exchanges", {})

    @property
    def symbols(self) -> list[str]:
        env_sym = self._env.get("SYMBOLS")
        if env_sym:
            return [s.strip() for s in env_sym.split(",")]
        return self._yaml.get("symbols", ["BTC/USDT"])

    @property
    def timeframes(self) -> list[str]:
        return self._yaml.get("timeframes", ["1h"])

    # ---- Strategy ----
    @property
    def strategy_preset(self) -> str:
        return self._env.get("STRATEGY_PRESET") or self._yaml.get("strategy", {}).get("preset", "regime-aware")

    @property
    def confluence_threshold(self) -> int:
        return self._yaml.get("strategy", {}).get("confluence_threshold", 65)

    @property
    def min_confidence(self) -> float:
        return self._yaml.get("strategy", {}).get("min_confidence", 58)

    # ---- Risk ----
    @property
    def max_daily_drawdown(self) -> float:
        return float(self._env.get("MAX_DAILY_DRAWDOWN", self._yaml.get("risk", {}).get("max_daily_drawdown", 0.15)))

    @property
    def max_consecutive_losses(self) -> int:
        return int(self._env.get("MAX_CONSECUTIVE_LOSSES", self._yaml.get("risk", {}).get("max_consecutive_losses", 5)))

    @property
    def max_positions(self) -> int:
        return int(self._env.get("MAX_POSITIONS", self._yaml.get("risk", {}).get("max_positions", 3)))

    @property
    def max_leverage(self) -> int:
        return int(self._env.get("MAX_LEVERAGE", self._yaml.get("risk", {}).get("max_leverage", 3)))

    @property
    def kelly_fraction(self) -> float:
        return self._yaml.get("risk", {}).get("position_size_kelly_fraction", 0.25)

    # ---- Backtest ----
    @property
    def backtest_initial_balance(self) -> float:
        return self._yaml.get("backtest", {}).get("initial_balance", 10000)

    @property
    def backtest_geometry(self) -> str:
        return self._yaml.get("backtest", {}).get("geometry", "atr")

    @property
    def tp_r_multiple(self) -> float:
        return self._yaml.get("backtest", {}).get("tp_r_multiple", 2.0)

    @property
    def sl_atr_multiple(self) -> float:
        return self._yaml.get("backtest", {}).get("sl_atr_multiple", 1.5)

    @property
    def cost_model(self) -> dict:
        return self._yaml.get("backtest", {}).get("cost_model", {
            "maker_fee": 0.001, "taker_fee": 0.0015, "slippage": 0.0005
        })

    # ---- RL ----
    @property
    def rl_algorithm(self) -> str:
        return self._yaml.get("rl", {}).get("algorithm", "PPO")

    @property
    def rl_learning_rate(self) -> float:
        return self._yaml.get("rl", {}).get("learning_rate", 3e-4)

    @property
    def rl_total_timesteps(self) -> int:
        return self._yaml.get("rl", {}).get("total_timesteps", 100000)

    # ---- Telegram ----
    @property
    def telegram_enabled(self) -> bool:
        return bool(self._env.get("TELEGRAM_BOT_TOKEN"))

    @property
    def telegram_token(self) -> str:
        return self._env.get("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return self._env.get("TELEGRAM_CHAT_ID", "")

    @property
    def telegram_alerts(self) -> dict:
        return self._yaml.get("telegram", {}).get("alerts", {
            "signals": True, "daily_pnl": True, "drawdown_alert": True,
            "regime_change": True, "weekly_report": True
        })

    # ---- Scheduler ----
    @property
    def signal_check_interval(self) -> int:
        return self._yaml.get("scheduler", {}).get("signal_check_interval", 300)

    # ---- Mode ----
    @property
    def trading_mode(self) -> str:
        return self._env.get("TRADING_MODE", "backtest")

    # ---- Database ----
    @property
    def database_url(self) -> str:
        return self._env.get("DATABASE_URL", "sqlite:///./data/nexus.db")