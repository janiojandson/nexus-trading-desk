"""
Risk manager — circuit breaker, drawdown tracker, position sizing.

Ported from TradeClaw's packages/signals/src/risk/.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskState:
    """Current risk state."""
    daily_pnl: float = 0.0
    daily_start_equity: float = 0.0
    consecutive_losses: int = 0
    is_circuit_breaker_active: bool = False
    current_drawdown: float = 0.0
    peak_equity: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker — pauses trading after N consecutive losses.
    Prevents emotional revenge trading.
    """

    def __init__(self, max_consecutive_losses: int = 5, cooldown_minutes: int = 60):
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self._consecutive_losses = 0
        self._activated_at: float | None = None

    def record_trade(self, win: bool):
        """Record a trade result."""
        if win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    @property
    def is_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if self._consecutive_losses >= self.max_consecutive_losses:
            return True
        if self._activated_at is not None:
            import time
            elapsed = (time.time() - self._activated_at) / 60
            if elapsed < self.cooldown_minutes:
                return True
            self._activated_at = None
        return False

    def check_and_activate(self) -> bool:
        """Check if breaker should activate. Returns True if newly activated."""
        if self._consecutive_losses >= self.max_consecutive_losses and self._activated_at is None:
            import time
            self._activated_at = time.time()
            return True
        return False


class DrawdownTracker:
    """
    Tracks drawdown and enforces hard stop at maximum daily drawdown.
    """

    def __init__(self, max_daily_drawdown: float = 0.15):
        self.max_daily_drawdown = max_daily_drawdown
        self._peak_equity = 0.0
        self._daily_start_equity = 0.0
        self._current_equity = 0.0

    def set_daily_start(self, equity: float):
        """Set the starting equity for the day."""
        self._daily_start_equity = equity
        self._peak_equity = equity
        self._current_equity = equity

    def update_equity(self, equity: float):
        """Update current equity and track peak."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction (0-1)."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    @property
    def daily_drawdown(self) -> float:
        """Daily drawdown from start of day."""
        if self._daily_start_equity <= 0:
            return 0.0
        return (self._daily_start_equity - self._current_equity) / self._daily_start_equity

    @property
    def is_hard_stop(self) -> bool:
        """Whether the hard stop has been hit."""
        return self.daily_drawdown >= self.max_daily_drawdown

    @property
    def is_warning(self) -> bool:
        """Whether drawdown is approaching the limit (10%+)."""
        return self.daily_drawdown >= self.max_daily_drawdown * 0.67


class RiskManager:
    """
    Unified risk manager.

    Combines circuit breaker, drawdown tracker, and position sizing.
    """

    def __init__(
        self,
        max_daily_drawdown: float = 0.15,
        max_consecutive_losses: int = 5,
        max_positions: int = 3,
        max_leverage: int = 3,
        kelly_fraction: float = 0.25,
    ):
        self.max_positions = max_positions
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.circuit_breaker = CircuitBreaker(max_consecutive_losses)
        self.drawdown_tracker = DrawdownTracker(max_daily_drawdown)
        self._open_positions: int = 0

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns (can_trade, reason).
        """
        if self.circuit_breaker.is_active:
            return False, "circuit-breaker-active"

        if self.drawdown_tracker.is_hard_stop:
            return False, "max-daily-drawdown-reached"

        if self._open_positions >= self.max_positions:
            return False, "max-positions-reached"

        return True, "ok"

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float = 0.55,
    ) -> float:
        """
        Calculate position size using fractional Kelly criterion.

        Kelly % = W - (1-W)/R
        where W = win rate, R = reward/risk ratio

        We use a fraction (default 25%) of full Kelly for safety.
        """
        risk = abs(entry_price - stop_loss)
        if risk <= 0 or equity <= 0:
            return 0.0

        reward_risk = 2.0  # Default R:R
        kelly_pct = win_rate - (1 - win_rate) / reward_risk
        if kelly_pct <= 0:
            return 0.0

        # Fractional Kelly
        position_risk = equity * kelly_pct * self.kelly_fraction
        quantity = position_risk / risk

        # Cap by max leverage
        max_notional = equity * self.max_leverage
        max_quantity = max_notional / entry_price if entry_price > 0 else 0

        return min(quantity, max_quantity)

    def record_trade(self, win: bool, pnl: float):
        """Record a trade result."""
        self.circuit_breaker.record_trade(win)
        if self.circuit_breaker.check_and_activate():
            pass  # Circuit breaker just activated

    def open_position(self):
        """Record that a position was opened."""
        self._open_positions += 1

    def close_position(self):
        """Record that a position was closed."""
        self._open_positions = max(0, self._open_positions - 1)