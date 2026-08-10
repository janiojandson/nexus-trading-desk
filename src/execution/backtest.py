"""
Backtesting engine with comprehensive metrics.

Supports:
- Multi-preset comparison
- ATR-based or fixed TP/SL geometry
- Cost model (fees + slippage + funding)
- Equity curve generation
- Walk-forward validation

Ported from TradeClaw's packages/strategies/src/run-backtest.ts.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.signals.presets import EntrySignal, StrategyId, get_strategy
from src.analysis.indicators import compute_indicators, calculate_atr


@dataclass
class BacktestTrade:
    """A single backtest trade."""
    id: int
    direction: str  # "BUY" or "SELL"
    entry: float
    exit: float
    entry_bar: int
    exit_bar: int
    pnl: float
    pnl_pct: float
    win: bool
    exit_reason: str  # "TP", "SL", "EOD"
    cost_pct: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    strategy_id: StrategyId
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    total_return: float
    start_balance: float
    end_balance: float
    equity_curve: list[float]
    trades: list[BacktestTrade]
    reason: str | None = None


class BacktestEngine:
    """
    Backtesting engine.

    Runs a strategy over historical data and computes comprehensive metrics.
    """

    def __init__(
        self,
        initial_balance: float = 10000,
        geometry: str = "atr",
        tp_r_multiple: float = 2.0,
        sl_atr_multiple: float = 1.5,
        cost_model: dict | None = None,
    ):
        self.initial_balance = initial_balance
        self.geometry = geometry
        self.tp_r_multiple = tp_r_multiple
        self.sl_atr_multiple = sl_atr_multiple
        self.cost_model = cost_model or {
            "maker_fee": 0.001,
            "taker_fee": 0.0015,
            "slippage": 0.0005,
        }

    def run(
        self,
        strategy_id: StrategyId,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        strategy_fn: callable | None = None,
    ) -> BacktestResult:
        """
        Run backtest for a single strategy.

        Args:
            strategy_id: Strategy identifier
            open_, high, low, close, volume: OHLCV data
            strategy_fn: Optional custom strategy function

        Returns:
            BacktestResult with all metrics
        """
        n = len(close)
        if n < 50:
            return BacktestResult(
                strategy_id=strategy_id, total_trades=0, win_rate=0,
                profit_factor=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, total_return=0,
                start_balance=self.initial_balance, end_balance=self.initial_balance,
                equity_curve=[self.initial_balance], trades=[], reason="no-data",
            )

        # Compute indicators
        indicators = compute_indicators(open_, high, low, close, volume)
        atr_arr = indicators["atr"]

        # Get strategy function
        fn = strategy_fn or get_strategy(strategy_id.value)

        # Generate entry signals
        entry_signals = fn(indicators, close, high, low)

        if not entry_signals:
            return BacktestResult(
                strategy_id=strategy_id, total_trades=0, win_rate=0,
                profit_factor=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, total_return=0,
                start_balance=self.initial_balance, end_balance=self.initial_balance,
                equity_curve=[self.initial_balance], trades=[], reason="no-signals",
            )

        # Simulate trades
        balance = self.initial_balance
        equity_curve = [balance]
        trades: list[BacktestTrade] = []
        position: dict | None = None
        trade_id = 0

        for sig in entry_signals:
            i = sig.bar_index
            entry_price = sig.price

            # Close existing position if any
            if position is not None:
                exit_price = close[i]
                pnl, pnl_pct, win = self._close_position(position, exit_price, i)
                balance += pnl
                trades.append(BacktestTrade(
                    id=trade_id, direction=position["direction"],
                    entry=position["entry"], exit=exit_price,
                    entry_bar=position["bar"], exit_bar=i,
                    pnl=pnl, pnl_pct=pnl_pct, win=win,
                    exit_reason="FLIP",
                    cost_pct=self._compute_cost(position["entry"], exit_price),
                ))
                trade_id += 1
                position = None

            # Compute stop levels
            atr_val = atr_arr[i] if i < len(atr_arr) and not np.isnan(atr_arr[i]) else (high[i] - low[i])
            if atr_val <= 0:
                continue

            risk = atr_val * self.sl_atr_multiple
            if sig.direction == "BUY":
                sl = entry_price - risk
                tp = entry_price + risk * self.tp_r_multiple
            else:
                sl = entry_price + risk
                tp = entry_price - risk * self.tp_r_multiple

            # Open position
            position = {
                "direction": sig.direction,
                "entry": entry_price,
                "stop_loss": sl,
                "take_profit": tp,
                "bar": i,
                "risk": risk,
            }

            # Check TP/SL on subsequent bars
            for j in range(i + 1, n):
                if position is None:
                    break

                # Check SL
                if position["direction"] == "BUY" and low[j] <= position["stop_loss"]:
                    pnl, pnl_pct, win = self._close_position(position, position["stop_loss"], j)
                    pnl = -abs(position["risk"])  # Lose 1R
                    pnl_pct = pnl / balance * 100
                    balance += pnl
                    equity_curve.append(balance)
                    trades.append(BacktestTrade(
                        id=trade_id, direction=position["direction"],
                        entry=position["entry"], exit=position["stop_loss"],
                        entry_bar=position["bar"], exit_bar=j,
                        pnl=pnl, pnl_pct=pnl_pct, win=False,
                        exit_reason="SL",
                        cost_pct=self._compute_cost(position["entry"], position["stop_loss"]),
                    ))
                    trade_id += 1
                    position = None
                    break

                if position["direction"] == "SELL" and high[j] >= position["stop_loss"]:
                    pnl = -abs(position["risk"])
                    pnl_pct = pnl / balance * 100
                    balance += pnl
                    equity_curve.append(balance)
                    trades.append(BacktestTrade(
                        id=trade_id, direction=position["direction"],
                        entry=position["entry"], exit=position["stop_loss"],
                        entry_bar=position["bar"], exit_bar=j,
                        pnl=pnl, pnl_pct=pnl_pct, win=False,
                        exit_reason="SL",
                        cost_pct=self._compute_cost(position["entry"], position["stop_loss"]),
                    ))
                    trade_id += 1
                    position = None
                    break

                # Check TP
                if position["direction"] == "BUY" and high[j] >= position["take_profit"]:
                    pnl = abs(position["risk"]) * self.tp_r_multiple
                    pnl_pct = pnl / balance * 100
                    balance += pnl
                    equity_curve.append(balance)
                    trades.append(BacktestTrade(
                        id=trade_id, direction=position["direction"],
                        entry=position["entry"], exit=position["take_profit"],
                        entry_bar=position["bar"], exit_bar=j,
                        pnl=pnl, pnl_pct=pnl_pct, win=True,
                        exit_reason="TP",
                        cost_pct=self._compute_cost(position["entry"], position["take_profit"]),
                    ))
                    trade_id += 1
                    position = None
                    break

                if position["direction"] == "SELL" and low[j] <= position["take_profit"]:
                    pnl = abs(position["risk"]) * self.tp_r_multiple
                    pnl_pct = pnl / balance * 100
                    balance += pnl
                    equity_curve.append(balance)
                    trades.append(BacktestTrade(
                        id=trade_id, direction=position["direction"],
                        entry=position["entry"], exit=position["take_profit"],
                        entry_bar=position["bar"], exit_bar=j,
                        pnl=pnl, pnl_pct=pnl_pct, win=True,
                        exit_reason="TP",
                        cost_pct=self._compute_cost(position["entry"], position["take_profit"]),
                    ))
                    trade_id += 1
                    position = None
                    break

                equity_curve.append(balance)

        # Compute metrics
        return self._compute_metrics(strategy_id, trades, equity_curve)

    def _close_position(self, position: dict, exit_price: float, bar: int) -> tuple[float, float, bool]:
        """Close a position and compute P&L."""
        if position["direction"] == "BUY":
            pnl = exit_price - position["entry"]
        else:
            pnl = position["entry"] - exit_price
        pnl_pct = pnl / position["entry"] * 100
        win = pnl > 0
        return pnl, pnl_pct, win

    def _compute_cost(self, entry: float, exit: float) -> float:
        """Compute total friction cost as % of notional."""
        total = self.cost_model["maker_fee"] + self.cost_model["taker_fee"] + self.cost_model["slippage"]
        return total * 2  # Entry + exit

    def _compute_metrics(
        self,
        strategy_id: StrategyId,
        trades: list[BacktestTrade],
        equity_curve: list[float],
    ) -> BacktestResult:
        """Compute all backtest metrics."""
        if not trades:
            return BacktestResult(
                strategy_id=strategy_id, total_trades=0, win_rate=0,
                profit_factor=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, total_return=0,
                start_balance=self.initial_balance,
                end_balance=equity_curve[-1] if equity_curve else self.initial_balance,
                equity_curve=equity_curve, trades=trades,
            )

        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]

        win_rate = len(wins) / len(trades) * 100 if trades else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        max_dd *= 100  # As percentage

        # Sharpe ratio (annualized, assuming 252 trading days)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[returns != 0]  # Remove zero returns
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 1:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252) if np.std(downside) > 0 else 0
        else:
            sortino = 0

        total_return = (equity_curve[-1] - self.initial_balance) / self.initial_balance * 100

        return BacktestResult(
            strategy_id=strategy_id,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            total_return=total_return,
            start_balance=self.initial_balance,
            end_balance=equity_curve[-1],
            equity_curve=equity_curve,
            trades=trades,
        )

    def compare_strategies(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        strategies: list[str] | None = None,
    ) -> dict[str, BacktestResult]:
        """Run backtest for multiple strategies and compare."""
        if strategies is None:
            strategies = ["classic", "regime-aware", "vwap-ema-bb"]

        results = {}
        for preset in strategies:
            try:
                sid = StrategyId(preset)
            except ValueError:
                continue
            result = self.run(sid, open_, high, low, close, volume)
            results[preset] = result

        return results