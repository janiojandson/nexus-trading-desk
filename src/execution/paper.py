"""
In-memory paper trading broker.
Simulates order execution with instant fills at the requested price.
Tracks positions, equity, and P&L entirely in memory.

Ported from TradeClaw's packages/agent/src/broker/paper-broker.ts.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    """Open position."""
    symbol: str
    direction: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float
    opened_at: float  # timestamp

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * (self.market_value - self.cost_basis)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis * 100


@dataclass
class Order:
    """Executed order."""
    id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    quantity: float
    filled_price: float
    status: str  # "filled", "rejected"
    created_at: float
    signal_id: str | None = None


@dataclass
class AccountInfo:
    """Paper trading account info."""
    equity: float
    cash: float
    buying_power: float
    positions_value: float
    currency: str = "USD"
    is_paper: bool = True


class PaperBroker:
    """
    In-memory paper trading broker.

    Simulates order execution with instant fills at the requested price.
    Tracks positions, equity, and P&L entirely in memory.
    """

    def __init__(self, initial_equity: float = 100_000):
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self._next_order_id = 1
        self._trade_history: list[dict] = []

    def get_account(self) -> AccountInfo:
        """Get current account info."""
        positions_value = sum(p.market_value for p in self.positions.values())
        equity = self.cash + positions_value
        return AccountInfo(
            equity=equity,
            cash=self.cash,
            buying_power=self.cash,
            positions_value=positions_value,
        )

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def open_position(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        signal_id: str | None = None,
    ) -> Order:
        """Open a new position."""
        # Check if position already exists for this symbol
        if symbol in self.positions:
            return Order(
                id=f"paper-{self._next_order_id}",
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                filled_price=price,
                status="rejected",
                created_at=time.time(),
                signal_id=signal_id,
            )

        cost = quantity * price
        if cost > self.cash:
            # Reduce quantity to fit available cash
            quantity = self.cash / price if price > 0 else 0
            if quantity <= 0:
                return Order(
                    id=f"paper-{self._next_order_id}",
                    symbol=symbol,
                    direction=direction,
                    quantity=0,
                    filled_price=price,
                    status="rejected",
                    created_at=time.time(),
                    signal_id=signal_id,
                )

        self.cash -= quantity * price
        self.positions[symbol] = Position(
            symbol=symbol,
            direction="long" if direction == "BUY" else "short",
            quantity=quantity,
            entry_price=price,
            current_price=price,
            opened_at=time.time(),
        )

        order = Order(
            id=f"paper-{self._next_order_id}",
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            filled_price=price,
            status="filled",
            created_at=time.time(),
            signal_id=signal_id,
        )
        self._next_order_id += 1
        self.orders.append(order)
        return order

    def close_position(
        self, symbol: str, price: float
    ) -> Order | None:
        """Close an existing position."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        realized_pnl = pos.unrealized_pnl
        self.cash += pos.quantity * price

        # Record trade
        self._trade_history.append({
            "symbol": symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "quantity": pos.quantity,
            "realized_pnl": realized_pnl,
            "opened_at": pos.opened_at,
            "closed_at": time.time(),
        })

        del self.positions[symbol]

        order = Order(
            id=f"paper-{self._next_order_id}",
            symbol=symbol,
            direction="SELL" if pos.direction == "long" else "BUY",
            quantity=pos.quantity,
            filled_price=price,
            status="filled",
            created_at=time.time(),
        )
        self._next_order_id += 1
        self.orders.append(order)
        return order

    def update_price(self, symbol: str, price: float):
        """Update current price for a position."""
        if symbol in self.positions:
            self.positions[symbol].current_price = price

    def get_trade_history(self) -> list[dict]:
        """Get all closed trade history."""
        return self._trade_history.copy()

    def get_daily_pnl(self) -> dict:
        """Get today's P&L summary."""
        account = self.get_account()
        realized = sum(t["realized_pnl"] for t in self._trade_history)
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        wins = sum(1 for t in self._trade_history if t["realized_pnl"] > 0)
        total = len(self._trade_history)

        return {
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
            "total_equity": account.equity,
            "win_rate": wins / total * 100 if total > 0 else 0,
            "total_trades": total,
            "open_positions": len(self.positions),
        }

    def reset(self):
        """Reset the paper trading account."""
        self.cash = self.initial_equity
        self.positions.clear()
        self.orders.clear()
        self._trade_history.clear()
        self._next_order_id = 1