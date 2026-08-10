"""
SQLite OHLCV store for caching candle data.
Avoids redundant API calls and enables offline backtesting.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.exchange import OHLCV


class OHLCVStore:
    """SQLite-backed OHLCV cache."""

    def __init__(self, db_path: str = "data/nexus.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv(symbol, timeframe);
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profits TEXT NOT NULL,
                confidence REAL NOT NULL,
                regime TEXT NOT NULL,
                reasons TEXT NOT NULL,
                preset TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            );
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                pnl REAL,
                status TEXT DEFAULT 'open',
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            );
        """)
        self._conn.commit()

    def save_candles(self, symbol: str, timeframe: str, candles: list[OHLCV]):
        """Upsert OHLCV candles into the store."""
        rows = [
            (symbol, timeframe, c.timestamp, c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def load_candles(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> list[OHLCV]:
        """Load cached OHLCV candles, newest last."""
        cursor = self._conn.execute(
            """SELECT timestamp, open, high, low, close, volume
               FROM ohlcv
               WHERE symbol = ? AND timeframe = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (symbol, timeframe, limit),
        )
        rows = cursor.fetchall()
        candles = [
            OHLCV(timestamp=r[0], open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5])
            for r in reversed(rows)
        ]
        return candles

    def load_candles_df(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> pd.DataFrame:
        """Load candles as a pandas DataFrame."""
        candles = self.load_candles(symbol, timeframe, limit)
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame([vars(c) for c in candles])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def save_signal(self, signal: dict) -> int:
        """Save a generated signal."""
        cursor = self._conn.execute(
            """INSERT INTO signals (timestamp, symbol, direction, entry, stop_loss,
               take_profits, confidence, regime, reasons, preset)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal["timestamp"],
                signal["symbol"],
                signal["direction"],
                signal["entry"],
                signal["stop_loss"],
                json.dumps(signal["take_profits"]),
                signal["confidence"],
                signal["regime"],
                json.dumps(signal["reasons"]),
                signal.get("preset", "unknown"),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_active_signals(self, symbol: str | None = None) -> list[dict]:
        """Get all active (unresolved) signals."""
        query = "SELECT * FROM signals WHERE status = 'active'"
        params: tuple = ()
        if symbol:
            query += " AND symbol = ?"
            params = (symbol,)
        query += " ORDER BY timestamp DESC"
        cursor = self._conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self):
        """Close the database connection."""
        self._conn.close()