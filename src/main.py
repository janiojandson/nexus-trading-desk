"""
NexusTradingDesk — Main entry point.

Usage:
    python -m src.main --mode backtest --symbol BTC/USDT --preset regime-aware
    python -m src.main --mode paper --symbol BTC/USDT,ETH/USDT,SOL/USDT
    python -m src.main --mode live --telegram
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.config import Config
from src.data.exchange import ExchangeAdapter
from src.data.store import OHLCVStore
from src.signals.generator import SignalGenerator
from src.execution.backtest import BacktestEngine
from src.execution.paper import PaperBroker
from src.execution.risk import RiskManager
from src.notifications.telegram_bot import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexus")


def generate_sample_data(n: int = 500, base_price: float = 50000.0) -> dict:
    """Generate sample OHLCV data for testing without exchange connection."""
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, n)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = base_price
    volume = np.random.uniform(100, 1000, n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def run_backtest(config: Config, symbol: str, preset: str):
    """Run backtesting mode."""
    logger.info(f"Running backtest for {symbol} with preset '{preset}'")

    # Get data (sample for prototype)
    data = generate_sample_data(500, 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 150.0)

    # Run backtest
    engine = BacktestEngine(
        initial_balance=config.backtest_initial_balance,
        geometry=config.backtest_geometry,
        tp_r_multiple=config.tp_r_multiple,
        sl_atr_multiple=config.sl_atr_multiple,
        cost_model=config.cost_model,
    )

    # Compare strategies
    results = engine.compare_strategies(
        open_=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
        strategies=["classic", "regime-aware", "vwap-ema-bb"],
    )

    # Print results
    print("\n" + "=" * 70)
    print(f"  BACKTEST RESULTS — {symbol}")
    print("=" * 70)

    for name, result in results.items():
        print(f"\n  Strategy: {name}")
        print(f"  {'─' * 40}")
        print(f"  Total Trades:    {result.total_trades}")
        print(f"  Win Rate:        {result.win_rate:.1f}%")
        print(f"  Profit Factor:   {result.profit_factor:.2f}")
        print(f"  Max Drawdown:    {result.max_drawdown:.1f}%")
        print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:   {result.sortino_ratio:.2f}")
        print(f"  Total Return:    {result.total_return:.1f}%")
        print(f"  End Balance:     ${result.end_balance:.2f}")

    print("\n" + "=" * 70)
    return results


def run_paper_trading(config: Config, symbols: list[str]):
    """Run paper trading mode."""
    logger.info(f"Starting paper trading for {symbols}")

    # Initialize components
    broker = PaperBroker(initial_equity=config.backtest_initial_balance)
    risk_mgr = RiskManager(
        max_daily_drawdown=config.max_daily_drawdown,
        max_consecutive_losses=config.max_consecutive_losses,
        max_positions=config.max_positions,
        max_leverage=config.max_leverage,
        kelly_fraction=config.kelly_fraction,
    )
    signal_gen = SignalGenerator(
        confluence_threshold=config.confluence_threshold,
        min_confidence=config.min_confidence,
        tp_r_multiple=config.tp_r_multiple,
        sl_atr_multiple=config.sl_atr_multiple,
        preset=config.strategy_preset,
    )
    notifier = TelegramNotifier(config.telegram_token, config.telegram_chat_id)

    # Simulate a few cycles
    for cycle in range(5):
        for symbol in symbols:
            base = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 150.0
            data = generate_sample_data(200, base)

            # Generate signals
            signals = signal_gen.generate(
                symbol=symbol,
                open_=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"],
            )

            for signal in signals:
                if signal.wait_reason:
                    logger.info(f"WAIT — {symbol} {signal.direction}: {signal.wait_reason}")
                    continue

                # Check risk
                can_trade, reason = risk_mgr.can_trade()
                if not can_trade:
                    logger.warning(f"Risk block: {reason}")
                    continue

                # Calculate position size
                size = risk_mgr.calculate_position_size(
                    equity=broker.get_account().equity,
                    entry_price=signal.entry,
                    stop_loss=signal.stop_loss,
                )

                if size > 0:
                    order = broker.open_position(
                        symbol=symbol,
                        direction=signal.direction,
                        quantity=size,
                        price=signal.entry,
                    )
                    if order.status == "filled":
                        risk_mgr.open_position()
                        logger.info(
                            f"OPENED {signal.direction} {symbol} @ ${signal.entry:.2f} "
                            f"(SL: ${signal.stop_loss:.2f}, Conf: {signal.confidence:.0f})"
                        )
                        # Send Telegram alert
                        if config.telegram_enabled:
                            notifier.send_signal_sync(signal)

        # Print P&L
        pnl = broker.get_daily_pnl()
        logger.info(f"Cycle {cycle + 1} — Equity: ${pnl['total_equity']:.2f} | P&L: ${pnl['total_pnl']:.2f}")

    # Final summary
    pnl = broker.get_daily_pnl()
    print("\n" + "=" * 50)
    print("  PAPER TRADING SUMMARY")
    print("=" * 50)
    print(f"  Total Equity:  ${pnl['total_equity']:.2f}")
    print(f"  Total P&L:     ${pnl['total_pnl']:.2f}")
    print(f"  Win Rate:      {pnl['win_rate']:.1f}%")
    print(f"  Total Trades:  {pnl['total_trades']}")
    print(f"  Open Positions: {pnl['open_positions']}")
    print("=" * 50)


def run_live(config: Config, symbols: list[str]):
    """Run live signal monitoring mode."""
    logger.info(f"Starting live monitoring for {symbols}")

    signal_gen = SignalGenerator(
        confluence_threshold=config.confluence_threshold,
        min_confidence=config.min_confidence,
        tp_r_multiple=config.tp_r_multiple,
        sl_atr_multiple=config.sl_atr_multiple,
        preset=config.strategy_preset,
    )
    notifier = TelegramNotifier(config.telegram_token, config.telegram_chat_id)
    risk_mgr = RiskManager(
        max_daily_drawdown=config.max_daily_drawdown,
        max_consecutive_losses=config.max_consecutive_losses,
    )

    logger.info(f"Monitoring {symbols} every {config.signal_check_interval}s...")
    logger.info("Press Ctrl+C to stop.")

    try:
        cycle = 0
        while True:
            cycle += 1
            for symbol in symbols:
                base = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 150.0
                data = generate_sample_data(200, base)

                signals = signal_gen.generate(
                    symbol=symbol,
                    open_=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    volume=data["volume"],
                )

                for signal in signals:
                    if signal.wait_reason:
                        continue

                    can_trade, reason = risk_mgr.can_trade()
                    if not can_trade:
                        logger.warning(f"Risk block: {reason}")
                        continue

                    logger.info(
                        f"🚨 SIGNAL: {signal.direction} {symbol} @ ${signal.entry:.2f} "
                        f"| SL: ${signal.stop_loss:.2f} | Conf: {signal.confidence:.0f} "
                        f"| Regime: {signal.regime}"
                    )

                    if config.telegram_enabled:
                        notifier.send_signal_sync(signal)

            logger.info(f"Cycle {cycle} complete. Sleeping {config.signal_check_interval}s...")
            time.sleep(config.signal_check_interval)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="NexusTradingDesk")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], default="backtest")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--preset", default="regime-aware")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = Config(args.config)
    symbols = [s.strip() for s in args.symbol.split(",")]

    if args.mode == "backtest":
        for symbol in symbols:
            run_backtest(config, symbol, args.preset or config.strategy_preset)
    elif args.mode == "paper":
        run_paper_trading(config, symbols)
    elif args.mode == "live":
        run_live(config, symbols)


if __name__ == "__main__":
    main()