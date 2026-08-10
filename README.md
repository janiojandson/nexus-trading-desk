# NexusTradingDesk

AI-Powered Crypto Trading System — SMC/ICT pattern detection, RL strategy optimization, advanced backtesting, paper trading, Telegram alerts.

**Unites the best of [QuantDesk](https://github.com/MOHAMMAD-a12/quantdesk) + [TradeClaw](https://github.com/naimkatiman/tradeclaw)**

## Features

- **SMC/ICT Analysis**: Order Blocks, Fair Value Gaps, Liquidity Pools/Sweeps, Break of Structure
- **Pattern Recognition**: Engulfing, Hammer, Doji, Morning/Evening Star
- **Regime Classification**: Trend / Volatile / Range via HMM-inspired classifier
- **5 Strategy Presets**: Classic, HMM Top-3, Regime-Aware, VWAP+EMA+BB, Full-Risk Pipeline
- **Advanced Backtesting**: Sharpe, Sortino, Max DD, Profit Factor, Win Rate, Equity Curve
- **Paper Trading**: Simulated execution with position tracking
- **RL Optimization**: PPO agent for strategy parameter tuning
- **Telegram Bot**: Real-time signals, daily P&L, drawdown alerts, regime changes
- **Multi-Exchange**: Binance, Bybit, OKX via ccxt
- **Risk Management**: Circuit breaker, drawdown tracker, Kelly sizing

## Quick Start

```bash
# Clone
git clone https://github.com/janiojandson/nexus-trading-desk.git
cd nexus-trading-desk

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys and Telegram token

# Run backtest
python -m src.main --mode backtest --symbol BTC/USDT --preset regime-aware

# Run paper trading
python -m src.main --mode paper --symbol BTC/USDT,ETH/USDT,SOL/USDT

# Run with Telegram alerts
python -m src.main --mode live --telegram
```

## Docker

```bash
docker build -t nexus-trading-desk .
docker run --env-file .env nexus-trading-desk
```

## Architecture

See [SPEC.md](./SPEC.md) for the full specification.

## Project Structure

```
nexus-trading-desk/
├── SPEC.md                    # Full specification
├── DEPLOY.md                  # Deployment plan
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── config.py             # Configuration loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── exchange.py       # ccxt multi-exchange adapter
│   │   └── store.py          # SQLite OHLCV store
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── indicators.py     # Technical indicators
│   │   ├── smc.py            # SMC/ICT detector
│   │   ├── patterns.py       # Candlestick patterns
│   │   ├── regime.py         # Regime classifier
│   │   └── confluence.py     # Confluence scorer
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── generator.py      # Signal generation engine
│   │   └── presets.py        # Strategy presets
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── backtest.py       # Backtesting engine
│   │   ├── paper.py          # Paper trading broker
│   │   └── risk.py           # Risk manager
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── env.py            # Gymnasium trading environment
│   │   └── agent.py          # PPO agent trainer
│   └── notifications/
│       ├── __init__.py
│       └── telegram_bot.py    # Telegram bot
└── tests/
    ├── __init__.py
    ├── test_indicators.py
    ├── test_smc.py
    ├── test_backtest.py
    └── test_signals.py
```

## License

MIT