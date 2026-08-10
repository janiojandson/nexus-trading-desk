# NexusTradingDesk — Specification Document

> AI-Powered Crypto Trading System uniting the best of **QuantDesk** (SMC/ICT analysis, AI signal generation) and **TradeClaw** (multi-strategy presets, regime-aware backtesting, paper trading, Telegram bot).

---

## 1. Architecture Overview

```
+---------------------+------------------+------------------+---------------------+
|     Data Layer      |  Analysis Engine |  Execution Engine|  Notification Layer |
+---------------------+------------------+------------------+---------------------+
| +-----------------+ | +--------------+ | +--------------+ | +-----------------+ |
| | Exchange        | | | SMC/ICT      | | | Backtest     | | | Telegram Bot    | |
| | Adapters        | | | Detector     | | | Engine       | | |  - Signals      | |
| |  - Binance      | | |              | | |              | | |  - P&L Daily    | |
| |  - Bybit        | | | Pattern      | | | Paper        | | |  - Drawdown     | |
| |  - OKX          | | | Recognizer   | | | Trading      | | |  - Regime       | |
| +-----------------+ | |              | | |              | | +-----------------+ |
|                     | | Indicator    | | | RL Agent     | | +-----------------+ |
| +-----------------+ | | Suite        | | | Optimizer    | | | Dashboard API   | |
| | OHLCV Store     | | |              | | |              | | |  REST + WS      | |
| |  - SQLite       | | | Regime       | | | Risk         | | |                 | |
| +-----------------+ | | Classifier   | | | Manager      | | +-----------------+ |
|                     | |              | | |              | |                     |
|                     | | Confluence   | | | Position     | |                     |
|                     | | Scorer       | | | Tracker     | |                     |
|                     | +--------------+ | +--------------+ |                     |
|                     | +--------------+ |                  |                     |
|                     | | RL Module    | |                  |                     |
|                     | |  PPO/DQN     | |                  |                     |
|                     | +--------------+ |                  |                     |
+---------------------+------------------+------------------+---------------------+
```

## 2. Module Specifications

### 2.1 Data Layer — Exchange Adapters

| Feature | Detail |
|---------|--------|
| Exchanges | Binance, Bybit, OKX |
| Data | OHLCV candles (1m-1d), ticker, order book depth L2 |
| Rate Limits | Per-exchange rate limiter with backoff |
| Caching | SQLite local store, 5-min candle cache |
| WebSocket | Real-time price streaming |

### 2.2 Analysis Engine

#### 2.2.1 SMC/ICT Detector

| Pattern | Description | Source |
|---------|-------------|--------|
| Swing Points | Fractal pivots with configurable strength (3-5 bars) | QuantDesk smc.ts |
| Break of Structure (BOS) | Close-based (not wick) structure breaks | QuantDesk smc.ts |
| Order Blocks | Last opposing candle before impulsive move | QuantDesk smc.ts |
| Fair Value Gaps | 3-candle gaps where body extremes don't overlap | QuantDesk smc.ts |
| Liquidity Pools | Clusters of equal highs/lows | QuantDesk smc.ts |
| Liquidity Sweeps | Wick beyond pool without close confirmation | QuantDesk smc.ts |
| Supply/Demand Zones | Consolidation before impulsive moves | QuantDesk smc.ts |

#### 2.2.2 Pattern Recognizer

| Pattern | Description |
|---------|-------------|
| Engulfing (Bullish/Bearish) | Body-conformity + ATR significance + context |
| Hammer/Shooting Star | Lower/upper wick ratio + prior trend |
| Doji | Body ratio < 10% of range |
| Morning/Evening Star | 3-candle reversal with gap |
| Three White Soldiers/Black Crows | 3 consecutive directional candles |

#### 2.2.3 Indicator Suite

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| RSI | 14 | Momentum, divergence |
| MACD | 12/26/9 | Trend, crossover signals |
| EMA | 9, 20, 50, 200 | Trend direction, dynamic S/R |
| Bollinger Bands | 20, 2-sigma | Volatility, mean-reversion |
| ATR | 14 | Volatility, stop sizing |
| Stochastic | 14/3/3 | Overbought/oversold |
| ADX | 14 | Trend strength |
| VWAP | Session | Institutional reference |
| Ichimoku | 9/26/52 | Multi-purpose trend system |
| Volume Profile | Session | High-volume nodes |

#### 2.2.4 Regime Classifier

| Regime | Characteristics | Allowed Directions |
|--------|----------------|-------------------|
| Trend | ADX > 25, directional EMA stack | BUY (uptrend) / SELL (downtrend) |
| Volatile | ATR spike, wide BB | Both (mean-revert) |
| Range | ADX < 20, BB squeeze | Both (fade extremes) |

Uses HMM (Hidden Markov Model) with Viterbi decoding for regime transitions.

#### 2.2.5 Confluence Scorer

Weighted scoring across categories:
- **SMC Confluence (30%)**: Order Block + FVG alignment
- **Pattern Confluence (20%)**: Candlestick at key level
- **Indicator Confluence (25%)**: RSI + MACD + EMA alignment
- **Multi-Timeframe (15%)**: Higher TF trend permission
- **Regime Alignment (10%)**: Signal matches regime bias

Minimum confluence threshold: **65/100** for signal emission.

### 2.3 Signal Generation

| Field | Type | Description |
|-------|------|-------------|
| symbol | string | BTC/USDT, ETH/USDT, SOL/USDT |
| direction | BUY/SELL | Trade direction |
| entry | number | Entry price |
| stopLoss | number | ATR-derived stop |
| takeProfits | number[] | 1:1, 1:2, 1:3 R:R targets |
| confidence | 0-100 | Weighted confluence score |
| regime | string | Current market regime |
| reasons | string[] | Machine-readable signal reasons |
| waitReason | string? | Why signal was rejected (if WAIT) |

**Strategy Presets** (from TradeClaw):
1. **Classic** — RSI + MACD + EMA + Stochastic + BB
2. **HMM Top-3** — Regime-classified with HMM
3. **Regime-Aware** — Classic gated by regime classifier
4. **VWAP+EMA+BB** — Intraday mean-reversion
5. **Full-Risk Pipeline** — All signals + circuit breaker + drawdown tracker

### 2.4 Backtesting Engine

| Metric | Formula | Target |
|--------|---------|--------|
| Win Rate | Wins / Total Trades | > 55% |
| Sharpe Ratio | (Mean Return - Rf) / Std Return | > 1.5 |
| Sortino Ratio | (Mean Return - Rf) / Downside Std | > 2.0 |
| Max Drawdown | Peak-to-trough decline | < 20% |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |
| Total Return | (End - Start) / Start | > 30% annualized |
| Avg Trade P&L | Mean trade profit | Positive |

**Features**:
- Multi-preset comparison (run all 5 strategies side-by-side)
- ATR-based or fixed TP/SL geometry
- Cost model (fees + slippage + funding)
- Equity curve generation
- Walk-forward validation

### 2.5 Paper Trading

| Feature | Detail |
|---------|--------|
| Initial Equity | $10,000 (configurable) |
| Order Types | Market, Limit |
| Position Tracking | Long/Short with unrealized P&L |
| Fill Simulation | Instant at requested price |
| Metrics | Win rate, P&L, drawdown, Sharpe |
| Reset | Full account reset capability |

### 2.6 RL Optimization Module

| Component | Detail |
|-----------|--------|
| Algorithm | PPO (Proximal Policy Optimization) |
| State Space | [RSI, MACD_hist, EMA_trend, ATR, BB_position, regime, P&L] |
| Action Space | [hold, buy, sell, close] |
| Reward | Risk-adjusted return (Sharpe-like) |
| Training | Stable-Baselines3 + gymnasium |
| Frequency | Weekly retraining on latest data |
| Deployment | Best model saved, loaded for live signals |

### 2.7 Risk Manager

| Component | Detail |
|-----------|--------|
| Circuit Breaker | Pause trading after N consecutive losses |
| Drawdown Tracker | Hard stop at 15% daily drawdown |
| Risk Veto | Reject signals below minimum confidence |
| Position Sizing | Kelly criterion (fractional, 25% Kelly) |
| Max Positions | 3 concurrent (configurable) |
| Max Leverage | 3x (configurable) |

### 2.8 Telegram Bot

| Alert Type | Frequency | Content |
|-----------|-----------|---------|
| Trade Signal | Real-time | Symbol, direction, entry, SL, TP, confidence, regime |
| P&L Daily | 21:00 UTC | Day's trades, realized P&L, win rate |
| Drawdown Alert | Real-time | When drawdown exceeds 10% |
| Regime Change | Real-time | New regime detected with implications |
| Weekly Report | Sunday 18:00 UTC | Full week stats, strategy comparison |
| System Health | Every 6h | Bot uptime, last signal time, API status |

### 2.9 Dashboard API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/signals` | GET | Active signals |
| `/api/signals/history` | GET | Historical signals with filters |
| `/api/backtest` | POST | Run backtest with params |
| `/api/paper-trading` | GET | Current positions & P&L |
| `/api/paper-trading/open` | POST | Open paper position |
| `/api/paper-trading/close` | POST | Close paper position |
| `/api/health` | GET | System health check |
| `/api/regime/{symbol}` | GET | Current regime |

## 3. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.12 | Ecosystem: pandas, numpy, ta-lib, stable-baselines3 |
| API | FastAPI | Async, auto-docs, WebSocket support |
| Data | SQLite + Redis | Lightweight, Railway-compatible |
| RL | Stable-Baselines3 + gymnasium | Industry standard PPO/DQN |
| Indicators | ta-lib + custom | Fast C library + SMC/ICT custom |
| Exchange | ccxt | Unified multi-exchange API |
| Telegram | python-telegram-bot | Mature, async, webhook support |
| Backtesting | Custom engine | Full control over metrics & comparison |
| Deployment | Docker + Railway | $5-20/month, auto-scaling |
| Monitoring | Prometheus + Grafana | Optional, for production |

## 4. Data Flow

1. Exchange (Binance/Bybit/OKX) -> ccxt WebSocket+REST -> OHLCV Store (SQLite)
2. OHLCV -> Indicators -> SMC/ICT Detector -> Pattern Recognizer -> Confluence Scorer
3. Confluence Score -> RL Agent + Risk Manager -> EXECUTE or WAIT
4. Execute -> Paper Trading + Telegram Bot + Dashboard API

## 5. Configuration

```yaml
# config.yaml
exchanges:
  binance:
    enabled: true
    testnet: true
  bybit:
    enabled: true
    testnet: true
  okx:
    enabled: false

symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT

timeframes:
  - 5m
  - 15m
  - 1h
  - 4h
  - 1d

strategy:
  preset: regime-aware
  confluence_threshold: 65
  min_confidence: 58

risk:
  max_daily_drawdown: 0.15
  max_consecutive_losses: 5
  position_size_kelly_fraction: 0.25
  max_positions: 3
  max_leverage: 3

backtest:
  initial_balance: 10000
  geometry: atr
  tp_r_multiple: 2.0
  sl_atr_multiple: 1.5
  cost_model:
    maker_fee: 0.001
    taker_fee: 0.0015
    slippage: 0.0005

rl:
  algorithm: PPO
  learning_rate: 0.0003
  total_timesteps: 100000
  retrain_interval_days: 7

telegram:
  enabled: true
  alerts:
    signals: true
    daily_pnl: true
    drawdown_alert: true
    regime_change: true
    weekly_report: true

scheduler:
  signal_check_interval: 300
  backtest_cron: "0 6 * * *"
  rl_retrain_cron: "0 2 * * 0"
```

## 6. Success Metrics

| KPI | Target | Measurement |
|-----|--------|-------------|
| Signal Accuracy | > 55% win rate | Backtested + Paper |
| Sharpe Ratio | > 1.5 | Backtested |
| Max Drawdown | < 20% | Paper trading |
| Signal Frequency | 3-8/day per symbol | Live monitoring |
| Alert Latency | < 30s | Signal -> Telegram |
| Uptime | > 99% | Health checks |
| Cost | < $20/month | Railway bill |

## 7. Roadmap

### Phase 1 (Week 1-2): Foundation
- Exchange adapters (ccxt)
- OHLCV store (SQLite)
- Indicator suite
- SMC/ICT detector
- Pattern recognizer
- Signal generator with confluence scoring

### Phase 2 (Week 3-4): Backtesting + Paper
- Backtesting engine with full metrics
- Multi-preset comparison
- Paper trading broker
- Risk manager (circuit breaker, drawdown tracker)

### Phase 3 (Week 5-6): RL + Telegram
- RL training environment (gymnasium)
- PPO agent training pipeline
- Telegram bot (signals, P&L, alerts)
- Dashboard API (FastAPI)

### Phase 4 (Week 7-8): Production
- Docker + Railway deployment
- Monitoring & alerting
- Walk-forward validation
- Live trading (testnet -> mainnet)