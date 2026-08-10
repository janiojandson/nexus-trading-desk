# NexusTradingDesk — Deployment Plan

## Target Platforms
- **Primary**: Railway (recommended for simplicity)
- **Secondary**: Render (alternative)
- **Budget**: $5-20/month

---

## 1. Railway Deployment

### 1.1 Prerequisites
- Railway account (railway.app)
- GitHub repository connected
- Environment variables configured

### 1.2 Railway Configuration

**railway.json** (place in repo root):
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python -m src.main --mode paper --symbol BTC/USDT,ETH/USDT,SOL/USDT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### 1.3 Environment Variables (Railway Dashboard)

| Variable | Value | Required |
|----------|-------|----------|
| `BINANCE_API_KEY` | Your Binance testnet key | Yes |
| `BINANCE_API_SECRET` | Your Binance testnet secret | Yes |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | Yes |
| `TELEGRAM_CHAT_ID` | Your chat ID | Yes |
| `TRADING_MODE` | `paper` | Yes |
| `SYMBOLS` | `BTC/USDT,ETH/USDT,SOL/USDT` | Yes |
| `STRATEGY_PRESET` | `regime-aware` | No |
| `MAX_DAILY_DRAWDOWN` | `0.15` | No |
| `MAX_CONSECUTIVE_LOSSES` | `5` | No |
| `MAX_POSITIONS` | `3` | No |
| `DATABASE_URL` | Railway PostgreSQL or SQLite | No |

### 1.4 Railway Resource Sizing

| Component | Plan | Cost |
|-----------|------|------|
| Trading Bot | Starter (512MB RAM, 1 vCPU) | $5/month |
| Redis (optional) | Starter | $5/month |
| **Total** | | **$5-10/month** |

### 1.5 Deploy Steps

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Link to repo
railway link

# 5. Set environment variables
railway variables set TRADING_MODE=paper
railway variables set SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
railway variables set TELEGRAM_BOT_TOKEN=your_token
railway variables set TELEGRAM_CHAT_ID=your_chat_id

# 6. Deploy
railway up

# 7. Check logs
railway logs
```

---

## 2. Render Deployment

### 2.1 Render Configuration

**render.yaml** (place in repo root):
```yaml
services:
  - type: web
    name: nexus-trading-desk
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python -m src.main --mode paper --symbol BTC/USDT,ETH/USDT,SOL/USDT
    envVars:
      - key: TRADING_MODE
        value: paper
      - key: PYTHON_VERSION
        value: 3.12.0
    plan: starter
```

### 2.2 Render Resource Sizing

| Component | Plan | Cost |
|-----------|------|------|
| Web Service | Starter (512MB RAM) | $7/month |
| **Total** | | **$7/month** |

---

## 3. Monitoring & Alerting

### 3.1 Health Check Endpoint

The FastAPI dashboard (Phase 3) will expose:
- `GET /api/health` — Returns system status
- Railway/Render can ping this for uptime monitoring

### 3.2 Telegram Self-Monitoring

The bot sends health checks every 6 hours:
- Uptime
- Last signal timestamp
- API connectivity status
- Current equity

### 3.3 Logging

All logs are structured JSON for easy parsing:
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": record.created,
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        })
```

---

## 4. Security Checklist

- [ ] Use exchange **testnet** keys initially
- [ ] Never commit `.env` file
- [ ] Rotate API keys monthly
- [ ] Use Railway's encrypted environment variables
- [ ] Enable 2FA on exchange accounts
- [ ] Set IP whitelist on exchange API keys
- [ ] Use read-only API keys for data fetching
- [ ] Separate trading API keys with minimal permissions
- [ ] Monitor for unauthorized API usage

---

## 5. Scaling Plan

### Phase 1 (Current): Single Instance
- 1 Railway service (512MB RAM)
- SQLite database
- Paper trading only
- **Cost: ~$5/month**

### Phase 2: Production
- 1 Railway service (1GB RAM)
- Railway PostgreSQL
- Redis for caching
- Paper trading + live signals
- **Cost: ~$15/month**

### Phase 3: Multi-Strategy
- 2 Railway services (signal generator + API)
- PostgreSQL + Redis
- RL training on-demand
- Full live trading
- **Cost: ~$20-30/month**

---

## 6. Backup & Recovery

### 6.1 Database Backups
- SQLite: Daily copy to `/data/backups/`
- PostgreSQL: Railway automated daily backups

### 6.2 Model Backups
- RL models saved to `/models/` with timestamps
- Keep last 5 model versions

### 6.3 Configuration
- All config in `config.yaml` (version controlled)
- Environment variables in Railway dashboard
- `.env.example` documents required variables

---

## 7. CI/CD Pipeline

### GitHub Actions (already in repo)

```yaml
name: Test & Deploy
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: railwayapp/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
```

---

## 8. Quick Start Checklist

- [ ] Fork/clone the repository
- [ ] Create `.env` from `.env.example`
- [ ] Get Binance testnet API keys
- [ ] Create Telegram bot via @BotFather
- [ ] Get your Telegram chat ID
- [ ] Run `pip install -r requirements.txt`
- [ ] Run `python -m src.main --mode backtest --symbol BTC/USDT`
- [ ] Verify backtest results
- [ ] Run `python -m src.main --mode paper`
- [ ] Verify Telegram alerts
- [ ] Deploy to Railway
- [ ] Monitor for 24h before considering live trading