FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/models

# Default: run backtest
ENV TRADING_MODE=backtest
ENV SYMBOLS=BTC/USDT

CMD ["python", "-m", "src.main", "--mode", "backtest", "--symbol", "BTC/USDT"]