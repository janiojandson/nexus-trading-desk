"""
Telegram bot for trading alerts.

Sends:
- Trade signals (real-time)
- Daily P&L summary (21:00 UTC)
- Drawdown alerts (>10%)
- Regime change notifications
- Weekly report (Sunday 18:00 UTC)
- System health checks (every 6h)

Ported from TradeClaw's packages/telegram-bot/.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from src.signals.generator import Signal

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification bot.

    Uses the Telegram Bot API directly (no python-telegram-bot dependency
    required for the prototype). Supports both polling and webhook modes.
    """

    API_BASE = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._enabled = bool(token and chat_id)

    async def _post(self, method: str, body: dict) -> dict:
        """Make a Telegram API request."""
        import aiohttp
        url = self.API_BASE.format(token=self.token, method=method)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.error(f"Telegram API error ({method}): {data.get('description')}")
                return data

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a text message."""
        if not self._enabled:
            logger.debug("Telegram disabled, skipping message")
            return False

        try:
            result = await self._post("sendMessage", {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            })
            return result.get("ok", False)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    # -----------------------------------------------------------------------
    # Signal Alerts
    # -----------------------------------------------------------------------

    async def send_signal(self, signal: Signal) -> bool:
        """Send a trade signal alert."""
        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
        tp_text = ", ".join(f"${tp:.2f}" for tp in signal.take_profits)

        text = (
            f"{direction_emoji} *{signal.direction}* — {signal.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 Entry: `${signal.entry:.2f}`\n"
            f"🛑 Stop Loss: `${signal.stop_loss:.2f}`\n"
            f"🎯 Take Profits: {tp_text}\n"
            f"💪 Confidence: `{signal.confidence:.0f}/100`\n"
            f"🔄 Regime: `{signal.regime}`\n"
            f"📐 Strategy: `{signal.preset}`\n"
            f"📝 Reasons: {', '.join(signal.reasons[:3])}\n"
        )

        if signal.wait_reason:
            text += f"⚠️ *WAIT*: {signal.wait_reason}\n"

        return await self.send_message(text)

    # -----------------------------------------------------------------------
    # Daily P&L
    # -----------------------------------------------------------------------

    async def send_daily_pnl(self, pnl_data: dict) -> bool:
        """Send daily P&L summary."""
        emoji = "📈" if pnl_data.get("total_pnl", 0) >= 0 else "📉"

        text = (
            f"{emoji} *Daily P&L Report*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Realized: `${pnl_data.get('realized_pnl', 0):.2f}`\n"
            f"📊 Unrealized: `${pnl_data.get('unrealized_pnl', 0):.2f}`\n"
            f"💵 Total: `${pnl_data.get('total_pnl', 0):.2f}`\n"
            f"🏦 Equity: `${pnl_data.get('total_equity', 0):.2f}`\n"
            f"✅ Win Rate: `{pnl_data.get('win_rate', 0):.1f}%`\n"
            f"📋 Trades: `{pnl_data.get('total_trades', 0)}`\n"
            f"📂 Open: `{pnl_data.get('open_positions', 0)}`\n"
        )

        return await self.send_message(text)

    # -----------------------------------------------------------------------
    # Drawdown Alert
    # -----------------------------------------------------------------------

    async def send_drawdown_alert(self, drawdown_pct: float, equity: float) -> bool:
        """Send drawdown warning alert."""
        severity = "⚠️" if drawdown_pct < 15 else "🚨"

        text = (
            f"{severity} *DRAWDOWN ALERT*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📉 Current Drawdown: `{drawdown_pct:.1f}%`\n"
            f"🏦 Current Equity: `${equity:.2f}`\n"
        )

        if drawdown_pct >= 15:
            text += "🛑 *HARD STOP ACTIVATED* — Trading paused\n"

        return await self.send_message(text)

    # -----------------------------------------------------------------------
    # Regime Change
    # -----------------------------------------------------------------------

    async def send_regime_change(self, symbol: str, old_regime: str, new_regime: str) -> bool:
        """Send regime change notification."""
        regime_emoji = {"trend": "📈", "volatile": "⚡", "range": "↔️", "unknown": "❓"}

        text = (
            f"🔄 *REGIME CHANGE* — {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{regime_emoji.get(old_regime, '❓')} From: `{old_regime}`\n"
            f"{regime_emoji.get(new_regime, '❓')} To: `{new_regime}`\n"
        )

        return await self.send_message(text)

    # -----------------------------------------------------------------------
    # Weekly Report
    # -----------------------------------------------------------------------

    async def send_weekly_report(self, report: dict) -> bool:
        """Send weekly performance report."""
        text = (
            f"📊 *Weekly Trading Report*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Total P&L: `${report.get('total_pnl', 0):.2f}`\n"
            f"✅ Win Rate: `{report.get('win_rate', 0):.1f}%`\n"
            f"📋 Total Trades: `{report.get('total_trades', 0)}`\n"
            f"📉 Max Drawdown: `{report.get('max_drawdown', 0):.1f}%`\n"
            f"📈 Sharpe Ratio: `{report.get('sharpe_ratio', 0):.2f}`\n"
        )

        # Strategy comparison
        if "strategies" in report:
            text += "\n*Strategy Comparison:*\n"
            for name, metrics in report["strategies"].items():
                text += f"  • {name}: WR={metrics.get('win_rate', 0):.0f}% | PF={metrics.get('profit_factor', 0):.2f}\n"

        return await self.send_message(text)

    # -----------------------------------------------------------------------
    # System Health
    # -----------------------------------------------------------------------

    async def send_health_check(self, status: dict) -> bool:
        """Send system health check."""
        uptime = status.get("uptime_hours", 0)
        last_signal = status.get("last_signal_ago", "N/A")

        text = (
            f"🏥 *System Health*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⏱ Uptime: `{uptime:.1f}h`\n"
            f"📡 Last Signal: `{last_signal}`\n"
            f"🟢 Status: `operational`\n"
        )

        return await self.send_message(text)

    # -----------------------------------------------------------------------
    # Synchronous wrapper for non-async contexts
    # -----------------------------------------------------------------------

    def send_signal_sync(self, signal: Signal) -> bool:
        """Synchronous wrapper for send_signal."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, schedule it
                asyncio.ensure_future(self.send_signal(signal))
                return True
            return loop.run_until_complete(self.send_signal(signal))
        except RuntimeError:
            return asyncio.run(self.send_signal(signal))