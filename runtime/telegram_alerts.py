from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import parse, request, error

from runtime.runtime_config import load_runtime_config


@dataclass(frozen=True)
class TelegramAlertConfig:
    enabled: bool
    configured: bool
    bot_token: str
    chat_id: str
    api_url: str
    min_score: int
    min_calibrated_score: int
    cooldown_minutes: int


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _cfg_value(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


def get_telegram_config(runtime_cfg: Optional[Dict[str, Any]] = None) -> TelegramAlertConfig:
    cfg = runtime_cfg or load_runtime_config()

    enabled = bool(_cfg_value(cfg, "telegram_alerts_enabled", False))
    if os.getenv("TELEGRAM_ALERTS_ENABLED", "").strip():
        enabled = _env_bool("TELEGRAM_ALERTS_ENABLED", enabled)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", str(_cfg_value(cfg, "telegram_bot_token", ""))).strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", str(_cfg_value(cfg, "telegram_chat_id", ""))).strip()

    configured = bool(bot_token and chat_id)
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage" if configured else ""

    return TelegramAlertConfig(
        enabled=enabled,
        configured=configured,
        bot_token=bot_token,
        chat_id=chat_id,
        api_url=api_url,
        min_score=int(_cfg_value(cfg, "telegram_alert_min_score", _cfg_value(cfg, "alert_min_score", 74))),
        min_calibrated_score=int(
            _cfg_value(cfg, "telegram_alert_min_calibrated_score", _cfg_value(cfg, "alert_min_score", 74))
        ),
        cooldown_minutes=int(_cfg_value(cfg, "telegram_alert_cooldown_minutes", _cfg_value(cfg, "alert_cooldown_minutes", 30))),
    )


def telegram_status(runtime_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_telegram_config(runtime_cfg)
    return {
        "enabled": cfg.enabled,
        "configured": cfg.configured,
        "bot_token_set": bool(cfg.bot_token),
        "chat_id_set": bool(cfg.chat_id),
        "min_score": cfg.min_score,
        "min_calibrated_score": cfg.min_calibrated_score,
        "cooldown_minutes": cfg.cooldown_minutes,
    }


def _truncate(text: str, limit: int = 3500) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_telegram_message(payload: Dict[str, Any]) -> str:
    title = str(payload.get("title", "Alert"))
    body = str(payload.get("body", ""))
    alert_type = str(payload.get("type", "alert"))
    severity = str(payload.get("severity", "info")).upper()
    timestamp = str(payload.get("timestamp", ""))
    signal_id = str(payload.get("signal_id", ""))
    lines = [
        f"<b>{title}</b>",
        f"<b>Type:</b> {alert_type}",
        f"<b>Severity:</b> {severity}",
    ]
    if timestamp:
        lines.append(f"<b>Time:</b> {timestamp}")
    if signal_id:
        lines.append(f"<b>Signal:</b> {signal_id}")
    if body:
        lines.append("")
        lines.append(body)
    return _truncate("\n".join(lines))


def send_telegram_message(
    message: str,
    *,
    runtime_cfg: Optional[Dict[str, Any]] = None,
    parse_mode: str = "HTML",
) -> bool:
    cfg = get_telegram_config(runtime_cfg)
    if not (cfg.enabled and cfg.configured):
        return False

    payload = {
        "chat_id": cfg.chat_id,
        "text": _truncate(message),
        "disable_web_page_preview": True,
        "parse_mode": parse_mode,
    }

    data = parse.urlencode(payload).encode("utf-8")
    req = request.Request(cfg.api_url, data=data, method="POST")
    try:
        with request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception:
        return False


def send_telegram_alert(
    payload: Dict[str, Any],
    *,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    message = format_telegram_message(payload)
    return send_telegram_message(message, runtime_cfg=runtime_cfg)


def send_telegram_test_message(
    *,
    message: str = "Telegram test from Trading Assistant",
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    return send_telegram_message(message, runtime_cfg=runtime_cfg)
