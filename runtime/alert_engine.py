from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


ALERT_HISTORY_PATH = "out/alerts.jsonl"
ALERT_STATE_PATH = "out/alert_state.json"
LATEST_ALERT_PATH = "out/latest_alert.json"


def _ensure_out_dir() -> None:
    os.makedirs("out", exist_ok=True)


def _load_state() -> Dict[str, str]:
    _ensure_out_dir()
    if not os.path.exists(ALERT_STATE_PATH):
        return {}

    try:
        with open(ALERT_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(data: Dict[str, str]) -> None:
    _ensure_out_dir()
    with open(ALERT_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _minutes_since(older_iso: str, now: datetime) -> Optional[float]:
    try:
        older = datetime.fromisoformat(older_iso)
        if now.tzinfo is not None and older.tzinfo is None:
            older = older.replace(tzinfo=now.tzinfo)
        elif now.tzinfo is None and older.tzinfo is not None:
            now = now.replace(tzinfo=older.tzinfo)
        return max(0.0, (now - older).total_seconds() / 60.0)
    except Exception:
        return None


def emit_alert(
    *,
    alert_type: str,
    title: str,
    body: str,
    severity: str,
    created_at: datetime,
    dedup_key: Optional[str] = None,
    cooldown_minutes: int = 30,
    signal_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    _ensure_out_dir()
    state = _load_state()

    if dedup_key:
        last_seen = state.get(dedup_key)
        if last_seen is not None:
            minutes = _minutes_since(last_seen, created_at)
            if minutes is not None and minutes < cooldown_minutes:
                return None

    payload = {
        "id": f"alert_{uuid.uuid4().hex[:12]}",
        "timestamp": created_at.isoformat(),
        "type": alert_type,
        "title": title,
        "body": body,
        "severity": severity,
        "signal_id": signal_id,
        "metadata": metadata or {},
    }

    with open(ALERT_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    with open(LATEST_ALERT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if dedup_key:
        state[dedup_key] = payload["timestamp"]
        _save_state(state)

    return payload


def load_alerts(limit: int = 100) -> List[Dict[str, Any]]:
    if not os.path.exists(ALERT_HISTORY_PATH):
        return []

    rows: List[Dict[str, Any]] = []
    try:
        with open(ALERT_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []

    return rows[-limit:][::-1]


def load_latest_alert() -> Optional[Dict[str, Any]]:
    if not os.path.exists(LATEST_ALERT_PATH):
        return None

    try:
        with open(LATEST_ALERT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
