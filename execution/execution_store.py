import json
import os
from typing import Dict, Any

EXECUTION_STORE_PATH = "out/execution_notes.json"


def _ensure_file() -> None:
    os.makedirs("out", exist_ok=True)
    if not os.path.exists(EXECUTION_STORE_PATH):
        with open(EXECUTION_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def load_execution_store() -> Dict[str, Any]:
    _ensure_file()
    try:
        with open(EXECUTION_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_execution_store(data: Dict[str, Any]) -> None:
    os.makedirs("out", exist_ok=True)
    with open(EXECUTION_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_execution_note(signal_id: str) -> Dict[str, Any]:
    store = load_execution_store()
    note = store.get(signal_id, {})
    return note if isinstance(note, dict) else {}


def set_execution_note(signal_id: str, note: Dict[str, Any]) -> None:
    store = load_execution_store()
    base = store.get(signal_id, {})
    if not isinstance(base, dict):
        base = {}

    merged = base.copy()
    merged.update(note)

    store[signal_id] = merged
    save_execution_store(store)