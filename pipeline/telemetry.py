import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TELEMETRY_FILE = os.path.join(ROOT, 'state', 'telemetry.json')

PRICING = {
    "grok-beta": {"prompt": 0.0000025, "completion": 0.00001},
    # Default pricing (USD per token). Can be overridden via env:
    # - GROK_4_1_FAST_INPUT_PRICE  (USD per 1K prompt tokens)
    # - GROK_4_1_FAST_OUTPUT_PRICE (USD per 1K completion tokens)
    # Requested: input=$0.20/1K, output=$0.50/1K
    "grok-4-1-fast": {"prompt": 0.0002, "completion": 0.0005},
    "grok-4-fast": {"prompt": 0.000003, "completion": 0.000015},
    "gpt-4": {"prompt": 0.00003, "completion": 0.00006},
    "gpt-4o": {"prompt": 0.0000025, "completion": 0.00001},
    "claude-3-opus": {"prompt": 0.000015, "completion": 0.000075},
    "claude-3-sonnet": {"prompt": 0.000003, "completion": 0.000015},
}

_state_lock = threading.Lock()


def _load_state() -> Dict[str, Any]:
    if os.path.exists(TELEMETRY_FILE):
        try:
            with open(TELEMETRY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "run_id": None,
        "status": "idle",
        "current_stage": "",
        "stage_index": 0,
        "total_stages": 4,
        "stage_percent": 0.0,
        "overall_percent": 0.0,
        "completed_units": 0,
        "total_units": 0,
        "eta_seconds": 0,
        "usage": {
            "provider": "",
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "spend_usd": 0.0
        }
    }


def _save_state(state: Dict[str, Any]):
    os.makedirs(os.path.dirname(TELEMETRY_FILE), exist_ok=True)
    with open(TELEMETRY_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def init_run(run_id: str, total_stages: int = 4):
    with _state_lock:
        state = _load_state()
        state["run_id"] = run_id
        state["status"] = "running"
        state["stage_index"] = 0
        state["total_stages"] = total_stages
        state["stage_percent"] = 0.0
        state["overall_percent"] = 0.0
        state["completed_units"] = 0
        state["total_units"] = 0
        state["usage"] = {
            "provider": "",
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "spend_usd": 0.0
        }
        _save_state(state)


def update_progress(stage_name: str, stage_index: int, total_stages: int,
                   completed_units: int, total_units: int, eta_seconds: int = 0):
    with _state_lock:
        state = _load_state()
        state["current_stage"] = stage_name
        state["stage_index"] = stage_index
        state["total_stages"] = total_stages
        state["completed_units"] = completed_units
        state["total_units"] = total_units
        if total_units > 0:
            state["stage_percent"] = (completed_units / total_units) * 100
        else:
            state["stage_percent"] = 0.0
        state["eta_seconds"] = eta_seconds
        if total_stages > 0:
            stage_weight = 100.0 / total_stages
            state["overall_percent"] = (stage_index * stage_weight) + (stage_weight * state["stage_percent"] / 100)
        _save_state(state)


def record_api_call(provider: str, model: str, prompt_tokens: int = 0,
                    completion_tokens: int = 0, success: bool = True):
    with _state_lock:
        state = _load_state()
        if state["usage"]["provider"] != provider:
            state["usage"]["provider"] = provider
        state["usage"]["model"] = model
        state["usage"]["request_count"] += 1
        if success:
            state["usage"]["successful_requests"] += 1
            state["usage"]["prompt_tokens"] += prompt_tokens
            state["usage"]["completion_tokens"] += completion_tokens
            total = prompt_tokens + completion_tokens
            state["usage"]["total_tokens"] += total
            spend = _calculate_spend(model, prompt_tokens, completion_tokens)
            state["usage"]["spend_usd"] += spend
        else:
            state["usage"]["failed_requests"] += 1
        _save_state(state)


def _calculate_spend(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Optional env overrides (USD per 1K tokens)
    if model and "grok-4-1-fast" in model.lower():
        try:
            in_per_1k = float(os.environ.get("GROK_4_1_FAST_INPUT_PRICE", ""))
            out_per_1k = float(os.environ.get("GROK_4_1_FAST_OUTPUT_PRICE", ""))
            return (prompt_tokens / 1000.0) * in_per_1k + (completion_tokens / 1000.0) * out_per_1k
        except Exception:
            pass

    model_key = model.lower().replace("-", "_").replace(".", "_")
    for key, prices in PRICING.items():
        if key.lower().replace("-", "_").replace(".", "_") in model_key:
            return (prompt_tokens * prices["prompt"]) + (completion_tokens * prices["completion"])
    return 0.0


def set_status(status: str):
    with _state_lock:
        state = _load_state()
        state["status"] = status
        _save_state(state)


def get_telemetry() -> Dict[str, Any]:
    return _load_state()


def reset():
    with _state_lock:
        if os.path.exists(TELEMETRY_FILE):
            os.remove(TELEMETRY_FILE)
