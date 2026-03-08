#!/usr/bin/env python3
"""
================================================================================
heidi_engine/telemetry.py - Event Bus and State Management for AutoTraining Pipeline
================================================================================

PURPOSE:
    Provides a centralized event emission system and state aggregation for
    the entire training pipeline. This enables:
    1. Real-time monitoring via dashboard
    2. Graceful pause/stop/resume functionality
    3. Accurate usage tracking (API tokens, costs)
    4. Debugging via event log replay

SECURITY:
    - HTTP server binds to 127.0.0.1 only (never 0.0.0.0)
    - All sensitive data is redacted from events and HTTP responses
    - File permissions are set to 0600 where possible
    - No env vars, tokens, or raw prompts exposed

EVENT SCHEMA (v1.0 - FROZEN):
    {
        "event_version": "1.0",
        "ts": "ISO8601 timestamp",
        "run_id": "unique run identifier",
        "round": 1,
        "stage": "generate|validate|train|eval|round_start|round_end",
        "level": "info|warn|error|success",
        "event_type": "stage_start|stage_end|progress|error|pipeline_start|pipeline_stop|pipeline_complete",
        "message": "human-readable message (truncated to 500 chars)",
        "counters_delta": {},
        "usage_delta": {},
        "artifact_paths": [],
        "prev_hash": "checksum of previous event"
    }

STATE SCHEMA (state.json):
    {
        "run_id": "unique identifier",
        "status": "running|paused|stopped|completed|error",
        "current_round": 1,
        "current_stage": "generate",
        "stop_requested": false,
        "pause_requested": false,
        "counters": {...},
        "usage": {...},
        "started_at": "ISO8601",
        "updated_at": "ISO8601"
    }

CONFIG VALIDATION:
    - Strict schema: BASE_MODEL, TEACHER_MODEL, SAMPLES_PER_ROUND, ROUNDS, etc.
    - Fails fast with clear error messages

================================================================================
"""

import atexit
import json
import os
import re
import stat
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from heidi_engine.state_machine import CANONICAL_AUTOTRAIN_DIR

try:
    import requests
except ImportError:
    requests = None

# Store remote states in memory
_remote_states: Dict[str, Any] = {}

# =============================================================================
# CONFIGURATION - Adjust these for your needs
# =============================================================================

AUTOTRAIN_DIR = os.environ.get("AUTOTRAIN_DIR", str(CANONICAL_AUTOTRAIN_DIR))

# Unique run identifier - set by loop.sh or menu.py
# TUNABLE: Auto-generated if not provided
RUN_ID = os.environ.get("RUN_ID", "")

# Batch size for event flushing - reduces IO overhead
# TUNABLE: Increase for less frequent writes, decrease for real-time updates
TELEMETRY_BATCH = int(os.environ.get("TELEMETRY_BATCH", "10"))

# HTTP status server port (0 to disable)
# TUNABLE: Change port if 7779 conflicts with another service
HTTP_STATUS_PORT = int(os.environ.get("HTTP_STATUS_PORT", "7779"))

# Pricing configuration file path
# TUNABLE: Point to custom pricing file for different API providers
PRICING_CONFIG_PATH = os.environ.get("PRICING_CONFIG_PATH", "")

# Event log rotation
# TUNABLE: Max size in MB before rotation
EVENT_LOG_MAX_SIZE_MB = int(os.environ.get("EVENT_LOG_MAX_SIZE_MB", "100"))
# Number of rotated log files to keep
EVENT_LOG_RETENTION = int(os.environ.get("EVENT_LOG_RETENTION", "5"))

# =============================================================================
# EVENT SCHEMA VERSION (FROZEN - DO NOT CHANGE)
# =============================================================================

EVENT_VERSION = "1.0"

# Allowed fields for event schema (enforce strict schema)
ALLOWED_EVENT_FIELDS: Set[str] = {
    "event_version",
    "ts",
    "run_id",
    "round",
    "stage",
    "level",
    "event_type",
    "message",
    "counters_delta",
    "usage_delta",
    "artifact_paths",
    "prev_hash",
}

# Allowed fields for HTTP status response (redacted, no secrets)
ALLOWED_STATUS_FIELDS: Set[str] = {
    "run_id",
    "status",
    "current_round",
    "current_stage",
    "stop_requested",
    "pause_requested",
    "counters",
    "usage",
    "gpu_summary",
    "last_event_ts",
    "health",
    "updated_at",
}


# =============================================================================
# SECRET REDACTION PATTERNS
# =============================================================================

# Patterns that indicate secrets - used to redact before writing events
SECRET_PATTERNS = [
    # Generic API keys and tokens
    (r"g[h]p_[a-zA-Z0-9]{36}", "[GITHUB_TOKEN]"),
    (r"glpat-[a-zA-Z0-9\-]{20,}", "[GITLAB_TOKEN]"),
    (r"s[k]-[a-zA-Z0-9]{20,}", "[OPENAI_KEY]"),
    (r"Bearer\s+[\w\-]{20,}", "[BEARER_TOKEN]"),
    (r'(?i)(api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*["\']?[\w\-]{20,}', "[API_KEY]"),
    (r"A[K]IA[0-9A-Z]{16}", "[AWS_KEY]"),
    (r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "[PRIVATE_KEY]"),
    (r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----", "[SSH_KEY]"),
    # Environment variable patterns
    (r"\$?(OPENAI_API_KEY|GITHUB_TOKEN|GITLAB_TOKEN|AWS_SECRET)[=]\S+", "[ENV_SECRET]"),
    # Generic token patterns
    (r'token[_-]?(id|key)?\s*[:=]\s*["\']?[\w\-]{20,}', "[TOKEN]"),
]

# ANSI escape sequence pattern for stripping
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Maximum string lengths for event fields
MAX_MESSAGE_LENGTH = 500
MAX_ERROR_LENGTH = 200
MAX_PATH_LENGTH = 100


def redact_secrets(text: str) -> str:
    """
    Redact secrets from text before writing to logs.

    HOW IT WORKS:
        - Iterates through SECRET_PATTERNS
        - Replaces matches with placeholder
        - Strips ANSI escape sequences

    SECURITY:
        - Never logs actual secrets
        - Uses consistent placeholders for auditing
    """
    if not isinstance(text, str):
        return str(text)

    # Strip ANSI first
    text = ANSI_ESCAPE.sub("", text)

    # Redact secrets
    for pattern, replacement in SECRET_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def truncate_string(text: str, max_length: int) -> str:
    """Truncate string to max length."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def sanitize_for_log(value: Any, max_length: int = MAX_MESSAGE_LENGTH) -> Any:
    """
    Sanitize value for logging.

    HOW IT WORKS:
        - Redacts secrets
        - Truncates long strings
        - Converts to safe types
    """
    if isinstance(value, str):
        return truncate_string(redact_secrets(value), max_length)
    elif isinstance(value, dict):
        return {k: sanitize_for_log(v, max_length) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_for_log(v, max_length) for v in value]
    elif isinstance(value, (int, float, bool, type(None))):
        return value
    else:
        return truncate_string(redact_secrets(str(value)), max_length)


def sanitize_artifact_paths(paths: List[str]) -> List[str]:
    """Sanitize artifact paths - truncate long paths."""
    if not paths:
        return []
    return [truncate_string(p, MAX_PATH_LENGTH) for p in paths]


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Event batch buffer - flushed periodically or on exit
_event_buffer: List[Dict[str, Any]] = []

# Lock for thread-safe operations
_lock = threading.RLock()

# Whether telemetry has been initialized
_initialized = False

# StateMachine instance for structured state transitions (Phase 2)
_state_machine: Optional[Any] = None


# =============================================================================
# DEFAULT COUNTERS AND USAGE TRACKING
# =============================================================================


def get_default_counters() -> Dict[str, Any]:
    """
    Get default counter values for a new run.

    HOW IT WORKS:
        Defines all counters tracked throughout the pipeline.
        Counters are incremental - delta values are added to existing values.

    TUNABLE:
        - Add new counters for new pipeline stages
        - Remove counters for stages you don't use
        - Counter names should be descriptive and unique

    COUNTER DESCRIPTIONS:
        - teacher_generated: Samples successfully generated by teacher
        - teacher_failed: Failed generation attempts
        - raw_written: Samples written to raw JSONL
        - validated_ok: Samples passing validation
        - rejected_schema: Samples rejected due to schema issues
        - rejected_secret: Samples rejected due to secret detection
        - rejected_dedupe: Samples rejected as duplicates
        - test_pass: Unit tests passing
        - test_fail: Unit tests failing
        - train_step: Current training step
        - train_loss: Latest training loss (stored as float)
        - eval_json_parse_rate: JSON parsing success rate
        - eval_format_rate: Format compliance rate
    """
    return {
        "teacher_generated": 0,
        "teacher_failed": 0,
        "raw_written": 0,
        "validated_ok": 0,
        "rejected_schema": 0,
        "rejected_secret": 0,
        "rejected_dedupe": 0,
        "test_pass": 0,
        "test_fail": 0,
        "train_step": 0,
        "train_loss": 0.0,
        "eval_json_parse_rate": 0.0,
        "eval_format_rate": 0.0,
    }


def get_default_usage() -> Dict[str, Any]:
    """
    Get default usage tracking values for a new run.

    HOW IT WORKS:
        Tracks teacher API usage across all rounds.
        Costs are estimated based on pricing configuration.

    TUNABLE:
        - Add new fields for new API providers
        - Adjust cost estimation logic for accuracy

    FIELDS:
        - requests_sent: Total API requests made
        - input_tokens: Total input tokens consumed
        - output_tokens: Total output tokens generated
        - rate_limits_hit: Number of rate limit errors
        - retries: Number of request retries
        - estimated_cost_usd: Estimated cost in USD
    """
    return {
        "requests_sent": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "rate_limits_hit": 0,
        "retries": 0,
        "estimated_cost_usd": 0.0,
    }


# =============================================================================
# PATH MANAGEMENT
# =============================================================================


def get_run_dir(run_id: Optional[str] = None) -> Path:
    """
    Get the run directory path.

    HOW IT WORKS:
        Creates runs/<run_id>/ directory structure.
        All run-specific files go here.

    TUNABLE:
        - Modify directory structure by changing path construction
    """
    if run_id is None:
        run_id = get_run_id()
    return Path(AUTOTRAIN_DIR) / "runs" / run_id


def get_events_path(run_id: Optional[str] = None) -> Path:
    """Get the event log file path."""
    return get_run_dir(run_id) / "events.jsonl"


def get_state_path(run_id: Optional[str] = None) -> Path:
    """Get the state file path."""
    return get_run_dir(run_id) / "state.json"


def get_config_path(run_id: Optional[str] = None) -> Path:
    """Get the config file path."""
    return get_run_dir(run_id) / "config.json"


def get_run_id() -> str:
    """
    Get or generate run ID.

    HOW IT WORKS:
        - Uses RUN_ID env var if set
        - Otherwise generates a new UUID
        - Stores in global for subsequent calls
    """
    global RUN_ID
    if not RUN_ID:
        RUN_ID = os.environ.get("RUN_ID", "")
    if not RUN_ID:
        RUN_ID = str(uuid.uuid4())[:8]
        RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_ID}"
    return RUN_ID


# =============================================================================
# PRICING CONFIGURATION
# =============================================================================

# Default pricing for common models (can be overridden by pricing.json)
# TUNABLE: Add new models here or create pricing.json
# prices are per 1M tokens
DEFAULT_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $/1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


def load_pricing_config() -> Dict[str, Dict[str, float]]:
    """
    Load pricing configuration from file or use defaults.

    HOW IT WORKS:
        - First checks for pricing.json in run directory
        - Falls back to DEFAULT_PRICING
        - Allows user to customize pricing per model

    TUNABLE:
        - Create pricing.json to override default prices
        - Format: {"model_name": {"input": 0.5, "output": 1.5}}
        - Prices are per 1M tokens
    """
    pricing = DEFAULT_PRICING.copy()

    # Check for pricing config file
    pricing_file = (
        Path(PRICING_CONFIG_PATH) if PRICING_CONFIG_PATH else get_run_dir() / "pricing.json"
    )

    if pricing_file.exists():
        try:
            with open(pricing_file) as f:
                custom = json.load(f)
                pricing.update(custom)
        except Exception as e:
            print(f"[WARN] Failed to load pricing config: {e}", file=sys.stderr)

    return pricing


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Estimate API cost based on token usage and model.

    HOW IT WORKS:
        - Looks up model in pricing config
        - Calculates cost: (input_tokens/1M * input_price) + (output_tokens/1M * output_price)
        - Returns 0 if model not found (tokens still tracked)

    TUNABLE:
        - Add models to DEFAULT_PRICING or pricing.json
        - Adjust prices for your API provider
    """
    pricing = load_pricing_config()

    if model not in pricing:
        return 0.0

    input_price = pricing[model].get("input", 0)
    output_price = pricing[model].get("output", 0)

    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

    return cost


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

# Allowed config fields with types and constraints
CONFIG_SCHEMA = {
    "BASE_MODEL": {"type": str, "required": False, "default": "mistralai/Mistral-7B-Instruct-v0.2"},
    "TEACHER_MODEL": {"type": str, "required": False, "default": "gpt-4o-mini"},
    "SAMPLES_PER_ROUND": {"type": int, "required": False, "default": 50, "min": 1, "max": 10000},
    "ROUNDS": {"type": int, "required": False, "default": 3, "min": 1, "max": 100},
    "VAL_RATIO": {"type": (int, float), "required": False, "default": 0.1, "min": 0.0, "max": 1.0},
    "SEQ_LEN": {"type": int, "required": False, "default": 2048, "min": 128, "max": 8192},
    "BATCH_SIZE": {"type": int, "required": False, "default": 1, "min": 1, "max": 64},
    "GRAD_ACCUM": {"type": int, "required": False, "default": 8, "min": 1, "max": 128},
    "TRAIN_STEPS": {"type": int, "required": False, "default": 500, "min": 1, "max": 100000},
    "LORA_R": {"type": int, "required": False, "default": 64, "min": 1, "max": 512},
    "LORA_ALPHA": {"type": int, "required": False, "default": 128, "min": 1, "max": 1024},
    "LR": {"type": str, "required": False, "default": "2e-4"},
    "RUN_UNIT_TESTS": {"type": str, "required": False, "default": "0", "allowed": ["0", "1"]},
    "SEED": {"type": (int, str), "required": False, "default": "42"},
    "OUT_DIR": {"type": str, "required": False},
}


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration against schema.

    SECURITY:
        - Rejects unknown fields
        - Validates types and ranges
        - Fails fast with clear error messages

    RAISES:
        ValueError: If config is invalid

    ARGS:
        config: Configuration dictionary to validate
    """
    errors = []

    # Check for unknown fields
    for key in config.keys():
        if key not in CONFIG_SCHEMA:
            errors.append(f"Unknown config field: {key}")

    # Validate known fields
    for key, schema in CONFIG_SCHEMA.items():
        if key not in config:
            if schema.get("required", False):
                errors.append(f"Missing required config field: {key}")
            continue

        value = config[key]
        expected_type = schema["type"]

        # Check type
        if not isinstance(value, expected_type):
            errors.append(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
            continue

        # Check min/max
        if "min" in schema and value < schema["min"]:
            errors.append(f"Value for {key} too small: {value} < {schema['min']}")
        if "max" in schema and value > schema["max"]:
            errors.append(f"Value for {key} too large: {value} > {schema['max']}")

        # Check allowed values
        if "allowed" in schema and value not in schema["allowed"]:
            errors.append(f"Invalid value for {key}: {value}. Allowed: {schema['allowed']}")

    if errors:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))


# =============================================================================
# INITIALIZATION
# =============================================================================


def init_telemetry(
    run_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None, force: bool = False
) -> str:
    """
    Initialize telemetry for a new run or resume an existing run.

    SECURITY:
        - Validates config against strict schema
        - Creates secure directory structure
        - Sets restrictive file permissions

    HOW IT WORKS:
        1. Validates config if provided
        2. Creates run directory structure
        3. Initializes state.json with default values
        4. Sets up event log file
        5. Registers cleanup handlers

    ARGS:
        run_id: Unique identifier (auto-generated if not provided)
        config: Configuration dict to store with run
        force: Reinitialize even if run exists

    RETURNS:
        The run_id used for this run

    RAISES:
        ValueError: If config validation fails
    """
    global _initialized, RUN_ID, _state_machine

    # Initialize StateMachine if available (Phase 2)
    try:
        from heidi_engine.state_machine import StateMachine

        if _state_machine is None:
            _state_machine = StateMachine(run_id=run_id)
            RUN_ID = _state_machine.run_id
        elif run_id and _state_machine.run_id != run_id:
            _state_machine = StateMachine(run_id=run_id)
    except ImportError:
        pass

    # Validate config if provided
    if config:
        validate_config(config)

    with _lock:
        if run_id:
            RUN_ID = run_id

        run_id = get_run_id()
        run_dir = get_run_dir(run_id)

        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)

        state_file = get_state_path(run_id)

        # Check if we should resume existing run
        if state_file.exists() and not force:
            # Resume from existing state - just reopen
            _initialized = True
            return run_id

        # Initialize new state (exclude config for security - don't store sensitive values)
        state = {
            "run_id": run_id,
            "status": "running",
            "current_round": 0,
            "current_stage": "initializing",
            "stop_requested": False,
            "pause_requested": False,
            "counters": get_default_counters(),
            "usage": get_default_usage(),
            "config": {},  # Don't store config in state for security
            "started_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Save initial state atomically
        save_state(state, run_id)

        # Save config if provided (but not to state.json)
        if config:
            config_file = get_config_path(run_id)
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            # Set restrictive permissions
            os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        # Initialize event log
        events_file = get_events_path(run_id)
        events_file.touch()

        # Set restrictive permissions
        os.chmod(events_file, stat.S_IRUSR | stat.S_IWUSR)

        _initialized = True

        # Register flush handler
        atexit.register(flush_events)

        return run_id


def get_state(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load current state from state.json.

    HOW IT WORKS:
        - Reads state.json file
        - Returns empty state if file doesn't exist

    ARGS:
        run_id: Run to read (defaults to current run)

    RETURNS:
        State dictionary
    """
    state_file = get_state_path(run_id)

    if not state_file.exists():
        return {
            "run_id": get_run_id(),
            "status": "idle",
            "counters": get_default_counters(),
            "usage": get_default_usage(),
        }

    try:
        with open(state_file) as f:
            state = json.load(f)
            # Resolve status from on-disk metadata
            state["status"] = resolve_status(state)
            return state
    except Exception as e:
        print(f"[WARN] Failed to load state: {e}", file=sys.stderr)
        return {"status": "error", "error": str(e)}


def resolve_status(state: Dict[str, Any]) -> str:
    """
    Resolve run status from on-disk metadata.

    STATUS VALUES:
        - idle: No active run, or run_id present but no events
        - running: Worker active / stop not requested / still processing
        - stopped: stop_requested=true or pipeline complete
        - error: last_error present or health degraded

    HOW IT WORKS:
        - Checks stop_requested flag
        - Checks status field
        - Checks for error indicators

    ARGS:
        state: State dictionary from state.json

    RETURNS:
        Status string: idle, running, stopped, error
    """
    # Explicit stop requested
    if state.get("stop_requested", False):
        return "stopped"

    # Check explicit status first
    status = state.get("status", "")
    if status in ("running", "completed", "stopped", "error"):
        return status

    # Check for errors
    if state.get("last_error") or state.get("health") == "degraded":
        return "error"

    # Check if there's an active run_id but no recent activity
    run_id = state.get("run_id")
    if run_id:
        # Has run_id - check for activity
        counters = state.get("counters", {})
        usage = state.get("usage", {})

        # If there's any activity, consider it running
        if counters.get("teacher_generated", 0) > 0:
            return "running"
        if usage.get("requests_sent", 0) > 0:
            return "running"
        if counters.get("train_step", 0) > 0:
            return "running"

        # No activity - idle
        return "idle"

    # Default to idle if no run_id
    return "idle"


def save_state(state: Dict[str, Any], run_id: Optional[str] = None) -> None:
    """
    Save state to state.json atomically.

    HOW IT WORKS:
        - Writes to temp file first
        - Uses os.rename for atomic update
        - Prevents corruption from partial writes

    TUNABLE:
        - N/A - just saves state

    ARGS:
        state: State dictionary to save
        run_id: Run ID (defaults to current)
    """
    run_id = run_id or get_run_id()
    state_file = get_state_path(run_id)
    temp_file = state_file.with_suffix(".tmp")

    # Update timestamp
    state["updated_at"] = datetime.utcnow().isoformat()

    # Write to temp file
    with open(temp_file, "w") as f:
        json.dump(state, f, indent=2)

    # Atomic rename
    os.replace(temp_file, state_file)


def update_counters(delta: Dict[str, Any], run_id: Optional[str] = None) -> None:
    """
    Update counters with incremental values.

    HOW IT WORKS:
        - Loads current state
        - Adds delta values to counters
        - Saves state atomically

    TUNABLE:
        - Pass negative values to decrement

    ARGS:
        delta: Dictionary of counter name -> value to add
        run_id: Run ID (defaults to current)
    """
    state = get_state(run_id)
    counters = state.get("counters", get_default_counters())

    for key, value in delta.items():
        if key not in counters:
            counters[key] = 0
        counters[key] += value

    # Handle train_loss specially (it's a float)
    if "train_loss" in delta and isinstance(delta["train_loss"], float):
        counters["train_loss"] = delta["train_loss"]

    state["counters"] = counters
    save_state(state, run_id)


def update_usage(
    delta: Dict[str, Any], model: Optional[str] = None, run_id: Optional[str] = None
) -> None:
    """
    Update usage statistics with incremental values.

    HOW IT WORKS:
        - Loads current state
        - Adds delta values to usage
        - Recalculates cost estimate
        - Saves state atomically

    TUNABLE:
        - Add new usage fields as needed
        - model parameter needed for cost estimation

    ARGS:
        delta: Dictionary of usage field -> value to add
        model: Model name for cost estimation
        run_id: Run ID (defaults to current)
    """
    state = get_state(run_id)
    usage = state.get("usage", get_default_usage())

    # Add deltas
    for key, value in delta.items():
        if key not in usage:
            usage[key] = 0
        usage[key] += value

    # Recalculate cost if we have model
    if model:
        cost = estimate_cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0), model)
        usage["estimated_cost_usd"] = cost

    state["usage"] = usage
    save_state(state, run_id)


def set_status(
    status: str,
    stage: Optional[str] = None,
    round_num: Optional[int] = None,
    run_id: Optional[str] = None,
) -> None:
    """
    Update run status.

    HOW IT WORKS:
        - Updates status, stage, and round in state
        - Used for tracking pipeline progress

    TUNABLE:
        - N/A

    ARGS:
        status: One of: running, paused, stopped, completed, error
        stage: Current pipeline stage
        round_num: Current round number
        run_id: Run ID (defaults to current)
    """
    state = get_state(run_id)

    if status:
        state["status"] = status

    if stage:
        state["current_stage"] = stage

    if round_num is not None:
        state["current_round"] = round_num

    save_state(state, run_id)


def sm_apply_event(event_name: str, **kwargs) -> Optional[str]:
    """
    Apply a StateMachine event (Phase 2 integration).

    This is the single writer for state transitions - all state changes
    should go through here when StateMachine is available.

    ARGS:
        event_name: Name of the event (START_FULL, START_COLLECT, TRAIN_NOW, etc.)
        **kwargs: Additional arguments for the event

    RETURNS:
        New phase name or None if StateMachine not available
    """
    global _state_machine

    if _state_machine is None:
        return None

    try:
        from heidi_engine.state_machine import Event

        event = Event[event_name]
        new_phase = _state_machine.apply(event, **kwargs)
        return new_phase.name
    except (KeyError, ValueError) as e:
        print(f"[WARN] Invalid state machine event {event_name}: {e}", file=sys.stderr)
        return None


def sm_set_mode(mode_name: str) -> None:
    """
    Set the operating mode via StateMachine (Phase 2).

    Modes: IDLE, COLLECT, TRAIN

    ARGS:
        mode_name: Name of the mode
    """
    global _state_machine

    if _state_machine is None:
        return

    try:
        from heidi_engine.state_machine import Mode

        mode = Mode[mode_name]
        _state_machine.set_mode(mode)
    except (KeyError, ValueError) as e:
        print(f"[WARN] Invalid mode {mode_name}: {e}", file=sys.stderr)


def sm_can_train() -> bool:
    """
    Check if training is allowed in current mode/phase (Phase 2).

    In COLLECT mode, training is only allowed from COMPLETE or INITIALIZING phases.

    RETURNS:
        True if training is allowed
    """
    global _state_machine

    if _state_machine is None:
        return True  # Default to allowing if no StateMachine

    return _state_machine.can_train()


def request_stop(run_id: Optional[str] = None) -> None:
    """
    Request graceful stop.

    HOW IT WORKS:
        - Sets stop_requested flag in state.json
        - Loop.sh checks this flag and exits at stage boundaries
        - Allows clean exit without corrupting outputs

    TUNABLE:
        - N/A

    ARGS:
        run_id: Run ID (defaults to current)
    """
    state = get_state(run_id)
    state["stop_requested"] = True
    save_state(state, run_id)


def request_pause(run_id: Optional[str] = None) -> None:
    """
    Request pause.

    HOW IT WORKS:
        - Sets pause_requested flag in state.json
        - Pipeline pauses at safe boundaries (between batches)
        - Can be resumed later

    TUNABLE:
        - N/A

    ARGS:
        run_id: Run ID (defaults to current)
    """
    state = get_state(run_id)
    state["pause_requested"] = True
    save_state(state, run_id)


def clear_pause(run_id: Optional[str] = None) -> None:
    """
    Clear pause request (resume).

    HOW IT WORKS:
        - Clears pause_requested flag
        - Pipeline continues from where it paused

    TUNABLE:
        - N/A

    ARGS:
        run_id: Run ID (defaults to current)
    """
    state = get_state(run_id)
    state["pause_requested"] = False
    save_state(state, run_id)


def check_stop_requested(run_id: Optional[str] = None) -> bool:
    """
    Check if stop has been requested.

    HOW IT WORKS:
        - Reads stop_requested flag from state
        - Called by pipeline scripts to check for stop

    TUNABLE:
        - N/A

    ARGS:
        run_id: Run ID (defaults to current)

    RETURNS:
        True if stop requested
    """
    state = get_state(run_id)
    return state.get("stop_requested", False)


def check_pause_requested(run_id: Optional[str] = None) -> bool:
    """
    Check if pause has been requested.

    HOW IT WORKS:
        - Reads pause_requested flag from state
        - Called by pipeline scripts to check for pause

    TUNABLE:
        - N/A

    ARGS:
        run_id: Run ID (defaults to current)

    RETURNS:
        True if pause requested
    """
    state = get_state(run_id)
    return state.get("pause_requested", False)


# =============================================================================
# EVENT EMISSION
# =============================================================================


def emit_event(
    event_type: str,
    message: str,
    level: str = "info",
    stage: Optional[str] = None,
    round_num: Optional[int] = None,
    counters_delta: Optional[Dict[str, Any]] = None,
    usage_delta: Optional[Dict[str, Any]] = None,
    artifact_paths: Optional[List[str]] = None,
    error: Optional[str] = None,
    model: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """
    Emit a telemetry event to the event stream.

    SECURITY:
        - Enforces event schema version (ALLOWED_EVENT_FIELDS)
        - Sanitizes all string fields (redacts secrets, strips ANSI, truncates)
        - Rejects unknown fields

    HOW IT WORKS:
        1. Creates event with all provided fields
        2. Validates against schema
        3. Sanitizes sensitive data
        4. Adds to event buffer
        5. Flushes buffer when it reaches TELEMETRY_BATCH size

    ARGS:
        event_type: Type of event (stage_start, stage_end, progress, error, etc.)
        message: Human-readable message
        level: Log level (info, warn, error, success)
        stage: Pipeline stage (generate, validate, train, eval)
        round_num: Current round number
        counters_delta: Counter increments to apply
        usage_delta: Usage increments to apply
        artifact_paths: Files created by this event
        error: Error message (if level=error)
        model: Model name (for cost estimation)
        run_id: Run ID (defaults to current)
    """
    global _event_buffer

    if not _initialized:
        init_telemetry(run_id)

    run_id = get_run_id()

    # Get current state for context
    state = get_state(run_id)

    # Build event with schema version
    event = {
        "event_version": EVENT_VERSION,
        "ts": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "round": round_num if round_num is not None else state.get("current_round", 0),
        "stage": stage or state.get("current_stage", "unknown"),
        "level": level,
        "event_type": event_type,
        "message": message,
        "counters_delta": counters_delta or {},
        "usage_delta": usage_delta or {},
        "artifact_paths": artifact_paths or [],
        "error": error,
    }

    # Remove None values
    event = {k: v for k, v in event.items() if v is not None}

    # Enforce schema: only allow known fields
    event = {k: v for k, v in event.items() if k in ALLOWED_EVENT_FIELDS}

    # Sanitize all string fields
    event["message"] = sanitize_for_log(event.get("message", ""), MAX_MESSAGE_LENGTH)
    if "error" in event:
        event["error"] = sanitize_for_log(event.get("error", ""), MAX_ERROR_LENGTH)
    if "artifact_paths" in event:
        event["artifact_paths"] = sanitize_artifact_paths(event.get("artifact_paths", []))

    # Sanitize counters_delta and usage_delta
    if "counters_delta" in event:
        event["counters_delta"] = sanitize_for_log(event["counters_delta"], 100)
    if "usage_delta" in event:
        event["usage_delta"] = sanitize_for_log(event["usage_delta"], 100)

    # Add to buffer
    with _lock:
        _event_buffer.append(event)

        # Update counters and usage if provided
        if counters_delta:
            update_counters(counters_delta, run_id)

        if usage_delta:
            update_usage(usage_delta, model, run_id)

        # Flush if batch is full
        if len(_event_buffer) >= TELEMETRY_BATCH:
            flush_events()


def flush_events() -> None:
    """
    Flush event buffer to disk with rotation support.

    HOW IT WORKS:
        - Writes all buffered events to events.jsonl
        - Uses file append for performance
        - Called automatically when batch is full or on exit
        - Rotates log file when max size exceeded
        - Maintains retention count of old files

    SECURITY:
        - Sets file permissions to 0600
    """
    global _event_buffer

    if not _event_buffer:
        return

    with _lock:
        if not _event_buffer:
            return

        events_file = get_events_path()

        try:
            # Check if rotation needed
            if events_file.exists():
                size_mb = events_file.stat().st_size / (1024 * 1024)
                if size_mb >= EVENT_LOG_MAX_SIZE_MB:
                    _rotate_events_log(events_file)

            # Ensure parent directory exists with proper permissions
            events_file.parent.mkdir(parents=True, exist_ok=True)

            with open(events_file, "a") as f:
                for event in _event_buffer:
                    f.write(json.dumps(event) + "\n")

            # Set restrictive permissions
            os.chmod(events_file, stat.S_IRUSR | stat.S_IWUSR)

        except Exception as e:
            print(f"[ERROR] Failed to write events: {e}", file=sys.stderr)

        _event_buffer = []


def _rotate_events_log(events_file: Path) -> None:
    """
    Rotate event log file when max size exceeded.

    HOW IT WORKS:
        - Renames current log to .1, .2, etc.
        - Deletes oldest if over retention limit
    """
    run_dir = events_file.parent

    # Remove oldest if at limit
    oldest = run_dir / f"events.jsonl.{EVENT_LOG_RETENTION}"
    if oldest.exists():
        oldest.unlink()

    # Rotate existing files
    for i in range(EVENT_LOG_RETENTION - 1, 0, -1):
        src = run_dir / f"events.jsonl.{i}"
        dst = run_dir / f"events.jsonl.{i + 1}"
        if src.exists():
            src.rename(dst)

    # Rename current to .1
    events_file.rename(run_dir / "events.jsonl.1")


def emit_stage_start(stage: str, round_num: int, message: str, **kwargs) -> None:
    """
    Emit a stage start event.

    HOW IT WORKS:
        - Convenience wrapper for stage_start events
        - Updates state with current stage/round

    TUNABLE:
        - N/A

    ARGS:
        stage: Pipeline stage name
        round_num: Current round
        message: Human-readable message
        **kwargs: Additional args passed to emit_event
    """
    set_status("running", stage, round_num)
    emit_event(
        event_type="stage_start",
        message=message,
        level="info",
        stage=stage,
        round_num=round_num,
        **kwargs,
    )


def emit_stage_end(
    stage: str, round_num: int, message: str, success: bool = True, **kwargs
) -> None:
    """
    Emit a stage end event.

    HOW IT WORKS:
        - Convenience wrapper for stage_end events
        - Records final counters and usage

    TUNABLE:
        - N/A

    ARGS:
        stage: Pipeline stage name
        round_num: Current round
        message: Human-readable message
        success: Whether stage completed successfully
        **kwargs: Additional args passed to emit_event
    """
    emit_event(
        event_type="stage_end",
        message=message,
        level="success" if success else "error",
        stage=stage,
        round_num=round_num,
        **kwargs,
    )


def emit_progress(
    stage: str, round_num: int, current: int, total: int, message: Optional[str] = None, **kwargs
) -> None:
    """
    Emit a progress event.

    HOW IT WORKS:
        - Convenience wrapper for progress events
        - Typically called every N items to avoid flooding

    TUNABLE:
        - Adjust frequency based on needs

    ARGS:
        stage: Pipeline stage name
        round_num: Current round
        current: Current item count
        total: Total items
        message: Optional message
        **kwargs: Additional args passed to emit_event
    """
    if message is None:
        message = f"Progress: {current}/{total}"

    emit_event(
        event_type="progress",
        message=message,
        level="info",
        stage=stage,
        round_num=round_num,
        **kwargs,
    )


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================


@contextmanager
def stage_context(stage: str, round_num: int, message: str, **kwargs):
    """
    Context manager for pipeline stages.

    HOW IT WORKS:
        - Emits stage_start on enter
        - Emits stage_end on exit
        - Handles exceptions automatically

    TUNABLE:
        - N/A

    EXAMPLE:
        with stage_context("generate", 1, "Generating samples"):
            # do work
            pass

    ARGS:
        stage: Pipeline stage name
        round_num: Current round
        message: Human-readable message
        **kwargs: Additional args passed to emit_event
    """
    try:
        emit_stage_start(stage, round_num, message, **kwargs)
        yield
    except Exception as e:
        emit_stage_end(stage, round_num, f"Failed: {str(e)}", success=False, error=str(e))
        raise
    else:
        emit_stage_end(stage, round_num, f"Completed: {message}", success=True, **kwargs)


# =============================================================================
# HTTP STATUS SERVER (OPTIONAL)
# =============================================================================


def get_last_event_ts() -> Optional[str]:
    """Get timestamp of last event."""
    try:
        events_file = get_events_path()
        if events_file.exists() and events_file.stat().st_size > 0:
            with open(events_file, "rb") as f:
                f.seek(-500, 2)  # Read last 500 bytes
                lines = f.read().decode().strip().split("\n")
                if lines:
                    last_line = lines[-1]
                    event = json.loads(last_line)
                    return event.get("ts")
    except Exception:
        pass
    return None


def redact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Redact state to only allowed fields."""
    redacted = {}
    for key in ALLOWED_STATUS_FIELDS:
        if key in state:
            value = state[key]
            # Sanitize any nested secrets
            if isinstance(value, dict):
                value = {k: sanitize_for_log(v, 100) for k, v in value.items()}
            redacted[key] = value
    return redacted


def start_reporter(dashboard_url: str):
    """Start background thread to push state to central dashboard."""
    if not requests:
        return

    def reporter_loop():
        error_count = 0
        telemetry_pass = os.environ.get("TELEMETRY_PASS")
        auth = ("admin", telemetry_pass) if telemetry_pass else None

        while True:
            try:
                state = get_state()
                # Redact before sending
                redacted = redact_state(state)
                # Add run_id to top level if not present
                if "run_id" not in redacted:
                    redacted["run_id"] = get_run_id()

                requests.post(f"{dashboard_url}/report", json=redacted, timeout=5, auth=auth)
                error_count = 0
            except Exception:
                error_count += 1
                if error_count > 5:
                    time.sleep(30)  # Backoff
            time.sleep(5)

    thread = threading.Thread(target=reporter_loop, daemon=True)
    thread.start()


def start_http_server(port: int = 7779) -> None:
    """
    Start HTTP status server.

    SECURITY:
        - Binds to 127.0.0.1 only (localhost-only) for security
        - Adds /report endpoint to receive remote states
        - Adds /runs endpoint to list all runs
    """
    if port <= 0:
        return

    try:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
    except ImportError:
        print("[WARN] HTTP server not available", file=sys.stderr)
        return

    # Use existing helper functions
    # (get_gpu_summary, get_last_event_ts, redact_state are defined in outer scope)

    def get_gpu_summary() -> Dict[str, Any]:
        """Get minimal GPU info without exposing sensitive data."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    return {
                        "vram_used_mb": int(parts[0].strip()),
                        "vram_total_mb": int(parts[1].strip()),
                        "util_pct": int(parts[2].strip()) if len(parts) > 2 else 0,
                    }
        except Exception:
            pass
        return {"available": False}

    class StateHandler(BaseHTTPRequestHandler):
        """HTTP handler with security restrictions."""

        def _send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"health": "ok"}).encode())
                return

            if self.path == "/":
                # Serve dashboard.html if it exists
                dashboard_path = Path(__file__).parent / "dashboard.html"
                if dashboard_path.exists():
                    self.send_response(200)
                    self._send_cors_headers()
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    with open(dashboard_path, "rb") as f:
                        self.wfile.write(f.read())
                    return

            # List all runs (local + remote)
            if self.path == "/runs":
                local_runs = list_runs()
                # Merge remote states
                all_runs = local_runs + list(_remote_states.values())
                # Deduplicate by run_id (prefer local)
                seen_ids = set()
                unique_runs = []
                for run in all_runs:
                    if run.get("run_id") not in seen_ids:
                        # Ensure it's redacted lightly if it came from list_runs (which reads state.json directly)
                        # list_runs usually returns raw state. we should redact.
                        redacted_run = redact_state(run) if "run_id" in run else run
                        unique_runs.append(redacted_run)
                        seen_ids.add(run.get("run_id"))

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(unique_runs).encode())
                return

            # Get specific run status
            if self.path.startswith("/status"):
                query_run_id = None
                if "?run_id=" in self.path:
                    try:
                        query_run_id = self.path.split("?run_id=")[1].split("&")[0]
                    except IndexError:
                        pass

                # Check remote states first if run_id provided
                if query_run_id and query_run_id in _remote_states:
                    state = _remote_states[query_run_id]
                else:
                    # Default to local state if no ID or ID not found remotely
                    # If ID is local, get_state takes argument
                    state = get_state(query_run_id)

                # Add GPU summary (no secrets)
                if "gpu_summary" not in state:
                    state["gpu_summary"] = get_gpu_summary()

                # Add last event timestamp
                state["last_event_ts"] = get_last_event_ts()

                # Add health status
                state["health"] = "ok" if state.get("status") != "error" else "degraded"

                # Redact to allowed fields only
                redacted = redact_state(state)

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(redacted).encode())
                return

            # Unknown endpoint
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

        def do_POST(self):
            if self.path == "/report":
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    data = json.loads(body)
                    run_id = data.get("run_id")
                    if run_id:
                        # Store in memory
                        _remote_states[run_id] = data
                        self.send_response(200)
                        self._send_cors_headers()
                        self.end_headers()
                        self.wfile.write(b'{"status":"ok"}')
                    else:
                        self.send_response(400)
                        self._send_cors_headers()
                        self.end_headers()
                except Exception as e:
                    self.send_response(500)
                    self._send_cors_headers()
                    self.end_headers()
                    print(f"[ERROR] Failed to process report: {e}", file=sys.stderr)
                return

            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self._send_cors_headers()
            self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress logging

    def run_server():
        try:
            # SECURITY: Bind to 0.0.0.0 for multi-machine support
            server = HTTPServer(("127.0.0.1", port), StateHandler)
            print(f"[INFO] HTTP status server running on http://0.0.0.0:{port}")
            server.serve_forever()
        except Exception as e:
            print(f"[WARN] HTTP server failed: {e}", file=sys.stderr)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Start reporter if configured
    dashboard_url = os.environ.get("DASHBOARD_URL")
    if dashboard_url:
        print(f"[INFO] Starting dashboard reporter to {dashboard_url}")
        start_reporter(dashboard_url)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def list_runs() -> List[Dict[str, Any]]:
    """
    List all runs in the heidi_engine directory.

    HOW IT WORKS:
        - Scans runs/ subdirectories
        - Returns basic info for each run

    TUNABLE:
        - N/A

    RETURNS:
        List of run info dictionaries
    """
    runs_dir = Path(AUTOTRAIN_DIR) / "runs"

    if not runs_dir.exists():
        return []

    runs = []

    for run_path in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_path.is_dir():
            continue

        state_file = run_path / "state.json"

        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    runs.append(state)
            except Exception:
                pass

    return runs


def get_latest_run() -> Optional[str]:
    """
    Get the most recent run ID.

    HOW IT WORKS:
        - Finds most recently modified run directory
        - Returns run_id from state.json

    TUNABLE:
        - N/A

    RETURNS:
        Run ID or None if no runs exist
    """
    runs = list_runs()

    if runs:
        return runs[0].get("run_id")

    return None


# =============================================================================
# MAIN / CLI
# =============================================================================


def main():
    """
    CLI for telemetry commands.

    TUNABLE:
        - Add new subcommands as needed

    COMMANDS:
        python -m heidi_engine.telemetry init [--run-id ID] [--config JSON]
        python -m heidi_engine.telemetry status [--run-id ID]
        python -m heidi_engine.telemetry list
        python -m heidi_engine.telemetry emit <type> <message>
    """
    import argparse

    parser = argparse.ArgumentParser(prog="heidi-engine telemetry")
    subparsers = parser.add_subparsers(dest="command")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new run")
    init_parser.add_argument("--run-id", help="Run ID")
    init_parser.add_argument("--config", help="Config JSON")
    init_parser.add_argument("--server", action="store_true", help="Start HTTP server")

    # status command
    status_parser = subparsers.add_parser("status", help="Show run status")
    status_parser.add_argument("--run-id", help="Run ID")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    subparsers.add_parser("list", help="List all runs")

    # emit command
    emit_parser = subparsers.add_parser("emit", help="Emit an event")
    emit_parser.add_argument("type", help="Event type")
    emit_parser.add_argument("message", help="Message")
    emit_parser.add_argument("--run-id", help="Run ID")
    emit_parser.add_argument("--stage", help="Stage")
    emit_parser.add_argument("--round", type=int, help="Round")

    # stop/pause commands
    subparsers.add_parser("stop", help="Request stop")
    subparsers.add_parser("pause", help="Request pause")
    subparsers.add_parser("resume", help="Clear pause")

    args = parser.parse_args()

    if args.command == "init":
        config = json.loads(args.config) if args.config else None
        run_id = init_telemetry(args.run_id, config)
        print(f"Initialized run: {run_id}")

        if args.server:
            start_http_server(HTTP_STATUS_PORT)

    elif args.command == "status":
        state = get_state(args.run_id)
        if args.json:
            print(json.dumps(state, indent=2))
        else:
            print(f"Run: {state.get('run_id')}")
            print(f"Status: {state.get('status')}")
            print(f"Stage: {state.get('current_stage')}")
            print(f"Round: {state.get('current_round')}")
            print(f"Counters: {state.get('counters')}")
            print(f"Usage: {state.get('usage')}")

    elif args.command == "list":
        runs = list_runs()
        for run in runs:
            print(f"{run.get('run_id')}: {run.get('status')} (round {run.get('current_round')})")

    elif args.command == "emit":
        emit_event(
            args.type, args.message, stage=args.stage, round_num=args.round, run_id=args.run_id
        )
        flush_events()
        print("Event emitted")

    elif args.command == "stop":
        request_stop(args.run_id)
        print("Stop requested")

    elif args.command == "pause":
        request_pause(args.run_id)
        print("Pause requested")

    elif args.command == "resume":
        clear_pause(args.run_id)
        print("Resume requested")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
