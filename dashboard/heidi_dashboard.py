#!/usr/bin/env python3
"""
================================================================================
heidi_engine/dashboard.py - Real-Time Terminal Dashboard (TUI) for AutoTraining
================================================================================

PURPOSE:
    Provides a live, auto-updating terminal dashboard that shows:
    1. Pipeline progress counters (samples generated, validated, trained, etc.)
    2. Teacher API usage (requests, tokens, rate limits, cost)
    3. Training metrics (loss, steps, VRAM usage)
    4. Recent events/logs
    5. GPU status (if available)

    The dashboard auto-refreshes and uses minimal CPU when idle.

HOW IT WORKS:
    1. Reads state.json for current counters and status
    2. Tails events.jsonl for streaming updates
    3. Polls nvidia-smi for GPU info (low frequency)
    4. Uses Rich library for beautiful terminal output
    5. Supports multiple views/panels

TUNABLE PARAMETERS:
    - REFRESH_RATE: Dashboard refresh rate in Hz (default: 2, max: 10)
    - GPU_POLL_INTERVAL: How often to poll GPU in seconds (default: 5)
    - MAX_EVENTS: Number of recent events to display (default: 20)
    - AUTOTRAIN_DIR: Base directory for heidi_engine outputs

EXTENDING THIS MODULE:
    - Add new panels in create_layout()
    - Add new metrics in update_counters_panel()
    - Add new views in switch_view()
    - Customize colors in PANEL_STYLES

KEYBOARD SHORTCUTS:
    - q: Quit dashboard
    - r: Refresh now
    - 1-5: Switch between views (Overview, Teacher, Trainer, Events, Logs)
    - up/down: Navigate in menus
    - Enter: Select item

REQUIREMENTS:
    - Rich library: pip install rich
    - Optional: psutil for better system stats

================================================================================
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box

# Rich imports for TUI
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from heidi_engine.state_machine import CANONICAL_AUTOTRAIN_DIR

# =============================================================================
# CONFIGURATION - Adjust these for your needs
# =============================================================================

# Dashboard refresh rate (Hz) - higher = more responsive but more CPU
# TUNABLE: Increase for faster updates, decrease for less CPU usage
# MAX RECOMMENDED: 10 Hz
REFRESH_RATE = int(os.environ.get("DASHBOARD_REFRESH_RATE", "2"))

# GPU polling interval in seconds
# TUNABLE: Increase to reduce GPU overhead, decrease for real-time VRAM
GPU_POLL_INTERVAL = int(os.environ.get("GPU_POLL_INTERVAL", "5"))

# Maximum events to show in event log panel
# TUNABLE: Adjust based on screen size
MAX_EVENTS = int(os.environ.get("DASHBOARD_MAX_EVENTS", "20"))

AUTOTRAIN_DIR = os.environ.get("AUTOTRAIN_DIR", str(CANONICAL_AUTOTRAIN_DIR))

# Console width (auto-detected if not set)
CONSOLE_WIDTH = int(os.environ.get("CONSOLE_WIDTH", "0"))

# Color scheme - Rich styles
# TUNABLE: Customize colors for your terminal
PANEL_STYLES = {
    "header": "cyan",
    "counters": "green",
    "usage": "blue",
    "trainer": "magenta",
    "events": "yellow",
    "error": "red",
    "success": "green",
}


# =============================================================================
# GLOBAL STATE
# =============================================================================

console = Console()
running = True
current_view = "overview"
run_id: Optional[str] = None
last_event_position = 0
events_cache: deque = deque(maxlen=MAX_EVENTS)
gpu_info: Dict[str, Any] = {"available": False}
gpu_lock = threading.Lock()

# Data tail settings
data_tail_lines = 20
data_tail_show_raw = False  # False = clean, True = raw
last_data_position = 0
data_cache: deque = deque(maxlen=data_tail_lines)


# =============================================================================
# PATH MANAGEMENT
# =============================================================================


def get_run_dir(run_id: str) -> Path:
    """Get the run directory path."""
    return Path(AUTOTRAIN_DIR) / "runs" / run_id


def get_events_path(run_id: str) -> Path:
    """Get the event log file path."""
    return get_run_dir(run_id) / "events.jsonl"


def get_latest_data_file(run_id: str, data_dir: Path, clean: bool = True) -> Optional[Path]:
    """Get the latest round data file (clean or raw)."""
    if not data_dir.exists():
        return None

    pattern = "clean_round_" if clean else "raw_round_"
    files = sorted(
        data_dir.glob(f"{pattern}*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return files[0] if files else None


def load_new_data_lines(run_id: str) -> List[str]:
    """Load new lines from the latest data file (incremental read)."""
    global last_data_position, data_cache

    data_dir = Path(AUTOTRAIN_DIR) / "data"
    if not data_dir.exists():
        return []

    # Toggle between clean and raw
    data_file = get_latest_data_file(run_id, data_dir, clean=not data_tail_show_raw)
    if not data_file or not data_file.exists():
        return []

    try:
        file_size = data_file.stat().st_size

        # If file was truncated/rotated, reset position
        if file_size < last_data_position:
            last_data_position = 0

        if file_size > last_data_position:
            with open(data_file, "r") as f:
                f.seek(last_data_position)
                new_lines = f.readlines()
                last_data_position = file_size

            # Add to cache (keep last N lines total)
            for line in new_lines:
                line = line.strip()
                if line:
                    data_cache.append(line)
                    if len(data_cache) > data_tail_lines:
                        data_cache.popleft()

            return new_lines
    except Exception:
        pass

    return []


def get_state_path(run_id: str) -> Path:
    """Get the state file path."""
    return get_run_dir(run_id) / "state.json"


def get_config_path(run_id: str) -> Path:
    """Get the config file path."""
    return get_run_dir(run_id) / "config.json"


# =============================================================================
# STATE LOADING
# =============================================================================


def load_state(run_id: str) -> Dict[str, Any]:
    """
    Load current state from state.json.

    HOW IT WORKS:
        - Reads state.json file
        - Returns empty state if file doesn't exist or is invalid

    TUNABLE:
        - N/A

    ARGS:
        run_id: Run to read

    RETURNS:
        State dictionary
    """
    state_file = get_state_path(run_id)

    if not state_file.exists():
        return get_default_state()

    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load state: {e}[/yellow]")
        return get_default_state()


def get_default_state() -> Dict[str, Any]:
    """Get default empty state."""
    return {
        "run_id": run_id or "unknown",
        "status": "unknown",
        "current_round": 0,
        "current_stage": "initializing",
        "stop_requested": False,
        "pause_requested": False,
        "counters": {
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
        },
        "usage": {
            "requests_sent": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "rate_limits_hit": 0,
            "retries": 0,
            "estimated_cost_usd": 0.0,
        },
    }


def load_config(run_id: str) -> Dict[str, Any]:
    """Load run configuration."""
    config_file = get_config_path(run_id)

    if not config_file.exists():
        return {}

    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# EVENT STREAMING
# =============================================================================


def load_new_events(run_id: str) -> List[Dict[str, Any]]:
    """
    Load new events from event log since last read.

    HOW IT WORKS:
        - Tracks file position
        - Reads new lines only
        - Updates events cache

    TUNABLE:
        - N/A

    ARGS:
        run_id: Run to read

    RETURNS:
        List of new events
    """
    global last_event_position

    events_file = get_events_path(run_id)

    if not events_file.exists():
        return []

    try:
        with open(events_file) as f:
            f.seek(last_event_position)
            new_events = []
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        new_events.append(event)
                        events_cache.append(event)
                    except json.JSONDecodeError:
                        pass
            last_event_position = f.tell()
        return new_events
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to read events: {e}[/yellow]")
        return []


def format_time(ts: str) -> str:
    """Format ISO timestamp to HH:MM:SS."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts[:8] if ts else ""


# =============================================================================
# GPU MONITORING
# =============================================================================


def poll_gpu_info() -> Dict[str, Any]:
    """
    Poll GPU information using nvidia-smi.

    HOW IT WORKS:
        - Runs nvidia-smi command
        - Parses output for VRAM usage
        - Caches result for display

    TUNABLE:
        - Adjust polling frequency via GPU_POLL_INTERVAL
        - Add more metrics as needed

    RETURNS:
        Dictionary with GPU info (empty if no GPU)
    """
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
                used = int(parts[0].strip())
                total = int(parts[1].strip())
                util = int(parts[2].strip()) if len(parts) > 2 else 0

                return {
                    "available": True,
                    "memory_used_mb": used,
                    "memory_total_mb": total,
                    "memory_used_pct": (used / total * 100) if total > 0 else 0,
                    "utilization_pct": util,
                }
    except Exception:
        pass

    return {"available": False}


def start_gpu_poller():
    """
    Start background thread for GPU polling.

    HOW IT WORKS:
        - Polls GPU at intervals
        - Updates global gpu_info
        - Runs in background to avoid blocking

    TUNABLE:
        - Adjust GPU_POLL_INTERVAL for polling frequency
    """

    def poll_loop():
        while running:
            with gpu_lock:
                global gpu_info
                gpu_info = poll_gpu_info()
            time.sleep(GPU_POLL_INTERVAL)

    thread = threading.Thread(target=poll_loop, daemon=True)
    thread.start()


# =============================================================================
# UI COMPONENTS
# =============================================================================


def create_header(state: Dict[str, Any]) -> Panel:
    """
    Create header panel with run info.

    HOW IT WORKS:
        - Shows run ID, status, current stage/round
        - Color codes status

    TUNABLE:
        - Customize colors in PANEL_STYLES

    ARGS:
        state: Current state dictionary

    RETURNS:
        Rich Panel
    """
    status = state.get("status", "unknown")
    stage = state.get("current_stage", "N/A")
    round_num = state.get("current_round", 0)
    total_rounds = state.get("config", {}).get("ROUNDS", "?")

    # Color code status
    status_colors = {
        "running": "green",
        "paused": "yellow",
        "stopped": "grey50",
        "completed": "cyan",
        "error": "red",
    }
    status_color = status_colors.get(status, "white")

    content = Text()
    content.append("Run: ", Style(dim=True))
    content.append(f"{state.get('run_id', 'unknown')}\n", Style(color="cyan", bold=True))
    content.append("Status: ", Style(dim=True))
    content.append(f"{status.upper()}\n", Style(color=status_color, bold=True))
    content.append("Stage: ", Style(dim=True))
    content.append(f"{stage}\n", Style(color="white"))
    content.append("Round: ", Style(dim=True))
    content.append(f"{round_num}/{total_rounds}", Style(color="white"))

    if state.get("stop_requested"):
        content.append(" [STOP REQUESTED]", Style(color="red", bold=True))
    if state.get("pause_requested"):
        content.append(" [PAUSED]", Style(color="yellow", bold=True))

    return Panel(
        content, title="[bold]Heidi AutoTrain Dashboard[/bold]", border_style=PANEL_STYLES["header"]
    )


def create_counters_panel(state: Dict[str, Any]) -> Panel:
    """
    Create counters panel showing pipeline progress.

    HOW IT WORKS:
        - Shows all pipeline counters in a table
        - Highlights important metrics
        - Shows pass/fail ratios

    TUNABLE:
        - Add/remove counters as needed
        - Customize column layout

    ARGS:
        state: Current state dictionary

    RETURNS:
        Rich Panel
    """
    counters = state.get("counters", {})

    table = Table(box=box.MINIMAL, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="white", justify="right")

    # Teacher generation
    generated = counters.get("teacher_generated", 0)
    failed = counters.get("teacher_failed", 0)
    total = generated + failed
    table.add_row("Teacher Generated", str(generated))
    table.add_row("Teacher Failed", str(failed))
    if total > 0:
        table.add_row("Success Rate", f"{generated / total * 100:.1f}%")

    table.add_row("", "")  # Spacer

    # Validation
    validated = counters.get("validated_ok", 0)
    rejected_schema = counters.get("rejected_schema", 0)
    rejected_secret = counters.get("rejected_secret", 0)
    rejected_dedupe = counters.get("rejected_dedupe", 0)
    raw = counters.get("raw_written", 0)

    table.add_row("Raw Written", str(raw))
    table.add_row("Validated OK", str(validated))
    table.add_row("Rejected (Schema)", str(rejected_schema))
    table.add_row("Rejected (Secret)", str(rejected_secret))
    table.add_row("Rejected (Dedup)", str(rejected_dedupe))

    if raw > 0:
        table.add_row("Validation Rate", f"{validated / raw * 100:.1f}%")

    table.add_row("", "")  # Spacer

    # Unit tests (if enabled)
    test_pass = counters.get("test_pass", 0)
    test_fail = counters.get("test_fail", 0)
    if test_pass > 0 or test_fail > 0:
        table.add_row("Tests Passed", str(test_pass))
        table.add_row("Tests Failed", str(test_fail))
        if test_pass + test_fail > 0:
            table.add_row("Test Pass Rate", f"{test_pass / (test_pass + test_fail) * 100:.1f}%")

    return Panel(
        table, title="[bold]Pipeline Progress[/bold]", border_style=PANEL_STYLES["counters"]
    )


def create_usage_panel(state: Dict[str, Any]) -> Panel:
    """
    Create panel showing teacher API usage.

    HOW IT WORKS:
        - Shows API request counts
        - Shows token usage (input/output)
        - Shows rate limits and retries
        - Shows estimated cost (if pricing configured)

    TUNABLE:
        - Add more usage metrics
        - Customize pricing display

    ARGS:
        state: Current state dictionary

    RETURNS:
        Rich Panel
    """
    usage = state.get("usage", {})
    config = state.get("config", {})
    model = config.get("TEACHER_MODEL", "unknown")

    table = Table(box=box.MINIMAL, show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="white", justify="right")

    requests = usage.get("requests_sent", 0)
    table.add_row("API Requests", str(requests))

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    table.add_row("Input Tokens", f"{input_tokens:,}")
    table.add_row("Output Tokens", f"{output_tokens:,}")
    table.add_row("Total Tokens", f"{input_tokens + output_tokens:,}")

    table.add_row("", "")  # Spacer

    rate_limits = usage.get("rate_limits_hit", 0)
    retries = usage.get("retries", 0)
    table.add_row("Rate Limits Hit", str(rate_limits))
    table.add_row("Retries", str(retries))

    table.add_row("", "")  # Spacer

    # Cost (if available)
    cost = usage.get("estimated_cost_usd", 0)
    if cost > 0:
        table.add_row("Model", model)
        table.add_row("Est. Cost (USD)", f"${cost:.4f}")
    else:
        table.add_row("Model", model)
        table.add_row("Cost", "N/A (configure pricing)")

    # Per-round stats if we have round info
    current_round = state.get("current_round", 0)
    if current_round > 0 and requests > 0:
        table.add_row("", "")
        table.add_row("This Round", f"Round {current_round}")

    return Panel(table, title="[bold]Teacher API Usage[/bold]", border_style=PANEL_STYLES["usage"])


def create_trainer_panel(state: Dict[str, Any]) -> Panel:
    """
    Create panel showing training metrics.

    HOW IT WORKS:
        - Shows current training step
        - Shows loss (if available)
        - Shows evaluation metrics
        - Shows GPU VRAM usage

    TUNABLE:
        - Add more training metrics
        - Customize GPU display

    ARGS:
        state: Current state dictionary

    RETURNS:
        Rich Panel
    """
    counters = state.get("counters", {})
    config = state.get("config", {})

    table = Table(box=box.MINIMAL, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="white", justify="right")

    # Training progress
    train_step = counters.get("train_step", 0)
    train_steps_config = config.get("TRAIN_STEPS", 0)

    if train_step > 0:
        table.add_row("Current Step", str(train_step))
        if train_steps_config > 0:
            pct = train_step / train_steps_config * 100
            table.add_row("Progress", f"{pct:.1f}%")

        loss = counters.get("train_loss", 0)
        if loss > 0:
            table.add_row("Loss", f"{loss:.4f}")
    else:
        table.add_row("Training", "Not started")

    table.add_row("", "")  # Spacer

    # Evaluation metrics
    json_rate = counters.get("eval_json_parse_rate", 0)
    format_rate = counters.get("eval_format_rate", 0)

    if json_rate > 0:
        table.add_row("JSON Parse Rate", f"{json_rate * 100:.1f}%")
    if format_rate > 0:
        table.add_row("Format Rate", f"{format_rate * 100:.1f}%")

    table.add_row("", "")  # Spacer

    # GPU info
    with gpu_lock:
        ginfo = gpu_info.copy()

    if ginfo.get("available"):
        used = ginfo.get("memory_used_mb", 0)
        total = ginfo.get("memory_total_mb", 0)
        util = ginfo.get("utilization_pct", 0)

        table.add_row("GPU VRAM", f"{used} / {total} MB")
        table.add_row("VRAM Usage", f"{used / total * 100:.1f}%" if total > 0 else "N/A")
        table.add_row("GPU Util", f"{util}%")
    else:
        table.add_row("GPU", "Not available")

    return Panel(table, title="[bold]Training Status[/bold]", border_style=PANEL_STYLES["trainer"])


def create_events_panel() -> Panel:
    """
    Create panel showing recent events.

    HOW IT WORKS:
        - Shows last N events from cache
        - Color codes by level
        - Shows timestamp, stage, message

    TUNABLE:
        - Adjust MAX_EVENTS
        - Customize formatting

    RETURNS:
        Rich Panel
    """
    table = Table(box=box.MINIMAL, show_header=True, header_style="bold yellow")
    table.add_column("Time", style="grey50", width=10)
    table.add_column("Stage", style="cyan", width=12)
    table.add_column("Event", style="white")

    # Show recent events in reverse order (newest first)
    events_list = list(events_cache)[-MAX_EVENTS:]
    events_list.reverse()

    for event in events_list:
        ts = format_time(event.get("ts", ""))
        stage = event.get("stage", "")
        event.get("event_type", "")
        message = event.get("message", "")[:50]

        # Color code by level
        level = event.get("level", "info")
        level_colors = {
            "info": "white",
            "warn": "yellow",
            "error": "red",
            "success": "green",
        }
        color = level_colors.get(level, "white")

        table.add_row(ts, stage, Text(message, style=color))

    if not events_list:
        table.add_row("", "", "No events yet...")

    return Panel(
        table, title="[bold]Recent Events[/bold]", border_style=PANEL_STYLES["events"], height=15
    )


def create_data_panel() -> Panel:
    """
    Create panel showing data tail (latest lines from data file).

    HOW IT WORKS:
        - Shows last N lines from latest clean_round_X.jsonl (or raw)
        - Toggle with data_tail_show_raw (False = clean, True = raw)
        - Updates incrementally like events

    TUNABLE:
        - Adjust data_tail_lines

    RETURNS:
        Rich Panel
    """
    global data_tail_show_raw

    file_type = "clean" if not data_tail_show_raw else "raw"
    title = f"[bold]Data Tail ({file_type})[/bold]"

    # Get latest data file
    if run_id:
        data_dir = Path(AUTOTRAIN_DIR) / "data"
        data_file = get_latest_data_file(run_id, data_dir, clean=not data_tail_show_raw)
    else:
        data_file = None

    if not data_file or not data_file.exists():
        return Panel(
            Text("No data file found yet"), title=title, border_style=PANEL_STYLES["events"]
        )

    # Show cached lines (truncated)
    lines = list(data_cache)[-data_tail_lines:]

    table = Table(box=box.MINIMAL, show_header=False)
    table.add_column("Line", style="white")

    for line in lines:
        # Truncate long lines
        display = line[:100] + "..." if len(line) > 100 else line
        table.add_row(display)

    if not lines:
        table.add_row("Waiting for data...")

    file_name = data_file.name if data_file else "none"
    subtitle = f"File: {file_name} (d to toggle raw/clean)"

    return Panel(
        table,
        title=title,
        subtitle=subtitle,
        border_style=PANEL_STYLES["events"],
        height=data_tail_lines + 3,
    )


def request_train_now() -> bool:
    """
    Request training to start via telemetry.

    Returns True if successful, False otherwise.
    """
    try:
        from heidi_engine.telemetry import request_train_now as do_request

        do_request()
        return True
    except Exception as e:
        console.print(f"[red]Failed to request train: {e}[/red]")
        return False


def create_config_panel(state: Dict[str, Any]) -> Panel:
    """
    Create panel showing current configuration.

    HOW IT WORKS:
        - Shows key config values
        - Helps user understand current settings

    TUNABLE:
        - Add/remove config keys

    ARGS:
        state: Current state dictionary

    RETURNS:
        Rich Panel
    """
    config = state.get("config", {})

    table = Table(box=box.MINIMAL, show_header=False)
    table.add_column("Key", style="cyan", width=20)
    table.add_column("Value", style="white")

    key_config = [
        ("BASE_MODEL", "base_model"),
        ("TEACHER_MODEL", "teacher_model"),
        ("SAMPLES_PER_ROUND", "samples_per_round"),
        ("ROUNDS", "rounds"),
        ("VAL_RATIO", "val_ratio"),
        ("SEQ_LEN", "seq_len"),
        ("LORA_R", "lora_r"),
        ("GRAD_ACCUM", "grad_accum"),
        ("RUN_UNIT_TESTS", "run_unit_tests"),
    ]

    for display_key, config_key in key_config:
        value = config.get(config_key, config.get(display_key, "N/A"))
        if value is not None:
            table.add_row(display_key, str(value))

    return Panel(table, title="[bold]Configuration[/bold]", border_style="grey50")


def create_keybindings_panel() -> Panel:
    """
    Create panel showing keyboard shortcuts.

    HOW IT WORKS:
        - Shows available keyboard shortcuts
        - Helps user navigate dashboard

    TUNABLE:
        - Add/remove shortcuts

    RETURNS:
        Rich Panel
    """
    table = Table(box=box.MINIMAL, show_header=False)
    table.add_column("Key", style="cyan", width=10)
    table.add_column("Action", style="white")

    shortcuts = [
        ("q", "Quit dashboard"),
        ("r", "Refresh now"),
        ("1", "Overview view"),
        ("2", "Teacher usage"),
        ("3", "Trainer metrics"),
        ("4", "Recent events"),
        ("5", "Data tail"),
        ("6", "Configuration"),
        ("d", "Toggle raw/clean data"),
        ("f", "Feed data (request train)"),
    ]

    for key, action in shortcuts:
        table.add_row(key, action)

    return Panel(table, title="[bold]Keyboard Shortcuts[/bold]", border_style="grey50")


# =============================================================================
# VIEW LAYOUTS
# =============================================================================


def create_overview_layout(state: Dict[str, Any]) -> Layout:
    """
    Create overview layout showing main metrics.

    HOW IT WORKS:
        - Shows header, counters, usage, trainer panels
        - Good default view

    TUNABLE:
        - Customize panel arrangement

    ARG:
        state: Current state dictionary

    RETURNS:
        Rich Layout
    """
    layout = Layout()

    # Top row: counters and usage
    top = Layout(name="top")
    top.split_column(
        Layout(name="counters", ratio=1),
        Layout(name="usage", ratio=1),
    )

    # Bottom row: trainer and events
    bottom = Layout(name="bottom")
    bottom.split_column(
        Layout(name="trainer", ratio=1),
        Layout(name="events", ratio=1),
    )

    layout.split_column(top, bottom)

    # Populate panels
    layout["counters"].update(create_counters_panel(state))
    layout["usage"].update(create_usage_panel(state))
    layout["trainer"].update(create_trainer_panel(state))
    layout["events"].update(create_events_panel())

    return layout


def create_teacher_layout(state: Dict[str, Any]) -> Layout:
    """Create layout focused on teacher usage."""
    layout = Layout()
    layout.split_column(
        Layout(name="usage", ratio=2),
        Layout(name="events", ratio=1),
    )
    layout["usage"].update(create_usage_panel(state))
    layout["events"].update(create_events_panel())
    return layout


def create_trainer_layout(state: Dict[str, Any]) -> Layout:
    """Create layout focused on training."""
    layout = Layout()
    layout.split_column(
        Layout(name="trainer", ratio=2),
        Layout(name="events", ratio=1),
    )
    layout["trainer"].update(create_trainer_panel(state))
    layout["events"].update(create_events_panel())
    return layout


def create_events_layout(state: Dict[str, Any]) -> Layout:
    """Create layout focused on events."""
    layout = Layout()
    layout.update(
        Panel(
            create_events_panel().renderable,
            title="[bold]Event Log[/bold]",
            border_style=PANEL_STYLES["events"],
        )
    )
    return layout


def create_data_layout(state: Dict[str, Any]) -> Layout:
    """Create layout focused on data tail."""
    layout = Layout()
    layout.split_column(
        Layout(name="data", ratio=3),
        Layout(name="help", ratio=1),
    )
    layout["data"].update(create_data_panel())
    layout["help"].update(create_keybindings_panel())
    return layout


def create_config_layout(state: Dict[str, Any]) -> Layout:
    """Create layout focused on configuration."""
    layout = Layout()
    layout.split_column(
        Layout(name="config", ratio=3),
        Layout(name="help", ratio=1),
    )
    layout["config"].update(create_config_panel(state))
    layout["help"].update(create_keybindings_panel())
    return layout


def create_main_layout(state: Dict[str, Any]) -> Tuple[Layout, str]:
    """
    Create main layout based on current view.

    HOW IT WORKS:
        - Returns appropriate layout for current view
        - Also returns panel title

    TUNABLE:
        - Add new views

    ARGS:
        state: Current state dictionary

    RETURNS:
        Tuple of (Layout, title)
    """
    global current_view

    view_titles = {
        "overview": "Overview",
        "teacher": "Teacher API Usage",
        "trainer": "Training Metrics",
        "events": "Event Log",
        "data": "Data Tail",
        "config": "Configuration",
    }

    layouts = {
        "overview": create_overview_layout,
        "teacher": create_teacher_layout,
        "trainer": create_trainer_layout,
        "events": create_events_layout,
        "data": create_data_layout,
        "config": create_config_layout,
    }

    builder = layouts.get(current_view, create_overview_layout)
    title = view_titles.get(current_view, "Overview")

    return builder(state), title


# =============================================================================
# MAIN DASHBOARD LOOP
# =============================================================================


def run_dashboard(run_id: str):
    """
    Main dashboard loop.

    HOW IT WORKS:
        1. Initializes state
        2. Starts GPU poller
        3. Enters render loop
        4. Handles keyboard input
        5. Exits gracefully on q or Ctrl+C

    TUNABLE:
        - Adjust REFRESH_RATE for update frequency

    ARGS:
        run_id: Run to monitor
    """
    global running, current_view, last_event_position, events_cache

    # Initialize
    console.clear()
    console.print(f"[cyan]Starting dashboard for run: {run_id}[/cyan]")

    # Load initial state
    state = load_state(run_id)
    config = load_config(run_id)
    state["config"] = config

    # Initialize event position
    events_file = get_events_path(run_id)
    if events_file.exists():
        last_event_position = events_file.stat().st_size

    # Start GPU poller
    start_gpu_poller()

    # Handle signals
    def signal_handler(sig, frame):
        global running
        console.print("\n[yellow]Shutting down dashboard...[/yellow]")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main render loop
    try:
        with Live(console=console, refresh_per_second=REFRESH_RATE, screen=True) as live:
            while running:
                # Load fresh state
                state = load_state(run_id)
                state["config"] = config

                # Check for new events
                load_new_events(run_id)

                # Check for new data lines
                load_new_data_lines(run_id)

                # Build layout
                layout, title = create_main_layout(state)

                # Create full screen
                screen = Layout(name="root")
                screen.split_column(
                    Layout(size=3, name="header"),
                    Layout(name="main"),
                )

                screen["header"].update(create_header(state))
                screen["main"].update(layout)

                # Render
                live.update(screen)

                # Sleep until next refresh
                time.sleep(1.0 / REFRESH_RATE)

    except Exception as e:
        console.print(f"[red]Error in dashboard: {e}[/red]")
        raise


def list_runs() -> List[str]:
    """
    List available runs.

    HOW IT WORKS:
        - Scans runs/ directory
        - Returns run IDs with state files

    TUNABLE:
        - N/A

    RETURNS:
        List of run IDs
    """
    runs_dir = Path(AUTOTRAIN_DIR) / "runs"

    if not runs_dir.exists():
        return []

    runs = []
    for run_path in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_path.is_dir():
            continue
        if (run_path / "state.json").exists():
            runs.append(run_path.name)

    return runs


def select_run() -> Optional[str]:
    """
    Interactive run selection.

    HOW IT WORKS:
        - Lists available runs
        - Prompts user to select one
        - Returns selected run ID

    TUNABLE:
        - N/A

    RETURNS:
        Selected run ID or None
    """
    runs = list_runs()

    if not runs:
        runs_dir = Path(AUTOTRAIN_DIR) / "runs"
        console.print(f"[yellow]No runs found in {runs_dir}[/yellow]")
        return None

    if len(runs) == 1:
        return runs[0]

    console.print("\n[cyan]Available runs:[/cyan]")
    for i, run_id in enumerate(runs):
        state = load_state(run_id)
        status = state.get("status", "unknown")
        round_num = state.get("current_round", 0)
        console.print(f"  [{i + 1}] {run_id} - {status} (round {round_num})")

    while True:
        try:
            choice = console.input("\nSelect run number (or Enter for latest): ").strip()
            if not choice:
                return runs[0]
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx]
        except ValueError:
            pass
        console.print("[red]Invalid selection, try again[/red]")


# =============================================================================
# CLI
# =============================================================================


def main():
    """
    CLI for dashboard.

    TUNABLE:
        - N/A

    COMMANDS:
        python -m heidi_engine.dashboard [--run RUN_ID]

        If no run specified, shows interactive selection
    """
    global run_id, current_view

    parser = argparse.ArgumentParser(
        prog="heidi-engine dashboard", description="Heidi Engine Real-Time Dashboard"
    )
    parser.add_argument("--run", "-r", help="Run ID to monitor")
    parser.add_argument(
        "--view",
        "-v",
        choices=["overview", "teacher", "trainer", "events", "config"],
        default="overview",
        help="Initial view",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List available runs")
    args = parser.parse_args()

    # Set view
    current_view = args.view

    # List runs if requested
    if args.list:
        runs = list_runs()
        if runs:
            console.print("[cyan]Available runs:[/cyan]")
            for run_id in runs:
                state = load_state(run_id)
                status = state.get("status", "unknown")
                round_num = state.get("current_round", 0)
                console.print(f"  {run_id}: {status} (round {round_num})")
        else:
            console.print("[yellow]No runs found[/yellow]")
        return

    # Select run
    if args.run:
        run_id = args.run
    else:
        run_id = select_run()

    if not run_id:
        console.print("[red]No run selected[/red]")
        sys.exit(1)

    # Check run exists
    if not get_state_path(run_id).exists():
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)

    # Run dashboard
    console.print(f"[green]Starting dashboard for run: {run_id}[/green]")
    run_dashboard(run_id)


if __name__ == "__main__":
    main()
