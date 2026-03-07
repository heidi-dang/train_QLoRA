#!/usr/bin/env python3
"""Live Dashboard for QLoRA Training."""

import os
import sys
import json
import time
import signal
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.resolve()
TELEMETRY_FILE = ROOT / "state" / "telemetry.json"
LOGS_DIR = ROOT / "logs"
LOG_FILE = LOGS_DIR / "loop.log"
MAX_LOG_LINES = 300
RENDER_INTERVAL = 0.2

ANSI_CLEAR_SCREEN = "\033[2J"
ANSI_CURSOR_HOME = "\033[H"
ANSI_CURSOR_HIDE = "\033[?25l"
ANSI_CURSOR_SHOW = "\033[?25h"


class RunState:
    def __init__(self):
        self.status = "idle"
        self.current_stage = ""
        self.stage_index = 0
        self.total_stages = 4
        self.stage_percent = 0.0
        self.overall_percent = 0.0
        self.completed_units = 0
        self.total_units = 0
        self.provider = ""
        self.model = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.spend_usd = 0.0
        self.cpu_percent = 0.0
        self.ram_percent = 0.0
        self.ram_used_gb = 0.0
        self.ram_total_gb = 0.0
        self.gpu_percent = 0.0
        self.gpu_memory_used_gb = 0.0
        self.gpu_memory_total_gb = 0.0
        self.log_buffer: deque = deque(maxlen=MAX_LOG_LINES)
        self.last_update: Optional[datetime] = None
        self.running = False
        self.stopped_cleanly = False
        self._last_log_size = 0
        self._last_telem_hash = ""

    def load_telemetry(self) -> bool:
        if not TELEMETRY_FILE.exists():
            return False
        try:
            with open(TELEMETRY_FILE) as f:
                telem = json.load(f)
            telem_str = json.dumps(telem, sort_keys=True)
            if telem_str == self._last_telem_hash:
                return False
            self._last_telem_hash = telem_str
            self.status = telem.get("status", "idle")
            self.current_stage = telem.get("current_stage", "")
            self.stage_index = telem.get("stage_index", 0)
            self.total_stages = telem.get("total_stages", 4)
            self.stage_percent = telem.get("stage_percent", 0.0) * 100
            self.overall_percent = telem.get("overall_percent", 0.0) * 100
            self.completed_units = telem.get("completed_units", 0)
            self.total_units = telem.get("total_units", 0)
            usage = telem.get("usage", {})
            self.provider = usage.get("provider", "")
            self.model = usage.get("model", "")
            self.prompt_tokens = usage.get("prompt_tokens", 0)
            self.completion_tokens = usage.get("completion_tokens", 0)
            self.total_tokens = usage.get("total_tokens", 0)
            self.request_count = usage.get("request_count", 0)
            self.success_count = usage.get("successful_requests", 0)
            self.failed_count = usage.get("failed_requests", 0)
            self.spend_usd = usage.get("spend_usd", 0.0)
            self.last_update = datetime.now()
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def load_resources(self):
        try:
            import psutil
            self.cpu_percent = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            self.ram_percent = mem.percent
            self.ram_used_gb = mem.used / (1024**3)
            self.ram_total_gb = mem.total / (1024**3)
        except ImportError:
            pass
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(", ")
                if len(parts) == 3:
                    self.gpu_percent = int(parts[0])
                    self.gpu_memory_used_gb = float(parts[1]) / 1024
                    self.gpu_memory_total_gb = float(parts[2]) / 1024
        except Exception:
            pass

    def append_logs(self):
        if not LOG_FILE.exists():
            return
        try:
            with open(LOG_FILE, "r") as f:
                f.seek(0, 2)
                file_size = f.tell()
                if file_size > self._last_log_size:
                    f.seek(self._last_log_size)
                    new_lines = f.readlines()
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            if len(line) > 200:
                                line = line[:197] + "..."
                            self.log_buffer.append(line)
                    self._last_log_size = file_size
                elif file_size < self._last_log_size:
                    self._last_log_size = 0
        except IOError:
            pass


class DashboardRenderer:
    def __init__(self, state: RunState):
        self.state = state
        self.width = 80
        self.height = 24

    def hide_cursor(self):
        sys.stdout.write(ANSI_CURSOR_HIDE)
        sys.stdout.flush()

    def show_cursor(self):
        sys.stdout.write(ANSI_CURSOR_SHOW)
        sys.stdout.flush()

    def render(self) -> str:
        s = self.state
        lines = []
        clock = datetime.now().strftime("%H:%M:%S")
        status_map = {
            "running": "RUNNING", "starting": "STARTING", "stopped": "STOPPED",
            "completed": "COMPLETED", "failed": "FAILED", "idle": "IDLE"
        }
        status_display = status_map.get(s.status, "IDLE")
        provider = s.provider or "N/A"
        model = s.model or "N/A"
        
        lines.append(f" QLoRA Training Dashboard | {status_display} | {clock} ")
        lines.append("=" * self.width)
        lines.append(" TEACHER USAGE & COST                  PROGRESS                    ")
        lines.append("-" * self.width)
        lines.append(f" Provider: {provider:<20} Stage: {s.current_stage} ({s.stage_index + 1}/{s.total_stages})")
        lines.append(f" Model:    {model:<20} Stage%: {s.stage_percent:>6.1f}%")
        lines.append(f" Prompt Tokens:   {s.prompt_tokens:>12,} Overall%: {s.overall_percent:>6.1f}%")
        lines.append(f" Completion Toks: {s.completion_tokens:>12,} Units: {s.completed_units:,}/{s.total_units:,}")
        lines.append(f" Total Tokens:    {s.total_tokens:>12,} Requests: {s.request_count:,}")
        lines.append(f" Success: {s.success_count:>8,} Failed: {s.failed_count:<8} Spend: ${s.spend_usd:>10.4f} USD")
        lines.append("-" * self.width)
        lines.append(" RESOURCES                            RECENT LOGS                 ")
        lines.append("-" * self.width)
        
        cpu = f"CPU: {s.cpu_percent:>5.1f}%"
        ram = f"RAM: {s.ram_percent:>5.1f}% ({s.ram_used_gb:.1f}/{s.ram_total_gb:.1f}GB)"
        res_lines = [cpu, ram]
        if s.gpu_percent > 0:
            res_lines.append(f"GPU: {s.gpu_percent:>5.0f}%")
        if s.gpu_memory_total_gb > 0:
            res_lines.append(f"GPU Mem: {s.gpu_memory_used_gb:.1f}/{s.gpu_memory_total_gb:.1f}GB")
        
        log_lines = list(s.log_buffer)[-8:]
        if not log_lines:
            log_lines = ["(no logs yet)"]
        
        max_res_lines = max(len(res_lines), len(log_lines))
        res_lines.extend([""] * (max_res_lines - len(res_lines)))
        log_lines_extended = log_lines.copy()
        log_lines_extended.extend([""] * (max_res_lines - len(log_lines)))
        
        for res_line, log_line in zip(res_lines, log_lines_extended):
            if len(log_line) > 40:
                log_line = log_line[:37] + "..."
            lines.append(f" {res_line:<35} {log_line:<40}")
        
        lines.append("=" * self.width)
        return "\n".join(lines)


class LiveDashboard:
    def __init__(self):
        self.state = RunState()
        self.renderer = DashboardRenderer(self.state)
        self.running = False
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.stop()

    def start(self):
        self.running = True
        self.state.running = True
        self.renderer.hide_cursor()
        self.state.load_telemetry()
        self.state.load_resources()
        self.state.append_logs()
        self._render()
        while not self._stop_event.is_set():
            self.state.load_telemetry()
            self.state.load_resources()
            self.state.append_logs()
            self._render()
            self._stop_event.wait(RENDER_INTERVAL)
        self._exit()

    def _render(self):
        output = self.renderer.render()
        sys.stdout.write(ANSI_CLEAR_SCREEN + ANSI_CURSOR_HOME)
        sys.stdout.write(output)
        sys.stdout.flush()

    def _exit(self):
        print("\n" + "=" * self.renderer.width, flush=True)
        print(f"Status: {self.state.status}", flush=True)
        print(f"Tokens: {self.state.total_tokens:,} | Spend: ${self.state.spend_usd:.4f}", flush=True)
        self.renderer.show_cursor()
        self.state.running = False
        self.state.stopped_cleanly = True
        self.running = False

    def stop(self):
        self._stop_event.set()


def main():
    dashboard = LiveDashboard()
    dashboard.start()


if __name__ == "__main__":
    main()
