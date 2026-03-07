#!/usr/bin/env python3
import os
import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

try:
    from rich.console import Console
except ImportError:
    print("Rich not available")
    exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TELEMETRY_FILE = os.path.join(ROOT, 'state', 'telemetry.json')
LOGS_DIR = os.path.join(ROOT, 'logs')
MAX_LOG_LINES = 50

console = Console()

PRICING = {
    "grok-beta": {"prompt": 0.0000025, "completion": 0.00001},
    "grok-4-1-fast": {"prompt": 0.000003, "completion": 0.000015},
    "gpt-4": {"prompt": 0.00003, "completion": 0.00006},
}


class DashboardState:
    def __init__(self):
        self.telemetry = self._default_telemetry()
        self.log_lines = deque(maxlen=MAX_LOG_LINES)
        self.last_log_size = 0
    
    def _default_telemetry(self):
        return {
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
    
    def load_telemetry(self):
        if os.path.exists(TELEMETRY_FILE):
            try:
                with open(TELEMETRY_FILE, 'r') as f:
                    self.telemetry = json.load(f)
            except:
                pass
    
    def append_logs(self, new_lines):
        for line in new_lines:
            if line.strip():
                self.log_lines.append(line.strip())


def get_new_log_lines():
    log_file = os.path.join(LOGS_DIR, 'loop.log')
    new_lines = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                f.seek(0, 2)
                file_size = f.tell()
                if file_size > 500000:
                    f.seek(file_size - 500000)
                else:
                    f.seek(0)
                lines = f.readlines()
                new_lines = [l.strip() for l in lines[-50:] if l.strip()]
        except:
            pass
    return new_lines


def render_dashboard(state):
    telem = state.telemetry
    lines = list(state.log_lines)
    
    usage = telem.get("usage", {})
    model = usage.get("model", "N/A") or "N/A"
    provider = usage.get("provider", "N/A") or "N/A"
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    requests = usage.get("request_count", 0)
    success = usage.get("successful_requests", 0)
    failed = usage.get("failed_requests", 0)
    spend = usage.get("spend_usd", 0.0)
    
    stage = telem.get("current_stage", "N/A")
    idx = telem.get("stage_index", 0)
    total = telem.get("total_stages", 4)
    stage_pct = telem.get("stage_percent", 0.0)
    overall_pct = telem.get("overall_percent", 0.0)
    completed = telem.get("completed_units", 0)
    total_units = telem.get("total_units", 0)
    eta = telem.get("eta_seconds", 0)
    
    status = telem.get("status", "idle")
    
    cpu = psutil.cpu_percent(interval=0)
    mem = psutil.virtual_memory()
    
    gpu = {'gpu_util': 0, 'gpu_mem_used': 0, 'gpu_mem_total': 0, 'gpu_mem_percent': 0}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(', ')
            if len(parts) == 3:
                gpu['gpu_util'] = int(parts[0])
                gpu['gpu_mem_used'] = float(parts[1]) / 1024
                gpu['gpu_mem_total'] = float(parts[2]) / 1024
                gpu['gpu_mem_percent'] = (float(parts[1]) / float(parts[2]) * 100) if float(parts[2]) > 0 else 0
    except:
        pass
    
    status_map = {
        "running": "RUNNING",
        "stopped": "STOPPED",
        "completed": "COMPLETED", 
        "failed": "FAILED",
        "idle": "IDLE"
    }
    
    status_color = {
        "running": "green",
        "stopped": "yellow",
        "completed": "blue", 
        "failed": "red",
        "idle": "white"
    }
    
    output = []
    output.append("")
    output.append("=" * 80)
    output.append(f"QLoRA Training Dashboard | [{status_color.get(status, 'white')}]{status_map.get(status, 'IDLE')}[/{status_color.get(status, 'white')}] | {datetime.now().strftime('%H:%M:%S')}")
    output.append("=" * 80)
    output.append("")
    
    output.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    output.append("│ 💰 TEACHER USAGE & COST                                                 │")
    output.append("├─────────────────────────────────────────────────────────────────────────────┤")
    output.append(f"│ Provider:       {provider:<20} Model: {model:<30}│")
    output.append(f"│ Prompt Tokens:  {prompt_tokens:>15,}   Completion Tokens: {completion_tokens:>15,}│")
    output.append(f"│ Total Tokens:   {total_tokens:>15,}   API Requests:    {requests:>15,}│")
    output.append(f"│ Success:       {success:>15,}   Failed:          {failed:>15,}│")
    output.append(f"│                                                                            │")
    output.append(f"│ 💵 SPEND: $[green]{spend:>10.4f}[/green] USD                                             │")
    output.append("└─────────────────────────────────────────────────────────────────────────────┘")
    output.append("")
    
    output.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    output.append("│ 📊 PROGRESS                                                            │")
    output.append("├─────────────────────────────────────────────────────────────────────────────┤")
    output.append(f"│ Stage: {stage} ({idx+1}/{total})                                              │")
    output.append(f"│ Stage Progress: {stage_pct:>6.1f}%                                                     │")
    output.append(f"│ Overall Progress: {overall_pct:>6.1f}%                                                 │")
    bar_len = 40
    filled = int(bar_len * stage_pct / 100)
    bar = '█' * filled + '░' * (bar_len - filled)
    output.append(f"│ [{bar}] {stage_pct:>5.1f}%                                           │")
    output.append(f"│ Units: {completed:,}/{total_units:,}                                                      │")
    if eta > 0:
        output.append(f"│ ETA: {str(timedelta(seconds=eta)):<45}│")
    output.append("└─────────────────────────────────────────────────────────────────────────────┘")
    output.append("")
    
    output.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    output.append("│ 🖥️ RESOURCES                                                            │")
    output.append("├─────────────────────────────────────────────────────────────────────────────┤")
    output.append(f"│ CPU: {cpu:>6.1f}%                                                           │")
    output.append(f"│ Memory: {mem.percent:>5.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB)                             │")
    if gpu['gpu_util'] > 0:
        output.append(f"│ GPU: {gpu['gpu_util']:>5}%                                                            │")
        output.append(f"│ GPU Mem: {gpu['gpu_mem_percent']:>5.1f}% ({gpu['gpu_mem_used']:.1f}/{gpu['gpu_mem_total']:.1f} GB)                         │")
    output.append("└─────────────────────────────────────────────────────────────────────────────┘")
    output.append("")
    
    output.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    output.append("│ 📋 RECENT LOGS (preserved, no flash)                                   │")
    output.append("├─────────────────────────────────────────────────────────────────────────────┤")
    for line in lines[-15:]:
        if len(line) > 70:
            line = line[:67] + "..."
        output.append(f"│ {line:<77}│")
    output.append("└─────────────────────────────────────────────────────────────────────────────┘")
    output.append("")
    
    return "\n".join(output)


def main():
    state = DashboardState()
    last_telem_hash = ""
    last_log_count = 0
    
    console.line()
    console.print("[bold green]Starting Dashboard...[/bold green] Logs and progress preserved - no flash")
    console.line()
    
    while True:
        try:
            state.load_telemetry()
            telem_str = json.dumps(state.telemetry, sort_keys=True)
            
            new_logs = get_new_log_lines()
            if len(new_logs) > last_log_count:
                state.append_logs(new_logs)
                last_log_count = len(new_logs)
            
            if telem_str != last_telem_hash:
                render = render_dashboard(state)
                console.print(render)
                last_telem_hash = telem_str
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped[/yellow]")
            break
        except Exception as e:
            time.sleep(1)


if __name__ == '__main__':
    main()
