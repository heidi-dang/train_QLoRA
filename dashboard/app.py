#!/usr/bin/env python3
import os
import json
import time
import threading
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
except ImportError:
    print("Rich not available")
    exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TELEMETRY_FILE = os.path.join(ROOT, 'state', 'telemetry.json')
LOGS_DIR = os.path.join(ROOT, 'logs')

console = Console()

PRICING = {
    "grok-beta": {"prompt": 0.0000025, "completion": 0.00001},
    "grok-4-1-fast": {"prompt": 0.000003, "completion": 0.000015},
    "gpt-4": {"prompt": 0.00003, "completion": 0.00006},
}


def load_telemetry():
    if os.path.exists(TELEMETRY_FILE):
        try:
            with open(TELEMETRY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
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


def get_gpu_stats():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].strip():
                gpu_util, mem_used, mem_total = lines[0].split(', ')
                return {
                    'gpu_util': int(gpu_util),
                    'gpu_mem_used': int(mem_used) / 1024,
                    'gpu_mem_total': int(mem_total) / 1024,
                    'gpu_mem_percent': (int(mem_used) / int(mem_total) * 100) if int(mem_total) > 0 else 0
                }
    except:
        pass
    return {'gpu_util': 0, 'gpu_mem_used': 0, 'gpu_mem_total': 0, 'gpu_mem_percent': 0}


def create_header(telem):
    status = telem.get("status", "idle")
    status_color = {"running": "green", "stopped": "yellow", "completed": "blue", "failed": "red", "idle": "dim"}
    status_text = f"[{status_color.get(status, 'white')}]{status.upper()}[/{status_color.get(status, 'white')}]"
    return Panel(
        f"[bold cyan]QLoRA Training[/bold cyan] | Status: {status_text} | {datetime.now().strftime('%H:%M:%S')}",
        style="blue"
    )


def create_usage_panel(telem):
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

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Provider", provider)
    table.add_row("Model", model)
    table.add_row("Prompt Tokens", f"{prompt_tokens:,}")
    table.add_row("Completion Tokens", f"{completion_tokens:,}")
    table.add_row("Total Tokens", f"{total_tokens:,}")
    table.add_row("API Requests", f"{requests:,}")
    table.add_row("Success", f"{success:,}")
    table.add_row("Failed", f"{failed:,}")
    table.add_row("[bold]Spend USD[/bold]", f"[bold green]${spend:.4f}[/bold green]")

    return Panel(table, title="💰 Teacher Usage & Cost", border_style="green")


def create_progress_panel(telem):
    stage = telem.get("current_stage", "N/A")
    idx = telem.get("stage_index", 0)
    total = telem.get("total_stages", 4)
    stage_pct = telem.get("stage_percent", 0.0)
    overall_pct = telem.get("overall_percent", 0.0)
    completed = telem.get("completed_units", 0)
    total_units = telem.get("total_units", 0)
    eta = telem.get("eta_seconds", 0)

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Stage", f"{stage} ({idx+1}/{total})")
    table.add_row("Stage Progress", f"{stage_pct:.1f}%")
    table.add_row("Overall Progress", f"{overall_pct:.1f}%")
    table.add_row("Units", f"{completed:,}/{total_units:,}")
    if eta > 0:
        table.add_row("ETA", str(timedelta(seconds=eta)))

    bar_len = 30
    filled = int(bar_len * stage_pct // 100)
    bar = '█' * filled + '░' * (bar_len - filled)
    table.add_row("Progress", f"[green]{bar}[/green] {stage_pct:.1f}%")

    return Panel(table, title="📊 Progress", border_style="cyan")


def create_resources_panel():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    gpu = get_gpu_stats()

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("CPU", f"{cpu:.1f}%")
    table.add_row("Memory", f"{mem.percent:.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB)")
    if gpu['gpu_util'] > 0:
        table.add_row("GPU", f"{gpu['gpu_util']}%")
        table.add_row("GPU Memory", f"{gpu['gpu_mem_percent']:.1f}% ({gpu['gpu_mem_used']:.1f}/{gpu['gpu_mem_total']:.1f}GB)")

    return Panel(table, title="🖥️ Resources", border_style="blue")


def create_status_panel(telem):
    status = telem.get("status", "idle")
    stage = telem.get("current_stage", "")

    status_map = {
        "running": "🟢 Running",
        "stopped": "🟡 Stopped",
        "completed": "🔵 Completed", 
        "failed": "🔴 Failed",
        "idle": "⚪ Idle"
    }

    services = []
    for name, pidfile in [("API", "api.pid"), ("Dashboard", "dashboard.pid"), ("Loop", "loop.pid"), ("MLflow", "mlflow.pid"), ("TB", "tb.pid")]:
        pid_path = os.path.join(ROOT, "state", "pids", pidfile)
        if os.path.exists(pid_path):
            try:
                with open(pid_path) as f:
                    pid = int(f.read().strip())
                    if psutil.pid_exists(pid):
                        services.append(f"{name}: ✓")
                    else:
                        services.append(f"{name}: ✗")
            except:
                services.append(f"{name}: ✗")
        else:
            services.append(f"{name}: -")

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Status", status_map.get(status, status))
    table.add_row("Stage", stage)
    table.add_row("Services", ", ".join(services))

    return Panel(table, title="📋 Run Status", border_style="yellow")


def main():
    console.clear()
    print("Starting Dashboard... Press Ctrl+C to exit")

    with Live(refresh_per_second=1) as live:
        try:
            while True:
                telem = load_telemetry()

                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="main"),
                )
                layout["main"].split_row(
                    Layout(name="left"),
                    Layout(name="right"),
                )
                layout["left"].split_column(
                    Layout(name="status"),
                    Layout(name="resources"),
                )
                layout["right"].split_column(
                    Layout(name="usage"),
                    Layout(name="progress"),
                )

                layout["header"].update(create_header(telem))
                layout["left"]["status"].update(create_status_panel(telem))
                layout["left"]["resources"].update(create_resources_panel())
                layout["right"]["usage"].update(create_usage_panel(telem))
                layout["right"]["progress"].update(create_progress_panel(telem))

                live.update(layout)
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped[/yellow]")


if __name__ == '__main__':
    main()
