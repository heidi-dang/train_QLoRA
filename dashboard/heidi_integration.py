#!/usr/bin/env python3
"""Heidi Engine Dashboard Integration for QLoRA Training Pipeline."""

import os
import sys
import json
import time
import signal
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add Heidi Engine to path
HEIDI_ENGINE_PATH = "/home/ubuntu/heidi-engine"
if HEIDI_ENGINE_PATH not in sys.path:
    sys.path.insert(0, HEIDI_ENGINE_PATH)

# Add current project to path
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich import box

# Import Heidi Engine components
try:
    from heidi_engine.dashboard import HeidiDashboard
    from heidi_engine.telemetry import TelemetryEmitter, get_state, get_events
    HEIDI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Heidi Engine not available: {e}")
    HEIDI_AVAILABLE = False

# Constants
LOGS_DIR = os.path.join(ROOT, 'logs')
DATA_DIR = os.path.join(ROOT, 'data/ai-lab')
CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints')
STATE_DIR = os.path.join(ROOT, 'state')
TELEMETRY_FILE = os.path.join(STATE_DIR, 'telemetry.json')

console = Console()

class QLoRAHeidiDashboard:
    """Integrated QLoRA Dashboard with Heidi Engine Strategy."""
    
    def __init__(self):
        self.running = True
        self.refresh_interval = 1.0
        self.last_update = 0
        self.update_cache = {}
        self.cache_ttl = 0.5
        
        # Heidi Engine integration
        self.heidi_dashboard = None
        if HEIDI_AVAILABLE:
            try:
                self.heidi_dashboard = HeidiDashboard()
            except Exception as e:
                print(f"Warning: Could not initialize Heidi Dashboard: {e}")
        
        # Fallback monitors
        self.qlora_monitors = self._init_qlora_monitors()
        
    def _init_qlora_monitors(self):
        """Initialize QLoRA-specific monitors as fallback."""
        from dashboard.app import ResourceMonitor, TrainingMonitor, LoRAMonitor, DataGenerationMonitor, LogMonitor
        return {
            'resource': ResourceMonitor(),
            'training': TrainingMonitor(),
            'lora': LoRAMonitor(),
            'data': DataGenerationMonitor(),
            'log': LogMonitor()
        }
    
    def get_heidi_state(self) -> Dict[str, Any]:
        """Get state from Heidi Engine telemetry."""
        if not HEIDI_AVAILABLE:
            return {}
        
        try:
            state = get_state()
            events = get_events(limit=50)
            
            return {
                'state': state,
                'events': events,
                'status': state.get('status', 'unknown'),
                'current_round': state.get('current_round', 0),
                'current_stage': state.get('current_stage', ''),
                'counters': state.get('counters', {}),
                'usage': state.get('usage', {}),
                'started_at': state.get('started_at'),
                'updated_at': state.get('updated_at')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_qlora_state(self) -> Dict[str, Any]:
        """Get state from QLoRA monitors."""
        return {
            'resources': self.qlora_monitors['resource'].get_current_stats(),
            'training': self.qlora_monitors['training'].get_training_status(),
            'lora': self.qlora_monitors['lora'].get_lora_info(),
            'data': self.qlora_monitors['data'].get_generation_status(),
            'logs': self.qlora_monitors['log'].get_recent_logs('loop')[-15:]
        }
    
    def create_layout(self) -> Layout:
        """Create integrated dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=10)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="heidi_status", size=8),
            Layout(name="resources", size=12),
            Layout(name="training", size=12)
        )
        
        layout["right"].split_column(
            Layout(name="pipeline", size=8),
            Layout(name="usage", size=8),
            Layout(name="progress", size=8)
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """Create header panel."""
        header_text = "[bold cyan]🚀 QLoRA + Heidi Engine Integrated Dashboard[/bold cyan]"
        if HEIDI_AVAILABLE:
            header_text += " [green]✓ Heidi Engine Connected[/green]"
        else:
            header_text += " [yellow]⚠ Heidi Engine Fallback[/yellow]"
        
        return Panel(
            Align.center(header_text),
            border_style="blue",
            box=box.ROUNDED
        )
    
    def create_heidi_status_panel(self) -> Panel:
        """Create Heidi Engine status panel."""
        if not HEIDI_AVAILABLE:
            return Panel(
                "[yellow]Heidi Engine not available\nUsing QLoRA fallback monitors[/yellow]",
                title="🔧 Heidi Engine Status",
                border_style="yellow"
            )
        
        heidi_state = self.get_heidi_state()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        if 'error' in heidi_state:
            table.add_row("Status", f"[red]{heidi_state['error']}[/red]")
        else:
            status = heidi_state.get('status', 'unknown')
            status_color = 'green' if status == 'running' else 'yellow'
            table.add_row("Status", f"[{status_color}]{status}[/{status_color}]")
            table.add_row("Current Stage", heidi_state.get('current_stage', 'N/A'))
            table.add_row("Round", str(heidi_state.get('current_round', 0)))
            
            # Show counters
            counters = heidi_state.get('counters', {})
            if counters:
                table.add_row("Generated", str(counters.get('generated', 0)))
                table.add_row("Validated", str(counters.get('validated', 0)))
                table.add_row("Trained", str(counters.get('trained', 0)))
        
        return Panel(table, title="🔧 Heidi Engine", border_style="magenta")
    
    def create_pipeline_panel(self) -> Panel:
        """Create pipeline progress panel."""
        heidi_state = self.get_heidi_state()
        
        table = Table(show_header=False, box=None)
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="white")
        
        if 'error' in heidi_state:
            table.add_row("Pipeline", f"[red]Error[/red]")
        else:
            current_stage = heidi_state.get('current_stage', '')
            stages = ['generate', 'validate', 'train', 'eval']
            
            for stage in stages:
                if stage == current_stage:
                    table.add_row(stage.title(), "[green]� Active[/green]")
                else:
                    table.add_row(stage.title(), "[dim]⚪ Pending[/dim]")
        
        return Panel(table, title="🔄 Pipeline", border_style="blue")
    
    def create_usage_panel(self) -> Panel:
        """Create usage/cost panel."""
        heidi_state = self.get_heidi_state()
        qlora_state = self.get_qlora_state()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Heidi Engine usage
        if 'error' not in heidi_state:
            usage = heidi_state.get('usage', {})
            if usage:
                table.add_row("Provider", usage.get('provider', 'N/A'))
                table.add_row("Model", usage.get('model', 'N/A'))
                table.add_row("Tokens", f"{usage.get('total_tokens', 0):,}")
                table.add_row("Cost", f"${usage.get('spend_usd', 0):.4f}")
                table.add_row("Requests", str(usage.get('request_count', 0)))
        
        # QLoRA fallback usage
        if not heidi_state.get('usage') and qlora_state.get('data'):
            data = qlora_state['data']
            if data.get('provider'):
                table.add_row("Provider", data.get('provider', 'N/A'))
                table.add_row("Model", data.get('model', 'N/A'))
                table.add_row("Tokens", f"{data.get('total_tokens', 0):,}")
                table.add_row("Cost", f"${data.get('spend_usd', 0):.4f}")
        
        return Panel(table, title="💰 Usage & Cost", border_style="green")
    
    def create_resources_panel(self) -> Panel:
        """Create resources panel."""
        qlora_state = self.get_qlora_state()
        resources = qlora_state.get('resources', {})
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("CPU Usage", f"{resources.get('cpu_percent', 0):.1f}%")
        table.add_row("Memory", f"{resources.get('memory_used_gb', 0):.1f}/{resources.get('memory_total_gb', 0):.1f}GB ({resources.get('memory_percent', 0):.1f}%)")
        table.add_row("Disk", f"{resources.get('disk_usage', 0):.1f}%")
        table.add_row("Uptime", f"{resources.get('uptime_hours', 0):.1f}h")
        
        if resources.get('gpu_utilization', 0) > 0:
            table.add_row("GPU", f"{resources.get('gpu_utilization', 0):.1f}%")
            table.add_row("GPU Memory", f"{resources.get('gpu_memory_used', 0):.1f}/{resources.get('gpu_memory_total', 0):.1f}GB")
        
        return Panel(table, title="🖥️ Resources", border_style="green")
    
    def create_training_panel(self) -> Panel:
        """Create training panel."""
        qlora_state = self.get_qlora_state()
        training = qlora_state.get('training', {})
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Status", "🟢 Training" if training.get('is_training') else "⚪ Idle")
        table.add_row("Round", f"{training.get('current_round', 0)}/{training.get('total_rounds', 0)}")
        table.add_row("Step", f"{training.get('training_step', 0)}/{training.get('total_steps', 0)}")
        table.add_row("Progress", f"{training.get('progress_percent', 0):.1f}%")
        table.add_row("Loss", f"{training.get('training_loss', 0):.4f}")
        table.add_row("Learning Rate", f"{training.get('learning_rate', 0):.2e}")
        
        if training.get('training_speed', 0) > 0:
            table.add_row("Speed", f"{training.get('training_speed', 0):.2f} steps/s")
        
        return Panel(table, title="🎯 Training", border_style="yellow")
    
    def create_progress_panel(self) -> Panel:
        """Create progress panel."""
        heidi_state = self.get_heidi_state()
        qlora_state = self.get_qlora_state()
        
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        
        # Heidi Engine progress
        if 'error' not in heidi_state:
            counters = heidi_state.get('counters', {})
            total_samples = counters.get('target_samples', 100)
            generated_samples = counters.get('generated', 0)
            
            if total_samples > 0:
                progress.add_task(
                    "Generation",
                    total=total_samples,
                    completed=generated_samples
                )
        
        # QLoRA fallback progress
        training = qlora_state.get('training', {})
        data = qlora_state.get('data', {})
        
        if training.get('total_steps', 0) > 0:
            progress.add_task(
                "Training",
                total=training.get('total_steps', 0),
                completed=training.get('training_step', 0)
            )
        
        if data.get('total_samples', 0) > 0:
            progress.add_task(
                "Data Generation",
                total=data.get('total_samples', 0),
                completed=data.get('processed_samples', 0)
            )
        
        return Panel(progress, title="📈 Overall Progress", border_style="cyan")
    
    def create_logs_panel(self) -> Panel:
        """Create logs panel."""
        heidi_state = self.get_heidi_state()
        qlora_state = self.get_qlora_state()
        
        # Use Heidi Engine events if available
        if 'error' not in heidi_state and heidi_state.get('events'):
            events = heidi_state['events'][-15:]  # Last 15 events
            
            formatted_logs = []
            for event in events:
                level = event.get('level', 'info').upper()
                message = event.get('message', '')[:80]  # Truncate long messages
                timestamp = event.get('ts', '')[-8:]  # Just time part
                
                if level == 'ERROR':
                    formatted_logs.append(f"[red]{timestamp} {message}[/red]")
                elif level == 'WARN':
                    formatted_logs.append(f"[yellow]{timestamp} {message}[/yellow]")
                elif level == 'SUCCESS':
                    formatted_logs.append(f"[green]{timestamp} {message}[/green]")
                else:
                    formatted_logs.append(f"[white]{timestamp} {message}[/white]")
            
            log_text = "\n".join(formatted_logs)
            return Panel(log_text, title="📋 Heidi Events", border_style="white")
        
        # Fallback to QLoRA logs
        logs = qlora_state.get('logs', [])
        formatted_logs = []
        for log in logs[-15:]:
            if 'ERROR' in log.upper():
                formatted_logs.append(f"[red]{log}[/red]")
            elif 'WARNING' in log.upper():
                formatted_logs.append(f"[yellow]{log}[/yellow]")
            elif 'INFO' in log.upper():
                formatted_logs.append(f"[white]{log}[/white]")
            else:
                formatted_logs.append(f"[dim]{log}[/dim]")
        
        log_text = "\n".join(formatted_logs)
        return Panel(log_text, title="📋 QLoRA Logs", border_style="white")
    
    def update(self):
        """Update all monitors with caching."""
        current_time = time.time()
        
        # Only update if cache expired
        if current_time - self.last_update > self.cache_ttl:
            self.update_cache = {
                'heidi': self.get_heidi_state(),
                'qlora': self.get_qlora_state()
            }
            self.last_update = current_time
        
        return self.update_cache
    
    def render(self):
        """Render integrated dashboard."""
        layout = self.create_layout()
        
        # Get cached data
        data = self.update()
        
        # Populate layout with cached data
        layout["header"].update(self.create_header())
        layout["left"]["heidi_status"].update(self.create_heidi_status_panel())
        layout["left"]["resources"].update(self.create_resources_panel())
        layout["left"]["training"].update(self.create_training_panel())
        layout["right"]["pipeline"].update(self.create_pipeline_panel())
        layout["right"]["usage"].update(self.create_usage_panel())
        layout["right"]["progress"].update(self.create_progress_panel())
        layout["footer"].update(self.create_logs_panel())
        
        return layout
    
    def run(self):
        """Run integrated dashboard."""
        console.clear()
        
        try:
            with Live(self.render(), refresh_per_second=1/self.refresh_interval, screen=True) as live:
                while self.running:
                    live.update(self.render())
                    time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            self.running = False
            console.print("\n[yellow]Integrated dashboard stopped by user[/yellow]")

def main():
    """Main entry point."""
    print("🚀 Starting QLoRA + Heidi Engine Integrated Dashboard")
    
    if HEIDI_AVAILABLE:
        print("✅ Heidi Engine connected - using advanced telemetry")
    else:
        print("⚠️  Heidi Engine not available - using QLoRA fallback")
    
    dashboard = QLoRAHeidiDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
