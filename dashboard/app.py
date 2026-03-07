#!/usr/bin/env python3
"""Rich Dashboard UI for QLoRA Training Pipeline.

Real-time monitoring dashboard showing:
1. Resource Utilization
2. Training Convergence Metrics  
3. LoRA Specifics
4. Progress % and Time Estimates
5. Teacher Data Generation Progress
6. Real-time Log Tailing
"""
import os
import json
import time
import threading
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not available. Install with: pip install rich")
    exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')
LOGS_DIR = os.path.join(ROOT, 'logs')
STATE_DIR = os.path.join(ROOT, 'state')
CHECKPOINTS_DIR = os.path.join(AI_LAB, 'checkpoints')
EVAL_DIR = os.path.join(AI_LAB, 'evaluation')

console = Console()

class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.max_history = 60  # Keep 60 data points
        self.start_time = time.time()
        
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_usage': psutil.disk_usage(ROOT).percent,
            'uptime_hours': (time.time() - self.start_time) / 3600,
        }
        
        # GPU stats if available
        try:
            gpu_stats = self._get_gpu_stats()
            stats.update(gpu_stats)
        except Exception:
            stats['gpu_utilization'] = 0
            stats['gpu_memory_used'] = 0
            stats['gpu_memory_total'] = 0
        
        # Network stats
        try:
            net_io = psutil.net_io_counters()
            stats['network_sent_mb'] = net_io.bytes_sent / (1024**2)
            stats['network_recv_mb'] = net_io.bytes_recv / (1024**2)
        except Exception:
            stats['network_sent_mb'] = 0
            stats['network_recv_mb'] = 0
        
        # Process count
        try:
            stats['process_count'] = len(psutil.pids())
        except Exception:
            stats['process_count'] = 0
        
        return stats
    
    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics using nvidia-smi."""
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
                        'gpu_utilization': int(gpu_util),
                        'gpu_memory_used': int(mem_used) / 1024,  # MB to GB
                        'gpu_memory_total': int(mem_total) / 1024,  # MB to GB
                    }
        except Exception:
            pass
        
        return {
            'gpu_utilization': 0,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0
        }

class TrainingMonitor:
    """Monitor training progress and metrics."""
    
    def __init__(self):
        self.current_round = 0
        self.training_step = 0
        self.total_steps = 0
        self.training_loss = []
        self.learning_rate = 0
        self.start_time = None
        self.estimated_completion = None
        self.total_rounds = 0
        self.samples_processed = 0
        self.total_samples = 0
        self.training_speed = 0  # steps per second
        self.last_step_time = None
        
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        # Read from MLflow or training logs
        status = {
            'current_round': self.current_round,
            'training_step': self.training_step,
            'total_steps': self.total_steps,
            'progress_percent': 0,
            'training_loss': self.training_loss[-1] if self.training_loss else 0,
            'learning_rate': self.learning_rate,
            'start_time': self.start_time,
            'estimated_completion': self.estimated_completion,
            'is_training': False,
            'total_rounds': self.total_rounds,
            'round_progress': 0,
            'training_speed': self.training_speed,
            'samples_processed': self.samples_processed,
            'total_samples': self.total_samples
        }
        
        # Check if training is active
        if self._is_training_active():
            status['is_training'] = True
            self._update_from_logs()
            
            # Calculate progress
            if status['total_steps'] > 0:
                status['progress_percent'] = (status['training_step'] / status['total_steps']) * 100
                
                # Calculate round progress
                if status['total_rounds'] > 0:
                    status['round_progress'] = (status['current_round'] / status['total_rounds']) * 100
                
                # Estimate completion time
                if status['start_time'] and status['progress_percent'] > 0:
                    elapsed = time.time() - status['start_time']
                    estimated_total = elapsed / (status['progress_percent'] / 100)
                    status['estimated_completion'] = status['start_time'] + estimated_total
                    
                    # Calculate training speed
                    if status['training_step'] > 0 and self.last_step_time:
                        status['training_speed'] = 1.0 / (time.time() - self.last_step_time)
                    self.last_step_time = time.time()
        
        return status
    
    def _is_training_active(self) -> bool:
        """Check if training is currently active."""
        # Check for training process
        try:
            result = subprocess.run(['pgrep', '-f', 'train_q_lora.py'], 
                                capture_output=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _update_from_logs(self):
        """Update training metrics from log files."""
        log_file = os.path.join(LOGS_DIR, 'loop.log')
        if not os.path.exists(log_file):
            return
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
            
            for line in lines:
                # Parse training step
                if 'step' in line.lower():
                    step_match = re.search(r'step[:\s]+(\d+)', line)
                    if step_match:
                        self.training_step = int(step_match.group(1))
                
                # Parse loss
                if 'loss' in line.lower():
                    loss_match = re.search(r'loss[:\s]+([\d.]+)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        if not self.training_loss or loss != self.training_loss[-1]:
                            self.training_loss.append(loss)
                            if len(self.training_loss) > 100:
                                self.training_loss = self.training_loss[-100:]
                
                # Parse learning rate
                if 'learning_rate' in line.lower() or 'lr' in line.lower():
                    lr_match = re.search(r'([0-9e.-]+)', line)
                    if lr_match:
                        try:
                            self.learning_rate = float(lr_match.group(1))
                        except ValueError:
                            pass
                
                # Parse round
                if 'round' in line.lower():
                    round_match = re.search(r'round[:\s]+(\d+)', line)
                    if round_match:
                        self.current_round = int(round_match.group(1))
        
        except Exception:
            pass

class LoRAMonitor:
    """Monitor LoRA-specific metrics."""
    
    def __init__(self):
        self.lora_config = {}
        self.adapter_size = 0
        self.trainable_params = 0
        self.total_params = 0
        
    def get_lora_info(self) -> Dict[str, Any]:
        """Get LoRA configuration and metrics."""
        # Try to read from latest checkpoint
        checkpoints = list(Path(CHECKPOINTS_DIR).glob('adapter_round_*'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            self._load_lora_info(latest_checkpoint)
        
        return {
            'lora_r': self.lora_config.get('r', 0),
            'lora_alpha': self.lora_config.get('alpha', 0),
            'lora_dropout': self.lora_config.get('dropout', 0),
            'adapter_size_mb': self.adapter_size,
            'trainable_params': self.trainable_params,
            'total_params': self.total_params,
            'trainable_percent': (self.trainable_params / self.total_params * 100) if self.total_params > 0 else 0
        }
    
    def _load_lora_info(self, checkpoint_path: Path):
        """Load LoRA information from checkpoint."""
        try:
            # Read metadata
            metadata_file = checkpoint_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.lora_config = metadata
            
            # Calculate adapter size
            adapter_files = list(checkpoint_path.glob('*.safetensors')) + list(checkpoint_path.glob('*.bin'))
            self.adapter_size = sum(f.stat().st_size for f in adapter_files) / (1024 * 1024)  # MB
            
            # Try to get parameter counts from training logs
            log_file = os.path.join(LOGS_DIR, 'loop.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read()
                    # Look for parameter count in logs
                    param_match = re.search(r'trainable params[:\s]+([\d,]+)', content)
                    if param_match:
                        self.trainable_params = int(param_match.group(1).replace(',', ''))
                    
                    total_match = re.search(r'total params[:\s]+([\d,]+)', content)
                    if total_match:
                        self.total_params = int(total_match.group(1).replace(',', ''))
        
        except Exception:
            pass

class DataGenerationMonitor:
    """Monitor teacher data generation progress."""
    
    def __init__(self):
        self.total_samples = 0
        self.processed_samples = 0
        self.generation_stage = 'idle'
        
    def get_generation_status(self) -> Dict[str, Any]:
        """Get data generation status."""
        self._update_from_logs()
        
        return {
            'total_samples': self.total_samples,
            'processed_samples': self.processed_samples,
            'progress_percent': (self.processed_samples / self.total_samples * 100) if self.total_samples > 0 else 0,
            'generation_stage': self.generation_stage,
            'samples_per_minute': self._calculate_generation_rate()
        }
    
    def _update_from_logs(self):
        """Update generation status from logs."""
        log_file = os.path.join(LOGS_DIR, 'loop.log')
        if not os.path.exists(log_file):
            return
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-50:]  # Last 50 lines
            
            for line in lines:
                if 'Stage: generate' in line:
                    self.generation_stage = 'generating'
                elif 'Generated' in line and 'samples' in line:
                    sample_match = re.search(r'Generated[:\s]+(\d+)', line)
                    if sample_match:
                        self.processed_samples = int(sample_match.group(1))
                elif 'samples to' in line and 'generate' in line:
                    # Try to get target from config
                    config_file = os.path.join(ROOT, 'config.json')
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            self.total_samples = config.get('samples_per_round', 100)
        
        except Exception:
            pass
    
    def _calculate_generation_rate(self) -> float:
        """Calculate samples per minute generation rate."""
        # This would need timestamp tracking for accurate calculation
        return 0.0

class LogMonitor:
    """Monitor and tail log files."""
    
    def __init__(self):
        self.log_lines = []
        self.max_lines = 20
        
    def get_recent_logs(self, process_name: str = None) -> List[str]:
        """Get recent log lines."""
        log_files = {
            'loop': os.path.join(LOGS_DIR, 'loop.log'),
            'api': os.path.join(LOGS_DIR, 'api.log'),
            'mlflow': os.path.join(LOGS_DIR, 'mlflow.log')
        }
        
        target_log = log_files.get(process_name, log_files['loop'])
        if not os.path.exists(target_log):
            return [f"Log file not found: {target_log}"]
        
        try:
            with open(target_log, 'r') as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-self.max_lines:]]
        except Exception:
            return [f"Error reading log file: {target_log}"]

class Dashboard:
    """Main dashboard application."""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.training_monitor = TrainingMonitor()
        self.lora_monitor = LoRAMonitor()
        self.data_monitor = DataGenerationMonitor()
        self.log_monitor = LogMonitor()
        self.running = True
        self.refresh_interval = 1.0  # seconds
        self.last_update = 0
        self.update_cache = {}
        self.cache_ttl = 0.5  # seconds
        
    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=10)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        layout["left"].split_column(
            Layout(name="resources", size=12),
            Layout(name="training", size=12)
        )
        
        layout["right"].split_column(
            Layout(name="lora", size=8),
            Layout(name="data_gen", size=8),
            Layout(name="progress", size=8)
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """Create header panel."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = f"[bold blue]QLoRA Training Dashboard[/bold blue] - {now}"
        return Panel(header_text, style="bold white on blue")
    
    def create_resources_panel(self) -> Panel:
        """Create resource utilization panel."""
        stats = self.resource_monitor.get_current_stats()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("CPU Usage", f"{stats['cpu_percent']:.1f}%")
        table.add_row("Memory Usage", f"{stats['memory_percent']:.1f}% ({stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f} GB)")
        table.add_row("Disk Usage", f"{stats['disk_usage']:.1f}%")
        table.add_row("Uptime", f"{stats['uptime_hours']:.1f}h")
        table.add_row("Processes", str(stats['process_count']))
        
        if stats.get('gpu_utilization', 0) > 0:
            table.add_row("GPU Usage", f"{stats['gpu_utilization']:.1f}%")
            table.add_row("GPU Memory", f"{stats['gpu_memory_used']:.1f}/{stats['gpu_memory_total']:.1f} GB")
        
        if stats.get('network_sent_mb', 0) > 0 or stats.get('network_recv_mb', 0) > 0:
            table.add_row("Network", f"↑{stats['network_sent_mb']:.1f}MB ↓{stats['network_recv_mb']:.1f}MB")
        
        return Panel(table, title="📊 Resource Utilization", border_style="green")
    
    def create_training_panel(self) -> Panel:
        """Create training metrics panel."""
        status = self.training_monitor.get_training_status()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Current Round", f"{status['current_round']}/{status['total_rounds']}")
        table.add_row("Training Step", f"{status['training_step']}/{status['total_steps']}")
        
        # Progress bar
        progress_color = "green" if status['progress_percent'] > 50 else "yellow"
        progress_text = f"[{progress_color}]{status['progress_percent']:.1f}%[/{progress_color}]"
        table.add_row("Step Progress", progress_text)
        
        if status['total_rounds'] > 0:
            round_color = "green" if status['round_progress'] > 50 else "yellow"
            round_text = f"[{round_color}]{status['round_progress']:.1f}%[/{round_color}]"
            table.add_row("Round Progress", round_text)
        
        if status['training_loss'] > 0:
            table.add_row("Current Loss", f"{status['training_loss']:.4f}")
        
        if status['learning_rate'] > 0:
            table.add_row("Learning Rate", f"{status['learning_rate']:.2e}")
        
        if status['training_speed'] > 0:
            table.add_row("Training Speed", f"{status['training_speed']:.2f} steps/s")
        
        if status['estimated_completion']:
            eta = status['estimated_completion'] - time.time()
            if eta > 0:
                eta_str = str(timedelta(seconds=int(eta)))
                table.add_row("ETA", eta_str)
        
        if status['total_samples'] > 0:
            table.add_row("Samples", f"{status['samples_processed']:,}/{status['total_samples']:,}")
        
        status_text = "[green]🟢 Training[/green]" if status['is_training'] else "[red]🔴 Idle[/red]"
        table.add_row("Status", status_text)
        
        return Panel(table, title="🎯 Training Metrics", border_style="blue")
    
    def create_lora_panel(self) -> Panel:
        """Create LoRA specifics panel."""
        info = self.lora_monitor.get_lora_info()
        
        table = Table(show_header=False, box=None)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("LoRA Rank", str(info['lora_r']))
        table.add_row("LoRA Alpha", str(info['lora_alpha']))
        table.add_row("LoRA Dropout", f"{info['lora_dropout']:.2f}")
        table.add_row("Adapter Size", f"{info['adapter_size_mb']:.1f} MB")
        table.add_row("Trainable Params", f"{info['trainable_params']:,}")
        table.add_row("Total Params", f"{info['total_params']:,}")
        
        if info['trainable_percent'] > 0:
            table.add_row("Trainable %", f"{info['trainable_percent']:.2f}%")
        
        return Panel(table, title="🔧 LoRA Specifics", border_style="yellow")
    
    def create_data_gen_panel(self) -> Panel:
        """Create data generation progress panel."""
        status = self.data_monitor.get_generation_status()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Stage", status['generation_stage'].title())
        table.add_row("Samples", f"{status['processed_samples']}/{status['total_samples']}")
        
        if status['progress_percent'] > 0:
            progress_color = "green" if status['progress_percent'] > 50 else "yellow"
            progress_text = f"[{progress_color}]{status['progress_percent']:.1f}%[/{progress_color}]"
            table.add_row("Progress", progress_text)
        
        if status['samples_per_minute'] > 0:
            table.add_row("Rate", f"{status['samples_per_minute']:.1f} samples/min")
        
        return Panel(table, title="📝 Data Generation", border_style="magenta")
    
    def create_progress_panel(self) -> Panel:
        """Create overall progress panel."""
        training_status = self.training_monitor.get_training_status()
        data_status = self.data_monitor.get_generation_status()
        
        # Create progress bars
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        
        # Training progress
        if training_status['is_training'] and training_status['total_steps'] > 0:
            training_task = progress.add_task(
                "Training", 
                total=training_status['total_steps'],
                completed=training_status['training_step']
            )
        
        # Data generation progress
        if data_status['total_samples'] > 0:
            data_task = progress.add_task(
                "Data Generation",
                total=data_status['total_samples'],
                completed=data_status['processed_samples']
            )
        
        return Panel(progress, title="📈 Overall Progress", border_style="cyan")
    
    def create_logs_panel(self) -> Panel:
        """Create real-time logs panel."""
        logs = self.log_monitor.get_recent_logs('loop')
        
        # Format logs with colors based on content
        formatted_logs = []
        for log in logs[-15:]:  # Show last 15 lines
            if 'ERROR' in log.upper():
                formatted_logs.append(f"[red]{log}[/red]")
            elif 'WARNING' in log.upper():
                formatted_logs.append(f"[yellow]{log}[/yellow]")
            elif 'INFO' in log.upper():
                formatted_logs.append(f"[white]{log}[/white]")
            else:
                formatted_logs.append(f"[dim]{log}[/dim]")
        
        log_text = "\n".join(formatted_logs)
        return Panel(log_text, title="📋 Real-time Logs", border_style="white")
    
    def update(self):
        """Update all monitors with caching."""
        current_time = time.time()
        
        # Only update if cache expired
        if current_time - self.last_update > self.cache_ttl:
            self.update_cache = {
                'resources': self.resource_monitor.get_current_stats(),
                'training': self.training_monitor.get_training_status(),
                'lora': self.lora_monitor.get_lora_info(),
                'data_gen': self.data_monitor.get_generation_status(),
                'logs': self.log_monitor.get_recent_logs('loop')[-15:]  # Limit logs
            }
            self.last_update = current_time
        
        return self.update_cache
    
    def render(self):
        """Render the dashboard."""
        layout = self.create_layout()
        
        # Get cached data
        data = self.update()
        
        # Populate layout with cached data
        layout["header"].update(self.create_header())
        layout["left"]["resources"].update(self.create_resources_panel())
        layout["left"]["training"].update(self.create_training_panel())
        layout["right"]["lora"].update(self.create_lora_panel())
        layout["right"]["data_gen"].update(self.create_data_gen_panel())
        layout["right"]["progress"].update(self.create_progress_panel())
        layout["footer"].update(self.create_logs_panel())
        
        return layout
    
    def run(self):
        """Run the dashboard."""
        console.clear()
        
        try:
            with Live(self.render(), refresh_per_second=1/self.refresh_interval, screen=True) as live:
                while self.running:
                    live.update(self.render())
                    time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            self.running = False
            console.print("\n[yellow]Dashboard stopped by user[/yellow]")

def main():
    """Main entry point."""
    if not RICH_AVAILABLE:
        print("Rich library is required. Install with: pip install rich")
        return
    
    dashboard = Dashboard()
    
    # Check if we're in the right directory
    if not os.path.exists(ROOT):
        print(f"Error: Cannot find project root at {ROOT}")
        return
    
    console.print("[bold green]Starting QLoRA Training Dashboard...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    
    dashboard.run()

if __name__ == "__main__":
    main()
