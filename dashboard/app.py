#!/usr/bin/env python3
"""Rich Dashboard for QLoRA Training with Live Monitoring."""

import os
import sys
import json
import time
import psutil
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List
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

# Constants
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(ROOT, 'logs')
DATA_DIR = os.path.join(ROOT, 'data/ai-lab')
CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints')
STATE_DIR = os.path.join(ROOT, 'state')
TELEMETRY_FILE = os.path.join(STATE_DIR, 'telemetry.json')
ENV_FILE = os.path.join(ROOT, '.env')

console = Console()

# Load environment variables
def load_env():
    """Load environment variables from .env file."""
    env_vars = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

# Global environment variables
ENV = load_env()

DEFAULT_LOG_SOURCE = (ENV.get('DASHBOARD_LOG_SOURCE', 'loop') or 'loop').strip().lower()

class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.max_history = 60
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
            
        return stats
    
    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                 '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization': float(gpu_util),
                    'gpu_memory_used': float(mem_used) / 1024,  # MB to GB
                    'gpu_memory_total': float(mem_total) / 1024,  # MB to GB
                }
        except Exception:
            pass
        return {
            'gpu_utilization': 0,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
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
        self.training_speed = 0
        self.last_step_time = None
        
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        # Check telemetry first
        telemetry_status = self._load_from_telemetry()
        if telemetry_status:
            # Telemetry currently doesn't carry loss/LR; enrich from logs so the UI is live.
            self._update_from_logs()
            telemetry_status['training_loss'] = self.training_loss[-1] if self.training_loss else 0
            telemetry_status['learning_rate'] = self.learning_rate
            return telemetry_status
            
        # Fallback to log parsing
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
    
    def _load_from_telemetry(self) -> Dict[str, Any]:
        """Load training status from telemetry file."""
        if not os.path.exists(TELEMETRY_FILE):
            return None
            
        try:
            with open(TELEMETRY_FILE, 'r') as f:
                telemetry = json.load(f)
            
            current_stage = (telemetry.get('current_stage', '') or '').lower()
            status = (telemetry.get('status', 'idle') or 'idle').lower()
            is_training = status in ['running', 'training'] or ('train' in current_stage)

            return {
                'current_round': telemetry.get('stage_index', 0),
                'training_step': telemetry.get('completed_units', 0),
                'total_steps': telemetry.get('total_units', 500),
                'progress_percent': telemetry.get('overall_percent', 0) * 100,
                'training_loss': 0,  # Not in telemetry
                'learning_rate': 2e-4,  # Default
                'start_time': None,
                'estimated_completion': None,
                'is_training': is_training,
                'total_rounds': telemetry.get('total_stages', 10),
                'round_progress': telemetry.get('stage_percent', 0) * 100,
                'training_speed': 0,
                'samples_processed': telemetry.get('completed_units', 0),
                'total_samples': telemetry.get('total_units', 100),
                'current_stage': current_stage,
                'status': status,
            }
        except Exception:
            return None
    
    def _is_training_active(self) -> bool:
        """Check if training is currently active."""
        try:
            # Check for train_loop.py process
            result = subprocess.run(['pgrep', '-f', 'train_loop.py'], 
                                capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return True
                
            # Also check for train_q_lora.py process
            result = subprocess.run(['pgrep', '-f', 'train_q_lora.py'], 
                                capture_output=True, text=True)
            return result.returncode == 0 and result.stdout.strip()
        except Exception:
            return False
    
    def _update_from_logs(self):
        """Update training status from logs."""
        log_file = os.path.join(LOGS_DIR, 'loop.log')
        if not os.path.exists(log_file):
            return
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
            
            for line in lines:
                # Parse training progress
                if 'Round' in line and '/' in line:
                    round_match = re.search(r'Round (\d+)/(\d+)', line)
                    if round_match:
                        self.current_round = int(round_match.group(1))
                        self.total_rounds = int(round_match.group(2))
                
                # Parse step progress (older format)
                if 'Step' in line and '/' in line:
                    step_match = re.search(r'Step (\d+)/(\d+)', line)
                    if step_match:
                        self.training_step = int(step_match.group(1))
                        self.total_steps = int(step_match.group(2))
                
                # Parse loss (new format from training logs)
                if "'loss':" in line:
                    loss_match = re.search(r"'loss':\s*'([0-9.]+)'", line)
                    if loss_match:
                        loss_value = float(loss_match.group(1))
                        self.training_loss.append(loss_value)
                        if len(self.training_loss) > 100:
                            self.training_loss.pop(0)
                
                # Parse step progress (new format)
                if "|" in line and "/" in line and "%" in line:
                    step_match = re.search(r'\|\s*([0-9]+)/([0-9]+)\s*\|', line)
                    if step_match:
                        self.training_step = int(step_match.group(1))
                        self.total_steps = int(step_match.group(2))
                
                # Parse learning rate (new format)
                if "'learning_rate':" in line:
                    lr_match = re.search(r"'learning_rate':\s*'([0-9.e-]+)'", line)
                    if lr_match:
                        self.learning_rate = float(lr_match.group(1))
        
        except Exception:
            pass

class LoRAMonitor:
    """Monitor LoRA adapter information."""
    
    def __init__(self):
        self.lora_config = {}
        self.adapter_size = 0
        self.trainable_params = 0
        self.total_params = 0
        
    def get_lora_info(self) -> Dict[str, Any]:
        """Get QLoRA adapter information."""
        # Read from environment first
        try:
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    env_content = f.read()
                    
                # Parse QLoRA config from env
                r_match = re.search(r'QLORA_R=([0-9]+)', env_content)
                alpha_match = re.search(r'QLORA_ALPHA=([0-9]+)', env_content)
                dropout_match = re.search(r'QLORA_DROPOUT=([0-9.]+)', env_content)
                
                if r_match:
                    self.lora_config['r'] = int(r_match.group(1))
                if alpha_match:
                    self.lora_config['alpha'] = int(alpha_match.group(1))
                if dropout_match:
                    self.lora_config['dropout'] = float(dropout_match.group(1))
        except Exception:
            pass
        
        # Try to read from latest checkpoint
        checkpoints = list(Path(CHECKPOINTS_DIR).glob('adapter_round_*'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            self._load_lora_info(latest_checkpoint)
        
        # Fallback to defaults if not found
        if not self.lora_config:
            self.lora_config = {'r': 64, 'alpha': 16, 'dropout': 0.1}
        
        return {
            'lora_r': self.lora_config.get('r', 64),
            'lora_alpha': self.lora_config.get('alpha', 16),
            'lora_dropout': self.lora_config.get('dropout', 0.1),
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
    """Monitor teacher data generation progress and cost."""
    
    def __init__(self):
        self.total_samples = 0
        self.processed_samples = 0
        self.generation_stage = 'idle'
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self.total_cost = 0.0
        
    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for the specified model."""
        pricing = {
            'grok-4-1-fast': {
                'input_price': float(ENV.get('GROK_4_1_FAST_INPUT_PRICE', '0.20')),
                'output_price': float(ENV.get('GROK_4_1_FAST_OUTPUT_PRICE', '0.50'))
            },
            'gpt-4': {
                'input_price': float(ENV.get('GPT_4_INPUT_PRICE', '0.03')),
                'output_price': float(ENV.get('GPT_4_OUTPUT_PRICE', '0.06'))
            },
            'gpt-3.5-turbo': {
                'input_price': float(ENV.get('GPT_3_5_TURBO_INPUT_PRICE', '0.0015')),
                'output_price': float(ENV.get('GPT_3_5_TURBO_OUTPUT_PRICE', '0.002'))
            },
            'gpt-5-mini': {
                'input_price': float(ENV.get('GPT_5_MINI_INPUT_PRICE', '0.015')),
                'output_price': float(ENV.get('GPT_5_MINI_OUTPUT_PRICE', '0.075'))
            },
            'claude-3-sonnet': {
                'input_price': float(ENV.get('CLAUDE_3_SONNET_INPUT_PRICE', '0.015')),
                'output_price': float(ENV.get('CLAUDE_3_SONNET_OUTPUT_PRICE', '0.075'))
            },
            'copilot': {
                'input_price': float(ENV.get('COPILOT_INPUT_PRICE', '0.015')),
                'output_price': float(ENV.get('COPILOT_OUTPUT_PRICE', '0.075'))
            },
            'github-copilot': {
                'input_price': float(ENV.get('COPILOT_INPUT_PRICE', '0.015')),
                'output_price': float(ENV.get('COPILOT_OUTPUT_PRICE', '0.075'))
            }
        }
        return pricing.get(model, pricing['copilot'])  # Default to copilot pricing
    
    def _format_provider(self, provider: str) -> str:
        """Format provider name for display."""
        provider = provider.lower().strip()
        if provider in ['copilot', 'github-copilot']:
            return 'GitHub Copilot'
        elif provider == 'openai':
            return 'OpenAI'
        elif provider == 'anthropic':
            return 'Anthropic'
        elif provider == 'xai':
            return 'xAI'
        else:
            return provider.title() or 'Unknown'
        
    def get_generation_status(self) -> Dict[str, Any]:
        """Get data generation status."""
        # Check telemetry first
        telemetry_status = self._load_from_telemetry()
        if telemetry_status:
            return telemetry_status
            
        self._update_from_logs()
        
        return {
            'total_samples': self.total_samples,
            'processed_samples': self.processed_samples,
            'progress_percent': (self.processed_samples / self.total_samples * 100) if self.total_samples > 0 else 0,
            'generation_stage': self.generation_stage,
            'samples_per_minute': self._calculate_generation_rate(),
            'provider': self._format_provider(ENV.get('TEACHER_PROVIDER', 'copilot')),
            'model': ENV.get('TEACHER_MODEL', 'gpt-5-mini'),
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'request_count': self.request_count,
            'total_cost': self.total_cost,
            'input_price': self.get_pricing(ENV.get('TEACHER_MODEL', 'gpt-5-mini'))['input_price'],
            'output_price': self.get_pricing(ENV.get('TEACHER_MODEL', 'gpt-5-mini'))['output_price']
        }
    
    def _load_from_telemetry(self) -> Dict[str, Any]:
        """Load generation status from telemetry."""
        if not os.path.exists(TELEMETRY_FILE):
            return None
            
        try:
            with open(TELEMETRY_FILE, 'r') as f:
                telemetry = json.load(f)
            
            usage = telemetry.get('usage', {})

            model = usage.get('model') or ENV.get('TEACHER_MODEL', 'gpt-5-mini')
            pricing = self.get_pricing(model)
            prompt_tokens = int(usage.get('prompt_tokens', 0) or 0)
            completion_tokens = int(usage.get('completion_tokens', 0) or 0)
            total_tokens = int(usage.get('total_tokens', prompt_tokens + completion_tokens) or 0)
            request_count = int(usage.get('request_count', 0) or 0)
            spend_usd = float(usage.get('spend_usd', 0.0) or 0.0)

            # If telemetry doesn't have spend, compute it from env pricing.
            # Pricing values are per 1K tokens.
            if spend_usd <= 0.0 and (prompt_tokens > 0 or completion_tokens > 0):
                spend_usd = (prompt_tokens / 1000.0) * pricing['input_price'] + (completion_tokens / 1000.0) * pricing['output_price']

            return {
                'total_samples': telemetry.get('total_units', 100),
                'processed_samples': telemetry.get('completed_units', 0),
                'progress_percent': telemetry.get('overall_percent', 0) * 100,
                'generation_stage': telemetry.get('current_stage', 'idle'),
                'samples_per_minute': 0,
                'provider': self._format_provider(usage.get('provider', ENV.get('TEACHER_PROVIDER', 'copilot'))),
                'model': model,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'request_count': request_count,
                'input_price': pricing['input_price'],
                'output_price': pricing['output_price'],
                'total_cost': spend_usd
            }
        except Exception:
            return None
    
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
                elif 'tokens' in line.lower() and ('prompt' in line.lower() or 'completion' in line.lower()):
                    # Extract token usage from logs
                    if 'prompt tokens:' in line.lower():
                        token_match = re.search(r'prompt tokens[:\s]+([\d,]+)', line)
                        if token_match:
                            self.prompt_tokens = int(token_match.group(1).replace(',', ''))
                    elif 'completion tokens:' in line.lower():
                        token_match = re.search(r'completion tokens[:\s]+([\d,]+)', line)
                        if token_match:
                            self.completion_tokens = int(token_match.group(1).replace(',', ''))
                    elif 'total tokens:' in line.lower():
                        token_match = re.search(r'total tokens[:\s]+([\d,]+)', line)
                        if token_match:
                            self.total_tokens = int(token_match.group(1).replace(',', ''))
                elif 'API request' in line or 'requests:' in line.lower():
                    # Count requests
                    req_match = re.search(r'(\d+)\s+request', line.lower())
                    if req_match:
                        self.request_count = int(req_match.group(1))
                elif 'samples to' in line and 'generate' in line:
                    # Try to get target from config
                    config_file = os.path.join(ROOT, 'config.json')
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            self.total_samples = config.get('samples_per_round', 100)
            
            # Calculate cost based on model and tokens
            teacher_model = ENV.get('TEACHER_MODEL', 'grok-4-1-fast')
            pricing = self.get_pricing(teacher_model)
            
            if self.prompt_tokens > 0 or self.completion_tokens > 0:
                input_cost = (self.prompt_tokens / 1000) * pricing['input_price']
                output_cost = (self.completion_tokens / 1000) * pricing['output_price']
                self.total_cost = input_cost + output_cost
        
        except Exception:
            pass
    
    def _calculate_generation_rate(self) -> float:
        """Calculate samples per minute generation rate."""
        return 0.0

class LogMonitor:
    """Monitor and tail log files."""
    
    def __init__(self):
        self.log_lines = []
        self.max_lines = 20

    def _tail_lines(self, path: str, lines: int = 20, max_bytes: int = 65536) -> List[str]:
        """Read last N lines from a potentially large file efficiently."""
        try:
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                read_size = min(max_bytes, end)
                f.seek(end - read_size)
                data = f.read(read_size)
        except Exception:
            return [f"Error reading log file: {path}"]

        try:
            text = data.decode('utf-8', errors='replace')
        except Exception:
            text = str(data)

        # If we didn't read from the beginning, drop a partial first line.
        if read_size < end:
            nl = text.find('\n')
            if nl != -1:
                text = text[nl + 1:]

        out = [line.rstrip('\n') for line in text.splitlines() if line is not None]
        if not out:
            return ["(log is empty)"]
        return out[-lines:]
        
    def get_recent_logs(self, process_name: str = None, lines: int = 20) -> List[str]:
        """Get recent log lines."""
        log_files = {
            'loop': os.path.join(LOGS_DIR, 'loop.log'),
            'api': os.path.join(LOGS_DIR, 'api.log'),
            'mlflow': os.path.join(LOGS_DIR, 'mlflow.log'),
            'tb': os.path.join(LOGS_DIR, 'tb.log'),
            'dashboard': os.path.join(LOGS_DIR, 'dashboard.log')
        }
        
        target_log = log_files.get(process_name, log_files['loop'])
        if not os.path.exists(target_log):
            return [f"Log file not found: {target_log}"]
        
        return [line.strip() for line in self._tail_lines(target_log, lines=lines)]

class Dashboard:
    """Main dashboard application."""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.training_monitor = TrainingMonitor()
        self.lora_monitor = LoRAMonitor()
        self.data_monitor = DataGenerationMonitor()
        self.log_monitor = LogMonitor()
        self.running = True
        self.refresh_interval = 1.0
        self.last_update = 0
        self.update_cache = {}
        self.cache_ttl = 0.5
        
    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=24)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
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

        # Full-width realtime logs split into 2 panels
        layout["footer"].split_row(
            Layout(name="training_logs"),
            Layout(name="other_logs"),
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """Create header panel."""
        header_text = "[bold cyan]🚀 QLoRA Training Pipeline Dashboard[/bold cyan]"
        return Panel(
            Align.center(header_text),
            border_style="blue",
            box=box.ROUNDED
        )
    
    def create_resources_panel(self) -> Panel:
        """Create resources panel."""
        resources = self.resource_monitor.get_current_stats()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("CPU Usage", f"{resources['cpu_percent']:.1f}%")
        table.add_row("Memory", f"{resources['memory_used_gb']:.1f}/{resources['memory_total_gb']:.1f}GB ({resources['memory_percent']:.1f}%)")
        table.add_row("Disk", f"{resources['disk_usage']:.1f}%")
        table.add_row("Uptime", f"{resources['uptime_hours']:.1f}h")
        
        if resources.get('gpu_utilization', 0) > 0:
            table.add_row("GPU", f"{resources['gpu_utilization']:.1f}%")
            table.add_row("GPU Memory", f"{resources['gpu_memory_used']:.1f}/{resources['gpu_memory_total']:.1f}GB")
        
        return Panel(table, title="🖥️ Resources", border_style="green")
    
    def create_training_panel(self) -> Panel:
        """Create training panel."""
        training = self.training_monitor.get_training_status()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        status_text = "[dim]⚪ Idle[/dim]"
        if training.get('is_training'):
            status_text = "[green]🟢 Training[/green]"
        table.add_row("Status", status_text)
        table.add_row("Round", f"{training['current_round']}/{training['total_rounds']}")
        table.add_row("Step", f"{training['training_step']}/{training['total_steps']}")
        table.add_row("Progress", f"{training['progress_percent']:.1f}%")
        
        # Add countdown timer if training is active
        if training.get('is_training') and training.get('training_speed', 0) > 0:
            remaining_steps = training['total_steps'] - training['training_step']
            if remaining_steps > 0:
                eta_seconds = remaining_steps / training['training_speed']
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                
                if eta_hours >= 1:
                    eta_text = f"{eta_hours:.1f}h"
                elif eta_minutes >= 1:
                    eta_text = f"{eta_minutes:.1f}m"
                else:
                    eta_text = f"{eta_seconds:.0f}s"
                
                table.add_row("ETA", eta_text)
        
        table.add_row("Loss", f"{float(training.get('training_loss', 0) or 0):.4f}")
        table.add_row("Learning Rate", f"{float(training.get('learning_rate', 0) or 0):.2e}")
        
        if training['training_speed'] > 0:
            table.add_row("Speed", f"{training['training_speed']:.2f} steps/s")
        
        return Panel(table, title="🎯 Training", border_style="yellow")
    
    def create_lora_panel(self) -> Panel:
        """Create LoRA panel."""
        lora = self.lora_monitor.get_lora_info()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Rank", str(lora['lora_r']))
        table.add_row("Alpha", str(lora['lora_alpha']))
        table.add_row("Dropout", f"{lora['lora_dropout']:.1f}")
        table.add_row("Size", f"{lora['adapter_size_mb']:.1f}MB")
        table.add_row("Trainable", f"{lora['trainable_percent']:.1f}%")
        
        return Panel(table, title="🔧 QLoRA", border_style="magenta")
    
    def create_data_gen_panel(self) -> Panel:
        """Create data generation panel."""
        data = self.data_monitor.get_generation_status()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Stage", data['generation_stage'].title())
        table.add_row("Samples", f"{data['processed_samples']}/{data['total_samples']}")
        table.add_row("Progress", f"{data['progress_percent']:.1f}%")
        
        # Token usage and cost information
        table.add_row("Provider", data.get('provider', 'N/A'))
        table.add_row("Model", data.get('model', 'N/A'))
        table.add_row("Requests", f"{data.get('request_count', 0):,}")
        table.add_row("Prompt Tokens", f"{data.get('prompt_tokens', 0):,}")
        table.add_row("Completion Tokens", f"{data.get('completion_tokens', 0):,}")
        table.add_row("Total Tokens", f"{data.get('total_tokens', 0):,}")
        
        # Pricing information
        if data.get('input_price') and data.get('output_price'):
            table.add_row("Input Price", f"${data.get('input_price', 0):.2f}/1K tokens")
            table.add_row("Output Price", f"${data.get('output_price', 0):.2f}/1K tokens")
        
        table.add_row("Cost", f"${data.get('total_cost', 0):.4f}")
        
        return Panel(table, title="📊 Data Gen & Cost", border_style="blue")
    
    def create_progress_panel(self) -> Panel:
        """Create progress panel."""
        training = self.training_monitor.get_training_status()
        data = self.data_monitor.get_generation_status()
        
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        
        # Training progress
        if training['total_steps'] > 0:
            training_task = progress.add_task(
                "Training",
                total=training['total_steps'],
                completed=training['training_step']
            )
        
        # Data generation progress
        if data['total_samples'] > 0:
            data_task = progress.add_task(
                "Data Generation",
                total=data['total_samples'],
                completed=data['processed_samples']
            )
        
        return Panel(progress, title="📈 Overall Progress", border_style="cyan")
    
    def create_logs_panel(self, log_source: str = 'loop', title: str = '📋 Real-time Logs', filter_mode: str = '') -> Panel:
        """Create real-time logs panel."""
        logs = self.log_monitor.get_recent_logs(log_source, lines=25)

        if filter_mode == 'training':
            filtered = []
            for line in logs:
                u = line.upper()
                if ('%|' in line) or ("'LOSS':" in u) or ("'LEARNING_RATE':" in u) or ('ROUND' in u) or ('STEP' in u):
                    filtered.append(line)
            logs = filtered[-25:] if filtered else logs[-25:]
        
        # Format logs with colors based on content
        formatted_logs = []
        for log in logs[-25:]:
            if 'ERROR' in log.upper():
                formatted_logs.append(f"[red]{log}[/red]")
            elif 'WARNING' in log.upper():
                formatted_logs.append(f"[yellow]{log}[/yellow]")
            elif 'INFO' in log.upper():
                formatted_logs.append(f"[white]{log}[/white]")
            else:
                formatted_logs.append(f"[dim]{log}[/dim]")
        
        log_text = "\n".join(formatted_logs)
        return Panel(log_text, title=title, border_style="white")
    
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
                'logs': self.log_monitor.get_recent_logs('loop')[-30:]
            }
            self.last_update = current_time
        
        return self.update_cache
    
    def render(self):
        """Render dashboard."""
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

        # Two pinned log sources (no auto-cycling)
        layout["footer"]["training_logs"].update(self.create_logs_panel('loop', '🎯 Training % / Progress', filter_mode='training'))
        layout["footer"]["other_logs"].update(self.create_logs_panel('api', '📋 Other Logs (API)'))
        
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
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
