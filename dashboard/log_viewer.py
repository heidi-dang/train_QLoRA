#!/usr/bin/env python3
"""Interactive log viewer for QLoRA training dashboard."""
import os
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.table import Table

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import LOGS_DIR

class LogViewer:
    """Interactive log viewer with multiple log sources."""
    
    def __init__(self):
        self.console = Console()
        self.current_log = 'loop'
        self.lines = 20
        self.follow = False
        self.running = True
        
    def get_log_files(self):
        """Get available log files."""
        log_files = {}
        for file in os.listdir(LOGS_DIR):
            if file.endswith('.log'):
                name = file.replace('.log', '')
                log_files[name] = os.path.join(LOGS_DIR, file)
        return log_files
    
    def read_logs(self, log_file, lines=None):
        """Read log lines from file."""
        if not os.path.exists(log_file):
            return [f"Log file not found: {log_file}"]
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                if lines:
                    return [line.strip() for line in all_lines[-lines:]]
                return [line.strip() for line in all_lines]
        except Exception as e:
            return [f"Error reading log file: {e}"]
    
    def format_log_line(self, line):
        """Format log line with colors."""
        if 'ERROR' in line.upper():
            return f"[red]{line}[/red]"
        elif 'WARNING' in line.upper():
            return f"[yellow]{line}[/yellow]"
        elif 'INFO' in line.upper():
            return f"[white]{line}[/white]"
        elif 'DEBUG' in line.upper():
            return f"[blue]{line}[/blue]"
        else:
            return f"[dim]{line}[/dim]"
    
    def create_help_panel(self):
        """Create help panel."""
        help_text = """
[yellow]Keyboard Commands:[/yellow]
[white]1-5[/white]     - Switch log source
[white]↑/↓[/white]     - Scroll up/down
[white]+/-[/white]      - Show more/fewer lines
[white]f[/white]        - Toggle follow mode
[white]r[/white]        - Refresh logs
[white]q[/white]        - Quit
        """
        return Panel(help_text.strip(), title="📖 Help", border_style="blue")
    
    def create_status_panel(self):
        """Create status panel."""
        log_files = self.get_log_files()
        table = Table(show_header=False, box=None)
        table.add_column("Log Source", style="cyan")
        table.add_column("Status", style="green")
        
        for i, (name, path) in enumerate(log_files.items(), 1):
            status = "🟢 Active" if name == self.current_log else "⚪ Available"
            marker = "→" if name == self.current_log else " "
            table.add_row(f"{marker} {i}. {name}", status)
        
        return Panel(table, title="📊 Log Sources", border_style="green")
    
    def create_log_panel(self):
        """Create log display panel."""
        log_files = self.get_log_files()
        if self.current_log not in log_files:
            return Panel("Log source not found", title="📋 Logs", border_style="red")
        
        log_file = log_files[self.current_log]
        logs = self.read_logs(log_file, self.lines)
        
        # Format logs with colors
        formatted_logs = [self.format_log_line(line) for line in logs]
        log_text = "\n".join(formatted_logs)
        
        title = f"📋 Logs: {self.current_log} ({self.lines} lines)"
        if self.follow:
            title += " [ Following ]"
        
        return Panel(log_text, title=title, border_style="white")
    
    def create_layout(self):
        """Create the viewer layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="main")
        )
        
        layout["main"].split_row(
            Layout(name="logs", ratio=3),
            Layout(name="sidebar", size=30)
        )
        
        layout["sidebar"].split_column(
            Layout(name="status", size=12),
            Layout(name="help")
        )
        
        # Populate layout
        layout["header"].update(self.create_header_panel())
        layout["logs"].update(self.create_log_panel())
        layout["sidebar"]["status"].update(self.create_status_panel())
        layout["sidebar"]["help"].update(self.create_help_panel())
        
        return layout
    
    def create_header_panel(self):
        """Create header panel."""
        header_text = f"""
[bold cyan]QLoRA Training Log Viewer[/bold cyan]

Current Log: [yellow]{self.current_log}[/yellow] | 
Lines: [yellow]{self.lines}[/yellow] | 
Follow: [yellow]{'ON' if self.follow else 'OFF'}[/yellow] | 
Time: [yellow]{time.strftime('%H:%M:%S')}[/yellow]
        """
        return Panel(header_text.strip(), border_style="blue")
    
    def handle_input(self):
        """Handle keyboard input."""
        # Simple input handling - in a real implementation you'd use 
        # keyboard libraries like curses or prompt_toolkit
        pass
    
    def run(self):
        """Run the log viewer."""
        self.console.clear()
        
        try:
            with Live(self.create_layout(), refresh_per_second=2, screen=True) as live:
                while self.running:
                    # Update layout
                    live.update(self.create_layout())
                    time.sleep(0.5)
        except KeyboardInterrupt:
            self.running = False
            self.console.print("\n[yellow]Log viewer stopped by user[/yellow]")

def main():
    """Main entry point."""
    viewer = LogViewer()
    viewer.run()

if __name__ == "__main__":
    main()
