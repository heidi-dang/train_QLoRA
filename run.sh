#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")"; pwd -P)
STATE="$ROOT/state"
LOGS="$ROOT/logs"
DATA="$ROOT/data/ai-lab"
PIDS="$STATE/pids"
CONFIG="$ROOT/config.json"

mkdir -p "$STATE" "$LOGS" "$PIDS" "$DATA/datasets/clean" "$DATA/checkpoints" "$DATA/evaluation"

VENV="$ROOT/venv"

VENV_ACTIVATE="$VENV/bin/activate"
VENV_PYTHON="$VENV/bin/python"

# Always ensure virtual environment is properly set up
echo "🔧 Setting up Python environment..."

# Create virtual environment if it doesn't exist (or is incomplete)
if [ ! -f "$VENV_ACTIVATE" ] || [ ! -f "$VENV_PYTHON" ]; then
  echo "Creating virtual environment..."
  rm -rf "$VENV" 2>/dev/null || true
  python3 -m venv "$VENV"
fi

# Always activate and install dependencies
echo "Activating virtual environment and installing dependencies..."
source "$VENV_ACTIVATE"
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"

# Ensure logs flush promptly for real-time dashboard tailing
export PYTHONUNBUFFERED=1

# Load configuration if exists
if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

VENV_UVICORN="$VENV/bin/uvicorn"
VENV_DASHBOARD="$VENV/bin/python"

# Always activate environment for all commands
echo "🚀 Activating Python virtual environment..."
source "$VENV_ACTIVATE"

CMD=${1:-help}

case $CMD in
  setup)
    # Interactive setup
    echo "🛑 Stopping all running services before setup..."
    # Stop all running services first
    touch "$STATE/STOP" 2>/dev/null || true
    for pidfile in "$PIDS"/*.pid; do
      if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
          echo "Stopping process $pid ($(basename $pidfile .pid))"
          kill "$pid" 2>/dev/null || true
          sleep 1
          kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
      fi
    done
    
    # Kill ALL remaining processes (force kill to be sure)
    echo "🔥 Force-killing all remaining processes..."
    pkill -9 -f "train_q_lora.py" 2>/dev/null || true
    pkill -9 -f "train_loop.py" 2>/dev/null || true
    pkill -9 -f "generate_samples.py" 2>/dev/null || true
    pkill -9 -f "train_api:app" 2>/dev/null || true
    pkill -9 -f "mlflow server" 2>/dev/null || true
    pkill -9 -f "tensorboard" 2>/dev/null || true
    
    # Kill ALL dashboard processes (comprehensive patterns)
    echo "🗑️ Force-killing all dashboard processes..."
    pkill -9 -f "dashboard.app" 2>/dev/null || true
    pkill -9 -f "python.*dashboard" 2>/dev/null || true
    pkill -9 -f "python.*app.*main" 2>/dev/null || true
    pkill -9 -f "from dashboard.app import main" 2>/dev/null || true
    pkill -9 -f "dashboard.*main" 2>/dev/null || true
    pkill -9 -f "$VENV/bin/python" 2>/dev/null || true
    
    # Remove STOP file after cleanup
    rm -f "$STATE/STOP" 2>/dev/null || true
    
    echo "✅ All services stopped. Starting setup..."
    $VENV_PYTHON setup_env.py
    ;;
  search)
    # GitHub repository search
    $VENV_PYTHON github_search.py
    ;;
  up)
    # Ensure .env exists
    if [ ! -f "$ROOT/.env" ]; then
      echo "No .env found. Running setup..."
      $VENV_PYTHON setup_env.py
    fi
    
    # Stop any existing services first
    echo "🛑 Stopping any existing services..."
    touch "$STATE/STOP" 2>/dev/null || true
    for pidfile in "$PIDS"/*.pid; do
      if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
          echo "Stopping process $pid ($(basename $pidfile .pid))"
          kill "$pid" 2>/dev/null || true
          sleep 1
          kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
      fi
    done
    
    # Kill ALL remaining processes (force kill to be sure)
    echo "🔥 Force-killing all remaining processes..."
    pkill -9 -f "train_q_lora.py" 2>/dev/null || true
    pkill -9 -f "train_loop.py" 2>/dev/null || true
    pkill -9 -f "generate_samples.py" 2>/dev/null || true
    pkill -9 -f "train_api:app" 2>/dev/null || true
    pkill -9 -f "mlflow server" 2>/dev/null || true
    pkill -9 -f "tensorboard" 2>/dev/null || true
    
    # Kill ALL dashboard processes (comprehensive patterns)
    echo "🗑️ Force-killing all dashboard processes..."
    pkill -9 -f "dashboard.app" 2>/dev/null || true
    pkill -9 -f "python.*dashboard" 2>/dev/null || true
    pkill -9 -f "python.*app.*main" 2>/dev/null || true
    pkill -9 -f "from dashboard.app import main" 2>/dev/null || true
    pkill -9 -f "dashboard.*main" 2>/dev/null || true
    pkill -9 -f "$VENV/bin/python" 2>/dev/null || true
    
    # Remove STOP file after cleanup
    rm -f "$STATE/STOP" 2>/dev/null || true
    
    echo "✅ Cleanup complete. Starting services..."
    
    # Load configuration
    if [ -f "$ROOT/.env" ]; then
      set -a
      source "$ROOT/.env"
      set +a
    fi
    
    # Clear stale STOP
    rm -f "$STATE/STOP"
    
    echo "🚀 Starting all services with proper environment..."
    
    # Start Dashboard if enabled
    if [ "${ENABLE_DASHBOARD:-true}" = "true" ]; then
      if ! pgrep -f "dashboard.app" > /dev/null; then
        nohup $VENV_DASHBOARD -u -c "from dashboard.app import main; main()" >> "$LOGS/dashboard.log" 2>&1 &
        echo $! > "$PIDS/dashboard.pid"
      fi
    fi
    
    # MLflow
    if [ "${ENABLE_MLFLOW:-true}" = "true" ]; then
      if ! pgrep -f "mlflow server" > /dev/null; then
        nohup $VENV_PYTHON -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"sqlite:///$STATE/mlflow.db\" --default-artifact-root file://$ROOT/mlartifacts --serve-artifacts >> "$LOGS/mlflow.log" 2>&1 &
        echo $! > "$PIDS/mlflow.pid"
      fi
    fi
    
    # TensorBoard
    if [ "${ENABLE_TENSORBOARD:-true}" = "true" ]; then
      if ! pgrep -f "tensorboard" > /dev/null; then
        mkdir -p "$ROOT/data/ai-lab/logs" 2>/dev/null || true
        nohup "$VENV/bin/tensorboard" --logdir "$ROOT/data/ai-lab/logs" --host 0.0.0.0 --port 6006 >> "$LOGS/tb.log" 2>&1 &
        echo $! > "$PIDS/tb.pid"
      fi
    fi
    
    # API
    if ! pgrep -f "train_api:app" > /dev/null; then
      nohup $VENV_UVICORN server.train_api:app --host 0.0.0.0 --port 8000 >> "$LOGS/api.log" 2>&1 &
      echo $! > "$PIDS/api.pid"
    fi
    
    # Loop
    if ! pgrep -f "train_loop.py" > /dev/null; then
      nohup $VENV_PYTHON -u -m pipeline.train_loop >> "$LOGS/loop.log" 2>&1 &
      echo $! > "$PIDS/loop.pid"
    fi
    
    echo "🎉 All services started successfully!"
    echo "📊 Dashboard: http://localhost:8888 (terminal)"
    echo "🔗 API: http://localhost:8000/docs"
    echo "📈 MLflow: http://localhost:5000"
    echo "📋 TensorBoard: http://localhost:6006"
    echo "📝 Logs: tail -f $LOGS/*"
    ;;
  stop)
    echo "🛑 Stopping all services..."
    touch "$STATE/STOP"
    echo "Signal file created: $STATE/STOP"
    
    # List of explicit patterns to kill
    PATTERNS=(
      "train_q_lora.py"
      "train_loop.py"
      "generate_samples.py"
      "clean_dataset.py"
      "evaluate_model.py"
      "run_round.py"
      "scrape_repos.py"
      "train_api:app"
      "mlflow server"
      "tensorboard"
      "dashboard.app"
      "heidi_integration"
      "log_viewer"
    )

    # 1. Kill processes with PID files first (clean stop)
    echo "🧹 Attempting clean stop of tracked services..."
    for pidfile in "$PIDS"/*.pid; do
      if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        service=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
          echo "Stopping $service (PID $pid)..."
          kill "$pid" 2>/dev/null || true
          sleep 0.5
        fi
        rm -f "$pidfile"
      fi
    done
    
    # 2. Check and kill by pattern (more thorough)
    echo "🔥 Identifying and stopping any remaining training processes..."
    for pattern in "${PATTERNS[@]}"; do
      # Find PIDs for the pattern
      PIDS_FOUND=$(pgrep -f "$pattern" || true)
      if [ -n "$PIDS_FOUND" ]; then
        for pid in $PIDS_FOUND; do
          if [ "$pid" != "$$" ]; then # Don't kill self
            # Get the full command line for logging
            cmdline=$(ps -p "$pid" -o args= | head -1)
            echo "Stopping PID $pid: $cmdline"
            kill -9 "$pid" 2>/dev/null || true
          fi
        done
      fi
    done
    
    # 3. Final catch-all for anything in the venv
    echo "🧹 Final cleanup of venv processes..."
    VENV_PIDS=$(pgrep -f "$VENV/bin/python" || true)
    if [ -n "$VENV_PIDS" ]; then
        for pid in $VENV_PIDS; do
            if [ "$pid" != "$$" ]; then
                echo "Killing leftover venv process: $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # Kill any runaway GPU processes if they look like ours
    if command -v nvidia-smi &> /dev/null; then
        echo "🔍 Checking for stray GPU processes..."
        GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader || true)
        for pid in $GPU_PIDS; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                cmdline=$(ps -p "$pid" -o args= | head -1)
                if [[ "$cmdline" == *"$ROOT"* ]] || [[ "$cmdline" == *"python"* ]]; then
                    echo "Stopping stray GPU PID $pid: $cmdline"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
    fi

    # Wait a moment for OS to catch up
    sleep 1
    
    echo "✅ All training processes identified and stopped!"
    ;;
  status)
    for pidfile in "$PIDS"/*.pid; do
      [ -f "$pidfile" ] || continue
      pid=$(cat "$pidfile")
      if kill -0 "$pid" 2>/dev/null; then
        echo "$(basename $pidfile .pid): alive"
      else
        echo "$(basename $pidfile .pid): dead/stale - cleaning"
        rm "$pidfile"
      fi
    done
    cat "$LOGS"/*.log | tail -5 || true
    ;;
  doctor)
    $VENV_PYTHON -m tools.doctor
    ;;
  train-once)
    $VENV_PYTHON -m pipeline.train_loop --once
    ;;
  dashboard)
    # Start standalone dashboard
    echo "Starting dashboard..."
    if [ ! -f "$VENV_PYTHON" ]; then
      echo "Virtual environment not found. Creating..."
      python3 -m venv "$VENV"
      source "$VENV/bin/activate"
      pip install --upgrade pip
      pip install -r requirements.txt
    fi
    $VENV_DASHBOARD -c "from dashboard.app import main; main()"
    ;;
  heidi)
    # Start Heidi Engine integrated dashboard
    echo "Starting Heidi Engine integrated dashboard..."
    if [ ! -f "$VENV_PYTHON" ]; then
      echo "Virtual environment not found. Creating..."
      python3 -m venv "$VENV"
      source "$VENV/bin/activate"
      pip install --upgrade pip
      pip install -r requirements.txt
    fi
    $VENV_DASHBOARD -c "from dashboard.heidi_integration import main; main()"
    ;;
  logs)
    # Start interactive log viewer
    echo "Starting log viewer..."
    if [ ! -f "$VENV_PYTHON" ]; then
      echo "Virtual environment not found. Creating..."
      python3 -m venv "$VENV"
      source "$VENV/bin/activate"
      pip install --upgrade pip
      pip install -r requirements.txt
    fi
    $VENV_DASHBOARD -c "from dashboard.log_viewer import main; main()"
    ;;
  *)
    echo "Usage: $0 {setup|search|up|stop|status|doctor|train-once|dashboard|heidi|logs}"
    echo ""
    echo "Commands:"
    echo "  setup     - Interactive environment configuration setup"
    echo "  search    - Search GitHub repositories"
    echo "  up        - Start all services"
    echo "  stop      - Stop all services"
    echo "  status    - Check service status"
    echo "  doctor    - Run health checks"
    echo "  train-once - Run one training round"
    echo "  dashboard - Start rich monitoring dashboard"
    echo "  heidi     - Start Heidi Engine integrated dashboard"
    echo "  logs      - Start interactive log viewer"
    echo ""
    echo "Configuration:"
    echo "  - Edit .env file directly for manual configuration"
    echo "  - Run './run.sh setup' for interactive configuration"
    echo "  - Copy .env.template to .env for template"
    ;;
esac