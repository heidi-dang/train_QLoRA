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
mkdir -p "$VENV"

# Always ensure virtual environment is properly set up
echo "🔧 Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
fi

# Always activate and install dependencies
echo "Activating virtual environment and installing dependencies..."
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"

# Load configuration if exists
if [ -f "$CONFIG" ]; then
  export $(cat "$ROOT/.env" | xargs)
fi

VENV_PYTHON="$VENV/bin/python"
VENV_UVICORN="$VENV/bin/uvicorn"
VENV_DASHBOARD="$VENV/bin/python"

# Verify virtual environment exists and dependencies are installed
if [ ! -f "$VENV_PYTHON" ]; then
  echo "Virtual environment not found. Creating..."
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# Always activate environment for all commands
echo "🚀 Activating Python virtual environment..."
source "$VENV/bin/activate"

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
    
    # Kill any remaining processes
    pkill -f "train_api:app" 2>/dev/null || true
    pkill -f "train_loop.py" 2>/dev/null || true
    pkill -f "mlflow server" 2>/dev/null || true
    pkill -f "tensorboard" 2>/dev/null || true
    pkill -f "dashboard.app" 2>/dev/null || true
    
    # Remove STOP file after cleanup
    rm -f "$STATE/STOP" 2>/dev/null || true
    
    echo "✅ All services stopped. Starting setup..."
    $VENV_PYTHON setup_config.py
    ;;
  search)
    # GitHub repository search
    $VENV_PYTHON github_search.py
    ;;
  up)
    # Always run setup to ensure proper configuration
    echo "🔧 Running configuration setup..."
    $VENV_PYTHON setup_config.py
    
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
    
    # Kill any remaining processes
    pkill -f "train_api:app" 2>/dev/null || true
    pkill -f "train_loop.py" 2>/dev/null || true
    pkill -f "mlflow server" 2>/dev/null || true
    pkill -f "tensorboard" 2>/dev/null || true
    pkill -f "dashboard.app" 2>/dev/null || true
    
    # Remove STOP file after cleanup
    rm -f "$STATE/STOP" 2>/dev/null || true
    
    echo "✅ Cleanup complete. Starting services..."
    
    # Load configuration
    if [ -f "$ROOT/.env" ]; then
      export $(cat "$ROOT/.env" | xargs)
    fi
    
    # Clear stale STOP
    rm -f "$STATE/STOP"
    
    echo "🚀 Starting all services with proper environment..."
    
    # Start Dashboard if enabled
    if [ "${ENABLE_DASHBOARD:-true}" = "true" ]; then
      if ! pgrep -f "dashboard.app" > /dev/null; then
        nohup $VENV_DASHBOARD -c "from dashboard.app import main; main()" >> "$LOGS/dashboard.log" 2>&1 &
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
      nohup $VENV_PYTHON -m pipeline.train_loop >> "$LOGS/loop.log" 2>&1 &
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
    touch "$STATE/STOP"
    for pidfile in "$PIDS"/*.pid; do
      if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
          kill "$pid"
        fi
        rm -f "$pidfile"
      fi
    done
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
    echo "Usage: $0 {setup|search|up|stop|status|doctor|train-once|dashboard|logs}"
    echo ""
    echo "Commands:"
    echo "  setup     - Interactive configuration setup"
    echo "  search    - Search GitHub repositories"
    echo "  up        - Start all services"
    echo "  stop      - Stop all services"
    echo "  status    - Check service status"
    echo "  doctor    - Run health checks"
    echo "  train-once - Run one training round"
    echo "  dashboard - Start rich monitoring dashboard"
    echo "  logs      - Start interactive log viewer"
    ;;
esac