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
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# Load configuration if exists
if [ -f "$CONFIG" ]; then
  export $(cat "$ROOT/.env" | xargs)
fi

VENV_PYTHON="$VENV/bin/python"
VENV_UVICORN="$VENV/bin/uvicorn"
VENV_DASHBOARD="$VENV/bin/python"

CMD=${1:-help}

case $CMD in
  setup)
    # Interactive setup
    $VENV_PYTHON setup_config.py
    ;;
  search)
    # GitHub repository search
    $VENV_PYTHON github_search.py
    ;;
  up)
    # Check if configuration exists
    if [ ! -f "$CONFIG" ]; then
      echo "No configuration found. Running setup first..."
      $VENV_PYTHON setup_config.py
    fi
    
    # Load configuration
    if [ -f "$ROOT/.env" ]; then
      export $(cat "$ROOT/.env" | xargs)
    fi
    
    # Clear stale STOP
    rm -f "$STATE/STOP"
    
    # Start Dashboard if enabled
    if [ "${ENABLE_DASHBOARD:-true}" = "true" ]; then
      if ! pgrep -f "dashboard/app.py" > /dev/null; then
        nohup $VENV_DASHBOARD -m dashboard.app >> "$LOGS/dashboard.log" 2>&1 &
        echo $! > "$PIDS/dashboard.pid"
      fi
    fi
    
    # Start MLflow if enabled
    if [ "${ENABLE_MLFLOW:-true}" = "true" ]; then
      if ! pgrep -f "mlflow server" > /dev/null; then
        nohup $VENV_PYTHON -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri "sqlite:///$STATE/mlflow.db" >> "$LOGS/mlflow.log" 2>&1 &
        echo $! > "$PIDS/mlflow.pid"
      fi
    fi
    
    # Start TensorBoard if enabled
    if [ "${ENABLE_TENSORBOARD:-true}" = "true" ]; then
      if ! pgrep -f "tensorboard" > /dev/null; then
        nohup $VENV_PYTHON -c "import sys; import importlib_metadata; sys.modules['pkg_resources'] = importlib_metadata; from tensorboard import main; main(['serve', '--logdir', '$ROOT/data/ai-lab/logs', '--host', '0.0.0.0', '--port', '6006'])" >> "$LOGS/tb.log" 2>&1 &
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
    
    echo "🚀 Services started:"
    echo "  - API: http://localhost:8000/docs"
    echo "  - MLflow: http://localhost:5000"
    echo "  - TensorBoard: http://localhost:6006"
    echo "  - Dashboard: http://localhost:8888 (terminal)"
    echo "  - Logs: tail -f $LOGS/*"
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
    $VENV_PYTHON -m tools.doctor one-click-runner
    ;;
  train-once)
    $VENV_PYTHON -m pipeline.train_loop --once
    ;;
  dashboard)
    # Start standalone dashboard
    $VENV_DASHBOARD -m dashboard.app
    ;;
  *)
    echo "Usage: $0 {setup|search|up|stop|status|doctor|train-once|dashboard}"
    echo ""
    echo "Commands:"
    echo "  setup     - Interactive configuration setup"
    echo "  search    - Search GitHub repositories"
    echo "  up        - Start all services"
    echo "  stop      - Stop all services"
    echo "  status    - Check service status"
    echo "  doctor    - Run health checks"
    echo "  train-once - Run one training round"
    echo "  dashboard - Start monitoring dashboard"
    ;;
esac