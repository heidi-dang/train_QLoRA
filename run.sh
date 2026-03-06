#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")"; pwd -P)
STATE="$ROOT/state"
LOGS="$ROOT/logs"
DATA="$ROOT/data/ai-lab"
PIDS="$STATE/pids"

mkdir -p "$STATE" "$LOGS" "$PIDS" "$DATA/datasets/clean" "$DATA/checkpoints" "$DATA/evaluation"

VENV="$ROOT/venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install torch transformers peft datasets trl accelerate bitsandbytes safetensors fastapi uvicorn psutil requests openai mlflow tensorboard wandb
fi

VENV_PYTHON="$VENV/bin/python"
VENV_UVICORN="$VENV/bin/uvicorn"

CMD=${1:-help}

case $CMD in
  up)
    # Clear stale STOP
    rm -f "$STATE/STOP"
    
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
    
    echo "Up: API http://localhost:8000/docs | TB http://localhost:6006"
    echo "Logs: tail -f $LOGS/*"
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
  *)
    echo "Usage: $0 {up|stop|status|doctor|train-once}"
    ;;
esac