#!/bin/bash
# Hot-reload script for run.sh services
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.."; pwd -P)
STATE="$ROOT/state"
PIDS="$STATE/pids"

echo "🔥 Starting hot-reload monitor for run.sh services..."

# Function to restart a service
restart_service() {
    local service_name="$1"
    local pidfile="$PIDS/${service_name}.pid"
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "🔄 Restarting $service_name (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
    
    # Restart the service
    case "$service_name" in
        "dashboard")
            cd "$ROOT"
            source venv/bin/activate
            nohup venv/bin/python -u -c "from dashboard.app import main; main()" >> "$ROOT/logs/dashboard.log" 2>&1 &
            echo $! > "$pidfile"
            echo "✅ Dashboard restarted"
            ;;
        "api")
            cd "$ROOT"
            source venv/bin/activate
            nohup venv/bin/uvicorn server.train_api:app --host 0.0.0.0 --port 8000 >> "$ROOT/logs/api.log" 2>&1 &
            echo $! > "$pidfile"
            echo "✅ API restarted"
            ;;
        "tensorboard")
            cd "$ROOT"
            source venv/bin/activate
            mkdir -p "$ROOT/data/ai-lab/logs" 2>/dev/null || true
            nohup "$ROOT/venv/bin/tensorboard" --logdir "$ROOT/data/ai-lab/logs" --host 0.0.0.0 --port 6006 >> "$ROOT/logs/tb.log" 2>&1 &
            echo $! > "$pidfile"
            echo "✅ TensorBoard restarted"
            ;;
        "mlflow")
            cd "$ROOT"
            source venv/bin/activate
            nohup venv/bin/python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri "sqlite:///$STATE/mlflow.db" --default-artifact-root file://$ROOT/mlartifacts --serve-artifacts >> "$ROOT/logs/mlflow.log" 2>&1 &
            echo $! > "$pidfile"
            echo "✅ MLflow restarted"
            ;;
        "loop")
            cd "$ROOT"
            source venv/bin/activate
            nohup venv/bin/python -u -m pipeline.train_loop >> "$ROOT/logs/loop.log" 2>&1 &
            echo $! > "$pidfile"
            echo "✅ Training loop restarted"
            ;;
    esac
}

# Monitor run.sh for changes
monitor_run_sh() {
    local run_sh_mtime=$(stat -c %Y "$ROOT/run.sh" 2>/dev/null || echo "0")
    local env_mtime=$(stat -c %Y "$ROOT/.env" 2>/dev/null || echo "0")
    
    while true; do
        sleep 2
        
        local current_run_sh_mtime=$(stat -c %Y "$ROOT/run.sh" 2>/dev/null || echo "0")
        local current_env_mtime=$(stat -c %Y "$ROOT/.env" 2>/dev/null || echo "0")
        
        if [ "$current_run_sh_mtime" != "$run_sh_mtime" ] || [ "$current_env_mtime" != "$env_mtime" ]; then
            echo "🔄 Detected changes in run.sh or .env, restarting services..."
            run_sh_mtime=$current_run_sh_mtime
            env_mtime=$current_env_mtime
            
            # Restart all services
            restart_service "dashboard"
            restart_service "api"
            restart_service "tensorboard"
            restart_service "mlflow"
            restart_service "loop"
            
            echo "✅ All services hot-reloaded!"
        fi
    done
}

# Start monitoring
monitor_run_sh
