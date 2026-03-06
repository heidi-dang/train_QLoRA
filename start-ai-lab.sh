#!/usr/bin/env bash

set -e

echo "================================="
echo " HEIDI AI TRAINING LAB STARTER"
echo "================================="

cd ~/trainer

# ------------------------------
# PROMPT FOR KEYS
# ------------------------------

read -p "Enter xAI API Key: " XAI_API_KEY
read -p "Enter HuggingFace API Token: " HF_TOKEN

export XAI_API_KEY=$XAI_API_KEY
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

echo ""
echo "Keys loaded."

# ------------------------------
# ACTIVATE ENV
# ------------------------------

echo "Activating environment..."

source qlora-env/bin/activate

# ------------------------------
# START MLFLOW
# ------------------------------

echo "Starting MLflow..."

pkill -f mlflow || true

nohup mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ~/trainer/mlruns \
--host 0.0.0.0 \
--port 5000 \
--allowed-hosts '*' \
> mlflow.log 2>&1 &

sleep 3

# ------------------------------
# START TENSORBOARD
# ------------------------------

echo "Starting TensorBoard..."

pkill -f tensorboard || true

nohup tensorboard \
--logdir runs \
--host 0.0.0.0 \
--port 6006 \
> tensorboard.log 2>&1 &

sleep 3

# ------------------------------
# START API SERVER
# ------------------------------

echo "Starting Training API..."

pkill -f server.py || true

nohup python server.py > server.log 2>&1 &

sleep 3

# ------------------------------
# START TRAIN LOOP
# ------------------------------

echo "Starting Continuous Training..."

pkill -f train_loop || true

nohup python -m pipeline.train_loop > train_loop.log 2>&1 &

sleep 5

# ------------------------------
# HEALTH CHECK
# ------------------------------

echo ""
echo "Running health checks..."
echo ""

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "NO GPU")

MLFLOW=$(curl -s http://localhost:5000/api/2.0/mlflow/experiments/search || echo "DOWN")
API=$(curl -s http://localhost:8080/train/status || echo "DOWN")

echo "================================="
echo " AI LAB STATUS"
echo "================================="

echo "GPU:"
echo "$GPU"
echo ""

echo "MLflow API:"
if [[ "$MLFLOW" == *"experiments"* ]]; then
echo "OK"
else
echo "FAILED"
fi

echo ""

echo "Training API:"
echo "$API"

echo ""

echo "Processes:"
ps aux | grep -E "mlflow|tensorboard|train_loop|server.py" | grep -v grep

echo ""

# ------------------------------
# SHOW MONITOR LINKS
# ------------------------------

IP=$(curl -s ifconfig.me)

echo "================================="
echo " MONITORING DASHBOARDS"
echo "================================="

echo ""
echo "MLflow:"
echo "http://$IP:5000"

echo ""
echo "TensorBoard:"
echo "http://$IP:6006"

echo ""
echo "Training API:"
echo "http://$IP:8080"

echo ""
echo "Logs:"
echo "tail -f train_loop.log"
echo "tail -f mlflow.log"

echo ""
echo "================================="
echo " AI LAB READY"
echo "================================="
