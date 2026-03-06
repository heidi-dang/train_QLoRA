#!/usr/bin/env bash
set -e

echo "======================================"
echo " HEIDI AI LAB PRO LAUNCHER"
echo "======================================"

cd ~/trainer

# --------------------------
# PROMPT API KEYS
# --------------------------

read -p "Enter xAI API key: " XAI_API_KEY
read -p "Enter HuggingFace token: " HF_TOKEN

export XAI_API_KEY
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# --------------------------
# GPU CHECK
# --------------------------

echo ""
echo "Checking GPU..."

if command -v nvidia-smi &> /dev/null
then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "GPU detected: $GPU"
else
    echo "WARNING: No GPU detected"
fi

# --------------------------
# VENV AUTO CREATE
# --------------------------

if [ ! -d "qlora-env" ]; then
    echo "Creating Python environment..."
    python3 -m venv qlora-env
fi

source qlora-env/bin/activate

# --------------------------
# INSTALL DEPENDENCIES
# --------------------------

echo "Installing dependencies..."

pip install --upgrade pip

pip install \
transformers \
datasets \
accelerate \
peft \
bitsandbytes \
trl \
mlflow \
tensorboard \
fastapi \
uvicorn \
tqdm \
gitpython \
requests

# --------------------------
# INSTALL TMUX
# --------------------------

if ! command -v tmux &> /dev/null
then
    sudo apt install -y tmux
fi

# --------------------------
# START SERVICES IN TMUX
# --------------------------

SESSION="ai-lab"

tmux kill-session -t $SESSION 2>/dev/null || true

tmux new-session -d -s $SESSION

# pane 1 MLflow
tmux send-keys -t $SESSION "
cd ~/trainer
source qlora-env/bin/activate
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ~/trainer/mlruns \
--host 0.0.0.0 \
--port 5000 \
--allowed-hosts '*'
" C-m

sleep 2

# pane 2 TensorBoard
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "
cd ~/trainer
source qlora-env/bin/activate
tensorboard --logdir runs --host 0.0.0.0 --port 6006
" C-m

sleep 2

# pane 3 API
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "
cd ~/trainer
source qlora-env/bin/activate
python server.py
" C-m

sleep 2

# pane 4 TRAIN LOOP
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "
cd ~/trainer
source qlora-env/bin/activate
while true; do
  python -m pipeline.train_loop || true
  echo 'Training crashed — restarting in 5 seconds'
  sleep 5
done
" C-m

tmux select-layout -t $SESSION tiled

# --------------------------
# HEALTH CHECK
# --------------------------

sleep 5

echo ""
echo "======================================"
echo " AI LAB RUNNING"
echo "======================================"

IP=$(curl -s ifconfig.me)

echo ""
echo "MLflow Dashboard:"
echo "http://$IP:5000"

echo ""
echo "TensorBoard:"
echo "http://$IP:6006"

echo ""
echo "Training API:"
echo "http://$IP:8080"

echo ""
echo "Attach to session:"
echo "tmux attach -t $SESSION"

echo ""
echo "GPU Monitor:"
echo "watch -n1 nvidia-smi"

echo ""
echo "======================================"
