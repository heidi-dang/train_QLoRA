#!/bin/bash

set -e

echo "===================================="
echo " AI LAB TRAINING PLATFORM INSTALLER"
echo "===================================="

read -p "Project directory [~/ai-lab]: " PROJECT_DIR
PROJECT_DIR=${PROJECT_DIR:-~/ai-lab}

read -p "Python venv name [qlora-env]: " VENV_NAME
VENV_NAME=${VENV_NAME:-qlora-env}

read -p "MLflow port [5000]: " MLFLOW_PORT
MLFLOW_PORT=${MLFLOW_PORT:-5000}

read -p "TensorBoard port [6006]: " TB_PORT
TB_PORT=${TB_PORT:-6006}

read -p "Training API port [8080]: " API_PORT
API_PORT=${API_PORT:-8080}

echo "Creating workspace..."

mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

mkdir -p runs checkpoints datasets mlflow/artifacts api

echo "Installing system dependencies..."

sudo apt update
sudo apt install -y python3-pip python3-venv git prometheus grafana

echo "Creating python environment..."

python3 -m venv ~/$VENV_NAME
source ~/$VENV_NAME/bin/activate

pip install --upgrade pip

echo "Installing AI stack..."

pip install \
mlflow \
ray[train] \
transformers \
datasets \
accelerate \
peft \
bitsandbytes \
trl \
tensorboard \
fastapi \
uvicorn \
psutil \
gpustat \
wandb

echo "Creating FastAPI control server..."

cat <<EOF > api/server.py
from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def root():
    return {"status": "AI training server running"}

@app.post("/train")
def train():
    subprocess.Popen(["python", "train.py"])
    return {"status": "training started"}

@app.get("/gpu")
def gpu():
    import subprocess
    result = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ])
    return {"gpu": result.decode()}
EOF

echo "Creating Mistral QLoRA training template..."

cat <<EOF > train.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="datasets/train.json")

training_args = TrainingArguments(
    output_dir="checkpoints",
    logging_dir="runs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)

trainer.train()
EOF

echo "Starting Ray cluster..."

ray start --head --port=6379

echo "Starting MLflow..."

mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root $PROJECT_DIR/mlflow/artifacts \
--host 0.0.0.0 \
--port $MLFLOW_PORT &

echo "Starting TensorBoard..."

tensorboard \
--logdir $PROJECT_DIR/runs \
--host 0.0.0.0 \
--port $TB_PORT &

echo "Starting FastAPI control server..."

uvicorn api.server:app \
--host 0.0.0.0 \
--port $API_PORT &

echo "Starting Grafana..."

sudo systemctl start grafana-server

IP=$(curl -s ifconfig.me)

echo ""
echo "===================================="
echo "AI LAB READY"
echo "===================================="
echo "MLflow:        http://$IP:$MLFLOW_PORT"
echo "TensorBoard:   http://$IP:$TB_PORT"
echo "Grafana:       http://$IP:3000"
echo "Training API:  http://$IP:$API_PORT"
echo ""
echo "Upload dataset to:"
echo "$PROJECT_DIR/datasets/train.json"
echo ""
echo "Start training:"
echo "curl -X POST http://$IP:$API_PORT/train"
echo ""
