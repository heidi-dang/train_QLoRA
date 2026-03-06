from fastapi import FastAPI
import subprocess
import os
import psutil

app = FastAPI()

@app.get("/")
def status():
    return {"status": "QLoRA trainer running"}

@app.get("/gpu")
def gpu():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]).decode()

    util, used, total = out.split(",")

    return {
        "gpu_utilization": util.strip(),
        "memory_used_mb": used.strip(),
        "memory_total_mb": total.strip()
    }

@app.get("/system")
def system():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent
    }

@app.get("/start_train")
def start_train():
    os.system("nohup python3 train.py > train.log 2>&1 &")
    return {"training": "started"}
