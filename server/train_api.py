from fastapi import FastAPI, BackgroundTasks
import json
import subprocess
import os
import uvicorn
import signal
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAINER = os.path.join(ROOT, 'pipeline', 'train_loop.py')
PID_FILE = os.path.join(ROOT, 'ai-lab', 'train_loop.pid')
STOP_FILE = os.path.join(ROOT, 'ai-lab', 'STOP')

app = FastAPI()
logging.basicConfig(level=logging.INFO)


def start_background_process():
    # start train_loop.py as background process
    proc = subprocess.Popen(['python3', TRAINER])
    with open(PID_FILE, 'w') as f:
        f.write(str(proc.pid))
    return proc.pid


def stop_background_process():
    # create STOP file to signal process to exit
    open(STOP_FILE, 'w').close()
    # also attempt to kill by pid if present
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


@app.post('/train/start')
def api_start():
    pid = start_background_process()
    return {'status': 'started', 'pid': pid}


@app.post('/train/stop')
def api_stop():
    stop_background_process()
    return {'status': 'stopped'}


@app.get('/train/status')
def api_status():
    running = os.path.exists(PID_FILE)
    return {'running': running}


@app.get('/train/metrics')
def api_metrics():
    # read latest evaluation file if exists
    eval_dir = os.path.join(ROOT, 'ai-lab', 'evaluation')
    if not os.path.exists(eval_dir):
        return {'metrics': {}}
    files = sorted([f for f in os.listdir(eval_dir) if f.startswith('results_round_')])
    if not files:
        return {'metrics': {}}
    latest = files[-1]
    with open(os.path.join(eval_dir, latest)) as f:
        try:
            return json.load(f)
        except Exception:
            return {'metrics': {}}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
