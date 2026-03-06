# QLoRA Continuous Trainer - One-Click Workflow

## Clean Clone Quickstart

```bash
git clone https://github.com/heidi-dang/train_QLoRA.git
cd train_QLoRA
./run.sh up
```

**Outputs:**
- API: http://localhost:8000/docs (Swagger, /train/start /status /metrics /gpu /system)
- TensorBoard: http://localhost:6006 (logs/tb)
- MLflow UI: http://localhost:5000 (state/mlflow.db)
- Logs: `tail -f logs/*`

**Stop:** `./run.sh stop`

**Status:** `./run.sh status` (pids alive, log tail)

**Doctor:** `./run.sh doctor` (checks + smoke)

## Architecture

See [docs/implementation-one_click_runner.md](docs/implementation-one_click_runner.md)

- **Root-relative:** `./state/` (pids/STOP/mlflow.db), `./logs/` (api/loop/tb), `./data/ai-lab/` (datasets/checkpoints/evaluation)
- **Venv:** `./venv/` auto-bootstrap (requirements.txt)
- **API:** `server/train_api.py` FastAPI 8000
- **Loop:** `pipeline/train_loop.py` continuous rounds (scrape→gen→clean→train→eval)
- **Runner:** `./run.sh` orchestrates all

## Recovery

- Stale pids/STOP auto-clean on up
- Failures: backoff (5s→10s→30s max5), no spam
- `run.sh status` shows alive/dead

## Manual Smoke (for dev)

1. `./run.sh up` → URLs printed, status alive
2. `curl localhost:8000/train/start`
3. `./run.sh status` → loop running
4. `./run.sh stop` → pids gone
5. `./run.sh up` → restarts clean

## Doctor Output Example

```
paths: OK (relative)
server: OK (train_api only)
lifecycle: OK
imports: OK
artifacts: OK (no tracked gen)
smoke: OK (up/stop cycle)
backoff: OK (sim failure no spam)
```

## Deps

`requirements.txt` - GPU: torch+CUDA, CPU fallback OK.