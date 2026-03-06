# One-Click Runner Implementation

## Overview

Canonical entrypoint: `./run.sh [up|stop|status|doctor|train-once]`

- **Repo-root relative**: State in `./state/`, logs `./logs/`, data `./data/ai-lab/`
- **Bootstrap**: venv in `./venv/`, pip install -r requirements.txt if missing.
- **Processes**: API (FastAPI train_api.py), Loop (train_loop.py), managed pids in `./state/pids/`
- **Lifecycle**: PID files + liveness check, stale clean, idempotent.
- **URLs**: http://localhost:8000/docs (API), http://localhost:6006 (TB)

## Flow Diagrams

```
clean clone
  ↓ ./run.sh up
bootstrap venv + dirs + reqs
validate config/repos.txt
start API (train_api.py) → pid + URL
start Loop (train_loop.py) → pid + log
print URLs/logs → tail -f logs/*
  ↓ Ctrl+C or ./run.sh stop
kill pids (alive check), rm stale, clear STOP
```

## Components

### run.sh
Bash orchestrator:
- up: mkdir -p state/logs/data/ai-lab/{datasets/checkpoints/eval}, venv bootstrap, API & loop Popen (bg), print.
- stop: kill pids (ps alive), rm pids/STOP
- status: ps pids + logs tail
- doctor: tools/doctor.py one-click
- train-once: loop single round

### server/train_api.py (canonical)
FastAPI /docs/health/status/start-round/stop-round

### pipeline/train_loop.py
Checked subprocess.run (check=True), fail→log/mark failed/STOP, backoff exp (5/10/30s max 5).

## Doctor Checks
tools/doctor_one_click_runner.py:
- Paths relative
- Only train_api.py server
- Pid liveness/lifecycle
- Imports all pipeline/*
- No tracked generated (git ls-files data/)
- Smoke run.sh up/stop (subprocess, no persist)

## Recovery
- Stale pid/STOP: clean on up
- Failures: backoff, status shows

## Clean Clone Steps
1. git clone ...
2. ./run.sh up
3. Visit URLs
4. ./run.sh stop