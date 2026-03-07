#!/usr/bin/env python3
import os
import json
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')

logging.basicConfig(level=logging.INFO)


def check_repo_list():
    p = os.path.join(AI_LAB, 'datasets', 'repos_premium.txt')
    return os.path.exists(p)


def check_dataset_generated():
    raw = os.path.join(AI_LAB, 'datasets', 'raw')
    return os.path.isdir(raw) and len(os.listdir(raw)) > 0


def check_mlflow():
    # placeholder: check if MLFLOW_TRACKING_URI set
    return 'MLFLOW_TRACKING_URI' in os.environ


def check_gpu():
    try:
        import subprocess
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
        return bool(out.strip())
    except Exception:
        return False


def check_training_loop_alive():
    pidf = os.path.join(ROOT, 'state', 'pids', 'train_loop.pid')
    return os.path.exists(pidf)


def main():
    report = {
        'repo_list': check_repo_list(),
        'dataset_raw_present': check_dataset_generated(),
        'mlflow_configured': check_mlflow(),
        'gpu_available': check_gpu(),
        'training_loop_pidfile': check_training_loop_alive(),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
