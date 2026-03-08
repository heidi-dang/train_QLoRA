#!/usr/bin/env python3
"""Main training loop: coordinates scraping, generation, cleaning, training, evaluation and checkpointing.

This script is designed to be started as a background process by the FastAPI server.
"""
import os
import time
import logging
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')
STOP_FILE = os.path.join(AI_LAB, 'STOP')
CHECKPOINTS = os.path.join(AI_LAB, 'checkpoints')
DATASETS_CLEAN = os.path.join(AI_LAB, 'datasets', 'clean')
TRAIN_FILE = os.path.join(DATASETS_CLEAN, 'train.json')
MLFLOW_EXPERIMENT = 'continuous_training'


def _preferred_python() -> str:
    """Prefer the repo venv Python for all subprocess stages.

    This avoids issues where the parent process is started with system python
    but ML deps (torch/transformers/trl/peft) are only installed in venv.
    """
    venv_py = os.path.join(ROOT, 'venv', 'bin', 'python')
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

logging.basicConfig(level=logging.INFO)


def stage_scrape():
    logging.info('Stage: scrape')
    subprocess.call([_preferred_python(), os.path.join(ROOT, 'pipeline', 'scrape_repos.py')])


def stage_generate():
    logging.info('Stage: generate samples')
    subprocess.call([_preferred_python(), os.path.join(ROOT, 'pipeline', 'generate_samples.py')])


def stage_clean():
    logging.info('Stage: clean dataset')
    subprocess.call([_preferred_python(), os.path.join(ROOT, 'pipeline', 'clean_dataset.py')])


def _safe_int_env(name: str, default: int) -> int:
    try:
        v = str(os.environ.get(name, '')).strip()
        return int(v) if v else default
    except Exception:
        return default


def _safe_bool_env(name: str, default: bool) -> bool:
    v = str(os.environ.get(name, '')).strip().lower()
    if not v:
        return default
    if v in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if v in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def get_clean_sample_count(train_file: str = TRAIN_FILE) -> int:
    """Count available cleaned samples. Returns 0 if missing/unreadable."""
    try:
        if not os.path.exists(train_file):
            return 0
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return int(len(data)) if isinstance(data, list) else 0
    except Exception:
        return 0


def stage_train(round_n: int):
    logging.info('Stage: train (round %d) - placeholder training', round_n)
    # In production: invoke training script that uses bitsandbytes + PEFT + transformers
    ckpt_dir = os.path.join(CHECKPOINTS, f'adapter_round_{round_n}')
    os.makedirs(ckpt_dir, exist_ok=True)
    # call training scaffold (train_q_lora) which will use real training if deps present
    train_script = os.path.join(ROOT, 'pipeline', 'train_q_lora.py')
    try:
        subprocess.check_call([
            _preferred_python(),
            train_script,
            os.path.join(AI_LAB, 'datasets', 'clean', 'train.json'),
            ckpt_dir,
        ])
    except subprocess.CalledProcessError:
        logging.exception('Training script failed; creating metadata only')
    meta = {'round': round_n, 'base_model': 'mistralai/Mistral-7B-Instruct-v0.2'}
    with open(os.path.join(ckpt_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)
    logging.info('Wrote checkpoint to %s', ckpt_dir)


def stage_evaluate(round_n: int):
    logging.info('Stage: evaluate')
    subprocess.call([_preferred_python(), os.path.join(ROOT, 'pipeline', 'evaluate_model.py'), str(round_n)])


def main_loop():
    round_n = 0
    Path(CHECKPOINTS).mkdir(parents=True, exist_ok=True)
    while True:
        try:
            if os.path.exists(STOP_FILE):
                logging.info('STOP file exists - exiting loop')
                break
            round_n += 1

            # Auto-orchestrator: if enough clean data exists, train first to save time,
            # then go back to generation/cleaning to replenish the buffer.
            auto_train_first = _safe_bool_env('AUTO_TRAIN_FIRST', True)
            min_samples_to_train = _safe_int_env('MIN_SAMPLES_TO_TRAIN', 200)
            target_buffer_samples = _safe_int_env('TARGET_BUFFER_SAMPLES', 500)
            clean_count = get_clean_sample_count(TRAIN_FILE)

            stages = []
            if auto_train_first and clean_count >= min_samples_to_train:
                stages.extend([
                    ("train", lambda: stage_train(round_n)),
                    ("evaluate", lambda: stage_evaluate(round_n)),
                ])

                # After training, top up data for the next round if below buffer target.
                # This is best-effort: scrape/generate/clean can still be slow depending on repos.
                if get_clean_sample_count(TRAIN_FILE) < target_buffer_samples:
                    stages.extend([
                        ("scrape", stage_scrape),
                        ("generate", stage_generate),
                        ("clean", stage_clean),
                    ])
            else:
                # Default behavior: generate/clean then train/evaluate.
                stages.extend([
                    ("scrape", stage_scrape),
                    ("generate", stage_generate),
                    ("clean", stage_clean),
                    ("train", lambda: stage_train(round_n)),
                    ("evaluate", lambda: stage_evaluate(round_n)),
                ])

            for i, (name, func) in enumerate(stages, 1):
                pct = (i / len(stages)) * 100
                bar_len = 20
                filled = int(bar_len * i // len(stages))
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f'\rRound {round_n} [{i}/5] |{bar}| {pct:.3f}% | {name}', end='', flush=True)
                func()
            print(f'\nRound {round_n} complete')
            # MLflow logging: read evaluation results and log metrics
            eval_file = os.path.join(AI_LAB, 'evaluation', f'results_round_{round_n}.json')
            metrics = {}
            try:
                with open(eval_file) as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}

            if mlflow is not None:
                try:
                    # configure local sqlite tracking server if not set
                    if 'MLFLOW_TRACKING_URI' not in os.environ:
                        db_path = os.path.abspath(os.path.join(ROOT, 'mlflow.db'))
                        os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{db_path}'
                    mlflow.set_experiment(MLFLOW_EXPERIMENT)
                    with mlflow.start_run(run_name=f'round_{round_n}'):
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(k, float(v))
                        # log checkpoint artifact if exists
                        ckpt_dir = os.path.join(CHECKPOINTS, f'adapter_round_{round_n}')
                        if os.path.exists(ckpt_dir):
                            try:
                                mlflow.log_artifacts(ckpt_dir, artifact_path=f'checkpoint_round_{round_n}')
                            except Exception:
                                logging.exception('Failed to log artifacts to MLflow')
                except Exception:
                    logging.exception('MLflow logging failed')

        except Exception:
            logging.exception('Round %d failed', round_n)
        # short sleep between rounds
        time.sleep(1)


if __name__ == '__main__':
    main_loop()
