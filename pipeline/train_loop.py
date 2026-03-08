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
RUNS_DIR = os.path.join(AI_LAB, 'runs')
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


def get_latest_checkpoint(runs_dir: str) -> Optional[str]:
    """Find the latest checkpoint across all runs to resume from."""
    import glob
    ckpt_patterns = [
        os.path.join(runs_dir, '*', 'checkpoints', 'checkpoint-*'),
        os.path.join(AI_LAB, 'checkpoints', 'adapter_round_*', 'checkpoint-*') # Legacy
    ]
    
    all_ckpts = []
    for pattern in ckpt_patterns:
        all_ckpts.extend(glob.glob(pattern))
    
    if not all_ckpts:
        return None
        
    # Sort by modification time to find the truly latest
    all_ckpts.sort(key=os.path.getmtime)
    latest = all_ckpts[-1]
    logging.info(f"Found latest checkpoint for resume: {latest}")
    return latest

def update_symlinks(run_dir: str):
    """Update latest and best symlinks in AI_LAB."""
    latest_link = os.path.join(AI_LAB, 'latest')
    if os.path.lexists(latest_link):
        os.remove(latest_link)
    os.symlink(run_dir, latest_link)
    logging.info(f"Updated 'latest' symlink to {run_dir}")

    # For 'best', we check eval_loss in run_metadata.json
    metadata_file = os.path.join(run_dir, 'run_metadata.json')
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                current_meta = json.load(f)
            
            best_link = os.path.join(AI_LAB, 'best')
            if not os.path.lexists(best_link):
                os.symlink(run_dir, best_link)
                logging.info(f"Created first 'best' symlink to {run_dir}")
            else:
                best_dir = os.path.realpath(best_link)
                best_meta_file = os.path.join(best_dir, 'run_metadata.json')
                if os.path.exists(best_meta_file):
                    with open(best_meta_file, 'r') as f:
                        best_meta = json.load(f)
                    
                    curr_loss = current_meta.get('best_eval_loss') or current_meta.get('final_loss', float('inf'))
                    best_loss = best_meta.get('best_eval_loss') or best_meta.get('final_loss', float('inf'))
                    
                    if curr_loss < best_loss:
                        os.remove(best_link)
                        os.symlink(run_dir, best_link)
                        logging.info(f"Updated 'best' symlink to {run_dir} (Loss: {curr_loss:.4f} < {best_loss:.4f})")
        except Exception as e:
            logging.error(f"Failed to update 'best' symlink: {e}")

def stage_train(run_id: str, round_n: int):
    logging.info('Stage: train (run %s, round %d)', run_id, round_n)
    
    run_dir = os.path.join(RUNS_DIR, run_id)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt = get_latest_checkpoint(RUNS_DIR)
    
    train_script = os.path.join(ROOT, 'pipeline', 'train_q_lora.py')
    env = os.environ.copy()
    if latest_ckpt:
        env['RESUME_FROM_CHECKPOINT'] = latest_ckpt

    try:
        # train_q_lora.py will handle HF resume via internal TrainingArguments if pointed to a checkpoint
        # but our script signature is <train_file> <output_dir>.
        # We need train_q_lora.py to know it should resume from latest_ckpt.
        # I'll update train_q_lora.py to check for an env var or just rely on HF's auto-detection if we use common output_dir.
        # But here output_dir is NEW for each run. 
        # So we MUST pass the specific checkpoint path.
        
        # ACTUALLY, TrainingArguments.resume_from_checkpoint can be a path.
        # Let's modify train_q_lora.py to take resume_from_checkpoint as an optional arg or env var.
        # I did add resume_from_checkpoint=True in train_q_lora.py, but that only works if ckpt is in output_dir.
        
        subprocess.check_call([
            _preferred_python(),
            train_script,
            os.path.join(AI_LAB, 'datasets', 'clean', 'train.json'),
            ckpt_dir,
        ], env=env)
        
        # After training, move/link final adapter to run_dir/adapter
        adapter_dest = os.path.join(run_dir, 'adapter')
        # trainer.save_model saves to output_dir (ckpt_dir here)
        # We want the final adapter in a clean 'adapter' folder
        if os.path.exists(ckpt_dir):
            import shutil
            # If trainer saved model directly to ckpt_dir, we might have files there.
            # But usually it saves to subfolders if checkpointing.
            # However, trainer.save_model(output_dir) saves the final state.
            os.makedirs(adapter_dest, exist_ok=True)
            for item in os.listdir(ckpt_dir):
                s = os.path.join(ckpt_dir, item)
                d = os.path.join(adapter_dest, item)
                if os.path.isfile(s) and not item.startswith('checkpoint-'):
                    shutil.copy2(s, d)
            logging.info(f"Copied final adapter to {adapter_dest}")
            
        update_symlinks(run_dir)
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Training script failed with exit code {e.returncode}")
        raise RuntimeError("Training failed")

def stage_evaluate(run_id: str):
    logging.info('Stage: evaluate run %s', run_id)
    subprocess.call([_preferred_python(), os.path.join(ROOT, 'pipeline', 'evaluate_model.py'), run_id])


def main_loop():
    round_n = 0
    os.makedirs(RUNS_DIR, exist_ok=True)
    while True:
        try:
            if os.path.exists(STOP_FILE):
                logging.info('STOP file exists - exiting loop')
                break
            round_n += 1
            
            # Generate a unique run ID for this round's training
            run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_r{round_n}"
            run_dir = os.path.join(RUNS_DIR, run_id)

            # Auto-orchestrator
            auto_train_first = _safe_bool_env('AUTO_TRAIN_FIRST', True)
            min_samples_to_train = _safe_int_env('MIN_SAMPLES_TO_TRAIN', 200)
            target_buffer_samples = _safe_int_env('TARGET_BUFFER_SAMPLES', 500)
            clean_count = get_clean_sample_count(TRAIN_FILE)

            stages = []
            if auto_train_first and clean_count >= min_samples_to_train:
                stages.extend([
                    ("train", lambda: stage_train(run_id, round_n)),
                    ("evaluate", lambda: stage_evaluate(run_id)),
                ])

                if get_clean_sample_count(TRAIN_FILE) < target_buffer_samples:
                    stages.extend([
                        ("scrape", stage_scrape),
                        ("generate", stage_generate),
                        ("clean", stage_clean),
                    ])
            else:
                stages.extend([
                    ("scrape", stage_scrape),
                    ("generate", stage_generate),
                    ("clean", stage_clean),
                    ("train", lambda: stage_train(run_id, round_n)),
                    ("evaluate", lambda: stage_evaluate(run_id)),
                ])

            for i, (name, func) in enumerate(stages, 1):
                pct = (i / len(stages)) * 100
                bar_len = 20
                filled = int(bar_len * i // len(stages))
                bar = '█' * filled + '░' * (bar_len - filled)
                sys.stdout.write(f'\rRound {round_n} [{i}/{len(stages)}] |{bar}| {pct:.1f}% | {name} ')
                sys.stdout.flush()
                func()
            print(f'\nRound {round_n} complete')

            # MLflow logging
            if mlflow is not None:
                try:
                    metadata_file = os.path.join(run_dir, 'run_metadata.json')
                    metrics = {}
                    if os.path.exists(metadata_file):
                        with open(metadata_file) as f:
                            metrics = json.load(f)

                    if 'MLFLOW_TRACKING_URI' not in os.environ:
                        db_path = os.path.abspath(os.path.join(ROOT, 'mlflow.db'))
                        os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{db_path}'
                    
                    mlflow.set_experiment(MLFLOW_EXPERIMENT)
                    with mlflow.start_run(run_name=run_id):
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(k, float(v))
                        if os.path.exists(run_dir):
                            mlflow.log_artifacts(run_dir, artifact_path=run_id)
                except Exception:
                    logging.exception('MLflow logging failed')

        except Exception:
            logging.exception('Round %d failed', round_n)
        time.sleep(1)


if __name__ == '__main__':
    main_loop()
