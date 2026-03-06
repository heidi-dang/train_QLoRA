#!/usr/bin/env python3
"""Main training loop: coordinates scraping, generation, cleaning, training, evaluation and checkpointing.

This script is designed to be started as a background process by the FastAPI server.
"""
import os
import time
import logging
import subprocess
import json
from pathlib import Path
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'ai-lab')
STOP_FILE = os.path.join(AI_LAB, 'STOP')
CHECKPOINTS = os.path.join(AI_LAB, 'checkpoints')
MLFLOW_EXPERIMENT = 'continuous_training'

logging.basicConfig(level=logging.INFO)


def stage_scrape():
    logging.info('Stage: scrape')
    subprocess.call(['python3', os.path.join(ROOT, 'pipeline', 'scrape_repos.py')])


def stage_generate():
    logging.info('Stage: generate samples')
    subprocess.call(['python3', os.path.join(ROOT, 'pipeline', 'generate_samples.py')])


def stage_clean():
    logging.info('Stage: clean dataset')
    subprocess.call(['python3', os.path.join(ROOT, 'pipeline', 'clean_dataset.py')])


def stage_train(round_n: int):
    logging.info('Stage: train (round %d) - placeholder training', round_n)
    # In production: invoke training script that uses bitsandbytes + PEFT + transformers
    ckpt_dir = os.path.join(CHECKPOINTS, f'adapter_round_{round_n}')
    os.makedirs(ckpt_dir, exist_ok=True)
    # call training scaffold (train_q_lora) which will use real training if deps present
    train_script = os.path.join(ROOT, 'pipeline', 'train_q_lora.py')
    try:
        subprocess.check_call(['python3', train_script, os.path.join(AI_LAB, 'datasets', 'clean', 'train.json'), ckpt_dir])
    except subprocess.CalledProcessError:
        logging.exception('Training script failed; creating metadata only')
    meta = {'round': round_n, 'base_model': 'mistralai/Mistral-7B-Instruct-v0.2'}
    with open(os.path.join(ckpt_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)
    logging.info('Wrote checkpoint to %s', ckpt_dir)


def stage_evaluate(round_n: int):
    logging.info('Stage: evaluate')
    subprocess.call(['python3', os.path.join(ROOT, 'pipeline', 'evaluate_model.py'), str(round_n)])


def main_loop():
    round_n = 0
    Path(CHECKPOINTS).mkdir(parents=True, exist_ok=True)
    while True:
        if os.path.exists(STOP_FILE):
            logging.info('STOP file exists - exiting loop')
            break
        round_n += 1
        try:
            stage_scrape()
            stage_generate()
            stage_clean()
            stage_train(round_n)
            stage_evaluate(round_n)
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
