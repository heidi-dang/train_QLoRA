#!/usr/bin/env python3
"""Run a single training round (scrape -> generate -> clean -> train -> evaluate -> log).

This script imports the orchestration functions from train_loop and runs them once for a chosen round number.
"""
import sys
import os
from pathlib import Path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from pipeline import train_loop
import json
import logging

logging.basicConfig(level=logging.INFO)


def run_once(round_n: int):
    train_loop.stage_scrape()
    train_loop.stage_generate()
    train_loop.stage_clean()
    train_loop.stage_train(round_n)
    train_loop.stage_evaluate(round_n)
    # MLflow logging (reuse train_loop logic)
    eval_file = os.path.join(train_loop.AI_LAB, 'evaluation', f'results_round_{round_n}.json')
    metrics = {}
    try:
        with open(eval_file) as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    if train_loop.mlflow is not None:
        try:
            if 'MLFLOW_TRACKING_URI' not in os.environ:
                db_path = os.path.abspath(os.path.join(ROOT, 'mlflow.db'))
                os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{db_path}'
            train_loop.mlflow.set_experiment(train_loop.MLFLOW_EXPERIMENT)
            with train_loop.mlflow.start_run(run_name=f'round_{round_n}'):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        train_loop.mlflow.log_metric(k, float(v))
                ckpt_dir = os.path.join(train_loop.CHECKPOINTS, f'adapter_round_{round_n}')
                if os.path.exists(ckpt_dir):
                    try:
                        train_loop.mlflow.log_artifacts(ckpt_dir, artifact_path=f'checkpoint_round_{round_n}')
                    except Exception:
                        logging.exception('Failed to log artifacts to MLflow')
        except Exception:
            logging.exception('MLflow logging failed')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_once(n)
