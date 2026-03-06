"""QLoRA training script scaffold.

This script implements a safe scaffold for training a QLoRA adapter using
transformers + bitsandbytes + peft. In this environment it will detect missing
dependencies and run a simulated training loop instead.

Configuration is read from TRAIN_CONFIG dict or environment variables.
"""
import os
import time
import logging

logging.basicConfig(level=logging.INFO)

TRAIN_CONFIG = {
    'base_model': os.environ.get('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2'),
    'r': int(os.environ.get('QLORA_R', '64')),
    'alpha': int(os.environ.get('QLORA_ALPHA', '16')),
    'dropout': float(os.environ.get('QLORA_DROPOUT', '0.05')),
    'batch': int(os.environ.get('BATCH', '1')),
    'grad_accum': int(os.environ.get('GRAD_ACCUM', '8')),
    'lr': float(os.environ.get('LR', '2e-4')),
}


def real_train(train_file: str, output_dir: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import peft
        import bitsandbytes as bnb
    except Exception:
        logging.warning('Missing heavy dependencies; running simulated training')
        simulated_train(train_file, output_dir)
        return

    # Placeholder: implement real QLoRA training here in production.
    logging.info('Dependencies present, but real training is not implemented in this scaffold.')
    simulated_train(train_file, output_dir)


def simulated_train(train_file: str, output_dir: str):
    logging.info('Simulated training on %s -> %s', train_file, output_dir)
    # Simulate epochs and create a fake adapter file
    Path = __import__('pathlib').Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 3):
        logging.info('Simulated epoch %d/2', epoch)
        time.sleep(1)
    adapter_path = os.path.join(output_dir, 'adapter.safetensors')
    with open(adapter_path, 'wb') as f:
        f.write(b'')
    logging.info('Wrote simulated adapter to %s', adapter_path)


if __name__ == '__main__':
    import sys
    train_file = sys.argv[1] if len(sys.argv) > 1 else 'ai-lab/datasets/clean/train.json'
    out = sys.argv[2] if len(sys.argv) > 2 else 'ai-lab/checkpoints/adapter_sim'
    real_train(train_file, out)
