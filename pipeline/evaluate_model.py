#!/usr/bin/env python3
"""Run simple evaluation benchmarks: HumanEval-style Python prompts and C++ compile tests."""
import os
import json
import logging
import subprocess
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'ai-lab')
EVAL_DIR = os.path.join(AI_LAB, 'evaluation')

logging.basicConfig(level=logging.INFO)


def run_cpp_compile_test(code: str) -> bool:
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
    src = os.path.join(EVAL_DIR, 'temp.cpp')
    exe = os.path.join(EVAL_DIR, 'temp.out')
    with open(src, 'w') as f:
        f.write(code)
    try:
        subprocess.check_call(['g++', src, '-O2', '-std=c++17', '-o', exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def evaluate_round(round_n: int) -> dict:
    # placeholder evaluation: sample checks
    results = {'round': round_n, 'human_eval_pass': 0.0, 'cpp_compile_rate': 0.0}
    # In production: run model on held-out prompts. Here we run a tiny cpp compile sanity check.
    code = 'int main(){return 0;}'
    ok = run_cpp_compile_test(code)
    results['cpp_compile_rate'] = 1.0 if ok else 0.0
    out = os.path.join(EVAL_DIR, f'results_round_{round_n}.json')
    with open(out, 'w') as f:
        json.dump(results, f)
    logging.info('Wrote evaluation results to %s', out)
    return results


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    evaluate_round(n)
