#!/usr/bin/env python3
"""Clean raw teacher outputs into a deduplicated, token-limited training file."""
import os
import json
from pathlib import Path
import logging
import hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'ai-lab')
RAW_DIR = os.path.join(AI_LAB, 'datasets', 'raw')
CLEAN_DIR = os.path.join(AI_LAB, 'datasets', 'clean')
OUT_FILE = os.path.join(CLEAN_DIR, 'train.json')

logging.basicConfig(level=logging.INFO)

MAX_TOKENS = 2048


def normalize(sample: dict) -> dict:
    # trim long fields
    for k in ('instruction', 'context', 'response'):
        if k in sample and isinstance(sample[k], str):
            sample[k] = sample[k].strip()
            if len(sample[k]) > 20000:
                sample[k] = sample[k][:20000]
    return sample


def simple_entropy(s: str) -> float:
    # crude entropy proxy: fraction of unique chars
    if not s:
        return 0.0
    return len(set(s)) / max(1, len(s))


def main():
    Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)
    seen = set()
    out_samples = []
    for p in Path(RAW_DIR).glob('*.jsonl'):
        try:
            with p.open() as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    obj = normalize(obj)
                    key = hashlib.sha1((obj.get('instruction','') + obj.get('context','') + obj.get('response','')).encode()).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    # filter low-quality
                    if obj.get('quality_score', 0.0) < 0.1:
                        continue
                    if simple_entropy(obj.get('response','')) < 0.01:
                        continue
                    out_samples.append(obj)
        except Exception:
            logging.exception('Failed processing %s', p)

    # enforce token-like limit (rough by chars)
    filtered = []
    max_chars = MAX_TOKENS * 4
    chars = 0
    for s in out_samples:
        l = len(s.get('instruction','')) + len(s.get('response','')) + len(s.get('context',''))
        if l > max_chars:
            continue
        filtered.append(s)

    with open(OUT_FILE, 'w') as f:
        json.dump(filtered, f)

    logging.info('Wrote %d samples to %s', len(filtered), OUT_FILE)


if __name__ == '__main__':
    main()
