#!/usr/bin/env python3
"""Generate training samples from source files using a teacher model (Grok/xAI API).

This script reads ai-lab/datasets/repos_filelist.txt and emits JSONL files into ai-lab/datasets/raw/.
Each JSON line follows the schema defined in docs/implementation_continuous_training.md
"""
import os
import json
import logging
import time
from pathlib import Path
import hashlib
import random
from pipeline import grok_client

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'ai-lab')
FILELIST = os.path.join(AI_LAB, 'datasets', 'repos_filelist.txt')
RAW_DIR = os.path.join(AI_LAB, 'datasets', 'raw')

logging.basicConfig(level=logging.INFO)

PROMPT_TEMPLATES = [
    ('explain', 'Explain the following code in detail and describe what each function does'),
    ('tests', 'Write unit tests that validate the behavior of the following code'),
    ('refactor', 'Refactor the following code to improve readability and maintainability'),
    ('optimize', 'Suggest optimizations for the following code and provide the improved code'),
    ('bugfind', 'Find potential bugs or edge cases in the following code and explain how to fix them'),
]


def read_filelist(path):
    if not os.path.exists(path):
        logging.warning('Filelist not found: %s', path)
        return []
    with open(path, 'r') as f:
        return [l.strip() for l in f if l.strip()]


def send_to_teacher(prompt: str, code: str) -> str:
    """Stub for teacher model (Grok/xAI). Replace with real API client.

    For now we return deterministic synthetic outputs to allow pipeline continuity.
    """
    # Simple synthetic reply to allow pipeline to run without external API
    seed = hashlib.sha1((prompt + code[:100]).encode()).digest()
    rnd = int.from_bytes(seed[:4], 'big')
    random.seed(rnd)
    # fabricate response
    if 'tests' in prompt.lower():
        return 'def test_example():\n    assert True\n'
    if 'explain' in prompt.lower():
        return 'This code implements ... (auto-generated explanation)'
    if 'refactor' in prompt.lower():
        return code  # return original as placeholder
    if 'optimize' in prompt.lower():
        return code  # placeholder
    if 'bug' in prompt.lower():
        return 'No obvious bugs found in this snippet.'
    return 'Generated sample.'


def configure_grok_from_env():
    """If GROK_API_KEY is present in environment, attempt to use a Grok client.

    This is a placeholder configuration function — in production replace with a
    proper Grok/xAI SDK client and implement batching and rate-limiting.
    """
    api_key = os.environ.get('GROK_API_KEY')
    if not api_key:
        return None
    # Placeholder: return a simple function wrapper that would call the real API
    def real_call(template, text):
        # Here you'd call the Grok API; we keep stub to avoid external calls
        return send_to_teacher(template, text)

    return real_call


def score_sample(instr: str, context: str, response: str) -> float:
    # lightweight heuristic score: longer response and non-empty context -> higher
    s = 0.0
    if context:
        s += 0.3
    s += min(len(response) / 1000.0, 0.7)
    return float(min(1.0, s))


def generate_for_file(path: str, out_dir: str, teacher_client=None):
    try:
        text = Path(path).read_text(errors='ignore')
    except Exception:
        logging.exception('Failed to read %s', path)
        return 0
    if not text.strip():
        return 0
    repo = 'unknown'
    # attempt to infer repo from path: ai-lab/repos/<owner_repo>/...
    parts = Path(path).parts
    if 'ai-lab' in parts and 'repos' in parts:
        try:
            idx = parts.index('repos')
            repo = parts[idx + 1]
        except Exception:
            repo = 'unknown'

    os.makedirs(out_dir, exist_ok=True)
    produced = 0
    for tag, template in PROMPT_TEMPLATES:
        prompt = f"{template}:\n\n{Path(path).name}\n---\n{ text[:2000] }"
        try:
            if teacher_client is not None:
                resp = teacher_client.generate(template, text)
            else:
                resp = send_to_teacher(template, text)
        except Exception:
            logging.exception('Teacher failed for %s', path)
            resp = ''
        sample = {
            'instruction': template,
            'context': text[:8192],
            'response': resp,
            'language': 'python' if path.endswith('.py') else 'cpp' if path.endswith(('.cpp', '.cc', '.hpp', '.h', '.c')) else 'mixed',
            'source_repo': repo,
            'quality_score': score_sample(template, text, resp),
        }
        fname = hashlib.sha1((path + tag).encode()).hexdigest() + '.jsonl'
        out_path = os.path.join(out_dir, fname)
        with open(out_path, 'a') as f:
            f.write(json.dumps(sample) + '\n')
        produced += 1
    return produced


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    files = read_filelist(FILELIST)
    # configure optional Grok client if available
    teacher_client = grok_client.get_client()
    total = 0
    for p in files:
        n = generate_for_file(p, RAW_DIR, teacher_client=teacher_client)
        total += n
    logging.info('Generated %d samples', total)


if __name__ == '__main__':
    main()
