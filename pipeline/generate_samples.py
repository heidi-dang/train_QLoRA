import os
import json
import logging
import time
from pathlib import Path
import hashlib
import random
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import grok_client

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')
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

MAX_WORKERS = 3

sample_lock = Lock()


def read_filelist(path):
    if not os.path.exists(path):
        logging.warning('Filelist not found: %s', path)
        return []
    with open(path, 'r') as f:
        return [l.strip() for l in f if l.strip()]


def send_to_teacher(prompt: str, code: str) -> str:
    seed = hashlib.sha1((prompt + code[:100]).encode()).digest()
    rnd = int.from_bytes(seed[:4], 'big')
    random.seed(rnd)
    if 'tests' in prompt.lower():
        return 'def test_example():\n    assert True\n'
    if 'explain' in prompt.lower():
        return 'This code implements ... (auto-generated explanation)'
    if 'refactor' in prompt.lower():
        return code
    if 'optimize' in prompt.lower():
        return code
    if 'bug' in prompt.lower():
        return 'No obvious bugs found in this snippet.'
    return 'Generated sample.'


def score_sample(instr: str, context: str, response: str) -> float:
    s = 0.0
    if context:
        s += 0.3
    s += min(len(response) / 1000.0, 0.7)
    return float(min(1.0, s))


def generate_for_file(path: str, out_dir: str, teacher_client=None):
    try:
        text = Path(path).read_text(errors='ignore')
    except Exception:
        return 0
    if not text.strip():
        return 0
    
    code = text[:4000] + "\n... [truncated]" if len(text) > 4000 else text
    
    repo = 'unknown'
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
        prompt = f"{template}:\n\n{Path(path).name}\n---\n{code}"
        try:
            if teacher_client is not None:
                resp = teacher_client.generate(template, text)
            else:
                resp = send_to_teacher(template, text)
        except Exception:
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
        with sample_lock:
            with open(out_path, 'a') as f:
                f.write(json.dumps(sample) + '\n')
        produced += 1
    
    return produced


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    files = read_filelist(FILELIST)
    teacher_client = grok_client.get_client()
    
    total_files = len(files)
    total_samples = 0
    
    print(f'Processing {total_files} files with {MAX_WORKERS} workers...')
    
    with tqdm(total=total_files, desc='Generating', unit='file') as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(generate_for_file, f, RAW_DIR, teacher_client): f for f in files}
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    n = future.result()
                    total_samples += n
                except Exception as e:
                    logging.error(f'Failed {path}: {e}')
                    n = 0
                
                pbar.update(1)
                fname = Path(path).name[:25] + '..' if len(Path(path).name) > 25 else Path(path).name
                pbar.set_postfix_str(f'samples={total_samples} file={fname}')
    
    print()
    logging.info('Generated %d samples from %d files', total_samples, total_files)


if __name__ == '__main__':
    main()
