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
try:
    from pipeline import telemetry
except ImportError:
    telemetry = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')
FILELIST = os.path.join(AI_LAB, 'datasets', 'repos_filelist.txt')
RAW_DIR = os.path.join(AI_LAB, 'datasets', 'raw')

logging.basicConfig(level=logging.INFO)

PREMIUM_SYSTEM_INSTRUCTIONS = (
    "You are a meticulous senior engineer and technical writer. "
    "You must be correct, concrete, and testable. "
    "Prefer runnable code and clearly separated sections. "
    "Do not invent missing APIs; if assumptions are needed, state them explicitly."
)

PREMIUM_OUTPUT_CONTRACT = (
    "Return your answer in the following format exactly:\n"
    "<ANSWER>\n"
    "...your content...\n"
    "</ANSWER>\n"
    "Do not include anything outside <ANSWER> tags."
)

PROMPT_TEMPLATES = [
    (
        'explain_deep',
        "Write a deep explanation of the code with a focus on invariants, data flow, and failure modes. "
        "Include: (1) high-level summary, (2) line-by-line walkthrough of key functions, "
        "(3) edge cases, (4) complexity/perf notes, (5) suggested improvements.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'tests_premium',
        "Write high-quality unit tests for the code. Requirements: "
        "(1) tests must be runnable, (2) cover normal + edge cases, "
        "(3) include at least one property-based or parameterized test if applicable, "
        "(4) mock network/filesystem/time where appropriate, (5) include clear assertions.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'code_review',
        "Perform a professional code review. Include: "
        "(1) correctness issues, (2) security issues, (3) performance issues, "
        "(4) readability/maintainability issues, (5) suggested refactor plan, "
        "(6) priority-ranked checklist of fixes.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'refactor_premium',
        "Refactor the code for clarity and maintainability. Requirements: "
        "(1) preserve behavior, (2) improve naming, structure, and cohesion, "
        "(3) remove duplication, (4) add small helper functions, "
        "(5) output the full refactored code.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'optimize_premium',
        "Optimize the code. Include: "
        "(1) a short list of bottlenecks, (2) improved implementation, "
        "(3) complexity comparison, (4) any trade-offs.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'bug_hunt',
        "Find subtle bugs and edge cases. Requirements: "
        "(1) list at least 5 potential issues, (2) show minimal reproducer inputs if possible, "
        "(3) provide concrete fixes.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'security_audit',
        "Do a security audit of the code. Include: "
        "(1) threat model assumptions, (2) vulnerabilities (injection, secrets, auth, unsafe deserialization, etc.), "
        "(3) mitigation steps, (4) safe-by-default code changes if relevant.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'debugging_trace',
        "Simulate a debugging session: "
        "(1) identify likely failure points, (2) propose targeted logging, "
        "(3) propose a minimal failing test, (4) provide a step-by-step fix plan.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
    (
        'architecture_notes',
        "Write architecture notes: "
        "(1) components, (2) responsibilities, (3) interfaces, (4) key invariants, "
        "(5) suggested modularization.\n" + PREMIUM_OUTPUT_CONTRACT,
    ),
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
        prompt = (
            f"{PREMIUM_SYSTEM_INSTRUCTIONS}\n\n"
            f"FILE: {Path(path).name}\n"
            f"REPO: {repo}\n\n"
            f"INSTRUCTION:\n{template}\n\n"
            f"CODE:\n{code}"
        )
        try:
            if teacher_client is not None:
                resp = teacher_client.generate(prompt, text)
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
    
    if telemetry:
        telemetry.update_progress("generate", 0, 4, 0, total_files)
    
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
                
                if telemetry:
                    telemetry.update_progress("generate", 0, 4, pbar.n, total_files)
    
    if telemetry:
        telemetry.update_progress("generate", 0, 4, total_files, total_files)
    
    print()
    logging.info('Generated %d samples from %d files', total_samples, total_files)


if __name__ == '__main__':
    main()
