# Continuous Premium Coding-Model Training Pipeline

This document describes the implementation of a continuous training pipeline that automatically builds a premium multi-language coding dataset and trains QLoRA adapters in an automated loop.

## 1. Pipeline architecture

- Orchestrator (pipeline/train_loop.py): main loop that coordinates scraping, teacher sample generation, cleaning, training, evaluation and checkpointing.
- Scraper (pipeline/scrape_repos.py): clones/updates repos listed in ai-lab/datasets/repos_premium.txt and extracts source files.
- Teacher generator (pipeline/generate_samples.py): uses a teacher (Grok or similar LLM) to generate instruction/response pairs per code file and writes JSONL into ai-lab/datasets/raw/.
- Cleaner (pipeline/clean_dataset.py): deduplicates, filters, normalizes and enforces token limits to produce ai-lab/datasets/clean/train.json.
- Trainer (pipeline/train_loop.py -> train_adapter): trains QLoRA adapters on the cleaned dataset, logs to MLflow/TensorBoard, and writes checkpoints into ai-lab/checkpoints/.
- Evaluator (pipeline/evaluate_model.py): runs benchmarks (HumanEval-style, C++ compile tests, reasoning prompts) and writes results to ai-lab/evaluation/.
- Control API (server/train_api.py): FastAPI endpoints to start/stop the background training loop and query status/metrics.
- Doctor/checks (tools/doctor_continuous_training.py): basic health checks and validations.

Components communicate through the filesystem (ai-lab/*) and MLflow for metrics.

## 2. Dataset schema

Each sample is a JSON object (JSONL files in datasets/raw/) with the following fields:

{
 "instruction": "...",
 "context": "...",
 "response": "...",
 "language": "python|cpp|mixed",
 "source_repo": "https://...",
 "quality_score": float
}

quality_score is in [0.0, 1.0] and is used to filter low-quality samples during cleaning.

File layout (root of project):

ai-lab/
  repos/                 # cloned repos (git clones)
  datasets/
    raw/                 # teacher outputs as JSONL
    clean/               # cleaned dataset (train.json)
    final/               # optionally formatted final dataset artifacts
    repos_premium.txt    # curated repo list
  checkpoints/
  evaluation/
  logs/

## 3. Repo scraping strategy

- Maintain ai-lab/datasets/repos_premium.txt as the canonical curated list.
- Clone repositories into ai-lab/repos/ using shallow clones when possible (git clone --depth 1).
- Update existing clones with git fetch + git reset --hard origin/HEAD when available.
- Filter source files by extension: .py, .cpp, .cc, .h, .hpp, .c, .rs, .go.
- Skip and ignore directories: tests, test, build, node_modules, .git, third_party.
- For large monorepos allow sampling (limit files per repo) to avoid overloading upstream.

Security / rate limits: respect GitHub API rate limits and prefer mirror caches when available. Use retry/backoff on network failures.

## 4. Teacher generation logic

- For each code file, create several prompt templates: explain code, write tests, refactor, optimize, and bug-finding.
- Send prompts to the teacher model (Grok / xAI API) and store teacher outputs in ai-lab/datasets/raw/<repo>_<file>.jsonl.
- If the teacher is unavailable, fallback to lightweight heuristics to produce synthetic training samples so the pipeline can continue.
- Each sample is scored with a lightweight heuristic (length, token-entropy, presence of code tokens) and annotated with quality_score.

Note: the repository includes a stub implementation for Grok—replace with an API client and proper credentials in production.

Grok (xAI API) integration notes:

- The included pipeline uses a deterministic teacher stub to keep the repository runnable without credentials. To integrate the real Grok API:
  - add GROK_API_KEY to environment and implement an API client wrapper (e.g. pipeline/grok_client.py) and replace the stub send_to_teacher with the real call
  - implement batching, concurrency, and retry/backoff for 429/5xx responses
  - add caching of teacher responses to avoid repeat API calls for identical prompts
  - monitor cost and rate limits; set quotas and prioritization for high-quality repos

## 5. Training loop lifecycle

- The training loop runs continuously until the STOP file (ai-lab/STOP) is created or the API /train/stop is called.
- Loop phases per round:
  1. scrape repos
  2. generate samples (teacher)
  3. clean dataset
  4. train QLoRA adapter
  5. evaluate model
  6. checkpoint adapter and upload metrics
  7. sleep/backoff or continue immediately

- Each completed round increments the round counter and saves a named checkpoint: ai-lab/checkpoints/adapter_round_{n}/

## 6. Failure recovery

- Each stage has try/except and writes clear logs to ai-lab/logs/.
- On transient failures (network, API) the loop retries with exponential backoff. On persistent failures it marks the round as failed, writes diagnostics to ai-lab/logs/ and continues to the next round (configurable).
- Checkpoints are flushed after training completes successfully. Partial training state is removed to avoid corrupt checkpoints.

## 7. Checkpoints & metrics

- Checkpoint directory structure: ai-lab/checkpoints/adapter_round_{n}/adapter.bin (or adapter.safetensors) + metadata.json
- Metrics (training loss, eval metrics, GPU utilization) are logged to MLflow under experiment "continuous_training" and to TensorBoard for visualization.

## 8. Safety & Controls

- The pipeline checks for a kill-switch file ai-lab/STOP at the start and end of each round.
- The Control API exposes endpoints to start/stop the loop and query status/metrics.
- Doctor module provides health-check endpoints and CLI checks to verify prerequisites.

## 9. Scaling & production notes

- Use orchestration (Kubernetes, systemd timers, or a supervisor) for production-grade reliability.
- Use distributed training (deepspeed/accelerate) for real QLoRA runs on multiple GPUs.
- Add access control to the FastAPI server.

## 10. Next steps for production

- Replace teacher stubs with a robust Grok/xAI client and batching.
- Implement robust repository prioritization and sampling to ensure a high-quality, diverse dataset.
- Integrate real QLoRA training (bitsandbytes + transformers + peft) and validate adapter compatibility with the base model.

---
Document created to provide a clear implementation plan and reference for the included pipeline scripts.
