#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
VENV_PY="$VENV/bin/python"

LOGS_DIR="$ROOT/logs"
PIDS_DIR="$ROOT/pids"

mkdir -p "$LOGS_DIR" "$PIDS_DIR"

usage() {
  cat <<'USAGE'
Usage:
  ./parallel_data_gen.sh start
  ./parallel_data_gen.sh stop
  ./parallel_data_gen.sh status

What it does:
  Runs a GitHub-based repo search + scrape + sample generation in the background,
  designed to run in parallel with training.

Requirements:
  - venv must exist at ./venv
  - optional: GITHUB_TOKEN env var for higher GitHub API rate limits
USAGE
}

require_venv() {
  if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: venv python not found at $VENV_PY"
    echo "Run: ./run.sh setup"
    exit 1
  fi
}

prompt_topic() {
  echo "Select what data you want to generate:"
  echo "  1) devops"
  echo "  2) github_workflows"
  echo "  3) penetration_security"
  echo "  4) hacking"
  echo "  5) whitehat_blackhat"
  echo "  6) custom"
  read -r -p "Enter choice [1-6]: " choice

  case "${choice:-}" in
    1) echo "devops" ;;
    2) echo "github_workflows" ;;
    3) echo "penetration_security" ;;
    4) echo "hacking" ;;
    5) echo "whitehat_blackhat" ;;
    6)
      read -r -p "Enter custom topic (free text): " custom
      custom="${custom// /_}"
      echo "${custom:-custom}" ;;
    *)
      echo "devops" ;;
  esac
}

topic_to_query() {
  case "$1" in
    devops) echo "devops infrastructure as code terraform kubernetes ci cd" ;;
    github_workflows) echo "github actions workflow ci cd" ;;
    penetration_security) echo "penetration testing security audit vulnerability" ;;
    hacking) echo "ctf exploit poc reverse engineering" ;;
    whitehat_blackhat) echo "ethical hacking red team blue team" ;;
    *) echo "$1" ;;
  esac
}

prompt_languages() {
  read -r -p "Programming languages to search on GitHub (comma-separated) [python,javascript]: " langs
  langs="${langs:-python,javascript}"
  echo "$langs"
}

prompt_run_name() {
  read -r -p "Name for output directory (will be created under data/ai-lab/datasets/parallel_runs/) [parallel_run_$(date +%Y%m%d_%H%M%S)]: " name
  name="${name:-parallel_run_$(date +%Y%m%d_%H%M%S)}"
  name="${name// /_}"
  echo "$name"
}

start() {
  require_venv

  if [ -f "$PIDS_DIR/parallel_data_gen.pid" ] && kill -0 "$(cat "$PIDS_DIR/parallel_data_gen.pid")" 2>/dev/null; then
    echo "parallel_data_gen already running (pid=$(cat "$PIDS_DIR/parallel_data_gen.pid"))"
    exit 0
  fi

  topic="$(prompt_topic)"
  query="$(topic_to_query "$topic")"
  languages_csv="$(prompt_languages)"
  run_name="$(prompt_run_name)"

  run_dir="$ROOT/data/ai-lab/datasets/parallel_runs/$run_name"
  repos_dir="$run_dir/repos"
  raw_dir="$run_dir/raw"
  repos_list="$run_dir/repos.txt"
  filelist_out="$run_dir/repos_filelist.txt"

  mkdir -p "$run_dir" "$repos_dir" "$raw_dir"

  log_file="$LOGS_DIR/parallel_data_gen_${run_name}.log"

  echo "Starting parallel data generation:"
  echo "  topic:      $topic"
  echo "  query:      $query"
  echo "  languages:  $languages_csv"
  echo "  output:     $run_dir"
  echo "  log:        $log_file"

  export PARALLEL_DATA_GEN_ROOT="$ROOT"
  export PARALLEL_DATA_GEN_RUN_DIR="$run_dir"
  export PARALLEL_DATA_GEN_REPOS_DIR="$repos_dir"
  export PARALLEL_DATA_GEN_RAW_DIR="$raw_dir"
  export PARALLEL_DATA_GEN_REPO_LIST="$repos_list"
  export PARALLEL_DATA_GEN_FILELIST_OUT="$filelist_out"
  export PARALLEL_DATA_GEN_QUERY="$query"
  export PARALLEL_DATA_GEN_LANGS="$languages_csv"
  # Ensure parallel gen uses same teacher provider/model as main pipeline
  export TEACHER_PROVIDER="${TEACHER_PROVIDER:-}"
  export TEACHER_MODEL="${TEACHER_MODEL:-}"
  export TEACHER_API_KEY="${TEACHER_API_KEY:-}"
  export TEACHER_COPILOT_MODEL="${TEACHER_COPILOT_MODEL:-}"

  nohup "$VENV_PY" -u - <<'PY' >"$log_file" 2>&1 &
import os
import re
from pathlib import Path

ROOT = os.path.abspath(os.environ["PARALLEL_DATA_GEN_ROOT"])
RUN_DIR = os.path.abspath(os.environ["PARALLEL_DATA_GEN_RUN_DIR"])
REPOS_DIR = os.path.abspath(os.environ["PARALLEL_DATA_GEN_REPOS_DIR"])
RAW_DIR = os.path.abspath(os.environ["PARALLEL_DATA_GEN_RAW_DIR"])
REPO_LIST = os.path.abspath(os.environ["PARALLEL_DATA_GEN_REPO_LIST"])
FILELIST_OUT = os.path.abspath(os.environ["PARALLEL_DATA_GEN_FILELIST_OUT"])

QUERY = os.environ.get("PARALLEL_DATA_GEN_QUERY", "").strip()
LANGS = [l.strip() for l in os.environ.get("PARALLEL_DATA_GEN_LANGS", "").split(",") if l.strip()]

# 1) Search repos
from github_search import GitHubSearcher

searcher = GitHubSearcher()
repos = searcher.search_repos(query=QUERY, languages=LANGS)
searcher.save_repo_list(repos, REPO_LIST)

# 2) Scrape repos (clone/update) + create filelist, with custom dirs
from pipeline import scrape_repos as sr

sr.REPOS_DIR = REPOS_DIR
sr.REPO_LIST = REPO_LIST
sr.FILELIST_OUT = FILELIST_OUT

# Narrow file extensions based on language selection
lang_ext = {
  "python": {".py"},
  "javascript": {".js", ".jsx", ".ts", ".tsx"},
  "typescript": {".ts", ".tsx"},
  "go": {".go"},
  "rust": {".rs"},
  "cpp": {".cpp", ".cc", ".c", ".h", ".hpp"},
  "c": {".c", ".h"},
}
include = set()
for l in LANGS:
  include |= lang_ext.get(l.lower(), set())
if include:
  sr.INCLUDE_EXT = include

sr.main()

# 3) Generate samples into custom RAW_DIR using custom FILELIST
from pipeline import generate_samples as gs

gs.FILELIST = FILELIST_OUT
gs.RAW_DIR = RAW_DIR

# If you want more parallelism for sample gen, bump this.
# Keep conservative so it doesn't starve training.
gs.MAX_WORKERS = int(os.environ.get("PARALLEL_DATA_GEN_WORKERS", "2"))

gs.main()

print("DONE. Output written to:", RUN_DIR)
PY

  pid=$!
  echo "$pid" > "$PIDS_DIR/parallel_data_gen.pid"
  echo "Started (pid=$pid)"
}

stop() {
  if [ ! -f "$PIDS_DIR/parallel_data_gen.pid" ]; then
    echo "No pid file found ($PIDS_DIR/parallel_data_gen.pid)"
    exit 0
  fi

  pid="$(cat "$PIDS_DIR/parallel_data_gen.pid" || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping parallel_data_gen (pid=$pid)"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  else
    echo "parallel_data_gen not running (stale pid=$pid)"
  fi

  rm -f "$PIDS_DIR/parallel_data_gen.pid"
}

status() {
  if [ -f "$PIDS_DIR/parallel_data_gen.pid" ]; then
    pid="$(cat "$PIDS_DIR/parallel_data_gen.pid" || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "parallel_data_gen running (pid=$pid)"
      exit 0
    fi
  fi
  echo "parallel_data_gen not running"
}

cmd="${1:-}"
case "$cmd" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 2
    ;;
esac
