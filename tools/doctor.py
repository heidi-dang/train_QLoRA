#!/usr/bin/env python3
"""Unified doctor tool that runs all checks including continuous training doctor."""
import json
import argparse
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
import doctor_continuous_training as dct


def check_dashboard_health():
    """Check for telemetry state file and its validity."""
    autotrain_dir = os.environ.get("AUTOTRAIN_DIR", "/home/heidi/work/train_QLoRA/data/ai-lab")
    run_id = os.environ.get("RUN_ID", "latest")
    
    # Check general state.json
    state_file = Path(autotrain_dir) / "runs" / run_id / "state.json"
    if not state_file.exists():
        # Try finding any state.json if 'latest' not found
        runs_dir = Path(autotrain_dir) / "runs"
        if runs_dir.exists():
            for rd in runs_dir.iterdir():
                if (rd / "state.json").exists():
                    state_file = rd / "state.json"
                    break

    if not state_file.exists():
        return "ERROR: No state.json found for telemetry."

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        
        if "sequence_number" not in state:
            return "WARNING: state.json missing sequence_number (v1 format)."
        
        return "OK"
    except Exception as e:
        return f"ERROR: Failed to read state.json: {e}"

def check_polling_wrappers():
    """Verify that dashboard codes use sequence-guarded polling."""
    dashboard_files = [
        "/home/heidi/work/train_QLoRA/dashboard/app.py",
        "/home/heidi/work/train_QLoRA/dashboard/heidi_dashboard.py"
    ]
    
    missing = []
    for fpath in dashboard_files:
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            content = f.read()
        if "sequence_number" not in content:
            missing.append(os.path.basename(fpath))
            
    if missing:
        return f"WARNING: {', '.join(missing)} might be missing sequence-guarded polling."
    return "OK"

def check_teacher_config():
    """Verify primary and failback teacher models."""
    primary = os.environ.get("TEACHER_MODEL", "github-copilot/gpt-5.3-codex")
    failback = os.environ.get("TEACHER_FAILBACK_MODEL", "xai/grok-4-1-fast")
    
    issues = []
    if primary != "github-copilot/gpt-5.3-codex":
        issues.append(f"Primary teacher mismatch: {primary}")
    if failback != "xai/grok-4-1-fast":
        issues.append(f"Failback teacher mismatch: {failback}")
        
    if issues:
        return "WARNING: " + "; ".join(issues)
    return "OK (Primary: github-copilot/gpt-5.3-codex, Failback: xai/grok-4-1-fast)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = {
        'continuous_training': dct.main.__doc__ is not None,
        'checks': None,
    }
    # reuse functions from doctor_continuous_training
    report['checks'] = {
        'repo_list': dct.check_repo_list(),
        'dataset_raw_present': dct.check_dataset_generated(),
        'mlflow_configured': dct.check_mlflow(),
        'gpu_available': dct.check_gpu(),
        'training_loop_pidfile': dct.check_training_loop_alive(),
        'dashboard_telemetry': check_dashboard_health(),
        'polling_verification': check_polling_wrappers(),
        'teacher_config': check_teacher_config(),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report['checks'].items():
            print(f"{k}: {v}")


if __name__ == '__main__':
    main()
