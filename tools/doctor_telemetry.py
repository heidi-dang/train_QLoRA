#!/usr/bin/env python3
import os
import json
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TELEMETRY_FILE = os.path.join(ROOT, 'state', 'telemetry.json')
TELEMETRY_MODULE = os.path.join(ROOT, 'pipeline', 'telemetry.py')


def check_telemetry_file():
    if not os.path.exists(TELEMETRY_FILE):
        return False, "telemetry.json not found"
    try:
        with open(TELEMETRY_FILE, 'r') as f:
            data = json.load(f)
        required_fields = ["status", "usage", "current_stage", "stage_percent"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing field: {field}"
        return True, "OK"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_telemetry_module():
    if not os.path.exists(TELEMETRY_MODULE):
        return False, "pipeline/telemetry.py not found"
    try:
        with open(TELEMETRY_MODULE, 'r') as f:
            content = f.read()
        required_funcs = ["record_api_call", "update_progress", "get_telemetry", "init_run"]
        for func in required_funcs:
            if f"def {func}" not in content:
                return False, f"Missing function: {func}"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_pricing_config():
    if not os.path.exists(TELEMETRY_MODULE):
        return False, "telemetry.py not found"
    try:
        with open(TELEMETRY_MODULE, 'r') as f:
            content = f.read()
        if "PRICING" not in content:
            return False, "PRICING config not found"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_dashboard_wiring():
    dashboard_file = os.path.join(ROOT, 'dashboard', 'app.py')
    if not os.path.exists(dashboard_file):
        return False, "dashboard/app.py not found"
    try:
        with open(dashboard_file, 'r') as f:
            content = f.read()
        if "load_telemetry" not in content:
            return False, "Dashboard doesn't use load_telemetry"
        if "spend_usd" not in content:
            return False, "Dashboard doesn't show spend"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    checks = [
        ("telemetry_file", check_telemetry_file),
        ("telemetry_module", check_telemetry_module),
        ("pricing_config", check_pricing_config),
        ("dashboard_wiring", check_dashboard_wiring),
    ]

    all_passed = True
    print("=" * 50)
    print("Teacher Cost & Progress Dashboard Doctor")
    print("=" * 50)

    for name, check_func in checks:
        passed, msg = check_func()
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {name}: {msg}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("✅ All checks passed!")
        return 0
    else:
        print("❌ Some checks failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
