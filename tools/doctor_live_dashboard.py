#!/usr/bin/env python3
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_FILE = os.path.join(ROOT, "dashboard", "app.py")


def check_no_mixed_print_rendering():
    if not os.path.exists(DASHBOARD_FILE):
        return False, "dashboard/app.py not found"
    try:
        with open(DASHBOARD_FILE, "r") as f:
            content = f.read()
        if "def render(self)" not in content:
            return False, "Missing render method"
        if "RENDER_INTERVAL" not in content:
            return False, "Missing fixed render cadence"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_log_buffer():
    if not os.path.exists(DASHBOARD_FILE):
        return False, "dashboard/app.py not found"
    try:
        with open(DASHBOARD_FILE, "r") as f:
            content = f.read()
        if "deque" not in content:
            return False, "Missing deque for log buffer"
        if "maxlen=" not in content:
            return False, "Missing maxlen for ring buffer"
        if "append_logs" not in content:
            return False, "Missing append_logs method"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_progress_fields():
    if not os.path.exists(DASHBOARD_FILE):
        return False, "dashboard/app.py not found"
    try:
        with open(DASHBOARD_FILE, "r") as f:
            content = f.read()
        required = ["stage_percent", "overall_percent", "completed_units", "total_units"]
        for field in required:
            if field not in content:
                return False, f"Missing field: {field}"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_teacher_cost_fields():
    if not os.path.exists(DASHBOARD_FILE):
        return False, "dashboard/app.py not found"
    try:
        with open(DASHBOARD_FILE, "r") as f:
            content = f.read()
        required = ["provider", "model", "prompt_tokens", "spend_usd"]
        for field in required:
            if field not in content:
                return False, f"Missing field: {field}"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_renderer_separation():
    if not os.path.exists(DASHBOARD_FILE):
        return False, "dashboard/app.py not found"
    try:
        with open(DASHBOARD_FILE, "r") as f:
            content = f.read()
        if "class RunState" not in content:
            return False, "Missing RunState class"
        if "class DashboardRenderer" not in content:
            return False, "Missing DashboardRenderer class"
        if "def load_telemetry" not in content:
            return False, "Missing load_telemetry method"
        if "def _render" not in content:
            return False, "Missing _render method"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_fixed_cadence():
    if not os.path.exists(DASHBOARD_FILE):
        return False, "dashboard/app.py not found"
    try:
        with open(DASHBOARD_FILE, "r") as f:
            content = f.read()
        if "RENDER_INTERVAL" not in content:
            return False, "Missing RENDER_INTERVAL"
        if "threading" not in content and "time.sleep" not in content:
            return False, "Missing render loop"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def check_terminal_control():
    if not os.path.exists(DASHBOARD_FILE):
        return False, "dashboard/app.py not found"
    try:
        with open(DASHBOARD_FILE, "r") as f:
            content = f.read()
        if "ANSI_CURSOR_HIDE" not in content:
            return False, "Missing cursor hide"
        if "hide_cursor" not in content:
            return False, "Missing hide_cursor method"
        if "show_cursor" not in content:
            return False, "Missing show_cursor method"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    checks = [
        ("no_mixed_rendering", check_no_mixed_print_rendering),
        ("log_buffer", check_log_buffer),
        ("progress_fields", check_progress_fields),
        ("teacher_cost_fields", check_teacher_cost_fields),
        ("renderer_separation", check_renderer_separation),
        ("fixed_cadence", check_fixed_cadence),
        ("terminal_control", check_terminal_control),
    ]

    all_passed = True
    print("=" * 60)
    print("Live Dashboard Doctor Checks")
    print("=" * 60)

    for name, check_func in checks:
        passed, msg = check_func()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All live dashboard checks passed!")
        return 0
    else:
        print("Some live dashboard checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
