# Implementation: Dashboard Polish and Realtime Data Correctness

## Overview
This document outlines the changes made to unify the truth source for the terminal dashboards and improve their real-time accuracy and visual polish.

## Core Architectural Changes

### 1. Centralized Telemetry and State Management (`heidi_telemetry.py`)
- **Single Source of Truth**: All dashboards now consume state from a centralized `heidi_telemetry` module.
- **Sequence Numbering**: Introduced a `sequence_number` in `state.json`. Every state update increments this number, allowing consumers to detect new data vs. stale data.
- **Unified Resource Monitoring**: GPU and system resource monitoring are now centralized in `heidi_telemetry`. Both dashboards share the same resource metrics (CPU, VRAM, etc.).
- **Atomic State Updates**: State changes are saved atomically to `state.json`.

### 2. Guarded Polling and Stale Data Protection
- **Sequence-Guarded Updates**: Both `app.py` and `heidi_dashboard.py` implement sequence-guarded polling. They skip UI re-renders if the `sequence_number` has not increased.
- **Stale Data Warnings**: If no sequence update occurs for >30 seconds, the UI displays a `[STALE DATA]` warning in the header.
- **Run Mismatch Detection**: If the telemetry `run_id` does not match the dashboard's `run_id`, a `[RUN MISMATCH]` alert is shown.

### 3. Backend Restart Recovery
- **Heuristic Detection**: Dashboards detect sequence number resets (e.g., jumping from 1000 back to 0) or `run_id` changes.
- **Cache Clearing**: Upon detecting a restart/reset, event logs and data caches are cleared to prevent mixing old and new run data.

## UI Polish

### Terminal Dashboard (`heidi_dashboard.py`)
- **Responsive Header**: Uses `Table.grid` to automatically align run info, status, and polling stats. Fits gracefully on narrow terminals.
- **Panel Ratios**: Main panels (Counters, Usage, Trainer) use `ratio` instead of fixed widths, allowing them to scale with the terminal size.
- **Compact View**: Labels and values are optimized for readability in tight spaces.
- **Flicker-Free Redraw**: Leverages Rich's `Live` context with intelligent panel-level updates.

### Web/Rich Dashboard (`app.py`)
- **Standardized Header**: Re-designed to match the terminal dashboard's layout and information density.
- **Shared Style**: Both dashboards now use consistent colors for statuses (GREEN for RUNNING, YELLOW for PAUSED/STALE, RED for ERROR/MISMATCH).

## Verification
- **Doctor Checks**: `tools/doctor.py` now includes checks for telemetry health and guarded polling implementations.
- **Reliability Tests**: `tests/test_dashboard_reliability.py` (Planned) will verify the polling loop singleton behavior and restart recovery.
