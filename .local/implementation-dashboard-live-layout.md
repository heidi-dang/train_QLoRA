# Live Dashboard Layout Implementation Note

## Current Failures

1. **Flashing** - Dashboard redraws on every telemetry update causing visible flicker
2. **Overlapping text** - Dashboard renders on top of existing stdout/pip/install text
3. **Destroyed logs** - Log panel loses context when progress updates
4. **Unreadable panels** - Mixed print() output bleeds through dashboard panels
5. **No cursor management** - Cursor remains visible during dashboard mode
6. **Event-driven redraw** - Rendering triggered by each event instead of fixed cadence

## Target Layout

```
Top row: title, run status, clock
Second row left: Teacher Usage & Cost
Second row right: Progress
Third row left: Resources
Bottom full width: Recent Logs
```

## Live State Model

```python
class RunState:
    # Status
    status: str              # idle, starting, running, stopped, completed, failed
    current_stage: str       # e.g., "generate", "filter", "train", "eval"
    stage_index: int         # 0-based
    total_stages: int        # e.g., 4
    
    # Progress (computed from backend)
    stage_percent: float     # 0.0-100.0
    overall_percent: float   # 0.0-100.0
    completed_units: int
    total_units: int
    
    # Teacher Usage
    provider: str            # e.g., "grok"
    model: str              # e.g., "grok-4-1-fast"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_count: int
    success_count: int
    failed_count: int
    spend_usd: float
    
    # Resources
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_percent: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    
    # Logs
    log_buffer: deque        # Ring buffer, last 300 lines
    last_update: datetime
    
    # Control
    running: bool            # Dashboard mode active
    stopped_cleanly: bool    # Final state preserved
```

## Render Cadence

- **Fixed refresh rate**: 4-5 times per second (200-250ms intervals)
- **State updates**: Immediate on events (token updates, log lines, resource samples)
- **Screen redraw**: Only on fixed cadence
- **Coalescing**: Multiple rapid updates render as single frame

## Stop/Restart/Reconnect Behavior

### Enter Dashboard Mode
1. Clear screen once
2. Hide cursor (ANSI hide cursor)
3. Initialize run state from telemetry file
4. Start render loop at fixed cadence
5. Begin resource sampling

### Exit Dashboard Mode
1. Stop render loop
2. Show cursor (ANSI show cursor)
3. If stopped/failed: freeze last state, print summary
4. If completed: print final metrics
5. Never leave blank screen

### On Run Stop/Fail
1. Freeze last known run state
2. Keep final logs visible
3. Display final status (STOPPED/FAILED)
4. Preserve token/spend totals
5. Stop background render loops

### On Restart/Resume
1. Preserve log buffer (don't clear)
2. Resume from current telemetry state
3. Continue appending to logs
4. No double-counting of tokens/spend

## Doctor Checks

The doctor check must validate:

1. **No mixed rendering**: Dashboard must not use direct print() that mixes with stdout
2. **Log buffer exists**: Must use in-memory ring buffer, not file re-reads
3. **Progress fields complete**: Must have stage_percent, overall_percent from backend
4. **Teacher cost fields**: Must have provider, model, tokens, spend
5. **Renderer separation**: State updates must be separate from screen render
6. **Fixed cadence**: No event-driven redraw spam

## Implementation Phases

### Phase 1: State Model (Complete first)
- Create RunState class with all fields
- Load from telemetry.json on start
- Update on events but don't render

### Phase 2: Log Buffer (Complete second)
- Create ring buffer for logs (300 lines)
- Append new log lines from file tail
- Never clear buffer on updates

### Phase 3: Renderer (Complete third)
- Fixed cadence loop (200ms)
- ANSI screen clear on enter
- Cursor hide/show control
- Single full redraw per frame

### Phase 4: Layout (Complete fourth)
- Fixed panel positions
- No overlapping text
- Compact mode for narrow terminals

### Phase 5: Doctor Check (Complete fifth)
- Add check_live_dashboard to tools/doctor.py
- Validate all requirements

### Phase 6: Validation (Complete last)
- Real run test
- Verify no flashing
- Verify logs preserved
- Verify stop behavior

## Acceptance Criteria

1. Dashboard enters with clean screen, no background text
2. Teacher tokens update live (no flicker)
3. Spend updates live
4. Progress bars move smoothly
5. Logs keep appending, never disappear
6. Stop leaves final state visible
7. Restart does not wipe logs
8. No overlapping text ever
9. Doctor check passes
