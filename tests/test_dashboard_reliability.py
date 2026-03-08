import unittest
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from dashboard.heidi_telemetry import init_telemetry, save_state, get_state

class TestDashboardReliability(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.run_dir = Path(self.tmp_dir.name) / "runs" / "test_run"
        self.run_dir.mkdir(parents=True)
        os.environ["AUTOTRAIN_DIR"] = self.tmp_dir.name
        os.environ["RUN_ID"] = "test_run"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_sequence_number_increment(self):
        """Verify that sequence number increments on every save."""
        init_telemetry(run_id="test_run")
        s1 = get_state()
        seq1 = s1.get("sequence_number", 0)
        
        save_state({"status": "running"})
        s2 = get_state()
        seq2 = s2.get("sequence_number", 0)
        
        self.assertGreater(seq2, seq1, "Sequence number should increment on save_state")

    def test_stale_data_detection(self):
        """Verify that sequence-guarded polling can detect stale data."""
        init_telemetry(run_id="test_run")
        save_state({"status": "running"})
        state = get_state()
        last_seq = state.get("sequence_number", 0)
        
        # Simulate polling without any new updates
        new_state = get_state()
        new_seq = new_state.get("sequence_number", 0)
        
        self.assertEqual(new_seq, last_seq, "Sequence number should stay same if no new saves")

    def test_restart_recovery(self):
        """Verify that a sequence reset (backend restart) can be detected."""
        # Start state
        init_telemetry(run_id="test_run")
        for _ in range(5): save_state({"step": 1})
        state = get_state()
        last_seq = state.get("sequence_number", 0)
        self.assertGreater(last_seq, 1)

        # Simulate backend restart (init_telemetry resets sequence with force=True)
        init_telemetry(run_id="test_run", force=True)
        new_state = get_state()
        new_seq = new_state.get("sequence_number", 0)
        
        self.assertLess(new_seq, last_seq, "Sequence number should reset after init_telemetry")
        self.assertTrue(new_seq < last_seq - 10 or new_seq == 0, "Reset detection heuristic should trigger")

if __name__ == "__main__":
    unittest.main()
