"""UI must not alter jobs, seeds, checkpoint contents or completion state."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import freeze, read_records, run_model
from progress_ui import run_with_progress
from protocol import default_config
from test_controls import FakeBackend, fixture_pool


class TrackingBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def generate(self, jobs, seed):
        self.inputs.append((jobs, seed))
        return super().generate(jobs, seed)


class TestProgress(unittest.TestCase):
    def prepare(self, folder):
        config = default_config()
        return freeze(folder, fixture_pool(config)["pools"]["smoke"], config)

    def test_wrapper_preserves_inputs_seeds_records_and_status(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as log:
            plain, shown = Path(tmp) / "plain", Path(tmp) / "shown"
            f1, f2 = self.prepare(plain), self.prepare(shown)
            b1, b2 = TrackingBackend(), TrackingBackend()
            one = run_model(plain, f1, "qwen3_4b", backend=b1)
            two = run_with_progress(shown, f2, "qwen3_4b", backend=b2)
            self.assertEqual(one, two)
            self.assertEqual(b1.inputs, b2.inputs)
            for stage in ["baseline", "followup"]:
                self.assertEqual(read_records(plain / "runs/primary/qwen3_4b", stage),
                                 read_records(shown / "runs/primary/qwen3_4b", stage))
            self.assertIn("실제 저장 258/258", log.getvalue())

    def test_partial_then_resume_counts_saved_without_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as log:
            frozen = self.prepare(tmp)
            partial = run_with_progress(tmp, frozen, "qwen3_4b", backend=TrackingBackend(), stop_after_batches=1)
            self.assertFalse(partial["complete"])
            self.assertIn("일시 중단: 실제 저장 4/258", log.getvalue())
            b = TrackingBackend()
            status = run_with_progress(tmp, frozen, "qwen3_4b", backend=b)
            self.assertTrue(status["complete"])
            self.assertEqual(len(b.inputs), 64)  # remaining baseline batch + 63 followup batches
            b = TrackingBackend()
            status = run_with_progress(tmp, frozen, "qwen3_4b", backend=b)
            self.assertTrue(status["complete"])
            self.assertEqual(b.inputs, [])
            self.assertIn("기존 저장 252/252", log.getvalue())

    def test_failure_is_not_completion_or_saved_batch(self):
        class Broken(TrackingBackend):
            def generate(self, jobs, seed):
                raise RuntimeError("synthetic error")
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as log:
            frozen = self.prepare(tmp)
            with self.assertRaisesRegex(RuntimeError, "synthetic error"):
                run_with_progress(tmp, frozen, "qwen3_4b", backend=Broken())
            self.assertNotIn("완료: 실제 저장", log.getvalue())
            self.assertEqual(read_records(Path(tmp) / "runs/primary/qwen3_4b", "baseline"), [])


if __name__ == "__main__":
    unittest.main()
