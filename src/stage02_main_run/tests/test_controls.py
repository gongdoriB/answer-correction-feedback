"""Synthetic fixtures test software only. Never scientific model results."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol import *
from engine import freeze, run_model, execute_batches


def fixture_pool(config):
    rows = []
    for s, subject in enumerate(config["subjects"]):
        for i in range(80):
            rows.append({"question_id": s * 1000 + i, "question": f"Synthetic fixture {subject} {i}",
                         "category": subject, "options": [f"Unique option {j}" for j in range(10)],
                         "answer": "A", "answer_index": 0, "cot_content": ""})
    return make_pool(rows, config)


class FakeBackend:
    environment = {"gpu": "SYNTHETIC_FIXTURE_NO_MODEL", "purpose": "software test only"}
    def __init__(self):
        self.calls = 0
    def preflight(self, jobs):
        return {"count": len(jobs), "synthetic": True}
    def generate(self, jobs, seed):
        self.calls += 1
        outputs = []
        for job in jobs:
            if job["phase"] == "baseline":
                answer = "B" if int(job["question_id"].split(":")[1]) // 1000 % 2 else "A"
            elif job["condition"] == "R1":
                answer = "A"
            elif job["defense"] == "S1":
                answer = json.loads(job["messages"][2]["content"])["answer"]
            else:
                answer = "C"
            outputs.append({"text": json.dumps({"answer": answer, "explanation": "SYNTHETIC FIXTURE"}), "truncated": False})
        return outputs


class TestControls(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.pool = fixture_pool(self.config)
        self.items = [{**i, "evidence": "Synthetic evidence for test only", "evidence_approved": False} for i in self.pool["pools"]["smoke"]]

    def test_sampling_is_reproducible_and_disjoint(self):
        self.assertEqual(self.pool, fixture_pool(self.config))
        ids = [i["question_id"] for p in self.pool["pools"].values() for i in p]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([len(self.pool["pools"][x]) for x in ["smoke", "pilot", "main"]], [6, 60, 300])
        self.assertTrue(all(i["gold"] != i["target_wrong"] for i in self.items))

    def test_complete_factorial_and_identical_history(self):
        baselines = {i["question_id"]: {"text": '{"answer":"B"}'} for i in self.items}
        jobs = followup_jobs(self.items, baselines, self.config)
        self.assertEqual(len(jobs), 6 * 7 * 2 * 3)
        self.assertTrue(audit_jobs(self.items, jobs, self.config)["identical_baseline_per_item"])
        self.assertEqual({j["condition"] for j in jobs}, set(CONDITIONS))
        j = jobs[0]
        j["messages"][2]["content"] = "changed"
        with self.assertRaises(AssertionError):
            audit_jobs(self.items, jobs, self.config)

    def test_system_only_difference_between_defenses(self):
        for condition in CONDITIONS:
            for v in range(3):
                a = messages(self.items[0], "original", condition, "S0", v)
                b = messages(self.items[0], "original", condition, "S1", v)
                self.assertEqual(a[1:], b[1:])
                self.assertNotEqual(a[0], b[0])
        self.assertNotIn(DEFENSES["S1"], messages(self.items[0])[0]["content"])

    def test_claim_exactly_shared(self):
        i = self.items[0]
        for v in range(3):
            stem = STEMS["claim"][v].format(target=i["target_wrong"])
            for c in ["C2", "C3", "C4", "C5"]:
                self.assertTrue(intervention(i, c, v).startswith(stem))
            self.assertNotIn(i["evidence"], intervention(i, "R1", v))

    def test_explicit_option_counts_and_matched_correct_suggestion(self):
        audit = audit_prompt_templates()
        self.assertEqual(audit["labels_checked"], 10)
        self.assertEqual(audit["explicit_option_mentions"], {c: 0 if c in {"C0", "C1"} else 1 for c in CONDITIONS})
        for index, target in enumerate("ABCDEFGHIJ"):
            gold = "ABCDEFGHIJ"[(index + 1) % 10]
            item = {**self.items[0], "target_wrong": target, "gold": gold}
            for v in range(3):
                for c in ["C2", "C3", "C4", "C5"]:
                    text = intervention(item, c, v)
                    self.assertEqual(re.findall(r"\boption ([A-J])\b", text), [target])
                self.assertEqual(intervention(item, "C2", v).replace("option " + target, "option " + gold), intervention(item, "R1", v))

    def test_repeated_target_in_followup_is_detected(self):
        baselines = {i["question_id"]: {"text": '{"answer":"B"}'} for i in self.items}
        jobs = followup_jobs(self.items, baselines, self.config)
        job = next(j for j in jobs if j["condition"] == "C3")
        job["messages"][3]["content"] += " I repeat: option A."
        with self.assertRaises(AssertionError):
            audit_jobs(self.items, jobs, self.config)

    def test_parser_does_not_guess_or_accept_duplicate_keys(self):
        self.assertEqual(parse_answer('```json\n{"answer":"a"}\n```', ["A", "B"]), ("A", "valid"))
        for text in ['The answer is A.', '{"answer":"A","answer":"B"}', '{"answer":"Z"}', '[]']:
            self.assertEqual(parse_answer(text, ["A", "B"])[1], "invalid_format")

    def test_lock_blocks_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lock.json"
            lock_json(p, {"x": 1})
            lock_json(p, {"x": 1})
            with self.assertRaises(RuntimeError):
                lock_json(p, {"x": 2})


    def test_resume_is_batch_stable(self):
        jobs = baseline_jobs(self.items)
        with tempfile.TemporaryDirectory() as d:
            backend = FakeBackend()
            first, done = execute_batches(jobs, d, backend, self.config, stop_after_batches=1)
            self.assertFalse(done)
            self.assertEqual(len(first), 4)
            records, done = execute_batches(jobs, d, backend, self.config)
            self.assertTrue(done)
            self.assertEqual(backend.calls, 2)
            self.assertEqual(records[:4], first)
            execute_batches(jobs, d, backend, self.config)
            self.assertEqual(backend.calls, 2)

    def test_corruption_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            execute_batches(baseline_jobs(self.items), d, FakeBackend(), self.config)
            path = Path(d) / "batch_000000.json"
            saved = read_json(path); saved["records"][0]["text"] = "tampered"
            atomic_json(path, saved)
            with self.assertRaises(RuntimeError):
                execute_batches(baseline_jobs(self.items), d, FakeBackend(), self.config)

    def test_complete_run_and_environment_block(self):
        with tempfile.TemporaryDirectory() as d:
            frozen = freeze(d, self.items, self.config)
            result = run_model(d, frozen, "qwen3_4b", backend=FakeBackend())
            self.assertTrue(result["complete"])
            self.assertEqual(result["followup_count"], 252)
            other = FakeBackend(); other.environment = {"gpu": "DIFFERENT_GPU"}
            with self.assertRaises(RuntimeError):
                run_model(d, frozen, "qwen3_8b", backend=other)


    def test_main_freezes_without_human_review(self):
        with tempfile.TemporaryDirectory() as d:
            config = default_config("main")
            items = fixture_pool(config)["pools"]["main"]
            frozen = freeze(d, items, config)
            self.assertEqual(len(frozen["items"]), 300)
            self.assertFalse((Path(d) / "evidence_review.csv").exists())
            self.assertNotIn("R2", frozen["config"]["conditions"])
            self.assertNotIn("evidence", frozen["items"][0])

    def test_evidence_condition_cannot_be_reintroduced(self):
        with self.assertRaises(ValueError):
            intervention(self.items[0], "R2", 0)
        with tempfile.TemporaryDirectory() as d:
            config = default_config()
            config["conditions"] = list(config["conditions"]) + ["R2"]
            with self.assertRaises(ValueError):
                freeze(d, self.items, config)


if __name__ == "__main__":
    unittest.main()
