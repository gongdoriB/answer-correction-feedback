import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from test_controls import fixture_pool, FakeBackend
from protocol import default_config
from engine import freeze, run_model
from analysis import analyze, bootstrap_interval, collect, revision_outcome


class RevisionBackend(FakeBackend):
    """Deliberately separates answer persistence, changed errors and invalid outputs."""
    def generate(self, jobs, seed):
        outputs = super().generate(jobs, seed)
        for job, output in zip(jobs, outputs):
            if job["phase"] == "followup" and job["condition"] == "R1":
                v = job["variant"]
                if job["defense"] == "S0":
                    output["text"] = json.dumps({"answer": ["A", "B", "C"][v]})
                elif v == 0:
                    output["text"] = json.dumps({"answer": "B"})
                elif v == 1:
                    output["text"] = "I cannot choose."
                else:
                    output.update(text='{"answer":"B"}', truncated=True)
        return outputs


class TestAnalysis(unittest.TestCase):
    def test_constant_zero_has_nonzero_uncertainty(self):
        lo, hi, method = bootstrap_interval([0.] * 100, ["math"] * 100)
        self.assertEqual(lo, 0)
        self.assertGreater(hi, .1)
        self.assertEqual(method, "conservative_Hoeffding")

    def test_synthetic_end_to_end(self):
        config = default_config()
        config["bootstrap_repetitions"] = 100
        items = fixture_pool(config)["pools"]["smoke"]
        with tempfile.TemporaryDirectory() as d:
            frozen = freeze(d, items, config)
            for model in config["models"]:
                run_model(d, frozen, model, backend=FakeBackend())
            _, frame, baseline, complete = collect(d)
            self.assertEqual(len(frame), 504)
            self.assertEqual(len(baseline), 12)
            folder, summary, effects = analyze(d, plot=False)
            self.assertTrue((folder / "paired_effects.csv").exists())
            tar = summary[summary.metric == "target_agreement"]
            self.assertTrue(set(tar.condition).issubset({"C2", "C3", "C4", "C5"}))
            conditional = summary[summary.stratum != "all_items"]
            self.assertEqual(set(conditional.n_items), {3})
            self.assertEqual(set(conditional.n_responses), {9})
            self.assertEqual(set(summary[summary.stratum == "all_items"].n_items), {6})
            self.assertEqual(len(effects), 14)
            self.assertTrue((effects[effects.contrast == "key_correction_change"].difference == 0).all())
            self.assertNotIn("correction_noninferiority_supported", effects.columns)
            self.assertEqual(set(effects[effects.contrast == "key_correction_change"].analysis_role), {"exploratory_diagnostic"})

    def test_revision_categories_exclude_invalid_baselines_and_truncations(self):
        for answer, status, expected in [("A", "valid", "corrected"), ("B", "valid", "retained_initial_wrong"),
                                         ("C", "valid", "changed_to_other_wrong"), (None, "invalid_format", "invalid"),
                                         ("B", "truncated", "invalid")]:
            self.assertEqual(revision_outcome("B", "valid", answer, status, "A"), expected)
        self.assertIsNone(revision_outcome(None, "invalid_format", None, "invalid_format", "A"))
        self.assertIsNone(revision_outcome("A", "valid", "A", "valid", "A"))

    def test_retention_is_not_one_minus_correction_and_categories_sum_to_one(self):
        config = default_config()
        config["bootstrap_repetitions"] = 100
        items = fixture_pool(config)["pools"]["smoke"]
        with tempfile.TemporaryDirectory() as d:
            frozen = freeze(d, items, config)
            run_model(d, frozen, "qwen3_4b", backend=RevisionBackend())
            folder, summary, effects = analyze(d, plot=False)
            r1 = summary[(summary.condition == "R1") & (summary.stratum == "initial_wrong")]
            s0 = r1[r1.defense == "S0"].set_index("metric").estimate
            s1 = r1[r1.defense == "S1"].set_index("metric").estimate
            self.assertAlmostEqual(s0["final_correct"], 1/3)
            self.assertAlmostEqual(s0["initial_wrong_retained"], 1/3)
            self.assertAlmostEqual(s0["other_wrong_change"], 1/3)
            self.assertEqual(s0["invalid"], 0)
            self.assertEqual(s1["final_correct"], 0)
            self.assertAlmostEqual(s1["initial_wrong_retained"], 1/3)
            self.assertEqual(s1["other_wrong_change"], 0)
            self.assertAlmostEqual(s1["invalid"], 2/3)
            self.assertAlmostEqual(s0.sum(), 1)
            self.assertAlmostEqual(s1.sum(), 1)
            self.assertNotEqual(s1["initial_wrong_retained"], 1 - s1["final_correct"])
            eff = effects.set_index("contrast")
            self.assertAlmostEqual(eff.loc["initial_wrong_retention_change", "difference"], 0)
            self.assertAlmostEqual(eff.loc["key_correction_change", "difference"], -1/3)
            self.assertEqual(eff.loc["all_item_invalid_change", "n_items"], 6)
            self.assertAlmostEqual(eff.loc["all_item_invalid_change", "difference"], (2/3)/7)
            self.assertAlmostEqual(eff.loc["initial_wrong_retention_change", "alpha_per_comparison"], .05/6)
            self.assertTrue((folder / "r1_revision_outcomes.csv").exists())
            self.assertTrue((folder / "format_failure_all_items.csv").exists())

    def test_empty_initial_strata_remain_missing_and_plots_render(self):
        class UniformBackend(FakeBackend):
            def generate(self, jobs, seed):
                return [{"text": json.dumps({"answer": self.answer}), "truncated": False} for _ in jobs]
        config = default_config()
        config["bootstrap_repetitions"] = 100
        items = fixture_pool(config)["pools"]["smoke"]
        for answer, missing_stratum in [("A", "initial_wrong"), ("B", "initial_correct")]:
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as d:
                frozen = freeze(d, items, config)
                backend = UniformBackend(); backend.answer = answer
                run_model(d, frozen, "qwen3_4b", backend=backend)
                folder, summary, effects = analyze(d, plot=True)
                empty = summary[summary.stratum == missing_stratum]
                self.assertTrue((empty.n_items == 0).all())
                self.assertTrue(empty.estimate.isna().all())
                self.assertTrue((folder / "03_r1_revision_outcomes.png").exists())


if __name__ == "__main__":
    unittest.main()
