"""Audit uploaded actual A-study outputs; uploaded code is compared, never executed.

Run with /opt/anaconda3/envs/tf/bin/python.
Workflow: Kassis et al. (2026), Scientific Agent Skills,
https://doi.org/10.48550/arXiv.2609.00065 (workflow attribution, not result evidence).
"""
from pathlib import Path
import hashlib
import io
import json
import platform
import sys
import zipfile
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
SUPPORT = WORKSPACE / "output/jupyter-notebook/sycophancy_qwen_8b_ablation_support"
ARCHIVE = Path("data/results_zips/sycophancy_qwen3_8b_ablation10800_a100_b16_v1_results.zip")  # REDACTED: was an absolute local Downloads path
sys.path.insert(0, str(SUPPORT))
from protocol import digest
from ablation_protocol import load_source, make_jobs, fingerprint, default_config, DEFENSES, ANALYSIS_PLAN
from ablation_run import batch_seed, generation_environment
from ablation_analysis import score_records, effect_table, comparison_table, diagnostic_summary
from analysis import summarize


def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        prefix = "sycophancy_qwen3_8b_ablation10800_a100_b16_v1/"
        read = lambda name: json.loads(z.read(prefix + name))
        frozen, source = read("study_lock.json"), read("source_snapshot.json")
        cfg = frozen["config"]
        assert digest({k: v for k, v in frozen.items() if k != "study_hash"}) == frozen["study_hash"]
        assert source == load_source(SUPPORT / "source"), "Original baseline or control records changed"
        assert cfg == default_config(source)
        assert frozen["code_sha256"] == fingerprint()
        assert frozen["defense_texts"] == DEFENSES
        assert frozen["analysis_plan"] == read("analysis_plan.json") == ANALYSIS_PLAN
        for name, expected in frozen["code_sha256"].items():
            assert hashlib.sha256(z.read(prefix + "code/" + name)).hexdigest() == expected
        env = read("environment_lock.json")
        assert env == source["run"]["environment"], "Original and follow-up environment differ"
        assert not env.get("synthetic_test", False)
        run_lock = read("run_lock.json")
        assert run_lock["study_hash"] == frozen["study_hash"] and run_lock["source_hash"] == source["source_hash"]
        assert run_lock["generation_environment"] == generation_environment(env)
        env_names = [n for n in z.namelist() if "/session_environments/" in n and n.endswith(".json")]
        env_hashes = set()
        for name in env_names:
            current = json.loads(z.read(name))
            assert generation_environment(current) == generation_environment(env)
            assert Path(name).stem == digest(current)[:20]
            env_hashes.add(digest(current))
        status = read("status.json")
        assert status["complete"] and status["saved_responses"] == 10800 and status["saved_batches"] == 675
        jobs = make_jobs(source, cfg)
        assert digest(jobs) == frozen["jobs_hash"]
        assert read("schedule.json") == {"jobs_hash": frozen["jobs_hash"], "batch_size": 16,
                                         "task_ids": [j["task_id"] for j in jobs]}
        new, resources, times = [], [], []
        for index in range(675):
            batch = read(f"checkpoints/batch_{index:06d}.json")
            scheduled = jobs[index*16:(index+1)*16]
            assert batch["batch_index"] == index
            assert batch["batch_hash"] == digest(scheduled)
            assert batch["records_hash"] == digest(batch["records"])
            assert batch["batch_seed"] == batch_seed(cfg, index, scheduled)
            assert batch["environment_hash"] in env_hashes
            assert len(batch["records"]) == len(scheduled) == 16
            for row, job in zip(batch["records"], scheduled):
                assert all(row.get(k) == v for k, v in job.items())
                assert row["batch_seed"] == batch["batch_seed"] and row["batch_index"] == index
                assert isinstance(row["text"], str) and isinstance(row["truncated"], bool)
                assert 0 <= row["output_tokens"] <= cfg["max_new_tokens"]
                assert 0 < row["input_tokens"] <= cfg["max_input_tokens"]
                assert "SYNTHETIC TEST ONLY" not in row["text"]
            new.extend(batch["records"])
            resources.append(batch["resource"])
            times.append(batch["saved_at_kst"])
        assert sum("/checkpoints/batch_" in n for n in z.namelist()) == 675
        frame = score_records(source, source["controls"] + new)
        assert len(frame) == 18000 and len(new) == 10800
        assert not frame.duplicated(["question_id", "condition", "variant", "defense"]).any()
        assert set(frame.groupby(["question_id", "condition", "defense"]).size()) == {3}
        assert set(frame.groupby("question_id").baseline_hash.nunique()) == {1}
        exported = pd.read_csv(io.BytesIO(z.read(prefix + "reports/scored_responses.csv")))
        rebuilt = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        pd.testing.assert_frame_equal(rebuilt, exported, check_exact=True)
        effects, details = effect_table(frame, cfg)
        stored = pd.read_csv(io.BytesIO(z.read(prefix + "reports/clause_effects.csv")))
        pd.testing.assert_frame_equal(effects, stored, check_dtype=False, atol=1e-12, rtol=1e-12)
        stored_details = pd.read_csv(io.BytesIO(z.read(prefix + "reports/item_contrasts.csv")))
        pd.testing.assert_frame_equal(details, stored_details, check_dtype=False, atol=1e-12, rtol=1e-12)
        summary = summarize(frame, cfg)
        stored_summary = pd.read_csv(io.BytesIO(z.read(prefix + "reports/metrics.csv"))).drop(columns="data_origin")
        pd.testing.assert_frame_equal(summary, stored_summary, check_dtype=False, atol=1e-12, rtol=1e-12)
        table = comparison_table(frame)
        stored_table = pd.read_csv(io.BytesIO(z.read(prefix + "reports/comparison_table.csv")))
        pd.testing.assert_frame_equal(table, stored_table, check_dtype=False, atol=1e-12, rtol=1e-12)
        result = diagnostic_summary(effects)
        stored_result = read("reports/diagnostic_summary.json")
        assert all(stored_result[k] == v for k, v in result.items())
        assert stored_result["execution_mode"] == "colab_ablation"
        resource = read("reports/resource_summary.json")
        assert np.isclose(sum(r["generation_seconds"] for r in resources), resource["generation_seconds_sum"])
        assert np.isclose(max(r["peak_allocated_bytes"] for r in resources)/2**30, resource["peak_allocated_GiB"])
        preflight = read("token_preflight.json")
        assert preflight["max_input_tokens_observed"] == max(r["input_tokens"] for r in new)
        assert preflight["jobs_hash"] == frozen["jobs_hash"] and preflight["count"] == 10800
        reports = OUT / "colab_reports"
        reports.mkdir(exist_ok=True)
        for name in z.namelist():
            if name.startswith(prefix + "reports/") and not name.endswith("/"):
                (reports / Path(name).name).write_bytes(z.read(name))

    table.to_csv(OUT / "verified_comparison_table.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUT / "verified_clause_effects.csv", index=False, encoding="utf-8-sig")
    r1 = frame[(frame.condition == "R1") & frame.baseline_valid & ~frame.baseline_correct]
    outcomes = pd.crosstab(r1.defense, r1.revision_outcome)
    outcomes.to_csv(OUT / "r1_outcome_counts.csv", encoding="utf-8-sig")
    formats = frame.groupby("defense")[["invalid", "truncated"]].sum()
    formats.to_csv(OUT / "format_counts.csv", encoding="utf-8-sig")
    primary_formats = []
    for condition, correct in [("R1", False), ("C3", True)]:
        sub = frame[(frame.condition == condition) & frame.baseline_valid & (frame.baseline_correct == correct)]
        grouped = sub.groupby("defense")[["invalid", "truncated"]].sum().reset_index()
        grouped["condition"] = condition
        primary_formats.append(grouped)
    primary_formats = pd.concat(primary_formats, ignore_index=True)
    primary_formats.to_csv(OUT / "primary_cohort_format_counts.csv", index=False)
    audit = {
        "archive": str(ARCHIVE), "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "all_checks_passed": True, "new_responses": 10800, "reused_responses": 7200,
        "original_baselines": 300, "baseline_counts": source["counts"],
        "source_and_existing_controls_unchanged": True, "generation_environment_identical": True,
        "all_code_files_match_delivered_local_code": True, "all_checkpoint_hashes_verified": True,
        "all_scored_rows_and_all_28_planned_contrasts_reproduced": True,
        "raw_response_start_end_kst": [min(times), max(times)], "resource_summary": resource,
        "primary_comparisons": result["primary_comparisons"],
        "method": "R1/C3 five contrasts each, family of 10, two-sided 99.5% intervals with 20,000 subject-stratified paired question draws; fixed phrasing triplets averaged first; other contrasts descriptive 95% intervals.",
        "interpretation": "Inhibition block lowers both R1 correction and C3 target agreement. Correction block partially offsets both effects when inhibition is present. Correction-alone R1 change remains uncertain.",
        "limitations": ["Adaptive same-300-item experiment", "One Qwen3-8B NF4/non-thinking setting",
                        "Historical controls and new random schedule", "Clause deletions also change length and position",
                        "Correct suggestion has no rationale", "One correction block contains two sentences; not an internal-mechanism test"],
        "audit_runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "workflow_reference": "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 (workflow attribution only).",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print('AUDIT PASSED')
    print(table.to_string(index=False))
    print('R1 outcomes:\n', outcomes.to_string())
    print('All format counts:\n', formats.to_string())
    print('Primary cohort format counts:\n', primary_formats.to_string(index=False))
    print('Primary contrasts:\n', effects[effects.analysis_role == 'primary'][['contrast','difference_pp','ci_low_pp','ci_high_pp']].to_string(index=False))
    print('First/last saved batches:', min(times), max(times))
    print('Saved audit:', OUT)


if __name__ == '__main__':
    main()
