# MAPPING: results -> notebooks / scripts / inputs


## Naming: paper vs. code

| Paper | Code | Meaning |
|---|---|---|
| G1 / G2 / G3 | `R1` / `R2` / `R3` | correct option only / + same-item rationale / + different-subject rationale (initially incorrect items) |
| W1 | `C2` | plain incorrect suggestion (initially correct items) |
| W2 | `C3` | incorrect suggestion + professor endorsement |
| W3 | `W3` | W1 suggestion + different-subject rationale (new in stage 11) |
| S0 / SC / SI / SR / S1 | same | defensive instruction: none / C / C+I / C+R / C+I+R |

The folder prefix `stageNN_` follows the numbering of the original experiment log (stages 01, 03, 04, 07 are not part of the paper and are not included).

## Pipeline (what feeds what)

stage02 (300 items, initial answers + S0/S1 follow-ups) -> stage05 (SC/SI/SR) -> stage06 (rationale extraction + R2 S0/S1) -> stage08 (R2 x SC/SI/SR) -> stage09 (R3 x SC/SI) -> stage11 (W3 x SC/SI) ; stage10 repeats the rationale design on Qwen3-4B.
Each later stage re-uses earlier responses (stored inside its `source/*.json`), so the stages must be re-run in this order, or the original inputs must be obtained.

## Paper item -> producer

| Paper item | Generation notebook | Source code | Analysis / audit script |
|---|---|---|---|
| Initial answers (133 correct / 164 incorrect / 3 invalid, Sec. 4.1); S0 and S1 rows of Table 2 | `notebooks/stage02_main_run_initial_S0_S1_4B_8B.ipynb` | `src/shared/`, `src/stage02_main_run/` (code is embedded in the notebook as a base64 ZIP; extracted copy here) | `analysis/audit_stage02_main_run_S0_S1.py` |
| Table 2 (SC, SI, SR rows; S0/S1 reused), Sec. 4.3 CIs | `notebooks/stage05_clause_ablation_SC_SI_SR_8B.ipynb` | `src/shared/`, `src/stage05_clause_ablation/` | `analysis/audit_stage05_clause_ablation.py` -> `verified_comparison_table.csv`, `verified_clause_effects.csv` |
| Rationale selection (164 -> 77 items, <=5 sentences, <=180 words; GPT-4o outputs); R2 under S0/S1 | `notebooks/stage06_rationale_R2_max5_8B.ipynb` | `src/stage06_rationale_extraction_R2/` | `analysis/audit_stage06_rationale_R2_max5.py`, `analysis/check_stage06_additional_item_availability.py` |
| G2 under SC/SI (Table 3, middle column) | `notebooks/stage08_clause_x_rationale_8B.ipynb` | `src/stage08_clause_x_rationale/` | `analysis/audit_stage08_clause_x_rationale.py`, `analysis/sensitivity_stage08_format.py` |
| G3 under SC/SI, one-to-one donor assignment, 74 length-matched pairs; Table 3; G3-G2 and G1-G3 contrasts | `notebooks/stage09_rationale_content_R3_8B.ipynb` | `src/stage09_rationale_content_R3/` (`prepare_matching.py` builds the donor assignment) | `analysis/audit_stage09_rationale_content_R3.py` -> `verified_comparison_table.csv`, `verified_control_effects.csv`, `verified_tight74_effects.csv` |
| Table 4 (W1 vs W3 retention, 133 items), SI-SC 14.79 vs 22.31 pp | `notebooks/stage11_retention_W3_8B.ipynb` | `src/stage11_retention_W3/` | `analysis/audit_stage11_retention_W3.py` -> `verified_comparison.csv`, `verified_effects.csv` |
| Supplementary Qwen3-4B check (80 items; 3.33 / 0.00 / 5.42 pp; lenient-JSON sensitivity) | `notebooks/stage10_supplementary_4B.ipynb` | `src/stage10_supplementary_4B/` | `analysis/audit_stage10_supplementary_4B.py`, `analysis/check_stage10_received_tokens.py`, `analysis/report_stage10_readable_summary.py` |

W1 rows of Table 2 and Table 4 are the same responses (condition `C2` under SC/SI from stage 05), counted once.
G1 for the 77 items (Table 3) are the stage 05 `R1` responses restricted to those items.

Spot checks (local audit CSVs vs. manuscript): Table 2 S0/SC/SI/SR/S1 G1 372/403/265/398/295 of 492; Table 3 G1/G2/G3 191/194/181 (SC) and 125/179/147 (SI) of 231; Table 4 W1 220/309 and W3 170/229 of 399. All match.

## How the pieces fit

* Notebooks (stages 05-11) expect a single `*_inputs.zip` chosen in cell 2 (name and SHA-256 are hard-coded in the notebook). That ZIP contained the flat set of Python modules (`src/shared/*` + the stage folder + `source/*.json`). To rebuild it, copy `src/shared/*` and the stage folder's `*.py` into one flat folder, add `source/` (see `DATA_NEEDED.md`), and zip. A rebuilt ZIP will have a different SHA-256, so the notebook's `INPUT_SHA256` check must be updated (this changes the notebook, not the analysis).
* `CODE_FILES` fingerprints inside each `*_protocol.py` read sibling files via `Path(__file__).parent`, so the modules must be flat in one directory when run.
* Stage 10 uses a different `runtime_check.py` (`src/stage10_supplementary_4B/runtime_check_4B_variant.py`; the 8B stages use `src/shared/runtime_check.py`).
* Tests (`tests/`) of each stage were included in the original ZIPs.
* Audit scripts verify raw checkpoints inside the Colab result ZIPs and rewrite the verified tables; they currently assume the original working-tree layout (see "Known portability problems").

## Known portability problems (not fixed, original behavior kept)

* `analysis/audit_*.py` and `check_*.py`: `sys.path` and `received/` paths are built from `Path(__file__).parents[...]` and `output/jupyter-notebook/<stage>_support` of the author's workspace; they will not run unmodified. Result-ZIP paths were redacted to `data/results_zips/<name>.zip`.
* `src/stage06_rationale_extraction_R2/prepare_evidence.py` expects a cached Qwen3 tokenizer at `../sycophancy_qwen_support/validation/tokenizers/qwen3_8b`.
* `src/stage10_supplementary_4B/prepare_inputs.py` (lines ~51-80) reads author-workspace paths (`output/sycophancy_results_audit/...`, cached 4B tokenizer).
* `analysis/check_stage06_additional_item_availability.py` reads `tmp/mmlu_official_sources/model_outputs_gpt4o(2024-05-13)_5shots.zip`.
* Notebooks write to `/content/drive/MyDrive/sycophancy_qwen/...` (Google Colab + Drive); generic, but Drive is required.
