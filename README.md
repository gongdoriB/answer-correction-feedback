# Answer Correction in Language Models Following User Feedback

Code for **"Answer Correction in Language Models Following User Feedback: Effects of Anti-Sycophancy Instructions and Rationale Conditions"**
Gihyun Bae and Sangsoon Lim (Chung-Ang University). *DigiCon-26*, 2026 (accepted, to appear).

## What the study asks

Instructions that tell a language model not to yield to users (anti-sycophancy instructions) keep it from accepting wrong suggestions. Do they also stop it from accepting **valid** corrections, and which part of the instruction is responsible?

Using **Qwen3-8B** on **MMLU-Pro** (300 items, six subjects), we split a defensive instruction into common review text (C), an inhibition clause (I) and a correction clause (R), and varied the feedback:

- initially correct answers received a wrong suggestion, plain (W1) or authority-backed (W2), or with a different-subject rationale (W3);
- initially incorrect answers received the correct option alone (G1), with a same-item rationale (G2), or with a different-subject rationale (G3).

## Main results (Qwen3-8B)

| Instruction | Valid correction (G1) ↑ | Authority-backed wrong agreement (W2) ↓ |
|---|---|---|
| SC (review text) | 81.9% | 77.4% |
| SI (+ inhibition clause) | 53.9% | 40.4% |

- Only the inhibition clause reduced valid correction; the correction clause could not restore it without also restoring wrong agreement.
- The cost of the clause depended on the feedback: 28.6 percentage points with the answer alone, 14.7 with a different-subject rationale, and 6.5 with a same-item rationale.
- Because every correction condition named the correct option, the results describe **revision behavior**, not independent reasoning.

Full numbers, confidence intervals and the supplementary Qwen3-4B check are in the paper.

## Repository layout

```
notebooks/   Colab generation notebooks, one per experimental stage (outputs cleared)
src/shared/  protocol, generation engine and analysis code shared by all stages
src/stageNN_*/  stage-specific run / protocol / analysis code and tests
analysis/    audit scripts that re-verify raw outputs and rebuild the tables
MAPPING.md   which notebook and script produce each table and number in the paper
DATA_NEEDED.md  every input the code expects and where it comes from
```

Stages follow the original experiment log: 02 initial answers and S0/S1, 05 clause ablation (SC/SI/SR), 06 rationale extraction, 08 same-item rationale × clauses, 09 different-subject rationale, 11 retention with W3, and 10 the Qwen3-4B check. Exploratory stages that are not in the paper are omitted.

## Setup used in the paper

Google Colab, one A100 GPU, Qwen3-8B with NF4 4-bit quantization, non-thinking mode (`enable_thinking=False`), 192 output tokens, temperature 0.7, top-p 0.8, top-k 20; three fixed feedback phrasings per condition, one response each. Package versions are pinned in `src/stage02_main_run/requirements-colab.txt` and in the first cell of each notebook. Statistics use a subject-stratified paired item bootstrap.

## Data

No data are included. MMLU-Pro and the Qwen3 models are downloaded from Hugging Face; rationales come from the public GPT-4o (2024-05-13) MMLU-Pro outputs released by TIGER-AI-Lab. The derived inputs (sampled items, earlier responses reused by later stages, selected rationales) and the raw model outputs are listed in `DATA_NEEDED.md`.

## Reproducibility notes

This is a cleaned snapshot of the research code, not a one-command pipeline.

- Each notebook expects an inputs ZIP whose name and SHA-256 are fixed in the notebook. The ZIP holds the modules from `src/shared/` and the stage folder in one flat directory plus a `source/` folder; rebuilding it changes the hash, so the check in the notebook must be updated.
- Later stages reuse responses from earlier ones, so stages must run in order.
- Some audit and preparation scripts still assume the original working-directory layout and a cached tokenizer; these are listed under *Known portability problems* in `MAPPING.md`.

## Citation

```
G. Bae and S. Lim, "Answer Correction in Language Models Following User Feedback:
Effects of Anti-Sycophancy Instructions and Rationale Conditions," DigiCon-26, 2026 (to appear).
```
