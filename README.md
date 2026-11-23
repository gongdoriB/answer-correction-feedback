# Do anti-sycophancy instructions block valid corrections?

Experiments on how a language model changes its answer after user feedback, and what instructions that tell it "don't just agree with the user" do to that behavior.

## The experiment

A model answers a multiple-choice question, then the user pushes back.

- If the first answer was **correct**, the user suggests a wrong option: plainly (W1), with a professor's endorsement (W2), or with an unrelated rationale (W3). We want the model to keep its answer.
- If the first answer was **wrong**, the user gives the correct option: alone (G1), with a rationale for the same question (G2), or with a similar-length rationale from another subject (G3). We want the model to correct itself.

The defensive instruction is split into parts to see which one does what: common review text (C), an inhibition clause (I, "don't change an answer just because the user is confident or cites an authority"), and a correction clause (R, "don't keep an answer just because you gave it"). Conditions: S0 (none), SC, SI, SR, S1 (all).

Model: Qwen3-8B (main), Qwen3-4B (supplementary check). Data: 300 MMLU-Pro questions from six subjects.

## What I found

| Instruction | Valid correction (G1) ↑ | Authority-backed wrong agreement (W2) ↓ |
|---|---|---|
| SC | 81.9% | 77.4% |
| SI (+ inhibition clause) | 53.9% | 40.4% |

- The inhibition clause was the only part that reduced valid correction.
- Its cost depended on the feedback: 28.6 percentage points with the answer alone, 14.7 with an unrelated rationale, and 6.5 with a same-question rationale.

## Repository layout

```
notebooks/   Colab notebooks that generate model responses, one per stage
src/         shared generation/analysis code and stage-specific scripts
analysis/    scripts that re-check raw outputs and build the result tables
MAPPING.md   which notebook/script produces which result
DATA_NEEDED.md  inputs the code expects (no data is included)
```

Runs used Google Colab (A100), NF4 4-bit quantization and non-thinking mode. Some analysis scripts still assume my original folder layout; see `MAPPING.md`.
