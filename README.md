# Model Alignment with DPO

CS 5326 PA2 by Ali Hasnain. Supervised fine-tuning and from-scratch Direct Preference Optimization on
`Qwen/Qwen2.5-0.5B`, a TRL `DPOTrainer` + LoRA persona run on `HuggingFaceTB/SmolLM2-360M-Instruct`, and blinded
pairwise evaluation with a Kaggle model judge. The assignment is in `docs/assignment_manual.pdf`.

**Report:** [`report/report.pdf`](report/report.pdf)

## Results

Judge: `google/gemini-3.6-flash` (temperature 0). The win rate excludes ties. The full statistics are in
[`outputs/report/judge_analysis.md`](outputs/report/judge_analysis.md).

| Comparison | Eval set | n | Wins / Ties / Losses | Win rate | Val reward accuracy |
|---|---|---|---|---|---|
| HH-RLHF DPO vs SFT | synthetic HH set | 300 | 111 / 85 / 104 | 0.516 | 0.637 |
| UltraFeedback DPO vs SFT | AlpacaEval (300, seed 42) | 300 | 100 / 85 / 115 | 0.465 | 0.748 |
| Persona LoRA-DPO vs base, style | persona style set | 28 | 12 / 6 / 10 | 0.545 | 1.000 |
| Persona LoRA-DPO vs base, helpfulness | persona style set | 28 | 9 / 1 / 18 | 0.333 | 1.000 |

None of these differences is statistically significant. Higher DPO reward accuracy did not give a better judged
model. The report covers why: length bias, truncation at the token cap, judge position bias, and reward
over-optimization.

## Submission checklist (manual section 16)

| Item | Location |
|---|---|
| GitHub repository | https://github.com/alihasnaink/model-alignment |
| Final report (PDF) | `report/report.pdf` (source `report/report.tex`) |
| Training plots for SFT and DPO | `outputs/plots/`, report Figures 1 and 2 |
| Evaluation summary, DPO vs SFT | `outputs/report/judge_analysis.md`, `outputs/eval/*_judged_summary.json`, report Section 5 |
| `submission_adapter.py` with public tests passing | `submission_adapter.py`, `tests/test_public_adapter_smoke.py` |

## Repository layout

```text
alignment/core.py              batching, response-only log-probs, SFT loss, DPO loss (no Trainer or TRL)
alignment/training.py          shared training utilities: validation, DPO rewards, checkpoints, run configs
alignment/prompting.py         the prompt template used by SFT, DPO and generation
submission_adapter.py          grading interface, thin wrappers around alignment.core
scripts/prepare_ultrachat.py   UltraChat 30k / 1k SFT split
scripts/prepare_preferences.py HH-RLHF and UltraFeedback 15k / 1k preference splits
scripts/prepare_alpaca_eval.py fixed AlpacaEval subset (300 prompts, seed 42)
scripts/train_sft.py           Task 3: SFT loop
scripts/train_dpo.py           Task 4: DPO loop (policy on cuda:0, frozen fp16 reference on cuda:1)
scripts/train_persona_lora.py  Task 5: TRL DPOTrainer + PEFT LoRA
scripts/generate.py            Tasks 2 and 6: batched greedy generation, resumable
scripts/build_pairwise.py      Task 6: blinded pair files with a seeded A/B order
scripts/judge_pairwise_kaggle.py, run_judging.py   Task 7: batched, resumable judging with kaggle_benchmarks
scripts/plot_metrics.py, summarize_results.py, analyze_judgments.py, preference_length_stats.py   Task 8 analysis
eval_sets/                     the three fixed evaluation sets
kaggle_orchestrator.ipynb      training and generation notebook (Kaggle, GPU T4 x2)
kaggle_continue_uf_dpo.ipynb   continuation run: UltraFeedback DPO and the remaining generations
kaggle_judge.ipynb             judging notebook (Kaggle benchmark notebook)
outputs/                       all run artifacts; see outputs/RUNS.md
report/                        report source, PDF and figure script
tests/                         public adapter tests, edge-case tests, end-to-end script tests
```

`outputs/` holds the metrics, configs, checkpoint records, generations, pair files, judgments, plots and logs of the
submitted runs. [`outputs/RUNS.md`](outputs/RUNS.md) explains which Kaggle run produced each file. Model weights and
raw training splits are not committed. They are in the Kaggle run outputs, and the data metadata files record how
to rebuild the splits.

## Reproducing

**Tests (CPU, offline):**

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py   # public adapter tests
PYTHONPATH=. python -m pytest -q tests/                               # all tests
```

`tests/test_pipeline_scripts.py` builds a tiny random model offline and runs every script end to end. It needs
`transformers` (and TRL and PEFT for the persona test); without them it is skipped.

**Training and generation (Kaggle).** Import `kaggle_orchestrator.ipynb`. Select GPU T4 x2 and turn Internet on,
then use Save Version and Save & Run All. Setting `SMOKE_TEST = True` first runs every stage on tiny data in about
20 minutes. Finished stages are skipped on a rerun, and `RESUME_FROM` continues from an attached earlier output.

**Judging (Kaggle).** Create a Kaggle benchmark notebook (model-proxy access, no GPU) from `kaggle_judge.ipynb` and
run all cells. It reads the pair files in `outputs/eval/` and never regenerates outputs.

**Report.** `python report/make_figures.py` rebuilds the figures. Compile `report/report.tex` with any LaTeX engine.

## Key implementation choices

- **Prompt format:** `### Instruction:\n{prompt}\n\n### Response:\n`, then the response and the EOS token. Prompt and
  response are tokenized separately, so the response boundary is exact. The persona model uses its chat template.
- **Masking:** prompt and padding labels are `-100`. EOS is added before truncation to `max_length`. Log
  probabilities are summed over response tokens only.
- **SFT loss reduction:** mean negative log likelihood over all unmasked response tokens.
- **DPO:** beta 0.1, constant learning rate 1e-6, AdamW, effective batch 32. The best checkpoint has the highest
  validation reward accuracy (strict `r_w > r_l`).
- **Decoding:** greedy, repetition penalty 1.1, at most 512 new tokens, the same for every model.
