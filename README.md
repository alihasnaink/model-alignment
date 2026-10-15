# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

CS 5326 PA2. Response-masked SFT and from-scratch DPO (HH-RLHF and UltraFeedback) on `Qwen/Qwen2.5-0.5B`, a TRL
`DPOTrainer` + LoRA persona run on `HuggingFaceTB/SmolLM2-360M-Instruct`, and blinded pairwise evaluation with a
Kaggle model judge. The full specification is in `docs/assignment_manual.pdf`.

## Layout

```text
alignment/core.py            batching, response-only log-probs, SFT loss, DPO loss (no Trainer / TRL)
alignment/training.py        loops' shared utilities: validation, DPO rewards, checkpoints, run configs
alignment/prompting.py       the single prompt template used by SFT, DPO, and generation
submission_adapter.py        stable grading interface -> alignment.core
scripts/prepare_ultrachat.py     UltraChat 30k / 1k single-turn SFT split (seed 42)
scripts/prepare_preferences.py   HH-RLHF and UltraFeedback 15k / 1k single-turn preference splits (seed 42)
scripts/prepare_alpaca_eval.py   fixed AlpacaEval subset (300 prompts, seed 42) -> eval_sets/
scripts/train_sft.py         Task 3   own SFT loop (fp32 weights, fp16 autocast, GradScaler, accumulation, clipping)
scripts/train_dpo.py         Task 4   own DPO loop (policy cuda:0, frozen fp16 reference cuda:1)
scripts/train_persona_lora.py Task 5  TRL DPOTrainer + PEFT LoRA (the one place TRL is allowed)
scripts/generate.py          Tasks 2/6 batched greedy generation, resumable, records decoding settings
scripts/build_pairwise.py    Task 6   blinded, seeded A/B pair files with hidden model mapping
scripts/judge_pairwise_kaggle.py Task 7  batched, resumable judge (kaggle_benchmarks)
scripts/run_judging.py       judges every pair file with the right rubric
scripts/plot_metrics.py      Task 8   training plots + judge result chart
scripts/summarize_results.py Task 8   config / results / examples tables -> outputs/report/results.md
scripts/analyze_judgments.py Task 8   significance, position/length bias, truncation, per-category -> outputs/report/judge_analysis.md
eval_sets/                   fixed eval sets: AlpacaEval 300/seed 42, HH eval 300, persona style set
kaggle_orchestrator.ipynb    run-all training + generation notebook (T4 x2)
kaggle_continue_uf_dpo.ipynb continuation: UltraFeedback DPO + remaining generations from a previous run's checkpoints
kaggle_judge.ipynb           Task 7 judging notebook (Kaggle benchmark notebook, model proxy)
tests/                       public adapter smoke tests + edge-case + end-to-end script tests
```

## Running on Kaggle

1. Push this repository to GitHub, then set `REPO_URL` in both notebooks if your fork has a different URL.
2. **Training notebook.** Import `kaggle_orchestrator.ipynb`, then set Accelerator **GPU T4 x2** and Internet **On**.
   - Optional but recommended: set `SMOKE_TEST = True` and click **Run All**. This takes about 15–25 minutes and runs every stage on tiny data.
   - Set `SMOKE_TEST = False`, then **Save Version → Save & Run All (Commit)**. This takes about 7–8 hours, which is within Kaggle's 12-hour limit.
   - If a run dies, attach its output as an input, set `RESUME_FROM` to its `.../model-alignment/outputs` path, and commit again. Finished stages are skipped.
3. **Judge notebook (Task 7).** Create a Kaggle *benchmark* notebook (model-proxy access; no GPU), import
   `kaggle_judge.ipynb`, optionally attach the `benchmark-inputs` dataset (generations + `outputs/eval/*_pairs.jsonl`),
   and **Run All**. Without an input it uses the pair files committed in `outputs/eval/`. The notebook:
   - checks proxy access and picks `google/gemini-3.6-flash`, falling back to Kaggle's default judge if it is
     unavailable (the model used is recorded in every summary);
   - uses temperature 0 if the model accepts it;
   - runs a 3-pair-per-file pilot in `eval_pilot/`, then judges all pairs and resumes from existing judgments;
   - writes `outputs/eval/*_judged*.json*`, plots, `report/results.md`, and `judge_bundle.zip`.

   Unzip `judge_bundle.zip` into this repository and commit it.
4. For a fresh full run, download `report_bundle.zip` (training notebook) and `judge_bundle.zip` (judge notebook).
   Unzip both into this repository (they contain an `outputs/` folder) and commit. `.gitignore` keeps weights and raw
   data copies out.

**Output size.** `OUTPUT_MODE = "lean"` (the default) deletes each Qwen checkpoint's weights as soon as every generation
that needs them is written. It keeps `config.json`, `checkpoint_info.json`, and a `PRUNED.txt` note, and drops the raw
training splits at the end. Without this, the saved version would hold about 12 GB of weights. The training notebook
writes two zips:
- `report_bundle.zip`: metrics, run configs, summaries, plots, `report/results.md`, logs, data metadata, generations,
  pair files, and the LoRA adapter.
- `benchmark_inputs.zip`: only `generations/`, `eval/*_pairs.jsonl`, and the eval sets, which is all the judge notebook needs.

Set `OUTPUT_MODE = "full"` only if you need the trained SFT/DPO weights afterwards. Lean mode does not change what is
trained or saved during the run; weights are removed only after their last use.

Local tests (CPU, offline):

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py   # required public suite
PYTHONPATH=. python -m pytest -q tests/                               # + edge cases and end-to-end scripts
```

`tests/test_pipeline_scripts.py` needs `transformers` (and TRL/PEFT for the persona test). Without them it is
skipped. It builds a tiny random Qwen2 model offline and runs every script end to end.

## Implementation choices to document in the report

- **Prompt format** (SFT, DPO, and generation for Qwen): `"### Instruction:\n{prompt}\n\n### Response:\n"`, followed by the
  response and the tokenizer EOS (`<|endoftext|>`). Prompt and response are tokenized separately without special
  tokens, so the response boundary is exact. The persona model uses its own chat template through TRL's conversational format.
- **SFT loss reduction**: mean NLL over all unmasked response tokens (`compute_sft_loss`; the training loop applies it per micro-batch).
- **Masking**: prompt and padding labels are `-100`. EOS is appended *before* truncation to `max_length`.
  Log-probabilities are summed over response tokens only.
- **DPO**: β = 0.1, constant LR 1e-6, AdamW, 1 pair per step × 32 accumulation. The policy runs fp32 + fp16 autocast + GradScaler,
  and the reference is fp16 on `cuda:1` under `no_grad`. Chosen and rejected rows go through one stacked forward pass.
  Reward accuracy uses strict `r_w > r_l`. In exact arithmetic step 0 gives 0, but here the fp16 reference and the fp32
  policy differ by rounding, so the measured step-0 values are noise (HH: accuracy 0.528, margin 0.0006). The best
  checkpoint has the highest validation reward accuracy.
- **Data filters**: single-turn only, non-empty, chosen ≠ rejected, deduplicated, formatted prompt ≤ 256 tokens.
  Filter counts are in `outputs/data/*/..._metadata.json`.
- **Decoding**: greedy, `repetition_penalty=1.1`, `max_new_tokens=512`, fp32 weights. Generations are cut at a
  `### Instruction:` / `### Response:` marker if the model starts a new turn. The same settings are used for every model,
  and each generation file has a `.meta.json` with the exact settings.
- **Judge**: blinded A/B prompts with seeded randomized order, 5 pairs per request, ≥ 5 s between attempts, 3 attempts,
  resumable. The rubric is safety for HH, general for AlpacaEval, and persona style plus general for the LoRA model.

## Output layout (`outputs/`)

The submitted results come from two Kaggle runs. `outputs/RUNS.md` says which run produced what and why there were two.

```text
RUNS.md                   provenance of the two Kaggle runs
kaggle_runs/              executed notebooks of both runs + run 1 kernel log
data/                     split metadata (filters, counts, seeds); eval_sets/ copy (raw splits not committed)
sft/  hh_dpo/  ultrafeedback_dpo/   metrics.jsonl, run_config.json, summary.json, best/ + final/ (checkpoint_info, config; no weights)
persona_lora/             metrics.jsonl (TRL log history), run_config.json, eval_split.json, final/ (LoRA adapter)
generations/              <model>_<evalset>.json + .meta.json (decoding settings)
eval/                     *_pairs.jsonl, *_judged.jsonl, *_judged_failures.jsonl, *_judged_summary.json
plots/                    sft.png, hh_dpo.png, ultrafeedback_dpo.png, persona_lora.png, judge_results.png
report/results.md         auto-generated tables for the report
report/judge_analysis.md  judge statistics: CIs, sign tests, position bias, truncation, per category
logs/run1_orchestrator/   full stdout of every stage of run 1
logs/run2_continue_uf_dpo/ full stdout of run 2
```
