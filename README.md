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
eval_sets/                   fixed eval sets: AlpacaEval 300/seed 42, HH eval 300, persona style set
kaggle_orchestrator.ipynb    run-all training + generation notebook (T4 x2)
kaggle_judge.ipynb           judging notebook (Kaggle utility benchmark)
tests/                       public adapter smoke tests + edge-case + end-to-end script tests
```

## Running on Kaggle

1. Push this repository to GitHub, then set `REPO_URL` in both notebooks if your fork has a different URL.
2. **Training notebook.** Import `kaggle_orchestrator.ipynb`, then set Accelerator **GPU T4 x2** and Internet **On**.
   - Optional but recommended: set `SMOKE_TEST = True` and click **Run All**. This takes about 15–25 minutes and runs every stage on tiny data.
   - Set `SMOKE_TEST = False`, then **Save Version → Save & Run All (Commit)**. This takes about 7–8 hours, which is within Kaggle's 12-hour limit.
   - If a run dies, attach its output as an input, set `RESUME_FROM` to its `.../model-alignment/outputs` path, and commit again. Finished stages are skipped.
3. **Judge notebook.** Create a new Kaggle notebook *as a utility benchmark*, paste in `kaggle_judge.ipynb`, add the
   training notebook's output as an input, and run it. It writes judgments, summaries, and `judge_bundle.zip`.
4. Download `submission_bundle.zip` (training notebook) and `judge_bundle.zip` (judge notebook). Unzip both into
   `outputs/` in this repository and commit. `.gitignore` keeps weights and raw data copies out.

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
  Reward accuracy uses strict `r_w > r_l`, so step 0 is exactly 0. The best checkpoint has the highest validation reward accuracy.
- **Data filters**: single-turn only, non-empty, chosen ≠ rejected, deduplicated, formatted prompt ≤ 256 tokens.
  Filter counts are in `outputs/data/*/..._metadata.json`.
- **Decoding**: greedy, `repetition_penalty=1.1`, `max_new_tokens=512`, fp32 weights. Generations are cut at a
  `### Instruction:` / `### Response:` marker if the model starts a new turn. The same settings are used for every model,
  and each generation file has a `.meta.json` with the exact settings.
- **Judge**: blinded A/B prompts with seeded randomized order, 5 pairs per request, ≥ 5 s between attempts, 3 attempts,
  resumable. The rubric is safety for HH, general for AlpacaEval, and persona style plus general for the LoRA model.

## Output layout (`outputs/`)

```text
data/                     split files + metadata (filters, counts, seeds); eval_sets/ copy
sft/  hh_dpo/  ultrafeedback_dpo/   metrics.jsonl, run_config.json, summary.json, best/, final/
persona_lora/             metrics.jsonl (TRL log history), run_config.json, eval_split.json, final/ (LoRA adapter)
generations/              <model>_<evalset>.json + .meta.json (decoding settings)
eval/                     *_pairs.jsonl, *_judged.jsonl, *_judged_failures.jsonl, *_judged_summary.json
plots/                    sft.png, hh_dpo.png, ultrafeedback_dpo.png, persona_lora.png, judge_results.png
report/results.md         auto-generated tables for the report
logs/                     full stdout of every stage
```
