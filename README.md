# Small-Model Alignment Starter

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Implemented Training Entry Points

The notebook is intentionally only an orchestrator: use the scripts below for
long-running work so checkpoints and metrics remain durable under
`/kaggle/working`.

```bash
# Full-parameter, response-masked SFT (JSON/JSONL rows: prompt, response)
python scripts/train_sft.py --train TRAIN.jsonl --validation VALID.jsonl \
  --output_dir /kaggle/working/sft

# Full-parameter DPO (rows: prompt, chosen, rejected). The frozen reference
# automatically uses cuda:1 when two GPUs are visible.
python scripts/train_dpo.py --sft_checkpoint /kaggle/working/sft/best \
  --train HH_TRAIN.jsonl --validation HH_VALID.jsonl --output_dir /kaggle/working/hh_dpo

# The one allowed TRL + LoRA experiment
python scripts/train_persona_lora.py --output_dir /kaggle/working/persona_lora
```

`scripts/generate.py` writes the required durable generation JSON format.
`scripts/build_pairwise.py` turns matching SFT/DPO output files into a seeded,
blinded JSONL comparison file; use its `mapping` only after a judge returns A/B.

The train files are intentionally generic JSON or JSONL to keep notebook data
preparation explicit: SFT expects `prompt` and `response`; DPO expects `prompt`,
`chosen`, and `rejected`. Field names and a stable prompt template can be
overridden with the corresponding CLI flags.

For the manual's SFT recipe, run `scripts/prepare_ultrachat.py` first. It
downloads `HuggingFaceH4/ultrachat_200k`, retains only single-turn
user/assistant records, and deterministically writes 30,000 train plus 1,000
validation rows (seed 42).

For Kaggle, start with [`kaggle_orchestrator.ipynb`](kaggle_orchestrator.ipynb).
It performs environment setup and invokes these scripts only; configure attached
dataset paths and turn on one `RUN_*` stage at a time.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.
