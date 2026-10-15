# How these outputs were produced

All results come from two Kaggle runs (GPU T4 x2, seed 42) with identical hyperparameters and decoding settings.

| Run | Notebook | Produced |
|---|---|---|
| 1 | `kaggle_orchestrator.ipynb` (repo commit `e9e8915`) | data splits + metadata, base generations, persona LoRA-DPO + its generations, SFT, HH-RLHF DPO, SFT generations, SFT sanity check |
| 2 | `kaggle_continue_uf_dpo.ipynb` (repo commit `e4dc352`) | UltraFeedback DPO, HH-DPO generations, UF-DPO generations |

Run 1 stopped when UltraFeedback DPO started. The JSONL reader split lines with `str.splitlines()`, which also
breaks on U+2028 characters inside UltraFeedback text. The fix (split on `"\n"` only) is in commit `e4dc352`, with a
regression test. Run 2 read the SFT `best` and HH-DPO `best` checkpoints and the UltraFeedback splits directly from
run 1's saved output, so UltraFeedback DPO starts from the same SFT checkpoint as HH-RLHF DPO.

- `kaggle_runs/` holds both executed notebooks and run 1's kernel log.
- `logs/run1_orchestrator/` and `logs/run2_continue_uf_dpo/` hold the full stdout of each stage. Run 1's `train_dpo.log`
  ends with the UltraFeedback crash; run 2's holds the successful UltraFeedback run.
- `*/best/` and `*/final/` keep only `config.json`, `generation_config.json` and `checkpoint_info.json` (step and
  validation metrics of the saved checkpoint). Model weights stay in the Kaggle run outputs. The tokenizers are the
  unchanged base-model tokenizers.
- `persona_lora/final/` keeps the LoRA adapter config and the chat template; `adapter_model.safetensors` stays in run 1's
  Kaggle output.
- Raw training splits (`data/*/*.jsonl`) are not committed; `data/*/*metadata.json` records the sources, filters, counts
  and seed needed to rebuild them with `scripts/prepare_*.py`.
