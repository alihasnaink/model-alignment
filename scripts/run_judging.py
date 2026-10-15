#!/usr/bin/env python
"""Judge every saved pairwise file in ``--eval_dir`` with the rubric that fits its eval set.

  hh_dpo_vs_sft_pairs.jsonl             -> safety rubric
  ultrafeedback_dpo_vs_sft_pairs.jsonl  -> general rubric
  persona_lora_vs_base_pairs.jsonl      -> persona_style rubric AND general rubric
                                           (style faithfulness vs. usefulness/safety)
  sft_vs_base_alpaca_pairs.jsonl        -> general rubric (optional extra analysis)

Always resumable (``--reuse_judgments``): rerunning only spends calls on pairs
without a valid verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import judge_pairwise_kaggle as judge

PLAN = [
    ("hh_dpo_vs_sft", "safety", "hh_dpo_vs_sft"),
    ("ultrafeedback_dpo_vs_sft", "general", "ultrafeedback_dpo_vs_sft"),
    ("persona_lora_vs_base", "persona_style", "persona_lora_vs_base_style"),
    ("persona_lora_vs_base", "general", "persona_lora_vs_base_helpfulness"),
    ("sft_vs_base_alpaca", "general", "sft_vs_base_alpaca"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True, type=Path)
    p.add_argument("--judge_model", default="google/gemini-3.6-flash")
    p.add_argument("--backend", default="kaggle", choices=sorted(judge.BACKENDS))
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--sleep_seconds", type=float, default=5.0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--skip_optional", action="store_true", help="skip the sft_vs_base analysis comparison")
    args = p.parse_args()
    for pairs_stem, rubric, output_stem in PLAN:
        pairs = args.eval_dir / f"{pairs_stem}_pairs.jsonl"
        if not pairs.exists() or (args.skip_optional and pairs_stem == "sft_vs_base_alpaca"):
            print(f"skip {pairs.name} (missing or optional)")
            continue
        print(f"\n=== {output_stem} [{rubric}] ===", flush=True)
        judge.main([
            "--pairs_path", str(pairs), "--output_path", str(args.eval_dir / f"{output_stem}_judged.jsonl"),
            "--judge_model", args.judge_model, "--backend", args.backend, "--rubric", rubric,
            "--batch_size", str(args.batch_size), "--sleep_seconds", str(args.sleep_seconds),
            "--max_retries", str(args.max_retries), "--reuse_judgments",
        ])


if __name__ == "__main__":
    main()
