#!/usr/bin/env python
"""Create the fixed AlpacaEval subset (default: 300 prompts, seed 42).

The raw ``alpaca_eval.json`` file is downloaded directly from the dataset repo,
which avoids the deprecated dataset loading script. Indices are drawn with
``random.Random(seed).sample`` and then sorted, so the subset keeps the source
order. The result is committed at ``eval_sets/alpaca_eval_300_seed42.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="eval_sets/alpaca_eval_300_seed42.json")
    p.add_argument("--num_prompts", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    source = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.num_prompts))
    subset = [
        {
            "id": f"alpaca-eval-{args.seed}-{position:04d}",
            "source_index": index,
            "instruction": rows[index]["instruction"],
            "dataset": "alpaca_eval",
            "category": rows[index].get("dataset", ""),
        }
        for position, index in enumerate(indices)
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(subset)} prompts (of {len(rows)}) to {out}")


if __name__ == "__main__":
    main()
