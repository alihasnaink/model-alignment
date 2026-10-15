#!/usr/bin/env python
"""Create blinded DPO-vs-SFT JSONL comparison pairs with deterministic order."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_outputs(path: str) -> dict[str, dict]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in values}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft", required=True)
    p.add_argument("--dpo", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    sft, dpo, rng = read_outputs(args.sft), read_outputs(args.dpo), random.Random(args.seed)
    if set(sft) != set(dpo): raise ValueError("SFT and DPO output ids must match exactly")
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for item_id in sft:  # preserve source evaluation ordering
            left, right = ("sft", "dpo") if rng.randrange(2) == 0 else ("dpo", "sft")
            answers = {"sft": sft[item_id]["output"], "dpo": dpo[item_id]["output"]}
            # `mapping` is hidden metadata: judge prompts should consume only instruction/A/B.
            handle.write(json.dumps({"id": item_id, "instruction": sft[item_id]["instruction"], "response_a": answers[left], "response_b": answers[right], "mapping": {"A": left, "B": right}}) + "\n")


if __name__ == "__main__":
    main()
