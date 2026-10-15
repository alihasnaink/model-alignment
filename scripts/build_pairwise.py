#!/usr/bin/env python
"""Create blinded pairwise-comparison JSONL files with a seeded, deterministic A/B order.

Every row carries ``instruction``, ``response_a``, ``response_b`` (the only fields a
judge prompt may use) plus hidden metadata: ``mapping`` (A/B -> model name), and the
``candidate``/``baseline`` names used to translate verdicts after judging.

Example:  --baseline sft.json --baseline_name sft --candidate dpo.json --candidate_name dpo
(``--sft`` / ``--dpo`` are kept as aliases for the DPO-vs-SFT case.)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_outputs(path: str) -> dict[str, dict]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in values}


def build_pairs(baseline: dict[str, dict], candidate: dict[str, dict], baseline_name: str, candidate_name: str,
                seed: int) -> list[dict]:
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) ^ set(candidate))[:5]
        raise ValueError(f"baseline and candidate output ids must match exactly (e.g. {missing})")
    if baseline_name == candidate_name:
        raise ValueError("baseline and candidate names must differ")
    rng = random.Random(seed)
    pairs = []
    for item_id in baseline:  # preserve source evaluation ordering
        if baseline[item_id]["instruction"] != candidate[item_id]["instruction"]:
            raise ValueError(f"instruction mismatch for id {item_id}")
        left, right = (baseline_name, candidate_name) if rng.randrange(2) == 0 else (candidate_name, baseline_name)
        answers = {baseline_name: baseline[item_id]["output"], candidate_name: candidate[item_id]["output"]}
        row = {
            "id": item_id,
            "instruction": baseline[item_id]["instruction"],
            "response_a": answers[left],
            "response_b": answers[right],
            # Hidden metadata: never shown to the judge; used only after a verdict comes back.
            "mapping": {"A": left, "B": right},
            "candidate": candidate_name,
            "baseline": baseline_name,
        }
        if "category" in baseline[item_id]:
            row["category"] = baseline[item_id]["category"]
        pairs.append(row)
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", "--sft", dest="baseline", required=True)
    p.add_argument("--candidate", "--dpo", dest="candidate", required=True)
    p.add_argument("--baseline_name", default="sft")
    p.add_argument("--candidate_name", default="dpo")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    pairs = build_pairs(read_outputs(args.baseline), read_outputs(args.candidate), args.baseline_name,
                        args.candidate_name, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    a_count = sum(row["mapping"]["A"] == args.candidate_name for row in pairs)
    print(f"Wrote {len(pairs)} pairs to {output} ({args.candidate_name} is Response A in {a_count})")


if __name__ == "__main__":
    main()
