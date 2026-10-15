#!/usr/bin/env python
"""Build the assignment's reproducible single-turn UltraChat SFT subset.

The source dataset's ``train_sft`` split includes multi-turn conversations. The
manual's reference recipe is single-turn, so this script retains exactly one
user message followed by one assistant message, then makes a seeded 30k/1k
train/validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def as_single_turn(row: dict) -> dict[str, str] | None:
    messages = row.get("messages", [])
    if len(messages) != 2:
        return None
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    prompt, response = user.get("content", "").strip(), assistant.get("content", "").strip()
    if not prompt or not response:
        return None
    return {"prompt": prompt, "response": response}


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--train_size", type=int, default=30_000)
    parser.add_argument("--validation_size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_size < 1 or args.validation_size < 1:
        raise ValueError("train_size and validation_size must be positive")

    source = load_dataset(args.dataset, split=args.split)
    filtered = source.map(as_single_turn, remove_columns=source.column_names).filter(lambda row: row["prompt"] is not None)
    needed = args.train_size + args.validation_size
    if len(filtered) < needed:
        raise RuntimeError(f"only {len(filtered)} valid single-turn rows, need {needed}")
    selected = filtered.shuffle(seed=args.seed).select(range(needed))
    records = list(selected)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "sft_train.jsonl", records[:args.train_size])
    write_jsonl(output_dir / "sft_valid.jsonl", records[args.train_size:])
    (output_dir / "metadata.json").write_text(json.dumps({
        "source_dataset": args.dataset,
        "source_split": args.split,
        "selection": "single-turn messages: user followed by assistant",
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "seed": args.seed,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
