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


def is_single_turn(row: dict) -> bool:
    """True iff the row is exactly one non-empty user turn then one non-empty assistant turn."""
    messages = row.get("messages") or []
    if len(messages) != 2:
        return False
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return False
    prompt = (user.get("content") or "").strip()
    response = (assistant.get("content") or "").strip()
    return bool(prompt) and bool(response)


def to_prompt_response(row: dict) -> dict[str, str]:
    """Convert an already-validated single-turn row to prompt/response fields."""
    user, assistant = row["messages"]
    return {
        "prompt": user["content"].strip(),
        "response": assistant["content"].strip(),
    }


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

    # Filter first, then map. The map function never returns None, which is
    # what crashed `datasets` with "'NoneType' object has no attribute 'write'".
    filtered = source.filter(is_single_turn).map(
        to_prompt_response,
        remove_columns=source.column_names,
    )
    print(f"Single-turn rows: {len(filtered)} / {len(source)}")

    needed = args.train_size + args.validation_size
    if len(filtered) < needed:
        raise RuntimeError(f"only {len(filtered)} valid single-turn rows, need {needed}")

    selected = filtered.shuffle(seed=args.seed).select(range(needed))
    records = list(selected)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "sft_train.jsonl", records[: args.train_size])
    write_jsonl(output_dir / "sft_valid.jsonl", records[args.train_size :])

    metadata = {
        "source_dataset": args.dataset,
        "source_split": args.split,
        "selection": "single-turn messages: user followed by assistant",
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()