#!/usr/bin/env python
"""Build the assignment's reproducible single-turn UltraChat SFT subset.

UltraChat's ``train_sft`` split is almost entirely multi-turn conversations
(only a handful have exactly one exchange). To get a single-turn dataset we keep
the FIRST user message and the assistant reply that follows it from each
conversation, then make a seeded 30k/1k train/validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def has_valid_first_exchange(row: dict) -> bool:
    """True iff the conversation starts with a non-empty user turn then a non-empty assistant turn."""
    messages = row.get("messages") or []
    if len(messages) < 2:
        return False
    user, assistant = messages[0], messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return False
    prompt = (user.get("content") or "").strip()
    response = (assistant.get("content") or "").strip()
    return bool(prompt) and bool(response)


def to_prompt_response(row: dict) -> dict[str, str]:
    """Take the first user/assistant exchange as a single-turn example."""
    user, assistant = row["messages"][0], row["messages"][1]
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

    # Filter first, then map (map must never return None).
    filtered = source.filter(has_valid_first_exchange).map(
        to_prompt_response,
        remove_columns=source.column_names,
    )
    print(f"Usable rows: {len(filtered)} / {len(source)}")

    needed = args.train_size + args.validation_size
    if len(filtered) < needed:
        raise RuntimeError(f"only {len(filtered)} usable rows, need {needed}")

    selected = filtered.shuffle(seed=args.seed).select(range(needed))
    records = list(selected)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "sft_train.jsonl", records[: args.train_size])
    write_jsonl(output_dir / "sft_valid.jsonl", records[args.train_size :])

    metadata = {
        "source_dataset": args.dataset,
        "source_split": args.split,
        "selection": "first user message + following assistant reply of each conversation",
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()