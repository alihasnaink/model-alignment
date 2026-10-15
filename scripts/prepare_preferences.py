#!/usr/bin/env python
"""Build reproducible single-turn preference splits for the two from-scratch DPO runs.

hh:            Anthropic/hh-rlhf (all subsets). Rows are full "\\n\\nHuman: ...\\n\\nAssistant: ..."
               transcripts; only single-turn exchanges are kept and parsed into prompt/chosen/rejected.
               train and validation are both drawn from the ``train`` split (disjoint, seeded).
ultrafeedback: HuggingFaceH4/ultrafeedback_binarized. Train from ``train_prefs``, validation from
               ``test_prefs``; only single-turn rows with score_chosen > score_rejected are kept.

Both tracks drop pairs with empty/identical responses and prompts whose formatted
length exceeds ``--max_prompt_length`` tokens (the manual's 256-token limit), so the
response is never truncated away entirely.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from alignment.prompting import format_prompt
from alignment.training import write_jsonl

HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def parse_hh(row: dict) -> dict | None:
    chosen, rejected = row["chosen"], row["rejected"]
    for text in (chosen, rejected):
        if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
            return None
    prompt_c, chosen_resp = chosen.split(ASSISTANT, 1)
    prompt_r, rejected_resp = rejected.split(ASSISTANT, 1)
    if prompt_c.strip() != prompt_r.strip():
        return None
    return {
        "prompt": prompt_c.replace(HUMAN, "", 1).strip(),
        "chosen": chosen_resp.strip(),
        "rejected": rejected_resp.strip(),
    }


def parse_ultrafeedback(row: dict) -> dict | None:
    chosen, rejected = row.get("chosen") or [], row.get("rejected") or []
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if [m["role"] for m in chosen] != ["user", "assistant"] or [m["role"] for m in rejected] != ["user", "assistant"]:
        return None
    if not row["score_chosen"] > row["score_rejected"]:
        return None
    return {
        "prompt": str(row["prompt"]).strip(),
        "chosen": chosen[1]["content"].strip(),
        "rejected": rejected[1]["content"].strip(),
        "score_chosen": row["score_chosen"],
        "score_rejected": row["score_rejected"],
    }


def clean(rows, parse, tokenizer, max_prompt_length: int, stats: dict) -> list[dict]:
    kept, seen = [], set()
    for row in rows:
        stats["source_rows"] += 1
        parsed = parse(row)
        if parsed is None:
            stats["dropped_multi_turn_or_malformed"] += 1
            continue
        if not parsed["prompt"] or not parsed["chosen"] or not parsed["rejected"] or parsed["chosen"] == parsed["rejected"]:
            stats["dropped_empty_or_identical"] += 1
            continue
        key = (parsed["prompt"], parsed["chosen"], parsed["rejected"])
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        prompt_tokens = len(tokenizer.encode(format_prompt(parsed["prompt"]), add_special_tokens=False))
        if prompt_tokens > max_prompt_length:
            stats["dropped_long_prompt"] += 1
            continue
        kept.append(parsed)
    stats["usable_rows"] = len(kept)
    return kept


def new_stats() -> dict:
    return {key: 0 for key in ("source_rows", "dropped_multi_turn_or_malformed", "dropped_empty_or_identical",
                               "dropped_duplicate", "dropped_long_prompt", "usable_rows")}


def main() -> None:
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["hh", "ultrafeedback"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_size", type=int, default=15_000)
    p.add_argument("--validation_size", type=int, default=1_000)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rng = random.Random(args.seed)
    if args.track == "hh":
        source_name, splits = "Anthropic/hh-rlhf", {"train": "train"}
        stats = new_stats()
        rows = clean(load_dataset(source_name, split="train"), parse_hh, tokenizer, args.max_prompt_length, stats)
        rng.shuffle(rows)
        needed = args.train_size + args.validation_size
        if len(rows) < needed:
            raise RuntimeError(f"only {len(rows)} usable single-turn HH pairs, need {needed}")
        train, valid = rows[: args.train_size], rows[args.train_size: needed]
        all_stats = {"train": stats}
    else:
        source_name, splits = "HuggingFaceH4/ultrafeedback_binarized", {"train": "train_prefs", "validation": "test_prefs"}
        train_stats, valid_stats = new_stats(), new_stats()
        train = clean(load_dataset(source_name, split="train_prefs"), parse_ultrafeedback, tokenizer, args.max_prompt_length, train_stats)
        valid = clean(load_dataset(source_name, split="test_prefs"), parse_ultrafeedback, tokenizer, args.max_prompt_length, valid_stats)
        rng.shuffle(train); rng.shuffle(valid)
        if len(train) < args.train_size or len(valid) < args.validation_size:
            raise RuntimeError(f"not enough usable UltraFeedback pairs: {len(train)} train, {len(valid)} validation")
        train, valid = train[: args.train_size], valid[: args.validation_size]
        all_stats = {"train": train_stats, "validation": valid_stats}

    for index, row in enumerate(train):
        row["id"] = f"{args.track}-train-{index:05d}"
    for index, row in enumerate(valid):
        row["id"] = f"{args.track}-valid-{index:05d}"
    out = Path(args.output_dir)
    write_jsonl(out / f"{args.track}_train.jsonl", train)
    write_jsonl(out / f"{args.track}_valid.jsonl", valid)
    metadata = {
        "track": args.track, "source_dataset": source_name, "source_splits": splits,
        "filters": ["single-turn only", "non-empty and chosen != rejected", "deduplicated",
                    f"formatted prompt <= {args.max_prompt_length} tokens ({args.tokenizer})"],
        "train_size": len(train), "validation_size": len(valid), "seed": args.seed, "filter_stats": all_stats,
    }
    (out / f"{args.track}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
