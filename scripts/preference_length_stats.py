#!/usr/bin/env python
"""Length statistics of chosen vs rejected responses in the DPO preference splits.

Reads <outputs>/data/<track>/<track>_{train,valid}.jsonl (the exact splits used for
training) and writes <outputs>/data/<track>/<track>_length_stats.json, a small file
that is committed so the length-bias analysis can be checked without the raw splits.
Lengths are whitespace-separated words.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    # Split on "\n" only (see alignment.training.load_json_records).
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def stats(rows: list[dict]) -> dict:
    chosen = [len(r["chosen"].split()) for r in rows]
    rejected = [len(r["rejected"].split()) for r in rows]
    mid = len(rows) // 2
    return {
        "pairs": len(rows),
        "chosen_mean_words": round(sum(chosen) / len(rows), 1),
        "rejected_mean_words": round(sum(rejected) / len(rows), 1),
        "chosen_median_words": sorted(chosen)[mid],
        "rejected_median_words": sorted(rejected)[mid],
        "share_chosen_longer": round(sum(c > r for c, r in zip(chosen, rejected)) / len(rows), 3),
        "share_chosen_shorter": round(sum(c < r for c, r in zip(chosen, rejected)) / len(rows), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", required=True, type=Path)
    args = p.parse_args()
    for track in ("hh", "ultrafeedback"):
        folder = args.outputs / "data" / track
        result = {split: stats(load(folder / f"{track}_{split}.jsonl"))
                  for split in ("train", "valid") if (folder / f"{track}_{split}.jsonl").exists()}
        if not result:
            print(f"{track}: no split files found, skipped")
            continue
        (folder / f"{track}_length_stats.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(track, json.dumps(result))


if __name__ == "__main__":
    main()
