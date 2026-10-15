#!/usr/bin/env python
"""Collect configs, training summaries, judge results, and example pairs into
``<outputs>/report/results.md`` (+ ``results.json``) for the final report.

Everything is read from files written by the other scripts, so the tables report
the values actually used rather than the guide configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = [("SFT", "sft"), ("HH-RLHF DPO", "hh_dpo"), ("UltraFeedback DPO", "ultrafeedback_dpo"), ("LoRA persona DPO", "persona_lora")]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def config_rows(outputs: Path) -> list[dict]:
    rows = []
    for label, name in RUNS:
        cfg = load(outputs / name / "run_config.json")
        if not cfg:
            continue
        lora = name == "persona_lora"
        rows.append({
            "run": label,
            "init / reference": cfg.get("model") if not lora else f"{cfg.get('model')} / adapter-disabled base",
            "trainable": (f"LoRA r={cfg['lora_rank']} α={cfg['lora_alpha']} drop={cfg['lora_dropout']} "
                          f"({', '.join(cfg.get('target_modules', []))})") if lora else "full fine-tuning",
            "optimizer": cfg.get("optimizer", "AdamW"),
            "peak LR": cfg.get("learning_rate"),
            "schedule / warmup": f"{cfg.get('scheduler', '')} / {cfg.get('warmup_steps', cfg.get('warmup_ratio', 0))}",
            "weight decay": cfg.get("weight_decay"),
            "clip": cfg.get("max_grad_norm", 1.0),
            "per-device batch × accum = eff.": f"{cfg.get('batch_size')} × {cfg.get('gradient_accumulation')} = "
                                              f"{cfg.get('effective_batch_size')}",
            "precision": cfg.get("precision"),
            "max length": cfg.get("max_length"),
            "epochs / steps": f"{cfg.get('epochs')} / {cfg.get('total_optimizer_steps', '–')}",
            "seed": cfg.get("seed"),
            "beta": cfg.get("beta", "–"),
            "reference ckpt": cfg.get("reference_checkpoint") or cfg.get("reference_model", "–"),
            "train / val size": f"{cfg.get('train_examples', cfg.get('train_pairs'))} / "
                                f"{cfg.get('validation_examples', cfg.get('validation_pairs', cfg.get('eval_pairs')))}",
        })
    return rows


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_none yet_\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)).replace("|", "/").replace("\n", " ") for key in keys) + " |")
    return "\n".join(lines) + "\n"


def judge_rows(eval_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(eval_dir.glob("*_summary.json")):
        data = load(path)
        o = data["overall"]
        rows.append({
            "comparison": path.stem.replace("_judged_summary", ""), "rubric": data["rubric"],
            "judge": data["judge_model"] + (" (MOCK)" if data["backend"] == "mock" else ""),
            "n": o["n"], "candidate wins": o["wins"], "ties": o["ties"], "baseline wins": o["losses"],
            "unknown": o["unknown"], "failures": o["failures"],
            "win rate (excl. ties)": o["win_rate_excluding_ties"], "win rate (ties=½)": o["win_rate_ties_as_half"],
            "retries (last run)": data.get("retries_this_run"),
        })
    return rows


def category_tables(eval_dir: Path) -> str:
    parts = []
    for path in sorted(eval_dir.glob("*_summary.json")):
        data = load(path)
        if len(data["by_category"]) <= 1:
            continue
        rows = [{"category": key, "n": v["n"], "wins": v["wins"], "ties": v["ties"], "losses": v["losses"],
                 "win rate (excl. ties)": v["win_rate_excluding_ties"]} for key, v in data["by_category"].items()]
        parts.append(f"#### {path.stem.replace('_judged_summary', '')}\n\n" + markdown_table(rows))
    return "\n".join(parts) or "_no per-category breakdown available_\n"


def examples(eval_dir: Path, generations: Path, per_kind: int = 2) -> str:
    """A few wins and losses per comparison, for the error analysis section."""
    parts = []
    for judged in sorted(eval_dir.glob("*_judged.jsonl")):
        # Judgment files may carry a rubric suffix (e.g. persona_lora_vs_base_style_judged.jsonl),
        # so match the longest pairs-file stem that prefixes the judgment stem.
        stem = judged.name[: -len("_judged.jsonl")]
        candidates = [path for path in eval_dir.glob("*_pairs.jsonl") if stem.startswith(path.name[: -len("_pairs.jsonl")])]
        if not candidates:
            continue
        pairs_path = max(candidates, key=lambda path: len(path.name))
        pairs = {json.loads(l)["id"]: json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()}
        verdicts = [json.loads(l) for l in judged.read_text().splitlines() if l.strip()]
        parts.append(f"### {judged.stem}\n")
        for outcome in ("win", "loss", "tie"):
            for v in [v for v in verdicts if v["candidate_outcome"] == outcome][:per_kind]:
                pair = pairs[v["id"]]
                by_model = {pair["mapping"]["A"]: pair["response_a"], pair["mapping"]["B"]: pair["response_b"]}
                parts.append(
                    f"**{outcome.upper()} for {v['candidate']}** — `{v['id']}` ({v.get('category')})\n\n"
                    f"> **Prompt:** {pair['instruction'][:400]}\n\n"
                    f"**{v['candidate']}:** {by_model[v['candidate']][:600]}\n\n"
                    f"**{v['baseline']}:** {by_model[v['baseline']][:600]}\n\n"
                    f"_Judge:_ {v['reason']}\n"
                )
    return "\n".join(parts) or "_no judgments yet_\n"


def generation_rows(generations: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(generations.glob("*.json.meta.json")):
        meta = load(meta_path)
        outputs = load(Path(str(meta_path)[: -len(".meta.json")])) or []
        lengths = [o.get("num_new_tokens", 0) for o in outputs]
        rows.append({
            "file": meta_path.name.replace(".meta.json", ""), "generator": meta["generator"], "n": len(outputs),
            "prompt format": meta["prompt_format"], "decoding": meta["decoding"], "max new tokens": meta["max_new_tokens"],
            "repetition penalty": meta["repetition_penalty"], "no-repeat n-gram": meta["no_repeat_ngram_size"],
            "dtype": meta["torch_dtype"], "mean new tokens": (sum(lengths) / len(lengths)) if lengths else None,
            "hit max tokens": sum(l >= meta["max_new_tokens"] for l in lengths),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", required=True, type=Path)
    args = p.parse_args()
    out = args.outputs
    training = {name: load(out / name / "summary.json") for _, name in RUNS}
    data_meta = {path.name: load(path) for path in sorted((out / "data").rglob("*metadata.json"))}
    report = [
        "# Results summary (auto-generated)\n",
        "Paste/adapt these tables into the report. All values are read from the run artifacts.\n",
        "## Data\n", "```json\n" + json.dumps(data_meta, indent=2) + "\n```\n",
        "## Experimental configuration (values actually used)\n", markdown_table(config_rows(out)),
        "## Training outcomes\n", "```json\n" + json.dumps(training, indent=2) + "\n```\n",
        "## Generation settings\n", markdown_table(generation_rows(out / "generations")),
        "## Pairwise judge results\n", markdown_table(judge_rows(out / "eval")),
        "### Per-category breakdown\n", category_tables(out / "eval"),
        "## Example comparisons\n", examples(out / "eval", out / "generations"),
    ]
    report_dir = out / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.md").write_text("\n".join(report), encoding="utf-8")
    (report_dir / "results.json").write_text(json.dumps({
        "data": data_meta, "configs": config_rows(out), "training": training,
        "generation": generation_rows(out / "generations"), "judging": judge_rows(out / "eval"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"wrote {report_dir / 'results.md'}")


if __name__ == "__main__":
    main()
