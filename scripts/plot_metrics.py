#!/usr/bin/env python
"""Training plots for the report (manual section 14.4).

Reads ``metrics.jsonl`` from each run directory and writes PNGs:
  sft.png               train loss (raw + EMA) and validation loss
  <dpo_run>.png         DPO loss, reward accuracy, reward margin (train EMA + validation)
  persona_lora.png      TRL loss and reward accuracy/margin (train + eval)
  judge_results.png     win / tie / loss shares per comparison, if judge summaries exist
Missing runs are skipped, so the script can run after any stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN, VAL, THIRD = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated palette)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
})


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def series(rows: list[dict], key: str, x: str = "step") -> tuple[list, list]:
    points = [(row[x], row[key]) for row in rows if key in row and row.get(key) is not None and x in row]
    return [p[0] for p in points], [p[1] for p in points]


def ema(values: list[float], alpha: float) -> list[float]:
    out, current = [], None
    for value in values:
        current = value if current is None else alpha * value + (1 - alpha) * current
        out.append(current)
    return out


def best_step(run_dir: Path) -> int | None:
    summary = run_dir / "summary.json"
    return json.loads(summary.read_text()).get("best_step") if summary.exists() else None


def mark_best(ax, step) -> None:
    if step:
        ax.axvline(step, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate(f"best ckpt (step {step})", (step, 0), xycoords=("data", "axes fraction"), xytext=(3, 4),
                    textcoords="offset points", fontsize=7, color=MUTED, rotation=90, va="bottom")


def plot_train_val(ax, rows, train_key, val_key, title, ylabel, alpha, best):
    x, y = series(rows, train_key)
    if x:
        ax.plot(x, y, color=TRAIN, alpha=0.25, linewidth=1)
        ax.plot(x, ema(y, alpha), color=TRAIN, label=f"train (EMA α={alpha})")
    x, y = series(rows, val_key)
    if x:
        ax.plot(x, y, color=VAL, marker="o", markersize=4, label="validation")
    mark_best(ax, best)
    ax.set_title(title); ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def plot_sft(run_dir: Path, out: Path) -> Path | None:
    rows = read_metrics(run_dir / "metrics.jsonl")
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    plot_train_val(ax, rows, "train_loss", "val_loss", "SFT: Qwen2.5-0.5B on UltraChat (response-token NLL)",
                   "loss (mean NLL / response token)", 0.1, best_step(run_dir))
    fig.text(0.01, 0.01, "Faint line: raw per-step train loss. Best checkpoint = lowest validation loss.", fontsize=7, color=MUTED)
    return save(fig, out / "sft.png")


def plot_dpo(run_dir: Path, out: Path, label: str) -> Path | None:
    rows = read_metrics(run_dir / "metrics.jsonl")
    if not rows:
        return None
    best = best_step(run_dir)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    plot_train_val(axes[0], rows, "train_dpo_loss", "val_dpo_loss", "DPO loss", "loss", 0.1, best)
    axes[0].axhline(0.6931, color=GRID, linewidth=1, linestyle="--")
    plot_train_val(axes[1], rows, "train_reward_accuracy", "val_reward_accuracy", "Reward accuracy (r_w > r_l)",
                   "fraction of pairs", 0.1, best)
    plot_train_val(axes[2], rows, "train_reward_margin", "val_reward_margin", "Reward margin (r_w − r_l)",
                   "mean margin", 0.1, best)
    fig.suptitle(f"{label} DPO from SFT best (β=0.1)", fontweight="bold")
    fig.text(0.01, 0.01, "Step 0 = policy identical to reference (accuracy exactly 0 by the strict comparison). "
             "Dashed line in the loss panel: log 2. Best checkpoint = highest validation reward accuracy.",
             fontsize=7, color=MUTED)
    return save(fig, out / f"{run_dir.name}.png")


def plot_persona(run_dir: Path, out: Path) -> Path | None:
    rows = read_metrics(run_dir / "metrics.jsonl")
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, train_key, eval_key, title in (
        (axes[0], "loss", "eval_loss", "LoRA-DPO loss"),
        (axes[1], "rewards/accuracies", "eval_rewards/accuracies", "Reward accuracy"),
    ):
        x, y = series(rows, train_key)
        if x:
            ax.plot(x, y, color=TRAIN, label="train (logged every 5 steps)")
        x, y = series(rows, eval_key)
        if x:
            ax.plot(x, y, color=VAL, marker="o", markersize=4, label="held-out eval")
        ax.set_title(title); ax.set_xlabel("optimizer step"); ax.legend(loc="best")
    x, y = series(rows, "eval_rewards/margins")
    if x:
        twin_note = ", ".join(f"{v:.2f}" for v in y[-3:])
        fig.text(0.01, 0.01, f"Last held-out reward margins: {twin_note}", fontsize=7, color=MUTED)
    fig.suptitle("Persona LoRA-DPO (TRL DPOTrainer) on SmolLM2-360M-Instruct", fontweight="bold")
    return save(fig, out / "persona_lora.png")


def plot_judges(eval_dir: Path, out: Path) -> Path | None:
    summaries = sorted(eval_dir.glob("*_summary.json"))
    if not summaries:
        return None
    labels, wins, ties, losses, other = [], [], [], [], []
    for path in summaries:
        data = json.loads(path.read_text())
        o = data["overall"]
        if not o["n"]:
            continue
        labels.append(f"{data['candidate']} vs {data['baseline']}\n({path.stem.replace('_judged_summary', '')}, n={o['n']})")
        wins.append(o["wins"] / o["n"]); ties.append(o["ties"] / o["n"]); losses.append(o["losses"] / o["n"])
        other.append((o["unknown"] + o["failures"]) / o["n"])
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(labels) + 1.4))
    left = [0.0] * len(labels)
    for values, color, name in ((wins, TRAIN, "candidate wins"), (ties, GRID, "tie"), (losses, VAL, "baseline wins"),
                                (other, MUTED, "unknown / failed")):
        ax.barh(labels, values, left=left, color=color, edgecolor="white", linewidth=2, label=name)
        for i, value in enumerate(values):
            if value >= 0.06:
                ax.text(left[i] + value / 2, i, f"{100 * value:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if color in (TRAIN, VAL, MUTED) else INK)
        left = [l + v for l, v in zip(left, values)]
    ax.set_xlim(0, 1); ax.set_xlabel("share of judged pairs"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.suptitle("Pairwise judge results", fontweight="bold")
    return save(fig, out / "judge_results.png")


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", required=True, type=Path, help="root output dir containing sft/, hh_dpo/, ...")
    p.add_argument("--plots_dir", type=Path, default=None)
    args = p.parse_args()
    out = args.plots_dir or args.outputs / "plots"
    plot_sft(args.outputs / "sft", out)
    plot_dpo(args.outputs / "hh_dpo", out, "HH-RLHF")
    plot_dpo(args.outputs / "ultrafeedback_dpo", out, "UltraFeedback")
    plot_persona(args.outputs / "persona_lora", out)
    plot_judges(args.outputs / "eval", out)


if __name__ == "__main__":
    main()
