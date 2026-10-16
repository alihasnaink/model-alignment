#!/usr/bin/env python
"""Build the report figures (vector PDF) from the committed run artifacts in outputs/.

python report/make_figures.py   ->  report/figures/{fig_sft_persona,fig_dpo,fig_length}.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "report" / "figures"
# Series colours (blue, orange) and greys for text and grid lines.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight",
})


def metrics(run: str) -> list[dict]:
    path = OUT / run / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def series(rows, key, x="step"):
    pts = [(r[x], r[key]) for r in rows if r.get(key) is not None and x in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def ema(values, alpha=0.1):
    out, cur = [], None
    for v in values:
        cur = v if cur is None else alpha * v + (1 - alpha) * cur
        out.append(cur)
    return out


def fig_sft_persona() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    sft = metrics("sft")
    x, y = series(sft, "train_loss")
    axes[0].plot(x, y, color=BLUE, alpha=0.2, linewidth=0.6)
    axes[0].plot(x, ema(y), color=BLUE, label="train (EMA 0.1)")
    x, y = series(sft, "val_loss")
    axes[0].plot(x, y, color=ORANGE, marker="o", markersize=3, label="validation")
    axes[0].set(title="(a) SFT loss", xlabel="optimizer step", ylabel="NLL per response token")
    axes[0].legend()
    persona = metrics("persona_lora")
    for ax, train_key, eval_key, title, ylabel in (
        (axes[1], "loss", "eval_loss", "(b) Persona LoRA-DPO loss", "DPO loss"),
        (axes[2], "rewards/margins", "eval_rewards/margins", "(c) Persona reward margin", r"$r_w - r_l$"),
    ):
        x, y = series(persona, train_key)
        ax.plot(x, y, color=BLUE, label="train")
        x, y = series(persona, eval_key)
        ax.plot(x, y, color=ORANGE, marker="o", markersize=3, label="held-out (20 pairs)")
        ax.set(title=title, xlabel="optimizer step", ylabel=ylabel)
        ax.legend()
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig_sft_persona.pdf")


def fig_dpo() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    panels = [("dpo_loss", "(a) DPO loss", "loss"), ("reward_accuracy", r"(b) Reward accuracy ($r_w > r_l$)", "fraction of pairs"),
              ("reward_margin", r"(c) Reward margin ($r_w - r_l$)", "mean margin")]
    for run, color, label in (("hh_dpo", BLUE, "HH-RLHF"), ("ultrafeedback_dpo", ORANGE, "UltraFeedback")):
        rows = metrics(run)
        best = json.loads((OUT / run / "summary.json").read_text())["best_step"]
        for ax, (key, title, ylabel) in zip(axes, panels):
            x, y = series(rows, f"train_{key}")
            ax.plot(x, ema(y), color=color, alpha=0.45, linewidth=1.0)
            x, y = series(rows, f"val_{key}")
            ax.plot(x, y, color=color, marker="o", markersize=3, label=f"{label} validation")
            bx = [xi for xi in x if xi == best]
            ax.plot(bx, [y[x.index(best)]], marker="*", markersize=9, color=color, linestyle="none")
            ax.set(title=title, xlabel="optimizer step", ylabel=ylabel)
    axes[0].axhline(0.6931, color=MUTED, linewidth=0.7, linestyle="--")
    axes[0].text(5, 0.697, "log 2", fontsize=6.5, color=MUTED)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig_dpo.pdf")


def fig_length() -> None:
    gen = OUT / "generations"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.6, 1]})
    sft = json.loads((gen / "sft_alpaca_eval.json").read_text())
    uf = json.loads((gen / "ultrafeedback_dpo_alpaca_eval.json").read_text())
    bins = list(range(0, 545, 32))
    for rows, color, label in ((sft, BLUE, "SFT"), (uf, ORANGE, "UF-DPO")):
        tokens = [min(r["num_new_tokens"], 512) for r in rows]
        mean = sum(tokens) / len(tokens)
        # Outlined histograms: overlapping filled bars would blend into a third, meaningless colour.
        axes[0].hist(tokens, bins=bins, histtype="step", color=color, linewidth=1.6, label=f"{label} (mean {mean:.0f} tokens)")
    axes[0].axvline(512, color=MUTED, linestyle="--", linewidth=0.8)
    axes[0].text(505, axes[0].get_ylim()[1] * 0.55, "512-token\ncap", fontsize=6.5, color=MUTED, ha="right")
    axes[0].set_xlabel("new tokens"); axes[0].set_ylabel("responses")
    axes[0].set_title("(a) Response length on AlpacaEval (300 prompts)", pad=16)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2)
    analysis = json.loads((OUT / "report" / "judge_analysis.json").read_text())["ultrafeedback_dpo_vs_sft"]["by_truncation"]
    groups = [("only UF-DPO\ntruncated", analysis["only_candidate_hit_cap"]), ("neither\ntruncated", analysis["neither_hit_cap"]),
              ("only SFT\ntruncated", analysis["only_baseline_hit_cap"])]
    labels = [g for g, _ in groups]
    rates = [s["win_rate_excl_ties"] or 0 for _, s in groups]
    bars = axes[1].bar(labels, rates, color=[ORANGE, MUTED, BLUE], width=0.6)
    for bar, (_, s) in zip(bars, groups):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"{s['wins']}/{s['ties']}/{s['losses']}\n(n={s['n']})",
                     ha="center", fontsize=6.5, color=INK, zorder=3,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axes[1].axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8, zorder=1)
    axes[1].set(ylabel="win rate (excl. ties)", ylim=(0, 1.3))
    axes[1].set_title("(b) UF-DPO win rate by truncation", pad=16)
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig_length.pdf")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_sft_persona()
    fig_dpo()
    fig_length()
    print("figures written to", FIG)
