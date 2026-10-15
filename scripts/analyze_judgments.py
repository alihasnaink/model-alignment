#!/usr/bin/env python
"""Statistical and error analysis of the pairwise judgments (report sections 14.5-14.7).

For each comparison: win/tie/loss with a two-sided sign test and a Wilson 95% interval on the
win rate excluding ties; per-category results; judge-reliability checks (position bias,
length preference, verdicts on byte-identical response pairs); and generation diagnostics
(length, truncation at the token cap, refusal rate, repetition). Writes
``<outputs>/report/judge_analysis.json`` and ``judge_analysis.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

COMPARISONS = [
    # judged stem, pairs stem, candidate generations, baseline generations
    ("hh_dpo_vs_sft", "hh_dpo_vs_sft", "hh_dpo_hh_eval", "sft_hh_eval"),
    ("ultrafeedback_dpo_vs_sft", "ultrafeedback_dpo_vs_sft", "ultrafeedback_dpo_alpaca_eval", "sft_alpaca_eval"),
    ("persona_lora_vs_base_style", "persona_lora_vs_base", "persona_lora_persona_eval", "persona_base_persona_eval"),
    ("persona_lora_vs_base_helpfulness", "persona_lora_vs_base", "persona_lora_persona_eval", "persona_base_persona_eval"),
    ("sft_vs_base_alpaca", "sft_vs_base_alpaca", "sft_alpaca_eval", "base_alpaca_eval"),
]
REFUSAL = re.compile(
    r"\b(i can(?:'|no)t|i cannot|i'm (?:not able|unable)|i am (?:not able|unable)|i won't|i will not|i'm sorry|"
    r"i apologi[sz]e|as an ai|not (?:able|appropriate) to (?:help|provide|assist)|i do not have the (?:capability|ability)|"
    r"(?:is|are|would be) (?:illegal|unethical|harmful)|seek (?:professional )?help|please (?:contact|reach out))", re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of wins vs losses (ties excluded) under p = 0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 3), round(centre + half, 3)


def outcome_stats(rows: list[dict]) -> dict:
    c = Counter(r["candidate_outcome"] for r in rows)
    w, l, t = c["win"], c["loss"], c["tie"]
    return {"n": len(rows), "wins": w, "ties": t, "losses": l,
            "win_rate_excl_ties": round(w / (w + l), 3) if w + l else None,
            "win_rate_ties_half": round((w + 0.5 * t) / len(rows), 3) if rows else None,
            "wilson95_excl_ties": wilson(w, w + l), "sign_test_p": round(sign_test_p(w, l), 4)}


def repetition_ratio(text: str, n: int = 4) -> float:
    """Share of word 4-grams that are repeats; high values indicate degenerate loops."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def position_balanced(judged: list[dict], pairs: dict) -> dict:
    """Candidate win rate when shown as A vs as B; their mean cancels a constant position preference."""
    def rate(rows):
        c = Counter(r["candidate_outcome"] for r in rows)
        return round(c["win"] / (c["win"] + c["loss"]), 3) if c["win"] + c["loss"] else None
    in_a = [r for r in judged if pairs[r["id"]]["mapping"]["A"] == r["candidate"]]
    in_b = [r for r in judged if pairs[r["id"]]["mapping"]["A"] != r["candidate"]]
    a, b = rate(in_a), rate(in_b)
    return {"candidate_as_A": {"n": len(in_a), "win_rate": a}, "candidate_as_B": {"n": len(in_b), "win_rate": b},
            "position_balanced_win_rate": round((a + b) / 2, 3) if a is not None and b is not None else None}


def similarity(cand: dict, base: dict) -> dict:
    """How far the candidate moved from its baseline: shared-prefix share, identical openings, sequence similarity."""
    ids = list(cand)
    prefix = [len(os.path.commonprefix([cand[i], base[i]])) / max(len(cand[i]), len(base[i]), 1) for i in ids]
    ratio = [SequenceMatcher(None, cand[i][:1500], base[i][:1500]).ratio() for i in ids]
    return {"mean_shared_prefix_share": round(sum(prefix) / len(ids), 3),
            "first_50_chars_identical": round(sum(cand[i][:50] == base[i][:50] for i in ids) / len(ids), 3),
            "mean_sequence_similarity": round(sum(ratio) / len(ids), 3)}


def generation_stats(rows: list[dict], cap: int) -> dict:
    words = [len(r["output"].split()) for r in rows]
    return {
        "mean_words": round(sum(words) / len(words), 1),
        "median_words": sorted(words)[len(words) // 2],
        "hit_token_cap": sum(r.get("num_new_tokens", 0) >= cap for r in rows),
        "empty": sum(not r["output"].strip() for r in rows),
        "refusal_rate": round(sum(bool(REFUSAL.search(r["output"][:400])) for r in rows) / len(rows), 3),
        "degenerate_repetition(>30% repeated 4-grams)": sum(repetition_ratio(r["output"]) > 0.3 for r in rows),
    }


def analyse(outputs: Path, generations: Path) -> dict:
    eval_dir = outputs / "eval"
    results = {}
    for judged_stem, pairs_stem, cand_gen, base_gen in COMPARISONS:
        judged_path = eval_dir / f"{judged_stem}_judged.jsonl"
        if not judged_path.exists():
            continue
        judged = read_jsonl(judged_path)
        pairs = {p["id"]: p for p in read_jsonl(eval_dir / f"{pairs_stem}_pairs.jsonl")}
        cand_rows = json.loads((generations / f"{cand_gen}.json").read_text())
        base_rows = json.loads((generations / f"{base_gen}.json").read_text())
        cap = json.loads((generations / f"{cand_gen}.json.meta.json").read_text())["max_new_tokens"]
        cand_out = {r["id"]: r["output"] for r in cand_rows}
        base_out = {r["id"]: r["output"] for r in base_rows}

        by_cat = defaultdict(list)
        for r in judged:
            by_cat[r.get("category") or "all"].append(r)

        identical = [r for r in judged if cand_out[r["id"]].strip() == base_out[r["id"]].strip()]
        differing = [r for r in judged if cand_out[r["id"]].strip() != base_out[r["id"]].strip()]

        decided = [r for r in judged if r["winner_label"] in ("A", "B")]
        picked_a = sum(r["winner_label"] == "A" for r in decided)
        longer_won = sum(
            len(pairs[r["id"]]["response_" + r["winner_label"].lower()]) >
            len(pairs[r["id"]]["response_" + ("b" if r["winner_label"] == "A" else "a")]) for r in decided)

        cand_tokens = {r["id"]: r.get("num_new_tokens", 0) for r in cand_rows}
        base_tokens = {r["id"]: r.get("num_new_tokens", 0) for r in base_rows}
        truncation = {
            "only_candidate_hit_cap": outcome_stats([r for r in judged if cand_tokens[r["id"]] >= cap > base_tokens[r["id"]]]),
            "only_baseline_hit_cap": outcome_stats([r for r in judged if base_tokens[r["id"]] >= cap > cand_tokens[r["id"]]]),
            "neither_hit_cap": outcome_stats([r for r in judged if max(cand_tokens[r["id"]], base_tokens[r["id"]]) < cap]),
        }
        results[judged_stem] = {
            "rubric": judged[0]["rubric"], "candidate": judged[0]["candidate"], "baseline": judged[0]["baseline"],
            "overall": outcome_stats(judged),
            "position_balanced": position_balanced(judged, pairs),
            "by_truncation": truncation,
            "output_similarity": similarity(cand_out, base_out),
            "excluding_identical_outputs": outcome_stats(differing),
            "identical_output_pairs": {"count": len(identical),
                                       "verdicts": dict(Counter(r["winner_label"] for r in identical))},
            "by_category": {cat: outcome_stats(rows) for cat, rows in sorted(by_cat.items())},
            "judge_reliability": {
                "decided_verdicts": len(decided),
                "share_picking_A": round(picked_a / len(decided), 3) if decided else None,
                "position_bias_p": round(sign_test_p(picked_a, len(decided) - picked_a), 4),
                "share_longer_response_won": round(longer_won / len(decided), 3) if decided else None,
                "length_bias_p": round(sign_test_p(longer_won, len(decided) - longer_won), 4),
            },
            "generations": {judged[0]["candidate"]: generation_stats(cand_rows, cap),
                            judged[0]["baseline"]: generation_stats(base_rows, cap)},
        }
    return results


def to_markdown(results: dict) -> str:
    out = ["# Judge analysis (auto-generated by scripts/analyze_judgments.py)\n",
           "Win rates exclude ties; 95% CI = Wilson interval; p = two-sided exact sign test (wins vs losses).\n",
           "| comparison | rubric | n | W / T / L | win rate | 95% CI | p | position-balanced win rate | identical pairs (verdicts) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        o = r["overall"]
        out.append(f"| {r['candidate']} vs {r['baseline']} ({name}) | {r['rubric']} | {o['n']} | "
                   f"{o['wins']} / {o['ties']} / {o['losses']} | {o['win_rate_excl_ties']} | {o['wilson95_excl_ties']} | "
                   f"{o['sign_test_p']} | {r['position_balanced']['position_balanced_win_rate']} | "
                   f"{r['identical_output_pairs']['count']} {r['identical_output_pairs']['verdicts'] or ''} |")
    out.append("\n## Judge reliability\n\n| comparison | decided | share picking A | position p | share longer won | length p |\n|---|---|---|---|---|---|")
    for name, r in results.items():
        j = r["judge_reliability"]
        out.append(f"| {name} | {j['decided_verdicts']} | {j['share_picking_A']} | {j['position_bias_p']} | "
                   f"{j['share_longer_response_won']} | {j['length_bias_p']} |")
    out.append("\n## Position effect, truncation, and how far the candidate moved from its baseline\n\n"
               "| comparison | win as A | win as B | W/T/L only candidate truncated | W/T/L neither truncated | "
               "shared prefix | first 50 chars identical | similarity |\n|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        pb, t, sim = r["position_balanced"], r["by_truncation"], r["output_similarity"]
        c, n = t["only_candidate_hit_cap"], t["neither_hit_cap"]
        out.append(f"| {name} | {pb['candidate_as_A']['win_rate']} (n={pb['candidate_as_A']['n']}) | "
                   f"{pb['candidate_as_B']['win_rate']} (n={pb['candidate_as_B']['n']}) | "
                   f"{c['wins']}/{c['ties']}/{c['losses']} | {n['wins']}/{n['ties']}/{n['losses']} ({n['win_rate_excl_ties']}) | "
                   f"{sim['mean_shared_prefix_share']} | {sim['first_50_chars_identical']} | {sim['mean_sequence_similarity']} |")
    out.append("\n## Generation diagnostics\n\n| comparison | model | mean words | median | hit cap | refusal rate | degenerate repetition |\n|---|---|---|---|---|---|---|")
    for name, r in results.items():
        for model, g in r["generations"].items():
            out.append(f"| {name} | {model} | {g['mean_words']} | {g['median_words']} | {g['hit_token_cap']} | "
                       f"{g['refusal_rate']} | {g['degenerate_repetition(>30% repeated 4-grams)']} |")
    for name, r in results.items():
        if len(r["by_category"]) > 1:
            out.append(f"\n## {name}: by category\n\n| category | n | W / T / L | win rate | 95% CI | p |\n|---|---|---|---|---|---|")
            for cat, s in r["by_category"].items():
                out.append(f"| {cat} | {s['n']} | {s['wins']} / {s['ties']} / {s['losses']} | {s['win_rate_excl_ties']} | "
                           f"{s['wilson95_excl_ties']} | {s['sign_test_p']} |")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", required=True, type=Path, help="folder with eval/ (judgments + pairs)")
    p.add_argument("--generations", type=Path, default=None, help="defaults to <outputs>/generations")
    args = p.parse_args()
    results = analyse(args.outputs, args.generations or args.outputs / "generations")
    report = args.outputs / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "judge_analysis.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (report / "judge_analysis.md").write_text(to_markdown(results), encoding="utf-8")
    print(to_markdown(results))


if __name__ == "__main__":
    main()
