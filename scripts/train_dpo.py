#!/usr/bin/env python
"""Explicit, from-scratch DPO training loop for JSON/JSONL preference pairs.

The policy (fp32 weights, fp16 autocast, GradScaler) lives on cuda:0 and the frozen
fp16 reference on cuda:1 when two GPUs are visible, following the manual's T4x2 tip.
Both are initialized from the same SFT checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from alignment.core import build_dpo_batch, compute_dpo_loss_from_logps
from alignment.prompting import PROMPT_TEMPLATE
from alignment.training import (
    ProgressClock, autocast_context, batches, dpo_logps, dpo_rewards, dpo_validation, format_rows, load_causal_lm,
    load_json_records, load_tokenizer, save_checkpoint, save_run_config, seed_everything, write_json_atomic,
    write_metric,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_checkpoint", required=True, help="SFT best checkpoint; initializes policy and reference")
    p.add_argument("--train", required=True)
    p.add_argument("--validation")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="dpo")
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--chosen_field", default="chosen")
    p.add_argument("--rejected_field", default="rejected")
    p.add_argument("--prompt_template", default=PROMPT_TEMPLATE)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validation passes")
    p.add_argument("--eval_at_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_train_examples", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--max_val_examples", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_interval", type=float, default=30.0)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reference_device", default=None, help="Defaults to cuda:1 when available, otherwise cuda:0")
    p.add_argument("--allow_cpu", action="store_true", help="run on CPU (local tests only)")
    return p


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("DPO is intended for a CUDA Kaggle runtime")
    seed_everything(args.seed)
    if torch.cuda.is_available():
        policy_device = torch.device("cuda:0")
        reference_device = torch.device(args.reference_device or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"))
    else:
        policy_device = reference_device = torch.device("cpu")
    args.fp16 = args.fp16 and policy_device.type == "cuda"
    train = load_json_records(args.train)[: args.max_train_examples]
    validation = load_json_records(args.validation)[: args.max_val_examples] if args.validation else []
    tokenizer = load_tokenizer(args.sft_checkpoint)
    policy = load_causal_lm(args.sft_checkpoint, torch.float32, policy_device)
    # The reference never receives gradients, so fp16 weights are safe and halve its memory.
    reference_dtype = torch.float16 if args.fp16 and reference_device.type == "cuda" else torch.float32
    reference = load_causal_lm(args.sft_checkpoint, reference_dtype, reference_device)
    reference.eval()
    reference.requires_grad_(False)
    policy.config.use_cache = False
    reference.config.use_cache = False
    policy.train()
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    out, metrics = Path(args.output_dir), Path(args.output_dir) / "metrics.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    metrics.unlink(missing_ok=True)
    micro_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    total_steps = max(1, ((micro_per_epoch + args.gradient_accumulation - 1) // args.gradient_accumulation) * args.epochs)
    save_run_config(out, args, {
        "stage": "dpo", "train_pairs": len(train), "validation_pairs": len(validation),
        "reference_checkpoint": str(args.sft_checkpoint), "policy_device": str(policy_device),
        "reference_device": str(reference_device), "reference_dtype": str(reference_dtype),
        "scheduler": "constant (no decay)", "warmup_steps": 0, "total_optimizer_steps": total_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "precision": "policy fp32 + fp16 autocast + GradScaler" if args.fp16 else "fp32",
        "checkpoint_selection": "best validation reward accuracy (strict r_w > r_l)",
    })
    best_accuracy, best_step, step = float("-inf"), None, 0
    clock = ProgressClock(total_steps)

    def validate(epoch: int) -> None:
        nonlocal best_accuracy, best_step
        result = dpo_validation(policy, reference, tokenizer, validation, args, policy_device)
        write_metric(metrics, {"step": step, "epoch": epoch, **result, **clock.status(step)})
        # At step 0 policy == reference so every pair ties and accuracy is exactly 0; never "best".
        if step > 0 and result["val_reward_accuracy"] > best_accuracy:
            best_accuracy, best_step = result["val_reward_accuracy"], step
            save_checkpoint(policy, tokenizer, out / "best", {"step": step, **result})
        tqdm.write(f"[{args.run_name} step {step}] val_loss={result['val_dpo_loss']:.4f} "
                   f"acc={result['val_reward_accuracy']:.3f} margin={result['val_reward_margin']:.4f} "
                   f"(best acc={best_accuracy:.3f} @ {best_step})")

    print(f"Train pairs: {len(train)} | validation pairs: {len(validation)} | optimizer steps: {total_steps}", flush=True)
    if validation and args.eval_at_start:
        validate(0)
    optimizer.zero_grad(set_to_none=True)
    overall = tqdm(total=total_steps, desc=f"{args.run_name} steps", ncols=110, mininterval=args.progress_interval, miniters=1)
    for epoch in range(args.epochs):
        window = {"loss": 0.0, "correct": 0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0, "pairs": 0, "micro": 0}
        for micro_step, rows in enumerate(batches(train, args.batch_size, True), start=1):
            prompts = format_rows(rows, args.prompt_field, args.prompt_template)
            chosen = [str(row[args.chosen_field]) for row in rows]
            rejected = [str(row[args.rejected_field]) for row in rows]
            batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, args.max_length)
            with autocast_context(args.fp16, policy_device):
                policy_chosen, policy_rejected = dpo_logps(policy, batch)
                with torch.no_grad():
                    ref_chosen, ref_rejected = dpo_logps(reference, batch)
                ref_chosen, ref_rejected = ref_chosen.to(policy_device), ref_rejected.to(policy_device)
                loss = compute_dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)
            # Only the policy graph exists, so backprop never touches the reference.
            scaler.scale(loss / args.gradient_accumulation).backward()
            with torch.no_grad():
                r_w, r_l = dpo_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)
                window["loss"] += float(loss.item()); window["micro"] += 1
                window["correct"] += int((r_w > r_l).sum()); window["margin"] += float((r_w - r_l).sum())
                window["chosen"] += float(r_w.sum()); window["rejected"] += float(r_l.sum()); window["pairs"] += len(rows)
            if micro_step % args.gradient_accumulation != 0 and micro_step != micro_per_epoch:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            step += 1
            pairs = max(window["pairs"], 1)
            record = {
                "step": step, "epoch": epoch, "train_dpo_loss": window["loss"] / max(window["micro"], 1),
                "train_reward_accuracy": window["correct"] / pairs, "train_reward_margin": window["margin"] / pairs,
                "train_reward_chosen": window["chosen"] / pairs, "train_reward_rejected": window["rejected"] / pairs,
                "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm), **clock.status(step),
            }
            write_metric(metrics, record)
            window = {key: 0 for key in window}
            overall.update(1)
            overall.set_postfix(loss=f"{record['train_dpo_loss']:.4f}", acc=f"{record['train_reward_accuracy']:.2f}")
            if validation and step % args.eval_every == 0 and step != total_steps:
                validate(epoch)
    overall.close()
    if validation:
        validate(args.epochs - 1)
    else:
        save_checkpoint(policy, tokenizer, out / "best", {"step": step, "note": "no validation set"})
        best_step = step
    save_checkpoint(policy, tokenizer, out / "final", {"step": step})
    write_json_atomic(out / "summary.json", {"best_val_reward_accuracy": best_accuracy, "best_step": best_step,
                                             "final_step": step, **clock.status(step)})
    (out / "DONE").write_text("ok\n")
    print(f"Done. Checkpoints in {out}", flush=True)


if __name__ == "__main__":
    main()
