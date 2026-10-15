#!/usr/bin/env python
"""From-scratch response-masked SFT for a JSON/JSONL single-turn dataset.

Example (Kaggle notebook):
python scripts/train_sft.py --model Qwen/Qwen2.5-0.5B --train data/train.jsonl \
  --validation data/valid.jsonl --output_dir /kaggle/working/sft

Loss reduction: mean negative log-likelihood over all unmasked response tokens in
each micro-batch (prompt and padding tokens are labelled -100 and ignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from alignment.core import IGNORE_INDEX, _response_logprobs_from_batch, build_sft_batch
from alignment.prompting import PROMPT_TEMPLATE
from alignment.training import (
    ProgressClock, autocast_context, batches, format_rows, load_causal_lm, load_json_records, load_tokenizer,
    save_checkpoint, save_run_config, seed_everything, sft_validation, write_json_atomic, write_metric,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train", required=True)
    p.add_argument("--validation")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--response_field", default="response")
    p.add_argument("--prompt_template", default=PROMPT_TEMPLATE)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validation passes")
    p.add_argument("--eval_at_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_train_examples", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--max_val_examples", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_interval", type=float, default=30.0, help="seconds between progress-bar refreshes")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow_cpu", action="store_true", help="run on CPU (local tests only)")
    return p


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("SFT is intended for a CUDA Kaggle runtime")
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.fp16 = args.fp16 and device.type == "cuda"
    train = load_json_records(args.train)[: args.max_train_examples]
    validation = load_json_records(args.validation)[: args.max_val_examples] if args.validation else []
    tokenizer = load_tokenizer(args.model)
    # Trainable weights stay in fp32; fp16 is used only inside autocast with a GradScaler.
    model = load_causal_lm(args.model, torch.float32, device)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    updates_per_epoch = (steps_per_epoch + args.gradient_accumulation - 1) // args.gradient_accumulation
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    out, metrics = Path(args.output_dir), Path(args.output_dir) / "metrics.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    metrics.unlink(missing_ok=True)  # a rerun starts a clean curve
    save_run_config(out, args, {
        "stage": "sft", "train_examples": len(train), "validation_examples": len(validation),
        "optimizer": "AdamW", "scheduler": "cosine with linear warmup", "warmup_steps": warmup_steps,
        "total_optimizer_steps": total_steps, "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "precision": "fp32 weights + fp16 autocast + GradScaler" if args.fp16 else "fp32",
        "loss_reduction": "mean NLL over unmasked response tokens per micro-batch",
    })
    best, best_step, step = float("inf"), None, 0
    clock = ProgressClock(total_steps)

    def validate(epoch: int) -> None:
        nonlocal best, best_step
        val_loss = sft_validation(model, tokenizer, validation, args, device)
        write_metric(metrics, {"step": step, "epoch": epoch, "val_loss": val_loss, **clock.status(step)})
        # Step 0 is the untouched base model; it is logged for the plot but never saved as "best".
        if step > 0 and val_loss < best:
            best, best_step = val_loss, step
            save_checkpoint(model, tokenizer, out / "best", {"step": step, "val_loss": val_loss})
        tqdm.write(f"[step {step}] val_loss={val_loss:.4f} (best={best:.4f} @ {best_step})")

    print(f"Train examples: {len(train)} | micro-batches/epoch: {steps_per_epoch} | optimizer steps (total): {total_steps}", flush=True)
    if validation and args.eval_at_start:
        validate(0)
    optimizer.zero_grad(set_to_none=True)
    # One bar over optimizer steps. Output from a notebook subprocess is not a TTY, so every refresh
    # becomes a new line; mininterval keeps that to roughly one line per progress_interval seconds.
    overall = tqdm(total=total_steps, desc="optimizer steps", dynamic_ncols=False, ncols=110,
                   mininterval=args.progress_interval, miniters=1)
    for epoch in range(args.epochs):
        window_losses: list[float] = []
        for micro_step, rows in enumerate(batches(train, args.batch_size, True), start=1):
            prompts = format_rows(rows, args.prompt_field, args.prompt_template)
            responses = [str(row[args.response_field]) for row in rows]
            batch = build_sft_batch(tokenizer, prompts, responses, args.max_length)
            with autocast_context(args.fp16, device):
                logps = _response_logprobs_from_batch(model, batch)
                token_count = batch["labels"][:, 1:].ne(IGNORE_INDEX).sum().clamp_min(1).to(device)
                loss = -logps.sum() / token_count
            scaler.scale(loss / args.gradient_accumulation).backward()
            window_losses.append(float(loss.item()))
            if micro_step % args.gradient_accumulation != 0 and micro_step != steps_per_epoch:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            # Average over the whole accumulation window (less noisy than the last micro-batch alone).
            train_loss = sum(window_losses) / len(window_losses)
            window_losses.clear()
            lr = scheduler.get_last_lr()[0]
            write_metric(metrics, {"step": step, "epoch": epoch, "train_loss": train_loss, "lr": lr,
                                   "grad_norm": float(grad_norm), **clock.status(step)})
            overall.update(1)
            overall.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{train_loss:.4f}", lr=f"{lr:.2e}",
                                best_val=("n/a" if best == float("inf") else f"{best:.4f}"))
            if validation and step % args.eval_every == 0 and step != total_steps:
                validate(epoch)
    overall.close()
    if validation:
        validate(args.epochs - 1)
    else:
        save_checkpoint(model, tokenizer, out / "best", {"step": step, "note": "no validation set"})
        best_step = step
    save_checkpoint(model, tokenizer, out / "final", {"step": step})
    write_json_atomic(out / "summary.json", {"best_val_loss": best, "best_step": best_step, "final_step": step,
                                             **clock.status(step)})
    (out / "DONE").write_text("ok\n")
    print(f"Done. Checkpoints in {out}", flush=True)


if __name__ == "__main__":
    main()
