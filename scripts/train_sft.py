#!/usr/bin/env python
"""From-scratch response-masked SFT for a JSON/JSONL single-turn dataset.

Example (Kaggle notebook):
python scripts/train_sft.py --model Qwen/Qwen2.5-0.5B --train data/train.jsonl \
  --validation data/valid.jsonl --output_dir /kaggle/working/sft
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from alignment.core import _response_logprobs_from_batch, build_sft_batch
from alignment.training import autocast_context, batches, load_json_records, save_checkpoint, seed_everything, sft_validation, write_metric


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train", required=True)
    p.add_argument("--validation")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--response_field", default="response")
    p.add_argument("--prompt_template", default="{prompt}")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SFT is intended for a CUDA Kaggle runtime")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    train, validation = load_json_records(args.train), load_json_records(args.validation) if args.validation else []
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    total_steps = max(1, (steps_per_epoch * args.epochs + args.gradient_accumulation - 1) // args.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    out, metrics = Path(args.output_dir), Path(args.output_dir) / "metrics.jsonl"
    best = float("inf")
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for micro_step, rows in enumerate(batches(train, args.batch_size, True), start=1):
            prompts = [args.prompt_template.format(prompt=row[args.prompt_field]) for row in rows]
            responses = [str(row[args.response_field]) for row in rows]
            batch = build_sft_batch(tokenizer, prompts, responses, args.max_length)
            with autocast_context(args.fp16, device):
                logps = _response_logprobs_from_batch(model, batch)
                token_count = batch["labels"][:, 1:].ne(-100).sum().clamp_min(1).to(device)
                loss = -logps.sum() / token_count / args.gradient_accumulation
            scaler.scale(loss).backward()
            if micro_step % args.gradient_accumulation != 0 and micro_step != steps_per_epoch:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            write_metric(metrics, {"step": step, "epoch": epoch, "train_loss": float(loss.item() * args.gradient_accumulation), "lr": scheduler.get_last_lr()[0]})
            if step % args.eval_every == 0 and validation:
                val_loss = sft_validation(model, tokenizer, validation, args, device)
                write_metric(metrics, {"step": step, "epoch": epoch, "val_loss": val_loss})
                if val_loss < best:
                    best = val_loss; save_checkpoint(model, tokenizer, out / "best")
    if validation:
        val_loss = sft_validation(model, tokenizer, validation, args, device)
        write_metric(metrics, {"step": step, "epoch": args.epochs, "val_loss": val_loss})
        if val_loss < best: save_checkpoint(model, tokenizer, out / "best")
    save_checkpoint(model, tokenizer, out / "final")


if __name__ == "__main__":
    main()
