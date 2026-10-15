#!/usr/bin/env python
"""Explicit, from-scratch DPO training loop for JSON/JSONL preference pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.core import _response_logprobs_from_batch, build_dpo_batch, compute_dpo_loss_from_logps
from alignment.training import autocast_context, batches, dpo_validation, load_json_records, save_checkpoint, seed_everything, write_metric


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_checkpoint", required=True, help="SFT best checkpoint; initializes policy and reference")
    p.add_argument("--train", required=True)
    p.add_argument("--validation")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--chosen_field", default="chosen")
    p.add_argument("--rejected_field", default="rejected")
    p.add_argument("--prompt_template", default="{prompt}")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reference_device", default=None, help="Defaults to cuda:1 when available, otherwise cuda:0")
    return p


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DPO is intended for a CUDA Kaggle runtime")
    seed_everything(args.seed)
    policy_device = torch.device("cuda:0")
    ref_name = args.reference_device or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    reference_device = torch.device(ref_name)
    train, validation = load_json_records(args.train), load_json_records(args.validation) if args.validation else []
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.float32).to(policy_device)
    # Reference can use fp16 because it never receives gradients.
    reference_dtype = torch.float16 if args.fp16 and reference_device.type == "cuda" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=reference_dtype).to(reference_device)
    reference.eval()
    reference.requires_grad_(False)
    policy.config.use_cache = False
    reference.config.use_cache = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    out, metrics = Path(args.output_dir), Path(args.output_dir) / "metrics.jsonl"
    best_accuracy, step = float("-inf"), 0
    micro_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for micro_step, rows in enumerate(batches(train, args.batch_size, True), start=1):
            prompts = [args.prompt_template.format(prompt=row[args.prompt_field]) for row in rows]
            chosen = [str(row[args.chosen_field]) for row in rows]
            rejected = [str(row[args.rejected_field]) for row in rows]
            batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, args.max_length)
            with autocast_context(args.fp16, policy_device):
                policy_chosen = _response_logprobs_from_batch(policy, batch["chosen"])
                policy_rejected = _response_logprobs_from_batch(policy, batch["rejected"])
                with torch.no_grad():
                    ref_chosen = _response_logprobs_from_batch(reference, batch["chosen"]).to(policy_device)
                    ref_rejected = _response_logprobs_from_batch(reference, batch["rejected"]).to(policy_device)
                loss = compute_dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if micro_step % args.gradient_accumulation != 0 and micro_step != micro_per_epoch:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            step += 1
            write_metric(metrics, {"step": step, "epoch": epoch, "train_dpo_loss": float(loss.item())})
            if validation and step % args.eval_every == 0:
                result = dpo_validation(policy, reference, tokenizer, validation, args, policy_device)
                result.update({"step": step, "epoch": epoch}); write_metric(metrics, result)
                if result["val_reward_accuracy"] > best_accuracy:
                    best_accuracy = result["val_reward_accuracy"]; save_checkpoint(policy, tokenizer, out / "best")
    if validation:
        result = dpo_validation(policy, reference, tokenizer, validation, args, policy_device)
        result.update({"step": step, "epoch": args.epochs}); write_metric(metrics, result)
        if result["val_reward_accuracy"] > best_accuracy: save_checkpoint(policy, tokenizer, out / "best")
    save_checkpoint(policy, tokenizer, out / "final")


if __name__ == "__main__":
    main()
