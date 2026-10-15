#!/usr/bin/env python
"""The assignment's allowed TRL DPOTrainer + PEFT LoRA persona experiment."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.training import load_json_records, seed_everything


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    rows = load_json_records(args.data)
    required = {"prompt", "chosen", "rejected"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"persona data must contain {sorted(required)}")
    # Use TRL's conversational preference schema so the instruct model's chat
    # template supplies the user/assistant delimiters consistently.
    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": str(row["prompt"])}],
            "chosen": [{"role": "assistant", "content": str(row["chosen"])}],
            "rejected": [{"role": "assistant", "content": str(row["rejected"])}],
        }
        for row in rows
    ])
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    config_kwargs = dict(
        output_dir=args.output_dir, learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=10, save_strategy="epoch", fp16=torch.cuda.is_available(), report_to="none", beta=args.beta,
    )
    # TRL renamed max_length in newer releases. Only pass accepted configuration keys.
    if "max_length" in inspect.signature(DPOConfig).parameters: config_kwargs["max_length"] = args.max_length
    if "max_prompt_length" in inspect.signature(DPOConfig).parameters: config_kwargs["max_prompt_length"] = args.max_length // 2
    train_args = DPOConfig(**config_kwargs)
    trainer_kwargs = dict(model=model, args=train_args, train_dataset=dataset, peft_config=lora)
    trainer_signature = inspect.signature(DPOTrainer).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(Path(args.output_dir) / "final")
    tokenizer.save_pretrained(Path(args.output_dir) / "final")


if __name__ == "__main__":
    main()
