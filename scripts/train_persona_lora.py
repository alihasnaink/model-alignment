#!/usr/bin/env python
"""The assignment's allowed TRL DPOTrainer + PEFT LoRA persona experiment (Task 5).

Run on ONE GPU (the notebook sets CUDA_VISIBLE_DEVICES): with two visible GPUs the
HF Trainer silently switches to DataParallel, which is unnecessary for a 360M model.

LoRA choices (justify in the report):
* r=16, alpha=32 (scale 2), dropout 0.05: enough capacity for a style shift learned
  from ~200 pairs while touching < 2% of the weights.
* target_modules = all attention and MLP projections: a persona changes vocabulary
  and tone, not just which context tokens are attended to, so MLP layers are adapted too.
* lr 1e-4 (LoRA adapters tolerate larger steps than full fine-tuning), effective
  batch 8, 3 epochs (~75 optimizer steps), beta 0.1.
The reference model is the frozen base (adapter disabled), so no second copy is loaded.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from trl import DPOConfig, DPOTrainer

from alignment.training import load_causal_lm, load_json_records, load_tokenizer, save_run_config, seed_everything, write_json_atomic

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def to_conversational(rows: list[dict]) -> Dataset:
    # TRL's conversational preference schema: the instruct model's chat template supplies
    # the user/assistant delimiters for prompt, chosen, and rejected consistently.
    return Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": str(row["prompt"])}],
            "chosen": [{"role": "assistant", "content": str(row["chosen"])}],
            "rejected": [{"role": "assistant", "content": str(row["rejected"])}],
        }
        for row in rows
    ])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--eval_size", type=int, default=20, help="held-out pairs for eval reward accuracy")
    p.add_argument("--max_steps", type=int, default=-1, help="cap for smoke tests")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()

    rows = load_json_records(args.data)
    required = {"prompt", "chosen", "rejected"}
    if not rows or not all(required.issubset(row) for row in rows):
        raise ValueError(f"persona data must contain {sorted(required)}")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    eval_ids = set(order[: args.eval_size])
    train_rows = [row for i, row in enumerate(rows) if i not in eval_ids]
    eval_rows = [row for i, row in enumerate(rows) if i in eval_ids]
    write_json_atomic(out / "eval_split.json", eval_rows)

    tokenizer = load_tokenizer(args.model)
    # fp32 base weights + fp16 autocast: fp16 master weights would make GradScaler fail to unscale.
    model = load_causal_lm(args.model, torch.float32, "cpu")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=TARGET_MODULES, bias="none",
    )
    config_kwargs = dict(
        output_dir=str(out / "trainer"), learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation, lr_scheduler_type="linear", warmup_ratio=0.1,
        logging_steps=5, save_strategy="no", eval_strategy="steps", eval_steps=10, fp16=use_cuda,
        report_to="none", beta=args.beta, seed=args.seed, max_grad_norm=1.0,
    )
    config_signature = inspect.signature(DPOConfig).parameters
    if "eval_strategy" not in config_signature:  # older transformers spelling
        config_kwargs["evaluation_strategy"] = config_kwargs.pop("eval_strategy")
    if "use_cpu" in config_signature and not use_cuda:
        config_kwargs["use_cpu"] = True
    # TRL renamed/removed length options across releases. Only pass accepted keys.
    if "max_length" in config_signature:
        config_kwargs["max_length"] = args.max_length
    if "max_prompt_length" in config_signature:
        config_kwargs["max_prompt_length"] = args.max_length // 2
    train_args = DPOConfig(**config_kwargs)
    trainer_kwargs = dict(model=model, args=train_args, train_dataset=to_conversational(train_rows),
                          eval_dataset=to_conversational(eval_rows), peft_config=lora)
    trainer_signature = inspect.signature(DPOTrainer).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_signature else "tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    import trl, peft

    save_run_config(out, args, {
        "stage": "persona_lora_dpo", "train_pairs": len(train_rows), "eval_pairs": len(eval_rows),
        "trainable_parameters": trainable, "total_parameters": total, "target_modules": TARGET_MODULES,
        "reference_model": f"{args.model} with LoRA adapter disabled (TRL default for PEFT models)",
        "optimizer": "AdamW (Trainer default)", "scheduler": "linear with 10% warmup", "weight_decay": 0.0,
        "effective_batch_size": args.batch_size * args.gradient_accumulation, "precision": "fp32 weights + fp16 autocast" if use_cuda else "fp32",
        "prompt_format": "tokenizer chat template (conversational DPO records)", "trl": trl.__version__, "peft": peft.__version__,
    })
    print(f"LoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)
    trainer.train()
    final_eval = trainer.evaluate()
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for entry in trainer.state.log_history:
            handle.write(json.dumps(entry) + "\n")
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(out / "final")
    write_json_atomic(out / "summary.json", {"final_eval": final_eval, "global_step": trainer.state.global_step})
    (out / "DONE").write_text("ok\n")
    print(json.dumps(final_eval, indent=2))


if __name__ == "__main__":
    main()
