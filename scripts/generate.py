#!/usr/bin/env python
"""Generate durable JSON outputs from a checkpoint and a fixed JSON/JSONL prompt set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.training import load_json_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt_field", default="instruction")
    p.add_argument("--id_field", default="id")
    p.add_argument("--prompt_template", default="{prompt}")
    p.add_argument("--generator", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
    output_path, outputs = Path(args.output), []
    for index, row in enumerate(load_json_records(args.inputs)):
        instruction = str(row[args.prompt_field])
        prompt = args.prompt_template.format(prompt=instruction)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            sequence = model.generate(encoded, do_sample=False, max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(sequence[0, encoded.shape[1]:], skip_special_tokens=True)
        outputs.append({"id": row.get(args.id_field, index), "instruction": instruction, "output": output, "generator": args.generator, "dataset": args.dataset})
        # Atomic-ish progress persistence makes interrupted Kaggle jobs resumable/auditable.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
