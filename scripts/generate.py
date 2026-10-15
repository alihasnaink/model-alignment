#!/usr/bin/env python
"""Generate durable JSON outputs from a checkpoint and a fixed JSON/JSONL prompt set.

Each entry has at least id, instruction, output, generator, dataset (manual Task 2).
The file is rewritten atomically after every batch, and a rerun skips ids that are
already present, so an interrupted Kaggle session resumes where it stopped. The
decoding settings are written to ``<output>.meta.json`` for the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alignment.prompting import PROMPT_TEMPLATE, clean_generation, format_prompt
from alignment.training import load_causal_lm, load_json_records, load_tokenizer, write_json_atomic


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint dir or hub id (base model when --adapter is used)")
    p.add_argument("--adapter", default=None, help="optional PEFT/LoRA adapter directory")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt_field", default="instruction")
    p.add_argument("--id_field", default="id")
    p.add_argument("--prompt_format", choices=["template", "chat"], default="template",
                   help="'template' = alignment.prompting.PROMPT_TEMPLATE; 'chat' = tokenizer chat template")
    p.add_argument("--prompt_template", default=PROMPT_TEMPLATE)
    p.add_argument("--generator", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity checks / smoke tests)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print_examples", type=int, default=0, help="print the first N outputs for manual inspection")
    return p


def build_prompt(tokenizer, instruction: str, args) -> str:
    if args.prompt_format == "chat":
        return tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)
    return format_prompt(instruction, args.prompt_template)


def stop_token_ids(tokenizer, model) -> list[int]:
    ids = {tokenizer.eos_token_id}
    configured = getattr(model.generation_config, "eos_token_id", None)
    ids.update(configured if isinstance(configured, (list, tuple)) else [configured])
    return sorted(int(i) for i in ids if i is not None)


def main() -> None:
    args = parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    rows = load_json_records(args.inputs)[: args.limit]
    output_path = Path(args.output)
    done: dict[str, dict] = {}
    if args.resume and output_path.exists():
        done = {str(item["id"]): item for item in load_json_records(output_path)}
    pending = [(index, row) for index, row in enumerate(rows) if str(row.get(args.id_field, index)) not in done]
    print(f"{args.generator} on {args.dataset}: {len(rows)} prompts, {len(done)} already done, {len(pending)} to generate", flush=True)

    meta = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    meta.update({"decoding": "greedy (do_sample=False)", "temperature": None, "top_p": None, "torch_dtype": str(dtype),
                 "post_processing": "decode new tokens, skip special tokens, cut at '### Instruction:'/'### Response:' markers, strip"})
    write_json_atomic(Path(str(output_path) + ".meta.json"), meta)

    def flush() -> None:
        ordered = [done[str(row.get(args.id_field, index))] for index, row in enumerate(rows) if str(row.get(args.id_field, index)) in done]
        write_json_atomic(output_path, ordered)

    if not pending:
        flush()
        return

    tokenizer = load_tokenizer(args.model if not args.adapter else args.adapter)
    tokenizer.padding_side = "left"  # decoder-only batched generation needs left padding
    model = load_causal_lm(args.model, dtype, device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    eos_ids = stop_token_ids(tokenizer, model)
    meta["eos_token_ids"] = eos_ids
    write_json_atomic(Path(str(output_path) + ".meta.json"), meta)

    # Longest prompts first keeps padding small and surfaces OOMs on the first batch.
    pending.sort(key=lambda item: -len(str(item[1][args.prompt_field])))
    printed = 0
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start: start + args.batch_size]
        prompts = [build_prompt(tokenizer, str(row[args.prompt_field]), args) for _, row in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        # Only pass what a causal LM accepts (some tokenizers also return token_type_ids).
        encoded = {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}
        with torch.no_grad():
            sequences = model.generate(
                **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
            )
        new_tokens = sequences[:, encoded["input_ids"].shape[1]:]
        for (index, row), tokens in zip(chunk, new_tokens):
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            item_id = str(row.get(args.id_field, index))
            done[item_id] = {
                "id": item_id, "instruction": str(row[args.prompt_field]), "output": clean_generation(raw),
                "generator": args.generator, "dataset": args.dataset,
                **({"category": row["category"]} if "category" in row else {}),
                "num_new_tokens": int(tokens.ne(tokenizer.pad_token_id).sum()),
            }
            if printed < args.print_examples:
                printed += 1
                print(f"\n--- [{item_id}] {row[args.prompt_field][:200]}\n>>> {done[item_id]['output'][:800]}\n", flush=True)
        flush()
        print(f"  {min(start + args.batch_size, len(pending))}/{len(pending)} generated", flush=True)


if __name__ == "__main__":
    main()
