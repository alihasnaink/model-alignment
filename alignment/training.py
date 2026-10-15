"""Small, explicit training utilities used by the Kaggle entry points."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from .core import _response_logprobs_from_batch, build_dpo_batch, build_sft_batch, compute_dpo_loss_from_logps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load either a JSON list or JSONL, keeping input order intact."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        result = json.loads(text)
        if not isinstance(result, list):
            raise ValueError("JSON dataset must be a list of objects")
        return result
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def batches(records: list[dict[str, Any]], batch_size: int, shuffle: bool) -> Iterable[list[dict[str, Any]]]:
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda rows: rows)


def write_metric(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values) + "\n")


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # save_pretrained is portable across standard Transformers causal LMs.
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def autocast_context(enabled: bool, device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def sft_validation(model: Any, tokenizer: Any, records: list[dict[str, Any]], args: Any, device: torch.device) -> float:
    if not records:
        return float("nan")
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for rows in batches(records, args.eval_batch_size, False):
            prompts = [args.prompt_template.format(prompt=row[args.prompt_field]) for row in rows]
            responses = [str(row[args.response_field]) for row in rows]
            batch = build_sft_batch(tokenizer, prompts, responses, args.max_length)
            with autocast_context(args.fp16, device):
                logps = _response_logprobs_from_batch(model, batch)
            count = int(batch["labels"][:, 1:].ne(-100).sum())
            total_nll += float(-logps.sum().item())
            total_tokens += count
    model.train()
    return total_nll / max(total_tokens, 1)


def dpo_validation(
    policy: Any, reference: Any, tokenizer: Any, records: list[dict[str, Any],], args: Any, policy_device: torch.device
) -> dict[str, float]:
    if not records:
        return {"val_dpo_loss": float("nan"), "val_reward_accuracy": float("nan"), "val_reward_margin": float("nan")}
    policy.eval()
    reference.eval()
    losses, correct, examples, margins = 0.0, 0, 0, 0.0
    with torch.no_grad():
        for rows in batches(records, args.eval_batch_size, False):
            prompts = [args.prompt_template.format(prompt=row[args.prompt_field]) for row in rows]
            chosen = [str(row[args.chosen_field]) for row in rows]
            rejected = [str(row[args.rejected_field]) for row in rows]
            batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, args.max_length)
            with autocast_context(args.fp16, policy_device):
                pc = _response_logprobs_from_batch(policy, batch["chosen"])
                pr = _response_logprobs_from_batch(policy, batch["rejected"])
            rc = _response_logprobs_from_batch(reference, batch["chosen"]).to(policy_device)
            rr = _response_logprobs_from_batch(reference, batch["rejected"]).to(policy_device)
            loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, args.beta)
            margin = args.beta * ((pc - rc) - (pr - rr))
            losses += float(loss.item()) * len(rows)
            correct += int((margin > 0).sum().item())  # strict inequality per assignment.
            margins += float(margin.sum().item())
            examples += len(rows)
    policy.train()
    return {
        "val_dpo_loss": losses / max(examples, 1),
        "val_reward_accuracy": correct / max(examples, 1),
        "val_reward_margin": margins / max(examples, 1),
    }
