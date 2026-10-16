"""Small, explicit training utilities used by the Kaggle entry points."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .core import IGNORE_INDEX, _response_logprobs_from_batch, build_dpo_batch, build_sft_batch, compute_dpo_loss_from_logps
from .prompting import format_prompt


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    # Split on "\n" only: str.splitlines() also breaks on U+2028, U+0085, etc., which json.dumps(ensure_ascii=False)
    # leaves unescaped inside strings (UltraFeedback contains them), cutting a record in half.
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Write via a temp file so an interrupted Kaggle job never leaves half a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def batches(records: list[dict[str, Any]], batch_size: int, shuffle: bool) -> Iterable[list[dict[str, Any]]]:
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda rows: rows)


def write_metric(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values) + "\n")
        handle.flush()


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, info: dict[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    if info is not None:
        write_json_atomic(output_dir / "checkpoint_info.json", info)


def save_run_config(output_dir: Path, args: Any, extra: dict[str, Any] | None = None) -> None:
    """Record the exact values used, for the report's experimental-configuration table."""
    import transformers

    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update(extra or {})
    config["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    write_json_atomic(output_dir / "run_config.json", config)


def load_causal_lm(name_or_path: str | Path, dtype: torch.dtype, device: torch.device | str):
    """Load a causal LM across Transformers versions (``torch_dtype`` was renamed ``dtype``)."""
    import transformers
    from transformers import AutoModelForCausalLM

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(str(name_or_path), **{key: dtype})
    return model.to(device)


def load_tokenizer(name_or_path: str | Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def autocast_context(enabled: bool, device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


class ProgressClock:
    """Elapsed time and a simple ETA, logged with metrics so runtimes go in the report."""

    def __init__(self, total_steps: int):
        self.start, self.total = time.time(), max(total_steps, 1)

    def status(self, step: int) -> dict[str, float]:
        elapsed = time.time() - self.start
        eta = elapsed / max(step, 1) * (self.total - step)
        return {"elapsed_min": round(elapsed / 60, 2), "eta_min": round(eta / 60, 2)}


def stack_dpo_batch(batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad chosen and rejected batches to one width and stack them as [chosen; rejected].

    One forward pass over 2B rows is faster than two passes over B rows and is
    numerically identical because every row is scored independently.
    """
    chosen, rejected = batch["chosen"], batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
    # Any id works for input padding because attention and labels both mask it.
    fill = {"input_ids": 0, "attention_mask": 0, "labels": IGNORE_INDEX}
    stacked = {}
    for key, pad_value in fill.items():
        parts = [F.pad(side[key], (0, width - side[key].shape[1]), value=pad_value) for side in (chosen, rejected)]
        stacked[key] = torch.cat(parts, dim=0)
    return stacked


def dpo_logps(model: Any, batch: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Response-only summed log-probabilities for chosen and rejected rows."""
    rows = batch["chosen"]["input_ids"].shape[0]
    logps = _response_logprobs_from_batch(model, stack_dpo_batch(batch))
    return logps[:rows], logps[rows:]


def dpo_rewards(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implicit DPO rewards r_w and r_l (manual equations 7 and 8)."""
    return beta * (policy_chosen - reference_chosen), beta * (policy_rejected - reference_rejected)


def format_rows(rows: list[dict[str, Any]], field: str, template: str) -> list[str]:
    return [format_prompt(row[field], template) for row in rows]


def sft_validation(model: Any, tokenizer: Any, records: list[dict[str, Any]], args: Any, device: torch.device) -> float:
    """Mean NLL over all unmasked response tokens in the validation set."""
    if not records:
        return float("nan")
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for rows in batches(records, args.eval_batch_size, False):
            prompts = format_rows(rows, args.prompt_field, args.prompt_template)
            responses = [str(row[args.response_field]) for row in rows]
            batch = build_sft_batch(tokenizer, prompts, responses, args.max_length)
            with autocast_context(args.fp16, device):
                logps = _response_logprobs_from_batch(model, batch)
            count = int(batch["labels"][:, 1:].ne(IGNORE_INDEX).sum())
            total_nll += float(-logps.sum().item())
            total_tokens += count
    model.train()
    return total_nll / max(total_tokens, 1)


def dpo_validation(
    policy: Any, reference: Any, tokenizer: Any, records: list[dict[str, Any]], args: Any, policy_device: torch.device
) -> dict[str, float]:
    if not records:
        return {"val_dpo_loss": float("nan"), "val_reward_accuracy": float("nan"), "val_reward_margin": float("nan")}
    policy.eval()
    reference.eval()
    losses, correct, examples, margins, chosen_total, rejected_total = 0.0, 0, 0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for rows in batches(records, args.eval_batch_size, False):
            prompts = format_rows(rows, args.prompt_field, args.prompt_template)
            chosen = [str(row[args.chosen_field]) for row in rows]
            rejected = [str(row[args.rejected_field]) for row in rows]
            batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, args.max_length)
            with autocast_context(args.fp16, policy_device):
                pc, pr = dpo_logps(policy, batch)
                rc, rr = dpo_logps(reference, batch)
            rc, rr = rc.to(policy_device), rr.to(policy_device)
            loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, args.beta)
            reward_chosen, reward_rejected = dpo_rewards(pc, pr, rc, rr, args.beta)
            losses += float(loss.item()) * len(rows)
            correct += int((reward_chosen > reward_rejected).sum().item())  # strict inequality per manual.
            margins += float((reward_chosen - reward_rejected).sum().item())
            chosen_total += float(reward_chosen.sum().item())
            rejected_total += float(reward_rejected.sum().item())
            examples += len(rows)
    policy.train()
    n = max(examples, 1)
    return {
        "val_dpo_loss": losses / n,
        "val_reward_accuracy": correct / n,
        "val_reward_margin": margins / n,
        "val_reward_chosen": chosen_total / n,
        "val_reward_rejected": rejected_total / n,
    }


__all__ = [
    "ProgressClock",
    "autocast_context",
    "batches",
    "dpo_logps",
    "dpo_rewards",
    "dpo_validation",
    "format_rows",
    "load_causal_lm",
    "load_json_records",
    "load_tokenizer",
    "save_checkpoint",
    "save_run_config",
    "seed_everything",
    "sft_validation",
    "stack_dpo_batch",
    "write_json_atomic",
    "write_jsonl",
    "write_metric",
]
