"""Framework-independent SFT/DPO batching and objectives.

The functions deliberately avoid ``Trainer`` and TRL.  They are shared by the
Kaggle training scripts and the root-level grading adapter.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize one already-formatted text without tokenizer-added special tokens."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    # Some tokenizers return tensors/arrays; a plain list keeps batching uniform.
    return [int(token) for token in encoded]


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        # Qwen tokenizers commonly use EOS as padding. Attention/labels still mask it.
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(pad_id)


def build_sft_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Build right-padded causal-LM inputs with labels only on response/EOS tokens.

    Prompts and responses are tokenized separately.  This makes the response
    boundary explicit (and avoids accidentally supervising prompt tokens), which
    is especially important for tokenizers that have a chat-template prefix.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    if not prompts:
        raise ValueError("cannot build an empty batch")
    if max_length < 1:
        raise ValueError("max_length must be positive")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    rows: list[tuple[list[int], list[int]]] = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = _ids(tokenizer, prompt)
        response_ids = _ids(tokenizer, response)
        if eos_id is not None:
            response_ids.append(int(eos_id))

        # Append EOS *before* truncation, as required by the assignment.
        input_ids = (prompt_ids + response_ids)[:max_length]
        response_start = min(len(prompt_ids), max_length)
        labels = [IGNORE_INDEX] * response_start + input_ids[response_start:]
        rows.append((input_ids, labels))

    width = min(max_length, max(len(input_ids) for input_ids, _ in rows))
    pad_id = _pad_token_id(tokenizer)
    input_tensor = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels_tensor = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for row, (input_ids, labels) in enumerate(rows):
        length = len(input_ids)
        if length:
            input_tensor[row, :length] = torch.tensor(input_ids, dtype=torch.long)
            attention_mask[row, :length] = 1
            labels_tensor[row, :length] = torch.tensor(labels, dtype=torch.long)

    return {"input_ids": input_tensor, "attention_mask": attention_mask, "labels": labels_tensor}


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _response_logprobs_from_batch(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum next-token log probabilities where labels are not ``IGNORE_INDEX``."""
    batch = _move_batch(batch, _model_device(model))
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    if logits.ndim != 3:
        raise ValueError("model must return logits shaped [batch, sequence, vocabulary]")

    # Logit t predicts input token t + 1. Position 0 cannot be scored by a
    # conventional causal LM, so it is deliberately excluded.
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    valid = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logps = F.log_softmax(shift_logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * valid).sum(dim=-1)


def compute_response_logprobs(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    """Return one differentiable response-only summed log probability per row."""
    return _response_logprobs_from_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def compute_sft_loss(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    """Response-token mean negative log likelihood (the documented token reduction)."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps = _response_logprobs_from_batch(model, batch)
    # The first position has no previous causal logit and is therefore excluded.
    token_count = batch["labels"][:, 1:].ne(IGNORE_INDEX).sum().clamp_min(1)
    return -logps.sum() / token_count.to(device=logps.device, dtype=logps.dtype)


def build_dpo_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Create identically formatted response-masked batches for a preference pair."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses, and rejected_responses must have equal length")
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Mean ``-logsigmoid(beta * ((pi_w-pi_l)-(ref_w-ref_l)))`` DPO objective."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps = torch.broadcast_tensors(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
    )
    logits = float(beta) * (
        (policy_chosen_logps - policy_rejected_logps)
        - (reference_chosen_logps - reference_rejected_logps)
    )
    return -F.logsigmoid(logits).mean()


def compute_dpo_loss(
    policy_model: Any,
    reference_model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    beta: float,
    max_length: int,
) -> torch.Tensor:
    """Compute DPO loss; the reference forward pass is strictly gradient-free."""
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    policy_chosen = _response_logprobs_from_batch(policy_model, batch["chosen"])
    policy_rejected = _response_logprobs_from_batch(policy_model, batch["rejected"])
    with torch.no_grad():
        reference_chosen = _response_logprobs_from_batch(reference_model, batch["chosen"]).to(policy_chosen.device)
        reference_rejected = _response_logprobs_from_batch(reference_model, batch["rejected"]).to(policy_chosen.device)
    return compute_dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
