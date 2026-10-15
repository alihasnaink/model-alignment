"""Edge-case checks beyond the public smoke suite (manual section 4.2).

Covers truncation boundaries, unusual tokenizer configurations, mixed sequence
lengths, row ordering, exact numerics, and small optimization steps. Offline, CPU.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment.training import dpo_logps, dpo_rewards, stack_dpo_batch
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def tokenizer_with(**overrides) -> ToyTokenizer:
    """ToyTokenizer with special-token attributes overridden after vocab construction."""
    tok = ToyTokenizer()
    for key, value in overrides.items():
        setattr(tok, key, value)
    return tok


def NoPadTokenizer():
    return tokenizer_with(pad_token_id=None)


def NoSpecialTokenizer():
    return tokenizer_with(pad_token_id=None, eos_token_id=None)


def ListEosTokenizer():
    return tokenizer_with(eos_token_id=[1, 2])


class PositionalLM(torch.nn.Module):
    """Logits depend on the previous token, so row mixing or off-by-one shifts change results."""

    def __init__(self, vocab_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.table = torch.nn.Parameter(torch.randn(vocab_size, vocab_size) * 0.5)

    def forward(self, input_ids, attention_mask=None):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.table[input_ids])


def manual_logp(model, tokenizer, prompt, response, max_length):
    """Reference: score each response token (and EOS) one position at a time."""
    p = tokenizer.encode(prompt, add_special_tokens=False)
    r = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = (p + r)[:max_length]
    logits = model(torch.tensor([ids])).logits[0]
    total = 0.0
    for t in range(max(len(p), 1), len(ids)):
        total += F.log_softmax(logits[t - 1].float(), -1)[ids[t]].item()
    return total


# ----------------------------------------------------------------------------- batching
def test_truncation_drops_eos_when_response_is_cut():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab"], ["cdefgh"], max_length=5)
    assert batch["input_ids"].shape == (1, 5)
    assert tok.eos_token_id not in batch["input_ids"][0].tolist()
    assert batch["labels"][0].tolist() == [-100, -100] + batch["input_ids"][0, 2:].tolist()


def test_eos_kept_at_exact_boundary():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab"], ["cd"], max_length=5)  # 2 + 2 + EOS == 5
    assert batch["input_ids"][0, -1].item() == tok.eos_token_id
    assert batch["labels"][0, -1].item() == tok.eos_token_id


def test_prompt_longer_than_max_length_has_no_labels():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["abcdefghij", "a"], ["xy", "xy"], max_length=6)
    assert batch["input_ids"].shape[1] <= 6
    assert batch["labels"][0].eq(-100).all()
    assert batch["labels"][1].ne(-100).sum() == 3
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefghij", "a"], ["xy", "xy"], max_length=6)
    assert logps[0].item() == 0.0 and logps[1].item() < 0


def test_mixed_lengths_padding_and_masks():
    tok = ToyTokenizer()
    prompts, responses = ["a", "abcdef", "abc"], ["x", "xyz", "xyzxyzxyz"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=64)
    for row, (p, r) in enumerate(zip(prompts, responses)):
        length = len(p) + len(r) + 1
        assert batch["attention_mask"][row].sum() == length
        assert batch["attention_mask"][row, :length].eq(1).all()
        assert batch["labels"][row].ne(-100).sum() == len(r) + 1
        assert batch["input_ids"][row, length:].eq(tok.pad_token_id).all()
    assert batch["input_ids"].shape[1] == max(len(p) + len(r) + 1 for p, r in zip(prompts, responses))


def test_tokenizer_without_pad_uses_eos_for_padding():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["x", "x"], max_length=16)
    assert batch["input_ids"][0, -1].item() == tok.eos_token_id
    assert batch["attention_mask"][0, -1].item() == 0
    assert batch["labels"][0, -1].item() == -100


def test_tokenizer_without_eos_or_pad():
    tok = NoSpecialTokenizer()
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["xy", "x"], max_length=16)
    assert batch["labels"][0].ne(-100).sum() == 2  # no EOS appended
    assert batch["labels"][1].ne(-100).sum() == 1


def test_list_valued_eos_uses_first_id():
    tok = ListEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["a"], ["x"], max_length=8)
    assert batch["input_ids"][0, -1].item() == 1


def test_dpo_batch_formats_prompt_identically():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["Q: ", "Longer q: "], ["good", "y"], ["no", "bad answer"], max_length=32)
    for row, prompt in enumerate(["Q: ", "Longer q: "]):
        n = len(prompt)
        assert torch.equal(batch["chosen"]["input_ids"][row, :n], batch["rejected"]["input_ids"][row, :n])
        assert batch["chosen"]["labels"][row, :n].eq(-100).all()
        assert batch["rejected"]["labels"][row, :n].eq(-100).all()


def test_mismatched_lengths_raise():
    tok = ToyTokenizer()
    with pytest.raises(ValueError):
        adapter.build_sft_batch(tok, ["a", "b"], ["x"], max_length=8)
    with pytest.raises(ValueError):
        adapter.build_dpo_batch(tok, ["a"], ["x", "y"], ["z"], max_length=8)


# ----------------------------------------------------------------------------- log-probs
@pytest.mark.parametrize("max_length", [3, 6, 9, 40])
def test_logprobs_match_manual_computation(max_length):
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    prompts, responses = ["ab: ", "q", "hello there "], ["xyz", "a longer reply", "k"]
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length)
    expected = [manual_logp(model, tok, p, r, max_length) for p, r in zip(prompts, responses)]
    torch.testing.assert_close(logps, torch.tensor(expected), rtol=1e-5, atol=1e-5)


def test_row_order_is_preserved():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    prompts, responses = ["a: ", "bb: ", "ccc: "], ["x", "yy", "zzz"]
    forward = adapter.compute_response_logprobs(model, tok, prompts, responses, 32)
    reverse = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], 32)
    torch.testing.assert_close(forward, reverse.flip(0))
    single = torch.stack([adapter.compute_response_logprobs(model, tok, [p], [r], 32)[0] for p, r in zip(prompts, responses)])
    torch.testing.assert_close(forward, single)


def test_padding_does_not_change_scores():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    alone = adapter.compute_response_logprobs(model, tok, ["a: "], ["x"], 64)
    padded = adapter.compute_response_logprobs(model, tok, ["a: ", "a very long prompt here: "], ["x", "a long response"], 64)
    torch.testing.assert_close(alone[0], padded[0])


def test_sft_loss_token_mean_reduction():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    prompts, responses = ["a: ", "bcd: "], ["x", "longer"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, 32)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, 32)
    tokens = sum(len(r) + 1 for r in responses)
    torch.testing.assert_close(loss, -logps.sum() / tokens)


# ----------------------------------------------------------------------------- DPO objective
def test_dpo_loss_matches_formula():
    pc, pr = torch.tensor([-1.0, -4.0, -2.0]), torch.tensor([-3.0, -1.0, -2.5])
    rc, rr = torch.tensor([-1.5, -3.0, -2.0]), torch.tensor([-2.0, -2.0, -2.0])
    beta = 0.3
    expected = torch.mean(torch.tensor([
        -math.log(1 / (1 + math.exp(-beta * ((a - b) - (c - d))))) for a, b, c, d in zip(pc, pr, rc, rr)
    ]))
    torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta), expected.float())


def test_dpo_loss_is_stable_for_large_gaps():
    big = torch.tensor([1000.0])
    zero = torch.tensor([0.0])
    assert torch.isfinite(adapter.compute_dpo_loss_from_logps(-big, zero, zero, zero, 1.0))
    assert adapter.compute_dpo_loss_from_logps(big, zero, zero, zero, 1.0).item() < 1e-6


def test_dpo_loss_gradient_direction():
    pc = torch.tensor([-2.0], requires_grad=True)
    pr = torch.tensor([-2.0], requires_grad=True)
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, torch.tensor([-2.0]), torch.tensor([-2.0]), 0.1)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()  # descent raises chosen, lowers rejected
    torch.testing.assert_close(pc.grad, torch.tensor([-0.05]))  # beta * (1 - sigmoid(0)) / batch


def test_reference_receives_no_gradient_even_if_trainable():
    tok = ToyTokenizer()
    policy, reference = PositionalLM(tok.vocab_size), PositionalLM(tok.vocab_size)
    assert reference.table.requires_grad
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["Q: "], ["good"], ["bad"], 0.1, 16)
    loss.backward()
    assert policy.table.grad is not None and policy.table.grad.abs().sum() > 0
    assert reference.table.grad is None


def test_identical_policy_and_reference_give_log_two():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    loss = adapter.compute_dpo_loss(model, model, tok, ["Q: ", "R: "], ["good", "y"], ["bad", "nope"], 0.5, 16)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


# ----------------------------------------------------------------------------- optimization steps
def test_sft_step_reduces_loss():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    args = (tok, ["Say: "], ["hello"], 32)
    before = adapter.compute_sft_loss(model, *args)
    before.backward(); opt.step(); opt.zero_grad()
    assert adapter.compute_sft_loss(model, *args).item() < before.item()


def test_dpo_step_improves_preference_and_reward_accuracy():
    tok = ToyTokenizer()
    policy, reference = PositionalLM(tok.vocab_size), PositionalLM(tok.vocab_size)
    reference.load_state_dict(policy.state_dict())
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    data = (tok, ["Q: ", "Ask: "], ["yes", "sure"], ["no", "never"])
    before = adapter.compute_dpo_loss(policy, reference, *data, 0.5, 32)
    torch.testing.assert_close(before, torch.log(torch.tensor(2.0)))
    for _ in range(5):
        loss = adapter.compute_dpo_loss(policy, reference, *data, 0.5, 32)
        loss.backward(); opt.step(); opt.zero_grad()
    after = adapter.compute_dpo_loss(policy, reference, *data, 0.5, 32)
    assert after.item() < before.item()
    batch = adapter.build_dpo_batch(tok, *data[1:], 32)
    with torch.no_grad():
        r_w, r_l = dpo_rewards(*dpo_logps(policy, batch), *dpo_logps(reference, batch), 0.5)
    assert (r_w > r_l).all()


# ----------------------------------------------------------------------------- training helpers
def test_stacked_forward_matches_separate_forward():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    batch = adapter.build_dpo_batch(tok, ["Q: ", "Long question: "], ["a", "long chosen answer"], ["rejected one", "b"], 64)
    stacked = stack_dpo_batch(batch)
    assert stacked["input_ids"].shape[0] == 4
    chosen, rejected = dpo_logps(model, batch)
    torch.testing.assert_close(chosen, adapter.compute_response_logprobs(model, tok, ["Q: ", "Long question: "], ["a", "long chosen answer"], 64))
    torch.testing.assert_close(rejected, adapter.compute_response_logprobs(model, tok, ["Q: ", "Long question: "], ["rejected one", "b"], 64))


def test_initial_reward_accuracy_is_zero_with_strict_comparison():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    batch = adapter.build_dpo_batch(tok, ["Q: "], ["a"], ["b"], 16)
    with torch.no_grad():
        logps = dpo_logps(model, batch)
        r_w, r_l = dpo_rewards(*logps, *logps, 0.1)
    assert not (r_w > r_l).any()
