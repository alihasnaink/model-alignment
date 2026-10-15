"""End-to-end run of every script on a tiny random Qwen2 model (offline, CPU or GPU).

Catches wiring bugs (CLI flags, file formats, checkpoint paths) before a long Kaggle
run. Skipped when transformers is not installed. TRL/PEFT parts are skipped if
those packages are missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}<|endoftext|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def make_tiny_model(path: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {"<|endoftext|>": 0, **{char: i + 1 for i, char in enumerate(sorted(alphabet))}}
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok.chat_template = CHAT_TEMPLATE
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                         eos_token_id=0, pad_token_id=0, bos_token_id=0, tie_word_embeddings=True)
    import torch

    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def run(*args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    result = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{args[0]} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    model = make_tiny_model(root / "tiny")
    sft = [{"prompt": f"Say item {i}", "response": f"Item {i} is here."} for i in range(12)]
    pref = [{"prompt": f"Question {i}?", "chosen": f"A helpful answer {i}.", "rejected": "no"} for i in range(12)]
    evals = [{"id": f"e{i}", "instruction": f"Tell me about {i}", "category": "c" + str(i % 2)} for i in range(5)]
    for name, rows in {"sft.jsonl": sft, "pref.jsonl": pref}.items():
        (root / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (root / "eval.json").write_text(json.dumps(evals))
    return root, model


def test_sft_dpo_generate_pair_judge_plot(workspace):
    root, model = workspace
    out = root / "outputs"
    run("scripts/train_sft.py", "--model", str(model), "--train", str(root / "sft.jsonl"), "--validation",
        str(root / "sft.jsonl"), "--output_dir", str(out / "sft"), "--batch_size", "2", "--gradient_accumulation", "2",
        "--eval_every", "1", "--learning_rate", "1e-3", "--allow_cpu", "--max_val_examples", "4")
    for sub in ("best", "final"):
        assert (out / "sft" / sub / "model.safetensors").exists()
    metrics = [json.loads(l) for l in (out / "sft" / "metrics.jsonl").read_text().splitlines()]
    assert any("val_loss" in m for m in metrics) and any("train_loss" in m for m in metrics)
    assert json.loads((out / "sft" / "run_config.json").read_text())["effective_batch_size"] == 4

    run("scripts/train_dpo.py", "--sft_checkpoint", str(out / "sft" / "best"), "--train", str(root / "pref.jsonl"),
        "--validation", str(root / "pref.jsonl"), "--output_dir", str(out / "hh_dpo"), "--gradient_accumulation", "4",
        "--eval_every", "1", "--learning_rate", "1e-3", "--allow_cpu")
    dpo_metrics = [json.loads(l) for l in (out / "hh_dpo" / "metrics.jsonl").read_text().splitlines()]
    assert dpo_metrics[0]["step"] == 0 and dpo_metrics[0]["val_reward_accuracy"] == 0.0
    assert abs(dpo_metrics[0]["val_dpo_loss"] - 0.6931) < 1e-3
    assert any("train_reward_margin" in m for m in dpo_metrics)
    assert (out / "hh_dpo" / "best" / "model.safetensors").exists() and (out / "hh_dpo" / "DONE").exists()

    gen = out / "generations"
    for name, ckpt in (("sft", out / "sft" / "best"), ("dpo", out / "hh_dpo" / "best")):
        run("scripts/generate.py", "--model", str(ckpt), "--inputs", str(root / "eval.json"), "--output",
            str(gen / f"{name}.json"), "--generator", name, "--dataset", "toy", "--max_new_tokens", "8", "--batch_size", "2")
    sft_out = json.loads((gen / "sft.json").read_text())
    assert [r["id"] for r in sft_out] == [f"e{i}" for i in range(5)]
    assert {"id", "instruction", "output", "generator", "dataset"} <= set(sft_out[0])
    # rerun resumes: nothing left to generate, file unchanged
    run("scripts/generate.py", "--model", str(out / "sft" / "best"), "--inputs", str(root / "eval.json"), "--output",
        str(gen / "sft.json"), "--generator", "sft", "--dataset", "toy", "--max_new_tokens", "8")
    assert json.loads((gen / "sft.json").read_text()) == sft_out

    pairs = out / "eval" / "toy_pairs.jsonl"
    run("scripts/build_pairwise.py", "--baseline", str(gen / "sft.json"), "--candidate", str(gen / "dpo.json"),
        "--baseline_name", "sft", "--candidate_name", "dpo", "--output", str(pairs))
    judged = out / "eval" / "toy_judged.jsonl"
    run("scripts/judge_pairwise_kaggle.py", "--pairs_path", str(pairs), "--output_path", str(judged), "--backend", "mock",
        "--batch_size", "2", "--reuse_judgments")
    summary = json.loads((out / "eval" / "toy_judged_summary.json").read_text())
    assert summary["overall"]["n"] == 5 and set(summary["by_category"]) == {"c0", "c1"}

    run("scripts/plot_metrics.py", "--outputs", str(out))
    assert (out / "plots" / "sft.png").exists() and (out / "plots" / "hh_dpo.png").exists()
    run("scripts/summarize_results.py", "--outputs", str(out))
    assert "Pairwise judge results" in (out / "report" / "results.md").read_text()


def test_persona_lora_with_trl(workspace):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    root, model = workspace
    data = [{"prompt": f"Cook {i}?", "chosen": f"Do it right, you donkey {i}!", "rejected": "Okay."} for i in range(10)]
    (root / "persona.json").write_text(json.dumps(data))
    out = root / "persona"
    run("scripts/train_persona_lora.py", "--model", str(model), "--data", str(root / "persona.json"), "--output_dir",
        str(out), "--eval_size", "2", "--max_steps", "2", "--batch_size", "2", "--gradient_accumulation", "1")
    assert (out / "final" / "adapter_config.json").exists() and (out / "metrics.jsonl").exists()
    run("scripts/generate.py", "--model", str(model), "--adapter", str(out / "final"), "--inputs", str(root / "eval.json"),
        "--output", str(root / "persona_gen.json"), "--generator", "lora", "--dataset", "toy", "--prompt_format", "chat",
        "--max_new_tokens", "4", "--limit", "2")
    assert len(json.loads((root / "persona_gen.json").read_text())) == 2


def test_judge_resume_retry_and_failure_logging(tmp_path, monkeypatch):
    import judge_pairwise_kaggle as judge

    pairs = [{"id": f"p{i}", "instruction": "q", "response_a": "a" * i, "response_b": "b", "mapping": {"A": "dpo", "B": "sft"},
              "candidate": "dpo", "baseline": "sft"} for i in range(4)]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    calls = {"n": 0}

    def flaky(_):
        def call(prompt):
            calls["n"] += 1
            if "a" * 3 in prompt:  # the batch with p3 never parses
                return "I refuse to answer in JSON"
            return judge.mock_backend("")(prompt)
        return call

    monkeypatch.setitem(judge.BACKENDS, "mock", flaky)
    out = tmp_path / "judged.jsonl"
    args = ["--pairs_path", str(pairs_path), "--output_path", str(out), "--backend", "mock", "--batch_size", "2",
            "--max_retries", "3", "--sleep_seconds", "0"]
    summary = judge.main(args + ["--reuse_judgments"])
    assert summary["overall"]["unknown"] == 2 and summary["overall"]["wins"] + summary["overall"]["losses"] + summary["overall"]["ties"] == 2
    assert calls["n"] == 1 + 3  # first batch once, second batch three attempts
    failures = [json.loads(l) for l in (tmp_path / "judged_failures.jsonl").read_text().splitlines()]
    assert {f["id"] for f in failures} == {"p2", "p3"}
    calls["n"] = 0
    judge.main(args + ["--reuse_judgments"])
    assert calls["n"] == 3  # only the unresolved batch is retried on resume


def test_parse_verdicts_variants():
    import judge_pairwise_kaggle as judge

    text = '```json\n[{"item": 1, "winner": "Response A"}, {"item": 2, "winner": "tie"}, {"item": 9, "winner": "B"}]\n```'
    verdicts = judge.parse_verdicts(text, 2)
    assert verdicts == {1: {"winner": "A", "reason": ""}, 2: {"winner": "tie", "reason": ""}}
    with pytest.raises(ValueError):
        judge.parse_verdicts("no json here", 2)


def test_judge_prompt_is_blinded():
    import judge_pairwise_kaggle as judge

    pair = {"instruction": "hi", "response_a": "x", "response_b": "y", "mapping": {"A": "dpo", "B": "sft"},
            "candidate": "dpo", "baseline": "sft", "id": "hh_dpo-42"}
    prompt = judge.build_judge_prompt([pair], "safety").lower()
    for leak in ("dpo", "sft", "checkpoint", "hh_dpo"):
        assert leak not in prompt


def test_build_pairs_balanced_and_deterministic():
    from build_pairwise import build_pairs

    base = {str(i): {"id": str(i), "instruction": f"q{i}", "output": f"s{i}"} for i in range(200)}
    cand = {str(i): {"id": str(i), "instruction": f"q{i}", "output": f"d{i}"} for i in range(200)}
    first = build_pairs(base, cand, "sft", "dpo", 42)
    assert first == build_pairs(base, cand, "sft", "dpo", 42)
    a_is_dpo = sum(p["mapping"]["A"] == "dpo" for p in first)
    assert 70 < a_is_dpo < 130
    for p in first:
        by_model = {p["mapping"]["A"]: p["response_a"], p["mapping"]["B"]: p["response_b"]}
        assert by_model["dpo"].startswith("d") and by_model["sft"].startswith("s")
