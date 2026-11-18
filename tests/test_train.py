"""Tests for kd.train (owner: T1). Run: pytest tests/test_train.py

Two kinds of test, all offline and on CPU:
  1. Fairness: the question order depends only on the seed (the v1 bug), and
     every arm gets the same questions at every step.
  2. End to end: a tiny randomly initialised Qwen3 (2 layers, 32 dims) is
     trained through every arm with the fake tokenizer from test_data.py.
     This proves the loop runs and logs correctly, not that it learns GSM8K;
     the real check is the smoke run on Kaggle.
"""

import copy
import json
import random

import pytest
import torch

from kd import data
from kd.train import (
    ARMS,
    question_ids_hash,
    question_order,
    sampling_seed,
    step_batches,
    train_loop,
)
from test_data import ROWS, FakeTokenizer  # the fake Qwen3 tokenizer, not copied

IDS = [f"train-{i:05d}" for i in range(40)]


# --------------------------------------------------------------------------
# 1. Order and fairness (no model needed)
# --------------------------------------------------------------------------

def test_order_is_a_permutation_and_depends_on_the_seed():
    a, b = question_order(IDS, seed=1), question_order(IDS, seed=2)
    assert sorted(a) == sorted(IDS) == sorted(b)
    assert a != b
    assert question_order(IDS, seed=1) == a


def test_order_ignores_the_global_rng_and_input_order():
    first = question_order(IDS, seed=1)
    random.seed(123)
    random.random()            # on-policy sampling used to draw from this in v1
    torch.manual_seed(7)
    assert question_order(list(reversed(IDS)), seed=1) == first


def test_epochs_repeat_the_pool_in_a_new_order():
    order = question_order(IDS, seed=1, epochs=2)
    assert len(order) == 80
    assert sorted(order[:40]) == sorted(order[40:]) and order[:40] != order[40:]


def test_step_batches_are_equal_sized_and_respect_max_steps():
    steps = step_batches(IDS[:10], 4)
    assert [len(s) for s in steps] == [4, 4]          # the last 2 questions are dropped
    assert len(step_batches(IDS, 4, max_steps=3)) == 3


def test_hash_and_sampling_seed_are_stable():
    assert question_ids_hash(["a", "b"]) == question_ids_hash(["a", "b"]) != question_ids_hash(["b", "a"])
    assert sampling_seed(1, 5) == sampling_seed(1, 5) != sampling_seed(1, 6)


# --------------------------------------------------------------------------
# 2. End to end with a tiny model
# --------------------------------------------------------------------------

VOCAB = 256  # covers every id the fake tokenizer hands out for these rows


def tiny_model(seed):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=1024, tie_word_embeddings=True)
    return Qwen3ForCausalLM(config)


def tiny_cfg(**train_overrides):
    cfg = {
        "train": {"questions_per_step": 4, "epochs": 1, "max_steps": None, "lr": 1e-3,
                  "grad_clip": 1.0, "loss": "forward_kl", "temperature": 1.0,
                  "student_dtype": "float32", "autocast_dtype": "float16",
                  "oom_retries": 2, "log_every": 100},
        "generation": {"max_new_tokens": 8,
                       "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}},
    }
    cfg["train"].update(train_overrides)
    return cfg


@pytest.fixture
def setup():
    tokenizer = FakeTokenizer()
    pool = data.examples_from_rows(ROWS * 2, "train")      # 8 questions -> 2 steps of 4
    for ex in pool:                                         # register every character now
        data.build_for_arm("human", tokenizer, ex)
    cache = {ex.question_id: {"question_id": ex.question_id,
                              "response": f"Short teacher answer. The final answer is {ex.gold}.",
                              "truncated": False} for ex in pool}
    for rec in cache.values():
        tokenizer(rec["response"])
    assert len(tokenizer.vocab) < VOCAB
    return tokenizer, pool, cache


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_arm_trains_logs_and_sees_the_same_questions(setup, tmp_path):
    tokenizer, pool, cache = setup
    hashes = {}
    for arm in ARMS:
        student, teacher = tiny_model(0), tiny_model(1)
        before = [p.detach().clone() for p in student.parameters()]
        summary = train_loop(tiny_cfg(), arm, 1, tmp_path / arm, student,
                             None if arm == "ref_hardlabel" else teacher, tokenizer, pool,
                             teacher_cache=cache)
        rows = read_log(tmp_path / arm / "train_log.jsonl")
        assert summary["steps"] == len(rows) == 2
        assert all(torch.isfinite(torch.tensor(r["loss"])) and r["tokens"] > 0 for r in rows)
        assert set(rows[0]) >= {"step", "loss", "tokens", "lr", "elapsed_s", "question_ids_hash"}
        assert any(not torch.equal(a, b) for a, b in zip(before, student.parameters()))
        hashes[arm] = [r["question_ids_hash"] for r in rows]
    assert len({tuple(h) for h in hashes.values()}) == 1   # identical order in all four arms


def test_kl_is_zero_when_the_student_is_a_copy_of_the_teacher(setup, tmp_path):
    tokenizer, pool, cache = setup
    teacher = tiny_model(1)
    student = copy.deepcopy(teacher)
    train_loop(tiny_cfg(max_steps=1), "human", 1, tmp_path, student, teacher, tokenizer, pool)
    assert abs(read_log(tmp_path / "train_log.jsonl")[0]["loss"]) < 1e-5


def test_overfit_the_kl_falls_on_a_few_questions(setup, tmp_path):
    """The plan's overfit check, in miniature: repeat 4 questions, the KL must drop."""
    tokenizer, pool, cache = setup
    cfg = tiny_cfg(lr=1e-2, epochs=15)
    train_loop(cfg, "human", 1, tmp_path, tiny_model(0), tiny_model(1), tokenizer, pool[:4])
    losses = [r["loss"] for r in read_log(tmp_path / "train_log.jsonl")]
    assert len(losses) == 15
    assert losses[-1] < 0.5 * losses[0]


def test_onpolicy_rollouts_repeat_for_the_same_seed_and_step(setup):
    from kd.train import build_step_sequences
    tokenizer, pool, _ = setup
    student = tiny_model(0)
    args = ("onpolicy", pool[:4], tokenizer, tiny_cfg(), 400)
    a = build_step_sequences(*args, student=student, gen_seed=sampling_seed(1, 1))
    torch.manual_seed(999)                                  # anything drawn in between
    b = build_step_sequences(*args, student=student, gen_seed=sampling_seed(1, 1))
    c = build_step_sequences(*args, student=student, gen_seed=sampling_seed(1, 2))
    assert [s.input_ids for s in a] == [s.input_ids for s in b]
    assert [s.input_ids for s in a] != [s.input_ids for s in c]


def test_missing_inputs_fail_early(setup, tmp_path):
    tokenizer, pool, _ = setup
    with pytest.raises(ValueError, match="needs the teacher model"):
        train_loop(tiny_cfg(), "human", 1, tmp_path, tiny_model(0), None, tokenizer, pool)
    with pytest.raises(ValueError, match="teacher cache"):
        train_loop(tiny_cfg(), "teacher", 1, tmp_path, tiny_model(0), tiny_model(1), tokenizer, pool)
    with pytest.raises(ValueError, match="unknown arm"):
        train_loop(tiny_cfg(), "bogus", 1, tmp_path, tiny_model(0), None, tokenizer, pool)


# --------------------------------------------------------------------------
# 3. T1 review additions: mixed lambda (stretch B8.1), OOM retry, vocabulary
# --------------------------------------------------------------------------

from kd.train import check_mixed_lambda, check_vocab, onpolicy_positions  # noqa: E402


@pytest.mark.parametrize("lmbda,k", [(0.0, 0), (0.25, 1), (0.5, 2), (0.75, 3), (1.0, 4)])
def test_onpolicy_positions_exact_count_and_reproducible(lmbda, k):
    n_rollouts = check_mixed_lambda(lmbda, 4)
    assert n_rollouts == k
    flags = [tuple(onpolicy_positions(1, step, 4, k)) for step in range(1, 30)]
    assert all(sum(f) == k for f in flags)
    assert flags == [tuple(onpolicy_positions(1, step, 4, k)) for step in range(1, 30)]
    if 0 < k < 4:
        assert len(set(flags)) > 1          # the positions change from step to step


def test_mixed_lambda_must_be_set_and_whole():
    with pytest.raises(ValueError, match="mixed_lambda"):
        check_mixed_lambda(None, 4)
    with pytest.raises(ValueError, match="whole number"):
        check_mixed_lambda(0.3, 4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        check_mixed_lambda(1.5, 4)


def test_mixed_arm_trains_with_exact_share_and_the_same_order(setup, tmp_path):
    tokenizer, pool, cache = setup
    human = tmp_path / "human"
    mixed = tmp_path / "mixed"
    train_loop(tiny_cfg(), "human", 1, human, tiny_model(0), tiny_model(1), tokenizer, pool)
    summary = train_loop(tiny_cfg(mixed_lambda=0.5), "mixed", 1, mixed, tiny_model(0),
                         tiny_model(1), tokenizer, pool, teacher_cache=cache)
    rows = read_log(mixed / "train_log.jsonl")
    assert summary["mixed_lambda"] == 0.5 and summary["steps"] == 2
    assert [r["n_rollouts"] for r in rows] == [2, 2]
    assert ([r["question_ids_hash"] for r in rows]
            == [r["question_ids_hash"] for r in read_log(human / "train_log.jsonl")])


def test_mixed_arm_needs_lambda_and_cache(setup, tmp_path):
    tokenizer, pool, cache = setup
    with pytest.raises(ValueError, match="teacher cache"):
        train_loop(tiny_cfg(mixed_lambda=0.5), "mixed", 1, tmp_path, tiny_model(0),
                   tiny_model(1), tokenizer, pool)
    with pytest.raises(ValueError, match="mixed_lambda"):
        train_loop(tiny_cfg(), "mixed", 1, tmp_path, tiny_model(0), tiny_model(1),
                   tokenizer, pool, teacher_cache=cache)


def test_mixed_step_sources_follow_the_flags(setup):
    from kd.train import build_step_sequences
    tokenizer, pool, cache = setup
    flags = [True, False, False, True]
    seqs = build_step_sequences("mixed", pool[:4], tokenizer, tiny_cfg(), 400,
                                teacher_cache=cache, student=tiny_model(0),
                                gen_seed=sampling_seed(1, 1), rollout_flags=flags)
    assert [s.arm for s in seqs] == ["onpolicy", "teacher", "teacher", "onpolicy"]
    assert [s.question_id for s in seqs] == [ex.question_id for ex in pool[:4]]


def test_out_of_memory_retries_the_same_step_with_a_fresh_scaler(setup, tmp_path, monkeypatch):
    import kd.train as train_mod
    tokenizer, pool, _ = setup
    real_step = train_mod.train_step
    seen_scalers, calls = [], {"n": 0}

    def flaky_step(seqs, arm, student, teacher, tok, optimizer, scaler, cfg):
        calls["n"] += 1
        seen_scalers.append(scaler)
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_step(seqs, arm, student, teacher, tok, optimizer, scaler, cfg)

    monkeypatch.setattr(train_mod, "train_step", flaky_step)
    summary = train_loop(tiny_cfg(), "human", 1, tmp_path, tiny_model(0), tiny_model(1),
                         tokenizer, pool)
    assert summary["oom_retries"] == 1 and summary["steps"] == 2
    assert calls["n"] == 3                        # step 1 twice, step 2 once
    assert seen_scalers[1] is not seen_scalers[0]  # the retry got a new scaler


def test_out_of_memory_too_often_stops_the_run(setup, tmp_path, monkeypatch):
    import kd.train as train_mod
    tokenizer, pool, _ = setup

    def always_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(train_mod, "train_step", always_oom)
    with pytest.raises(RuntimeError, match="instead of skipping"):
        train_loop(tiny_cfg(), "human", 1, tmp_path, tiny_model(0), tiny_model(1), tokenizer, pool)


def test_rollouts_restore_train_mode_after_an_error(setup, monkeypatch):
    from kd.train import generate_rollouts
    tokenizer, pool, _ = setup
    student = tiny_model(0)
    student.train()

    def broken_generate(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(student, "generate", broken_generate)
    prompts = [data.prompt_ids(tokenizer, ex.question) for ex in pool[:2]]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        generate_rollouts(student, tokenizer, prompts, tiny_cfg(), seed=1)
    assert student.training


def test_vocab_mismatch_is_reported():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    def model(vocab):
        torch.manual_seed(0)
        return Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                                            num_hidden_layers=1, num_attention_heads=4,
                                            num_key_value_heads=2, head_dim=8,
                                            max_position_embeddings=64))

    assert check_vocab(model(64), model(64)) is None
    assert check_vocab(model(64), model(80)) == 64
    assert check_vocab(model(64), None) is None   # ref_hardlabel has no teacher
