"""Tests for kd.data (A2). Run: pytest tests/test_data.py

Everything except the last block runs offline with a fake tokenizer that mimics
the Qwen3 chat template. Set KD_RUN_DATA_TESTS=1 (internet + `datasets` +
the real Qwen tokenizers) to also run the checks on the real data.
"""

import json
import os
import random
import re

import pytest

from kd import data
from kd.data import (
    INSTRUCTION,
    build_for_arm,
    build_human_sequence,
    build_onpolicy_sequence,
    build_sequence,
    build_teacher_sequence,
    clean_solution,
    examples_from_rows,
    gold_number,
    prompt_ids,
    prompt_text,
)

# --------------------------------------------------------------------------
# Fixtures: real GSM8K-style rows and a fake Qwen3-like tokenizer
# --------------------------------------------------------------------------

ROWS = [
    {   # train row 0 of GSM8K
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as "
                    "many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
                  "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    },
    {   # annotation directly after a currency sign
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
                    "babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
                  "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
    },
    {   # thousands separator, trailing spaces, blank line
        "question": "A house costs 800 a month for 2 years. What is the total?",
        "answer": "He pays 800*12 = <<800*12=9600>>9600 a year.  \n\nOver 2 years: 9600*2 = "
                  "<<9600*2=19200>>19200.\n#### 19,200",
    },
    {   # negative number
        "question": "The temperature fell from 3 to -2. What is the change?",
        "answer": "3 - 5 = <<3-5=-2>>-2.\n#### -2",
    },
]


class FakeTokenizer:
    """Char-level stand-in with Qwen3's chat template behaviour."""

    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]

    def __init__(self, ignore_enable_thinking=False):
        self.vocab = {s: i for i, s in enumerate(self.SPECIALS)}
        self.unk_token_id = None
        self.eos_token_id = self.vocab["<|im_end|>"]
        self.pad_token_id = self.vocab["<|endoftext|>"]
        self._ignore = ignore_enable_thinking
        self._re = re.compile("|".join(map(re.escape, self.SPECIALS)))

    def _id(self, ch):
        return self.vocab.setdefault(ch, len(self.vocab))

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True, **kwargs):
        assert tokenize is False
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
            if not enable_thinking and not self._ignore:
                out += "<think>\n\n</think>\n\n"
        return out

    def __call__(self, text, add_special_tokens=True):
        ids, pos = [], 0
        for m in self._re.finditer(text):
            ids += [self._id(c) for c in text[pos:m.start()]]
            ids.append(self.vocab[m.group()])
            pos = m.end()
        ids += [self._id(c) for c in text[pos:]]
        return {"input_ids": ids}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def examples():
    return examples_from_rows(ROWS, "train")


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def test_clean_removes_annotations_and_marker():
    assert clean_solution(ROWS[0]["answer"]) == (
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "The final answer is 72."
    )


def test_clean_keeps_result_after_currency_annotation():
    assert clean_solution(ROWS[1]["answer"]) == (
        "Weng earns 12/60 = $0.2 per minute.\n"
        "Working 50 minutes, she earned 0.2 x 50 = $10.\n"
        "The final answer is 10."
    )


def test_clean_normalises_commas_whitespace_and_blank_lines():
    cleaned = clean_solution(ROWS[2]["answer"])
    assert cleaned.splitlines() == [
        "He pays 800*12 = 9600 a year.",
        "Over 2 years: 9600*2 = 19200.",
        "The final answer is 19200.",
    ]


def test_clean_negative_number():
    assert clean_solution(ROWS[3]["answer"]).endswith("The final answer is -2.")
    assert gold_number(ROWS[3]["answer"]) == "-2"


def test_clean_leaves_no_markers():
    for row in ROWS:
        out = clean_solution(row["answer"])
        assert "<<" not in out and ">>" not in out and "####" not in out


def test_clean_rejects_bad_input():
    with pytest.raises(ValueError):
        clean_solution("no marker here")
    with pytest.raises(ValueError):
        clean_solution("step <<1+1=2>>2\n#### about two")
    with pytest.raises(ValueError):
        clean_solution("step <<1+1=2\n#### 2")  # unmatched annotation
    with pytest.raises(ValueError):
        clean_solution("step\n#### ")


def test_last_number_of_cleaned_solution_is_gold():
    """Independent re-implementation of the answer rule's core: last number."""
    for ex in examples():
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", ex.human_response)
        assert nums[-1].replace(",", "") == ex.gold


def test_cleaned_solutions_grade_correct_with_answer_extraction():
    """Plan (E2 handoff): every cleaned human solution grades as correct."""
    ae = pytest.importorskip("kd.answer_extraction")

    def grader(text, gold):
        pred = ae.extract_last_number(text)
        return pred is not None and float(str(pred).replace(",", "")) == float(gold)

    assert data.find_ungradable_solutions(examples(), grader) == []


def test_find_ungradable_reports_failures():
    exs = examples()
    assert data.find_ungradable_solutions(exs, lambda t, g: False) == [e.question_id for e in exs]


def test_examples_from_rows_ids_and_error_context():
    exs = examples()
    assert [e.question_id for e in exs] == ["train-00000", "train-00001", "train-00002", "train-00003"]
    assert exs[0].gold == "72"
    with pytest.raises(ValueError, match="test-00001"):
        examples_from_rows([ROWS[0], {"question": "q", "answer": "x"}], "test")


# --------------------------------------------------------------------------
# Prompt and chat template
# --------------------------------------------------------------------------

def test_prompt_contains_question_instruction_and_thinking_off():
    tok = FakeTokenizer()
    text = prompt_text(tok, "How many?")
    assert "How many?" in text and INSTRUCTION in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert prompt_ids(tok, "How many?") == tok(text, add_special_tokens=False)["input_ids"]


def test_prompt_raises_if_template_ignores_thinking_flag():
    with pytest.raises(ValueError, match="think"):
        prompt_text(FakeTokenizer(ignore_enable_thinking=True), "How many?")


def test_tokenizer_compatibility_check():
    data.assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer())
    other = FakeTokenizer()
    other.vocab["extra"] = 999
    with pytest.raises(AssertionError):
        data.assert_tokenizers_compatible(FakeTokenizer(), other)


# --------------------------------------------------------------------------
# Sequence builders and masks
# --------------------------------------------------------------------------

def test_human_sequence_mask_and_end_token():
    tok, ex = FakeTokenizer(), examples()[0]
    seq = build_human_sequence(tok, ex)
    p = prompt_ids(tok, ex.question)
    end = data.end_token_id(tok)
    assert seq.input_ids[: len(p)] == p
    assert seq.response_mask == [0] * len(p) + [1] * (len(seq) - len(p))
    assert seq.input_ids[-1] == end and seq.finished and not seq.truncated
    assert tok.decode(seq.input_ids[len(p):-1]) == ex.human_response
    assert seq.n_prompt == len(p) and seq.n_response == len(seq) - len(p)


def test_arms_share_the_same_prompt_prefix_and_mask_boundary():
    tok, ex = FakeTokenizer(), examples()[1]
    p = prompt_ids(tok, ex.question)
    answer = "12/60 = 0.2 per minute.\nThe final answer is 10."
    rollout = tok(answer, add_special_tokens=False)["input_ids"]
    seqs = [
        build_for_arm("human", tok, ex),
        build_for_arm("ref_hardlabel", tok, ex),
        build_for_arm("teacher", tok, ex, {"question_id": ex.question_id, "response": answer}),
        build_for_arm("onpolicy", tok, ex, rollout),
    ]
    for s in seqs:
        assert s.input_ids[: len(p)] == p
        assert sum(s.response_mask[: len(p)]) == 0
        assert all(m == 1 for m in s.response_mask[len(p):])
    assert seqs[0].input_ids == seqs[1].input_ids          # ref_hardlabel == human sequences
    assert [s.arm for s in seqs] == ["human", "ref_hardlabel", "teacher", "onpolicy"]


def test_teacher_record_truncated_gets_no_end_token_and_ids_are_preferred():
    tok, ex = FakeTokenizer(), examples()[0]
    ids = tok("Step one", add_special_tokens=False)["input_ids"]
    rec = {"question_id": ex.question_id, "response": "IGNORED", "response_ids": ids, "truncated": True}
    seq = build_teacher_sequence(tok, ex, rec)
    assert seq.input_ids[seq.n_prompt:] == ids
    assert not seq.finished
    done = build_teacher_sequence(tok, ex, {**rec, "truncated": False})
    assert done.finished and done.input_ids[-1] == data.end_token_id(tok)


def test_teacher_record_for_wrong_question_raises():
    tok, ex = FakeTokenizer(), examples()[0]
    with pytest.raises(ValueError):
        build_teacher_sequence(tok, ex, {"question_id": "train-09999", "response": "x"})


def test_onpolicy_ids_are_trimmed_at_end_token_and_padding_dropped():
    tok, ex = FakeTokenizer(), examples()[0]
    body = tok("abc", add_special_tokens=False)["input_ids"]
    end, pad = data.end_token_id(tok), tok.pad_token_id
    seq = build_onpolicy_sequence(tok, ex, body + [end, pad, pad, pad])
    assert seq.input_ids[seq.n_prompt:] == body + [end]
    assert seq.finished


def test_onpolicy_without_stop_token_is_unfinished():
    tok, ex = FakeTokenizer(), examples()[0]
    body = tok("abc", add_special_tokens=False)["input_ids"]
    seq = build_onpolicy_sequence(tok, ex, body)
    assert seq.input_ids[seq.n_prompt:] == body and not seq.finished


def test_max_seq_len_truncates_and_flags_and_long_prompt_raises():
    tok, ex = FakeTokenizer(), examples()[0]
    full = build_human_sequence(tok, ex)
    cut = build_human_sequence(tok, ex, max_seq_len=full.n_prompt + 5)
    assert cut.truncated and not cut.finished and len(cut) == full.n_prompt + 5
    assert cut.response_mask[full.n_prompt:] == [1] * 5
    assert not build_human_sequence(tok, ex, max_seq_len=10_000).truncated
    with pytest.raises(ValueError):
        build_human_sequence(tok, ex, max_seq_len=full.n_prompt)


def test_build_for_arm_validates_source_and_arm():
    tok, ex = FakeTokenizer(), examples()[0]
    with pytest.raises(ValueError):
        build_for_arm("human", tok, ex, source=[1, 2])
    with pytest.raises(ValueError):
        build_for_arm("teacher", tok, ex)
    with pytest.raises(ValueError):
        build_for_arm("onpolicy", tok, ex)
    with pytest.raises(ValueError):
        build_for_arm("nonsense", tok, ex)


def test_build_sequence_text_unfinished():
    tok, ex = FakeTokenizer(), examples()[0]
    seq = build_sequence(tok, ex, "partial", arm="teacher", finished=False)
    assert not seq.finished and seq.input_ids[-1] != data.end_token_id(tok)


def test_collate_pads_right_and_masks_padding():
    torch = pytest.importorskip("torch")
    tok, exs = FakeTokenizer(), examples()
    seqs = [build_human_sequence(tok, exs[0]), build_human_sequence(tok, exs[1])]
    batch = data.collate_sequences(seqs, tok.pad_token_id)
    width = max(len(s) for s in seqs)
    assert batch["input_ids"].shape == (2, width)
    short = min(range(2), key=lambda i: len(seqs[i]))
    n = len(seqs[short])
    assert batch["attention_mask"][short, n:].sum() == 0
    assert batch["response_mask"][short, n:].sum() == 0
    assert (batch["input_ids"][short, n:] == tok.pad_token_id).all()
    assert batch["response_mask"].tolist()[short][:n] == seqs[short].response_mask


def test_collate_rejects_empty_response():
    pytest.importorskip("torch")
    tok, ex = FakeTokenizer(), examples()[0]
    empty = build_sequence(tok, ex, [], arm="onpolicy")
    with pytest.raises(ValueError):
        data.collate_sequences([empty], tok.pad_token_id)


# --------------------------------------------------------------------------
# Pool, held-out set, subsets
# --------------------------------------------------------------------------

N_TRAIN = 7473


def test_pool_is_deterministic_and_independent_of_global_rng():
    a = data.select_pool_indices(N_TRAIN, 2000, pool_seed=7)
    random.seed(123)
    [random.random() for _ in range(1000)]
    b = data.select_pool_indices(N_TRAIN, 2000, pool_seed=7)
    assert a == b and len(set(a)) == 2000


def test_pool_depends_on_pool_seed_only():
    assert data.select_pool_indices(N_TRAIN, 2000, 7) != data.select_pool_indices(N_TRAIN, 2000, 8)


def test_pools_are_nested():
    big = data.select_pool_indices(N_TRAIN, 2000, 7)
    small = data.select_pool_indices(N_TRAIN, 1000, 7)
    assert small == big[:1000]


def test_heldout_is_disjoint_from_largest_pool():
    pool = set(data.select_pool_indices(N_TRAIN, 2000, 7))
    held = data.select_heldout_indices(N_TRAIN, 500, 7)
    assert len(held) == 500 and not pool & set(held)


def test_load_heldout_checks_disjointness_before_loading():
    fake_train = [None] * 100
    with pytest.raises(ValueError):
        data.load_heldout(60, 7, n_pool=50, train=fake_train)


def test_pool_size_validation():
    with pytest.raises(ValueError):
        data.select_pool_indices(100, 0, 1)
    with pytest.raises(ValueError):
        data.select_pool_indices(100, 101, 1)


def test_select_subset_is_deterministic_sorted_and_sized():
    rows = [{"question": f"q{i}", "answer": f"a\n#### {i}"} for i in range(50)]
    exs = examples_from_rows(rows, "test")
    s1, s2 = data.select_subset(exs, 10, seed=42), data.select_subset(exs, 10, seed=42)
    assert [e.question_id for e in s1] == [e.question_id for e in s2]
    assert len(s1) == 10 and [e.index for e in s1] == sorted(e.index for e in s1)


# --------------------------------------------------------------------------
# Teacher cache
# --------------------------------------------------------------------------

def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_teacher_cache_roundtrip_and_validation(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [{"question_id": "train-00000", "response": "a"}, {"question_id": "train-00001", "response": "b"}])
    cache = data.load_teacher_cache(p, expected_ids=["train-00000", "train-00001"])
    assert set(cache) == {"train-00000", "train-00001"}
    with pytest.raises(ValueError, match="missing"):
        data.load_teacher_cache(p, expected_ids=["train-00000", "train-00002"])
    _write(p, [{"question_id": "x", "response": "a"}, {"question_id": "x", "response": "b"}])
    with pytest.raises(ValueError, match="duplicate"):
        data.load_teacher_cache(p)
    _write(p, [{"question_id": "x"}])
    with pytest.raises(ValueError):
        data.load_teacher_cache(p)


# --------------------------------------------------------------------------
# Opt-in: real data and real tokenizers (needs internet / HF cache)
# --------------------------------------------------------------------------

RUN_REAL = os.environ.get("KD_RUN_DATA_TESTS") == "1"


@pytest.mark.skipif(not RUN_REAL, reason="set KD_RUN_DATA_TESTS=1 to run on real GSM8K")
def test_real_gsm8k_every_cleaned_solution_is_valid():
    pytest.importorskip("datasets")
    train = data.load_split("train")
    assert len(train) == 7473 and len(data.load_test()) == 1319
    for ex in train:
        assert ex.human_response.endswith(f"The final answer is {ex.gold}.")
        assert "<<" not in ex.human_response and "####" not in ex.human_response
    ae = pytest.importorskip("kd.answer_extraction")
    bad = data.find_ungradable_solutions(
        train, lambda t, g: float(str(ae.extract_last_number(t)).replace(",", "")) == float(g))
    assert bad == []


@pytest.mark.skipif(not RUN_REAL, reason="set KD_RUN_DATA_TESTS=1 to run with real Qwen3 tokenizers")
def test_real_qwen3_tokenizers_agree_and_render_thinking_off():
    transformers = pytest.importorskip("transformers")
    teacher = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    student = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    data.assert_tokenizers_compatible(teacher, student)
    ex = examples()[0]
    seq = build_human_sequence(student, ex)
    assert seq.input_ids[-1] == student.convert_tokens_to_ids("<|im_end|>")
    assert student.decode(seq.input_ids[seq.n_prompt:-1]) == ex.human_response
