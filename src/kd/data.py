"""kd.data -- deliverable A2 (owner: T2, reviewer: E2).

GSM8K loading, the fixed training pool, the prompt / chat-template builder,
human-solution cleaning, and one sequence builder per arm.

This module is SHARED: ``evaluate.py`` (E2) and ``metrics.py`` (E1) import the
prompt builder from here and never keep a copy.  It must not import torch,
transformers or datasets at module level, so that the cleaning rule and the
pool selection can be unit-tested anywhere.

---------------------------------------------------------------------------
THE CLEANING RULE (``clean_solution``)
---------------------------------------------------------------------------
A raw GSM8K solution looks like::

    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72

``clean_solution`` turns it into::

    Natalia sold 48/2 = 24 clips in May.
    Natalia sold 48+24 = 72 clips altogether in April and May.
    The final answer is 72.

1. Split at the LAST ``####``.  Everything before it is the reasoning, the
   text after it is the final answer.  No ``####`` -> ValueError.
2. Remove every calculator annotation ``<<...>>`` from the reasoning.  The
   result that follows the annotation is kept, so ``$<<12/60=0.2>>0.2`` becomes
   ``$0.2``.  Any ``<<`` or ``>>`` left over -> ValueError (never silent).
3. Tidy whitespace: collapse runs of spaces/tabs, strip each line, drop blank
   lines.  Line breaks between reasoning steps are kept.
4. Normalise the final answer: strip spaces, ``$`` and a trailing ``.``,
   remove thousands separators (``1,600`` -> ``1600``).  The result must be a
   plain number (``-?digits[.digits]``), otherwise ValueError.
5. Append a new last line ``The final answer is <number>.``  The sentence
   contains no digits other than the number, so the LAST NUMBER of the cleaned
   solution is the gold answer.  That is exactly what the answer rule
   (``answer_extraction.py``: last number, commas / currency / trailing period
   stripped) looks for, so every cleaned solution grades as correct.

The rule is applied identically for the ``human`` arm and the ``ref_hardlabel``
arm (they use the same sequences).  The ``teacher`` and ``onpolicy`` arms never
see human text; they are only asked the same prompt.

---------------------------------------------------------------------------
PROMPT AND CHAT TEMPLATE
---------------------------------------------------------------------------
One user message: ``"<question>\\n\\nSolve the problem step by step. Put the
final number at the end."``  The text is rendered by the tokenizer's own chat
template with ``add_generation_prompt=True`` and ``enable_thinking=False``.
For Qwen3 that ends the prompt with an empty think block
(``<think>\\n\\n</think>\\n\\n``); ``prompt_text`` raises if it does not, so a
template that silently switches thinking back on can never reach a run.

---------------------------------------------------------------------------
SEQUENCES AND MASKS
---------------------------------------------------------------------------
Every arm returns a ``TokenizedSequence`` with

* ``input_ids``      prompt ids + response ids (+ end token if the response
                     finished)
* ``response_mask``  1 for response tokens (end token included), 0 for prompt.

``response_mask`` is aligned with ``input_ids`` (token i <-> mask i).  Logits
at position t predict token t+1, so a loss over logits ``[:, :-1]`` must use
``response_mask[:, 1:]``.  Padding is added by ``collate_sequences`` and is
always mask 0.  Prompt and response are tokenised separately and concatenated,
so the mask boundary is exact and equals what ``generate`` produces for the
on-policy arm.

Response end: ``<|im_end|>`` (``end_token_id``).  A response that did not end
by itself (truncated at ``max_new_tokens`` or at ``max_seq_len``) gets NO end
token, so a student is never trained to stop mid-sentence.

---------------------------------------------------------------------------
THE FIXED POOL
---------------------------------------------------------------------------
Questions are ranked by ``sha256("kd-pool:<pool_seed>:<split>:<index>")``.
The pool is the first N ranks, the held-out set (for KL on rollouts) is the
last H ranks.  Consequences: no global RNG is used anywhere; the result does
not depend on library versions; the pool of 1,000 is a subset of the pool of
2,000 (so the "cut to 1,000" fallback keeps every shared question); pool and
held-out are disjoint whenever N + H <= 7,473.

``question_id`` is ``"<split>-<row index in the HF split>"``, e.g.
``train-00000``.  It is the key used by the teacher cache, the logs and the
predictions files.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence as SeqABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"

#: The instruction of the briefing pack (section 1, "Prompt").
INSTRUCTION = "Solve the problem step by step. Put the final number at the end."
FINAL_ANSWER_TEMPLATE = "The final answer is {answer}."

END_TOKEN = "<|im_end|>"
ENDOFTEXT_TOKEN = "<|endoftext|>"
#: What a Qwen3 template renders last when thinking is off.
THINKING_OFF_SUFFIX = "</think>\n\n"

#: Arms that train on sequences produced by ``build_for_arm``.
TRAINING_ARMS = ("human", "teacher", "onpolicy", "ref_hardlabel")

_ANNOTATION_RE = re.compile(r"<<.*?>>")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------
# Human-solution cleaning
# --------------------------------------------------------------------------

def normalize_final_number(raw: str) -> str:
    """Normalise the text after ``####`` to a plain number string.

    ``" 1,600 "`` -> ``"1600"``, ``"$18."`` -> ``"18"``, ``"-5"`` -> ``"-5"``.
    Raises ValueError if the result is not a plain number.
    """
    s = raw.strip().replace(",", "").replace("$", "").strip()
    if s.endswith("."):
        s = s[:-1].strip()
    if not _NUMBER_RE.fullmatch(s):
        raise ValueError(f"final answer is not a plain number: {raw!r}")
    return s


def split_solution(solution: str) -> tuple[str, str]:
    """Split a raw GSM8K solution at the last ``####`` into (reasoning, final)."""
    marker = solution.rfind("####")
    if marker < 0:
        raise ValueError("solution has no '####' final-answer line")
    return solution[:marker], solution[marker + 4:]


def gold_number(solution: str) -> str:
    """Normalised gold answer (a number string) of a raw GSM8K solution."""
    return normalize_final_number(split_solution(solution)[1])


def clean_solution(solution: str) -> str:
    """Raw GSM8K solution -> cleaned human response. See the module docstring."""
    body, final_raw = split_solution(solution)
    final = normalize_final_number(final_raw)

    body = _ANNOTATION_RE.sub("", body)
    if "<<" in body or ">>" in body:
        raise ValueError("unmatched calculator annotation left after cleaning")

    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if ln]
    lines.append(FINAL_ANSWER_TEMPLATE.format(answer=final))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Examples and loading
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    """One GSM8K question. Identical object for every arm."""

    question_id: str      # "<split>-<row index>", e.g. "train-00042"
    split: str            # "train" or "test"
    index: int            # row index in the HF split
    question: str
    solution: str         # raw human solution (with <<...>> and ####)
    gold: str             # normalised gold number as a string
    human_response: str   # cleaned human solution (see clean_solution)


def make_question_id(split: str, index: int) -> str:
    return f"{split}-{index:05d}"


def examples_from_rows(rows: Iterable[Mapping[str, Any]], split: str) -> list[Example]:
    """Build Examples from rows with ``question`` and ``answer`` fields.

    The position in ``rows`` is the row index, so pass rows in HF order.
    Fails loudly (with the question id) if a row cannot be cleaned.
    """
    out: list[Example] = []
    for i, row in enumerate(rows):
        qid = make_question_id(split, i)
        try:
            solution = row["answer"]
            out.append(
                Example(
                    question_id=qid,
                    split=split,
                    index=i,
                    question=row["question"].strip(),
                    solution=solution,
                    gold=gold_number(solution),
                    human_response=clean_solution(solution),
                )
            )
        except ValueError as err:
            raise ValueError(f"{qid}: {err}") from err
    return out


def load_split(split: str, *, cache_dir: str | None = None) -> list[Example]:
    """Load a GSM8K split ("train": 7,473 rows, "test": 1,319 rows).

    Needs the ``datasets`` package and either internet (Kaggle: switch it on in
    the notebook settings) or a populated ``cache_dir`` / HF cache.
    """
    from datasets import load_dataset  # lazy: keeps this module importable anywhere

    ds = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=split, cache_dir=cache_dir)
    return examples_from_rows(ds, split)


# --------------------------------------------------------------------------
# Fixed pool, held-out set, subsets (no global RNG anywhere)
# --------------------------------------------------------------------------

def _ranked_indices(n_total: int, seed: int, split: str) -> list[int]:
    def key(i: int) -> str:
        return hashlib.sha256(f"kd-pool:{seed}:{split}:{i}".encode()).hexdigest()

    return sorted(range(n_total), key=key)


def select_pool_indices(n_total: int, n_pool: int, pool_seed: int, split: str = "train") -> list[int]:
    """First ``n_pool`` ranks. Pools are nested: pool(1000) is a subset of pool(2000)."""
    if not 0 < n_pool <= n_total:
        raise ValueError(f"n_pool must be in 1..{n_total}, got {n_pool}")
    return _ranked_indices(n_total, pool_seed, split)[:n_pool]


def select_heldout_indices(n_total: int, n_heldout: int, pool_seed: int, split: str = "train") -> list[int]:
    """Last ``n_heldout`` ranks: the slice of GSM8K train outside the pool."""
    if not 0 < n_heldout <= n_total:
        raise ValueError(f"n_heldout must be in 1..{n_total}, got {n_heldout}")
    return _ranked_indices(n_total, pool_seed, split)[-n_heldout:]


def load_pool(n_pool: int, pool_seed: int, *, train: list[Example] | None = None,
              cache_dir: str | None = None) -> list[Example]:
    """The fixed training pool: ``n_pool`` train questions, chosen once by ``pool_seed``.

    The pool does not depend on the run seed. The ORDER in which a run visits
    the pool comes from ``seeding.py`` (``data_order``), not from here; the
    list returned here is in rank order and must not be iterated directly.
    """
    train = train if train is not None else load_split("train", cache_dir=cache_dir)
    return [train[i] for i in select_pool_indices(len(train), n_pool, pool_seed)]


def load_heldout(n_heldout: int, pool_seed: int, *, n_pool: int,
                 train: list[Example] | None = None,
                 cache_dir: str | None = None) -> list[Example]:
    """Held-out train questions outside the pool (for KL on student rollouts).

    ``n_pool`` is required so that disjointness is checked, not assumed.
    """
    train = train if train is not None else load_split("train", cache_dir=cache_dir)
    if n_pool + n_heldout > len(train):
        raise ValueError(
            f"pool ({n_pool}) + held-out ({n_heldout}) exceeds the {len(train)} train questions"
        )
    return [train[i] for i in select_heldout_indices(len(train), n_heldout, pool_seed)]


def select_subset(examples: list[Example], n: int, seed: int) -> list[Example]:
    """Deterministic subset of ``n`` examples (hash-ranked, no global RNG).

    Used e.g. for the small test slice in ``configs/smoke.yaml``. NOTE: this is
    NOT guaranteed to equal the seed-42, 300-question subset of the headroom
    notebook; the harness gate must use whatever selection that notebook used.
    """
    if not 0 < n <= len(examples):
        raise ValueError(f"n must be in 1..{len(examples)}, got {n}")
    idx = _ranked_indices(len(examples), seed, examples[0].split + "-subset")[:n]
    return [examples[i] for i in sorted(idx)]


def load_test(*, cache_dir: str | None = None) -> list[Example]:
    """The full GSM8K test set (1,319 questions), in dataset order."""
    return load_split("test", cache_dir=cache_dir)


# --------------------------------------------------------------------------
# Prompt and chat template (SHARED with evaluate.py -- import, never copy)
# --------------------------------------------------------------------------

def build_user_message(question: str) -> str:
    return f"{question.strip()}\n\n{INSTRUCTION}"


def build_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_user_message(question)}]


def prompt_text(tokenizer, question: str) -> str:
    """Chat-templated prompt string, generation prompt included, thinking off."""
    text = tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not text.endswith(THINKING_OFF_SUFFIX):
        raise ValueError(
            "chat template did not render a closed (empty) think block; thinking "
            f"may be on. Prompt ends with {text[-40:]!r}"
        )
    return text


def _encode(tokenizer, text: str) -> list[int]:
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def prompt_ids(tokenizer, question: str) -> list[int]:
    """Token ids of the prompt. This is what ``generate`` is called with."""
    return _encode(tokenizer, prompt_text(tokenizer, question))


def end_token_id(tokenizer) -> int:
    """Id of ``<|im_end|>``, the token that ends an assistant turn."""
    tid = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        tid = tokenizer.eos_token_id
    if tid is None:
        raise ValueError("tokenizer has no end-of-turn token")
    return int(tid)


def stop_token_ids(tokenizer) -> frozenset[int]:
    """Ids that end a generation: ``<|im_end|>``, eos_token_id, ``<|endoftext|>``."""
    ids = {end_token_id(tokenizer)}
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eot = tokenizer.convert_tokens_to_ids(ENDOFTEXT_TOKEN)
    if eot is not None and eot != getattr(tokenizer, "unk_token_id", None):
        ids.add(int(eot))
    return frozenset(ids)


def assert_tokenizers_compatible(tok_a, tok_b) -> None:
    """Teacher and student must see identical ids: same vocab, end token, prompt."""
    if tok_a.get_vocab() != tok_b.get_vocab():
        raise AssertionError("teacher and student tokenizers have different vocabularies")
    if end_token_id(tok_a) != end_token_id(tok_b):
        raise AssertionError("teacher and student disagree on the end token")
    probe = "What is 1 + 1?"
    if prompt_text(tok_a, probe) != prompt_text(tok_b, probe):
        raise AssertionError("teacher and student render different chat prompts")


# --------------------------------------------------------------------------
# Sequences and per-arm builders
# --------------------------------------------------------------------------

@dataclass
class TokenizedSequence:
    question_id: str
    arm: str
    input_ids: list[int]
    response_mask: list[int]   # token-aligned with input_ids; 1 = response token
    n_prompt: int
    finished: bool             # the response ends with the end token
    truncated: bool            # cut by max_seq_len (always implies finished=False)

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def n_response(self) -> int:
        return len(self.input_ids) - self.n_prompt


def build_sequence(
    tokenizer,
    example: Example,
    response: str | SeqABC[int],
    *,
    arm: str,
    finished: bool | None = None,
    max_seq_len: int | None = None,
) -> TokenizedSequence:
    """Prompt + response -> ids and response mask. The one place masks are made.

    ``response`` is text (human / cached teacher text) or token ids (cached
    teacher ids / on-policy rollout).

    * text: the end token is appended unless ``finished=False``.
    * ids: cut at the first stop token if there is one (padding after the end
      token disappears) and finished=True; with no stop token the response is
      treated as unfinished unless ``finished=True`` is passed.
    * ``max_seq_len``: the response is cut to fit and ``truncated`` is set; a
      prompt that alone reaches ``max_seq_len`` raises.
    """
    p_ids = prompt_ids(tokenizer, example.question)
    end_id = end_token_id(tokenizer)

    if isinstance(response, str):
        r_ids = _encode(tokenizer, response)
        if finished is None:
            finished = True
    else:
        r_ids = [int(t) for t in response]
        stops = stop_token_ids(tokenizer)
        cut = next((k for k, t in enumerate(r_ids) if t in stops), None)
        if cut is not None:
            r_ids = r_ids[:cut]
            finished = True
        elif finished is None:
            finished = False
    if finished:
        r_ids = r_ids + [end_id]

    ids = p_ids + r_ids
    truncated = False
    if max_seq_len is not None:
        if len(p_ids) >= max_seq_len:
            raise ValueError(
                f"{example.question_id}: prompt has {len(p_ids)} tokens, max_seq_len is {max_seq_len}"
            )
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            truncated = True
            finished = False  # the end token (if any) was cut off

    mask = [0] * len(p_ids) + [1] * (len(ids) - len(p_ids))
    return TokenizedSequence(
        question_id=example.question_id,
        arm=arm,
        input_ids=ids,
        response_mask=mask,
        n_prompt=len(p_ids),
        finished=bool(finished),
        truncated=truncated,
    )


def build_human_sequence(tokenizer, example: Example, *, arm: str = "human",
                         max_seq_len: int | None = None) -> TokenizedSequence:
    """``human`` / ``ref_hardlabel``: the cleaned human solution."""
    return build_sequence(tokenizer, example, example.human_response, arm=arm,
                          finished=True, max_seq_len=max_seq_len)


def build_teacher_sequence(tokenizer, example: Example, record: Mapping[str, Any], *,
                           max_seq_len: int | None = None) -> TokenizedSequence:
    """``teacher``: a cached teacher answer.

    ``record`` is one row of ``data/teacher_sequences.jsonl``: ``question_id``,
    ``response`` (text), optionally ``response_ids`` (preferred when present, so
    no re-tokenisation drift) and ``truncated`` (True if the teacher hit
    ``max_new_tokens``; then no end token is added).
    """
    if record.get("question_id", example.question_id) != example.question_id:
        raise ValueError(f"teacher record {record.get('question_id')} is not for {example.question_id}")
    finished = not bool(record.get("truncated", False))
    ids = record.get("response_ids")
    response: str | SeqABC[int] = ids if ids is not None else record["response"]
    return build_sequence(tokenizer, example, response, arm="teacher",
                          finished=finished, max_seq_len=max_seq_len)


def build_onpolicy_sequence(tokenizer, example: Example, rollout_ids: SeqABC[int], *,
                            max_seq_len: int | None = None) -> TokenizedSequence:
    """``onpolicy``: the student's own rollout (ids straight from ``generate``)."""
    return build_sequence(tokenizer, example, rollout_ids, arm="onpolicy",
                          max_seq_len=max_seq_len)


def build_for_arm(arm: str, tokenizer, example: Example, source: Any = None, *,
                  max_seq_len: int | None = None) -> TokenizedSequence:
    """Uniform entry point so that ``train.py`` only picks the arm.

    ``source``: ignored (must be None) for human / ref_hardlabel; the teacher
    cache record for ``teacher``; the rollout token ids for ``onpolicy``.
    """
    if arm in ("human", "ref_hardlabel"):
        if source is not None:
            raise ValueError(f"arm {arm!r} takes no external source")
        return build_human_sequence(tokenizer, example, arm=arm, max_seq_len=max_seq_len)
    if arm == "teacher":
        if source is None:
            raise ValueError("arm 'teacher' needs its cache record")
        return build_teacher_sequence(tokenizer, example, source, max_seq_len=max_seq_len)
    if arm == "onpolicy":
        if source is None:
            raise ValueError("arm 'onpolicy' needs the rollout token ids")
        return build_onpolicy_sequence(tokenizer, example, source, max_seq_len=max_seq_len)
    raise ValueError(f"unknown arm {arm!r}; expected one of {TRAINING_ARMS}")


# --------------------------------------------------------------------------
# Teacher cache, batching, answer-rule check
# --------------------------------------------------------------------------

def load_teacher_cache(path: str | Path, *, expected_ids: Iterable[str] | None = None) -> dict[str, dict]:
    """Read ``data/teacher_sequences.jsonl`` into ``{question_id: record}``.

    Raises on duplicate ids, rows without ``question_id`` / ``response``, and
    (if ``expected_ids`` is given, e.g. the pool) on any missing question.
    """
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "question_id" not in rec or "response" not in rec:
                raise ValueError(f"{path}:{line_no}: row needs 'question_id' and 'response'")
            if rec["question_id"] in records:
                raise ValueError(f"{path}:{line_no}: duplicate {rec['question_id']}")
            records[rec["question_id"]] = rec
    if expected_ids is not None:
        missing = [q for q in expected_ids if q not in records]
        if missing:
            raise ValueError(f"teacher cache is missing {len(missing)} questions, e.g. {missing[:3]}")
    return records


def collate_sequences(sequences: list[TokenizedSequence], pad_token_id: int, *, device=None) -> dict:
    """Right-pad a batch. Returns long ``input_ids`` / ``attention_mask`` and
    ``response_mask`` (token-aligned; padding is 0). Needs torch.
    """
    import torch  # lazy

    if not sequences:
        raise ValueError("empty batch")
    for s in sequences:
        if s.n_response == 0:
            raise ValueError(f"{s.question_id}: no response tokens; skip it and log the skip")
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(sequences), width), dtype=torch.long)
    rmask = torch.zeros((len(sequences), width), dtype=torch.long)
    for b, s in enumerate(sequences):
        n = len(s)
        ids[b, :n] = torch.tensor(s.input_ids, dtype=torch.long)
        attn[b, :n] = 1
        rmask[b, :n] = torch.tensor(s.response_mask, dtype=torch.long)
    batch = {"input_ids": ids, "attention_mask": attn, "response_mask": rmask}
    return {k: v.to(device) for k, v in batch.items()} if device is not None else batch


def find_ungradable_solutions(examples: Iterable[Example], grader) -> list[str]:
    """Question ids whose cleaned human solution does NOT grade as correct.

    ``grader(response_text: str, gold: str) -> bool`` is built by the caller
    from ``kd.answer_extraction`` (the answer rule is not copied here). The
    plan requires this list to be empty for the whole training split.
    """
    return [ex.question_id for ex in examples if not grader(ex.human_response, ex.gold)]
