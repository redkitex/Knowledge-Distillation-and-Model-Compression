"""Generate the teacher's answers for the training pool, once, and save them.

    python scripts/10_cache_teacher_sequences.py --config configs/default.yaml

Output (both files are committed, so nobody has to regenerate them):

    data/teacher_sequences.jsonl      one row per pool question, in pool order
    data/teacher_sequences.meta.json  the settings used, a hash of the file, statistics

One row of the JSONL file (this is the ``record`` that
``kd.data.build_teacher_sequence`` reads):

    question_id   "train-00042"
    response      the teacher's answer as text
    response_ids  the same answer as token ids, WITHOUT the end token
    truncated     true if the teacher hit max_new_tokens before it stopped
    n_tokens      number of answer tokens (= len(response_ids))

Decoding is greedy and answers are NOT filtered for correctness, so that the
``teacher`` arm differs from ``human`` only in where the text comes from.

Re-running is safe:
* finished file + same settings  -> nothing is generated, statistics are printed
* unfinished file + same settings -> continues where it stopped (a Kaggle
  session that dies costs only the current batch)
* different settings              -> stops with an error; use --force to redo

The prompt comes from ``kd.data``, the generator from ``kd.evaluate`` and the
answer rule from ``kd.answer_extraction``; none of them is copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

from kd import data

# --------------------------------------------------------------------------
# Small pure helpers (no GPU needed; these are what the tests cover)
# --------------------------------------------------------------------------

def settings_fingerprint(cfg: dict, question_ids: list[str]) -> dict:
    """Everything the teacher answers depend on.

    If any of these changes, the saved answers are stale. On-policy sampling
    settings, learning rate etc. are deliberately absent: they do not affect
    greedy teacher answers.
    """
    ids_hash = hashlib.sha256("\n".join(question_ids).encode()).hexdigest()
    return {
        "teacher": cfg["teacher"],
        "pool_size": cfg["data"]["pool_size"],
        "pool_seed": cfg["data"]["pool_seed"],
        "max_new_tokens": cfg["generation"]["max_new_tokens"],
        "decoding": "greedy",
        "instruction": data.INSTRUCTION,
        "question_ids_sha256": ids_hash,
    }


def split_generated(new_ids: list[int], stop_ids: frozenset[int]) -> tuple[list[int], bool]:
    """Cut generated ids at the first stop token.

    Returns (answer ids without the stop token, truncated). ``truncated`` is
    True when there is no stop token at all, i.e. the teacher was still
    writing when it reached max_new_tokens.
    """
    for k, tok in enumerate(new_ids):
        if tok in stop_ids:
            return list(new_ids[:k]), False
    return list(new_ids), True


def make_record(example: data.Example, new_ids: list[int], tokenizer) -> dict:
    """Generated ids for one question -> one JSONL row."""
    ids, truncated = split_generated([int(t) for t in new_ids], data.stop_token_ids(tokenizer))
    return {
        "question_id": example.question_id,
        "response": tokenizer.decode(ids, skip_special_tokens=True),
        "response_ids": ids,
        "truncated": truncated,
        "n_tokens": len(ids),
    }


def read_rows(path: Path) -> dict[str, dict]:
    """Rows already on disk, keyed by question_id ({} if there is no file).

    A line that is not valid JSON (half-written when a session died) is
    dropped with a warning; that question is simply generated again.
    """
    rows: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARNING: {path}:{line_no} is incomplete; dropped.", file=sys.stderr)
                    continue
                rows[row["question_id"]] = row
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write all rows, replacing the file in one step (never half a file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(pool: list[data.Example], rows: dict[str, dict], grader=None) -> dict:
    """Mean length, truncation rate and (if a grader is given) fraction correct.

    ``grader(response_text, gold) -> bool``. Correctness is reported only; it
    is never stored per row and never used to filter answers.
    """
    lengths = [rows[ex.question_id]["n_tokens"] for ex in pool]
    n = len(pool)
    stats = {
        "n": n,
        "mean_tokens": round(sum(lengths) / n, 1),
        "max_tokens": max(lengths),
        "truncation_rate": round(sum(rows[ex.question_id]["truncated"] for ex in pool) / n, 4),
        "fraction_correct": None,
    }
    if grader is not None:
        correct = sum(bool(grader(rows[ex.question_id]["response"], ex.gold)) for ex in pool)
        stats["fraction_correct"] = round(correct / n, 4)
    return stats


def load_grader():
    """The shared answer rule from kd.answer_extraction, or None if it is not there yet."""
    try:
        from kd.answer_extraction import extract_last_number
    except ImportError:
        print("WARNING: kd.answer_extraction.extract_last_number not found; "
              "fraction correct is not reported.", file=sys.stderr)
        return None

    # extract_last_number(text) returns a float, or None if there is no number.
    # This is the same comparison evaluate.py makes. ``gold`` is data.py's
    # normalised gold string, so it is converted here.
    def grader(response: str, gold: str) -> bool:
        pred = extract_last_number(response)
        return pred is not None and abs(pred - float(gold)) < 1e-6

    return grader


# --------------------------------------------------------------------------
# The cache logic: skip / resume / generate / finalise
# --------------------------------------------------------------------------

def cache_answers(pool, out_path: Path, fingerprint: dict, generate_fn, make_row,
                  *, batch_size: int, force: bool = False, grader=None) -> dict:
    """Make sure ``out_path`` holds one answer per pool question. Returns the meta dict.

    ``generate_fn(examples) -> list of generated-id lists`` does the GPU work
    and ``make_row(example, new_ids) -> dict`` builds a row. Passing them in
    keeps this function free of torch, so the skip and resume logic can be
    tested with a fake generator. ``generate_fn`` may be a zero-argument
    factory result; it is only called if something is missing.
    """
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    old_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if force:
        out_path.unlink(missing_ok=True)
        old_meta = None
    if (old_meta is None) != (not out_path.exists()):
        raise SystemExit(f"{out_path} and {meta_path.name} must exist together; "
                         "restore the missing one or rerun with --force.")
    if old_meta is not None and old_meta["settings"] != fingerprint:
        changed = sorted(k for k in fingerprint if old_meta["settings"].get(k) != fingerprint[k])
        raise SystemExit(f"{out_path} was made with different settings ({', '.join(changed)}). "
                         "Rerun with --force to regenerate it.")

    rows = read_rows(out_path)
    todo = [ex for ex in pool if ex.question_id not in rows]

    if old_meta is not None and old_meta.get("complete") and not todo:
        if file_sha256(out_path) != old_meta["sha256"]:
            raise SystemExit(f"{out_path} does not match the hash in {meta_path.name}: "
                             "the file was edited. Restore it or rerun with --force.")
        print(f"Up to date: {out_path} ({len(pool)} answers). Nothing generated.")
        return old_meta

    # Record the settings BEFORE generating, so a dead session can be resumed
    # only with the same settings.
    meta = {"settings": fingerprint, "complete": False}
    meta_path.write_text(json.dumps(meta, indent=2))
    if rows:
        print(f"Resuming: {len(rows)} answers on disk, {len(todo)} to go.")
        # Rewrite the good rows, so new rows are never appended to a broken line.
        write_rows(out_path, list(rows.values()))

    # Longest questions first: if the batch size is too big for the GPU, we
    # find out in the first minute, not after an hour.
    todo.sort(key=lambda ex: len(ex.question), reverse=True)
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            for ex, new_ids in zip(batch, generate_fn(batch), strict=True):
                rows[ex.question_id] = make_row(ex, new_ids)
                f.write(json.dumps(rows[ex.question_id], ensure_ascii=False) + "\n")
            f.flush()
            done = start + len(batch)
            print(f"  {done}/{len(todo)} answers, {time.time() - started:.0f} s", flush=True)

    # Rewrite in pool order so the file (and its hash) is the same no matter
    # how often the run was interrupted.
    write_rows(out_path, [rows[ex.question_id] for ex in pool])
    meta.update(
        complete=True,
        sha256=file_sha256(out_path),
        stats=summarize(pool, rows, grader),
        batch_size=batch_size,
        seconds_this_session=round(time.time() - started),
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------
# GPU part: E2's batched greedy generator, imported from kd.evaluate
# --------------------------------------------------------------------------

def make_generate_fn(model, tokenizer, *, device: str, dtype, max_new_tokens: int):
    """Returns ``generate(examples) -> list[list[int]]`` using ``kd.evaluate.generate_batch``.

    The teacher answers are generated by exactly the same code that evaluates
    every model, so the cache and evaluation cannot drift apart. That function
    does the left padding, greedy decoding and out-of-memory halving. It
    returns each answer cut just AFTER the first stop token (the stop token is
    kept); a row with no stop token hit ``max_new_tokens``. ``make_record``
    turns that into a row without the stop token plus a ``truncated`` flag.
    """
    # Imported here, not at the top: kd.evaluate imports torch and
    # transformers, and the tests of this script must run without them.
    from kd.evaluate import generate_batch

    stop_ids = data.stop_token_ids(tokenizer)

    def generate(examples):
        prompts = [data.prompt_ids(tokenizer, ex.question) for ex in examples]
        return generate_batch(model, tokenizer, device, prompts, stop_ids, max_new_tokens, dtype)

    return generate


def load_teacher(name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="speed only; halves itself if the GPU runs out of memory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="delete existing answers and regenerate")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
        
    out_path = Path(args.out or cfg["data"]["teacher_sequences"])
    out_path.parent.mkdir(parents=True, exist_ok=True) # create /data if it doesn't exist
    
    pool = data.load_pool(cfg["data"]["pool_size"], cfg["data"]["pool_seed"])
    fingerprint = settings_fingerprint(cfg, [ex.question_id for ex in pool])

    # The model is loaded only when the first batch is needed, so an
    # up-to-date cache costs seconds and no GPU.
    state: dict = {}

    def ensure_model():
        if not state:
            import torch

            model, tok = load_teacher(cfg["teacher"], args.device)
            state["tok"] = tok
            state["gen"] = make_generate_fn(
                model, tok, device=args.device, dtype=torch.float16,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
            )
        return state

    meta = cache_answers(
        pool, out_path, fingerprint,
        generate_fn=lambda batch: ensure_model()["gen"](batch),
        make_row=lambda ex, new_ids: make_record(ex, new_ids, ensure_model()["tok"]),
        batch_size=args.batch_size, force=args.force, grader=load_grader(),
    )
    print(json.dumps(meta["stats"], indent=2))


if __name__ == "__main__":
    main()
