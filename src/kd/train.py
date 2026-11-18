"""Training loop: one arm, one seed. Owner: T1 (with T3 for the run bookkeeping).

STATUS: DRAFT for T1 to review. It runs end to end (tests/test_train.py trains a
tiny model on CPU through every arm), but T1 owns it: read it top to bottom,
change what you disagree with, keep the tests passing. Open points are marked
"TODO(T1)" and collected at the end of this docstring.

Called by scripts/20_train.py as  run_training(cfg, arm, seed, out_dir).
It writes   out_dir/train_log.jsonl   one row per optimizer step
and saves   out_dir/checkpoint/        the trained student (30_evaluate.py loads it)

What differs between the arms, and NOTHING else does:

    arm            training text                    loss
    human          GSM8K solutions (cleaned)        forward KL to the teacher
    teacher        cached teacher answers           forward KL to the teacher
    onpolicy       the student's own answers        forward KL to the teacher
    ref_hardlabel  GSM8K solutions (cleaned)        cross-entropy on the correct token

Same for every arm: the questions, their order, the number of steps, the
optimizer, the learning rate and the maximum sequence length.

The two bugs of the v1 notebook, and how this file avoids them:
  1. Shared randomness. v1 used one global RNG for the question order and for
     on-policy sampling, so arms saw different orders. Here the order comes
     from its own random.Random("data-order:<seed>"), and on-policy sampling is
     re-seeded from (seed, step) before every generation. The hash of the
     question ids of each step is logged, so E1 can check it across arms.
  2. Skipped steps. v1 skipped a step after an out-of-memory error. Here the
     same step is retried (cfg train.oom_retries times) and then the run stops.

DECISIONS taken in this draft (check them before the Nov 20 lock):
  - Micro-batches of ONE sequence, gradients accumulated over the step, as in
    v1: memory stays bounded by the longest single sequence.
  - The loss is averaged over all response tokens of the step (briefing pack 2.4).
  - Constant learning rate, no warm-up (as in v1). AdamW defaults otherwise.
  - Max sequence length = longest prompt in the pool + max_new_tokens, for all arms.
  - Gradient checkpointing on (as in v1).
  - The last incomplete batch of the pool is dropped, so every step has the
    same number of questions (2000 / 4 leaves nothing to drop).

TODO(T1):
  - Check the memory on Kaggle with max_new_tokens = 512 (v1 was measured at 384).
  - Decide whether to save the checkpoint in fp16 (half the size; evaluation
    loads fp16 anyway). Currently saved as trained (fp32).
  - Confirm the on-policy re-seeding gives identical rollouts when a step is
    retried after an out-of-memory error (it should on the same GPU).

T1 review (Nov 18), changes on top of the draft:
  - Stretch arm "mixed" (B8.1): exactly train.mixed_lambda * questions_per_step
    questions of every step get a student rollout, the rest their cached
    teacher answer. Which positions is a hash of (seed, step, position), so it
    is reproducible and independent of every RNG. Same question order as all
    other arms. Not one of the four main arms; scripts/20_train.py accepts it.
  - Out-of-memory retry gets a fresh GradScaler (same scale). Without it, an
    OOM after scaler.unscale_() -- e.g. when AdamW allocates its state at the
    first optimizer step, the largest single allocation -- made the retry fail
    with "unscale_() has already been called".
  - The student's train/eval mode is restored in a finally block after rollouts.
  - Vocabulary sizes are compared once at the start. The loss still slices to
    the shared size (briefing pack 2.4 allows it), but a mismatch is now printed
    and recorded in the summary (vocab_sliced_to) instead of happening silently.
"""

import contextlib
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from kd import data
from kd.losses import distillation_loss

ARMS = data.TRAINING_ARMS          # the four main arms
STRETCH_ARMS = ("mixed",)          # B8.1, only with spare quota


# ---------------------------------------------------------------------------
# 1. Randomness and data order (pure Python, tested without a model)
# ---------------------------------------------------------------------------

def question_order(question_ids, seed, epochs=1):
    """The order in which a run visits the pool. Depends ONLY on the seed.

    Not on the arm, and not on how many random numbers anything else drew,
    because it has its own generator. The input is sorted first, so the
    order of the list passed in does not matter either.
    """
    rng = random.Random(f"data-order:{seed}")
    order = []
    for _ in range(epochs):
        ids = sorted(question_ids)
        rng.shuffle(ids)
        order.extend(ids)
    return order


def step_batches(order, questions_per_step, max_steps=None):
    """Split the order into equal steps; drop an incomplete last one; stop at max_steps."""
    n_steps = len(order) // questions_per_step
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)
    return [order[i * questions_per_step:(i + 1) * questions_per_step] for i in range(n_steps)]


def question_ids_hash(question_ids):
    """Short fingerprint of one step's questions; must match across arms (E1 checks)."""
    return hashlib.sha256(",".join(question_ids).encode()).hexdigest()[:16]


def sampling_seed(seed, step):
    """Seed for on-policy sampling at one step, derived from (seed, step) only."""
    return int(hashlib.sha256(f"sampling:{seed}:{step}".encode()).hexdigest()[:8], 16)


def check_mixed_lambda(lmbda, questions_per_step):
    """Number of rollouts per step for the mixed arm; lambda * step size must be whole."""
    if lmbda is None:
        raise ValueError("arm 'mixed' needs train.mixed_lambda in the config (e.g. 0.5)")
    if not 0.0 <= lmbda <= 1.0:
        raise ValueError(f"train.mixed_lambda must be in [0, 1], got {lmbda}")
    k = round(lmbda * questions_per_step)
    if abs(lmbda * questions_per_step - k) > 1e-9:
        raise ValueError(f"train.mixed_lambda * questions_per_step = {lmbda * questions_per_step} "
                         f"is not a whole number; use multiples of 1/{questions_per_step}")
    return k


def onpolicy_positions(seed, step, n, k):
    """Which k of the n positions of a step get a student rollout (mixed arm).

    A hash of (seed, step, position): reproducible, independent of every RNG,
    and different from step to step. Exactly k positions are True.
    """
    ranked = sorted(range(n), key=lambda p: hashlib.sha256(f"mix:{seed}:{step}:{p}".encode()).hexdigest())
    chosen = set(ranked[:k])
    return [p in chosen for p in range(n)]


# ---------------------------------------------------------------------------
# 2. Devices and precision
# ---------------------------------------------------------------------------

def resolve_device(name):
    """'cuda:1' etc. from the config; falls back so the code also runs on CPU (tests)."""
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        index = int(name.split(":")[1]) if ":" in name else 0
        if index >= torch.cuda.device_count():
            print(f"warning: {name} not available, using cuda:0 (select GPU T4 x2 on Kaggle)")
            return torch.device("cuda:0")
    return torch.device(name)


def autocast(device, cfg):
    """fp16 compute on the GPU (a T4 has no fast bf16); plain fp32 on CPU."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=getattr(torch, cfg["train"]["autocast_dtype"]))
    return contextlib.nullcontext()


def model_device(model):
    return next(model.parameters()).device


def pad_token_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else data.end_token_id(tokenizer)


# ---------------------------------------------------------------------------
# 3. Training text for one step
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_rollouts(student, tokenizer, prompts, cfg, seed):
    """On-policy answers: the student samples a reply to each prompt.

    Prompts are left-padded so all replies start at the same column. Returns
    the raw generated ids per prompt; data.build_sequence cuts them at the
    first stop token and drops the padding after it.
    """
    device = model_device(student)
    pad = pad_token_id(tokenizer)
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for i, p in enumerate(prompts):
        input_ids[i, width - len(p):] = torch.tensor(p, device=device)
        attention[i, width - len(p):] = 1

    # Own sampling stream for this step: rollouts depend on (seed, step) only.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    gen = cfg["generation"]
    was_training = student.training
    student.eval()
    try:
        with autocast(device, cfg):
            out = student.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=gen["max_new_tokens"],
                do_sample=True,
                temperature=gen["sampling"]["temperature"],
                top_p=gen["sampling"]["top_p"],
                top_k=gen["sampling"]["top_k"],
                pad_token_id=pad,
                eos_token_id=sorted(data.stop_token_ids(tokenizer)),
                use_cache=True,
            )
    finally:  # also after an out-of-memory error, so a retry starts from the same mode
        student.train(was_training)
    return [row.tolist() for row in out[:, width:]]


def build_step_sequences(arm, examples, tokenizer, cfg, max_seq_len, *,
                         teacher_cache=None, student=None, gen_seed=None, rollout_flags=None):
    """The training sequences of one step. The ONLY place where arms differ in data.

    rollout_flags (mixed arm only): True where the position gets a student
    rollout, False where it gets its cached teacher answer.
    """
    if arm == "onpolicy":
        rollout_flags = [True] * len(examples)
    elif arm == "mixed":
        if rollout_flags is None or len(rollout_flags) != len(examples):
            raise ValueError("arm 'mixed' needs one rollout flag per example")
    else:
        rollout_flags = [False] * len(examples)

    rollouts = iter(())
    if any(rollout_flags):
        prompts = [data.prompt_ids(tokenizer, ex.question)
                   for ex, f in zip(examples, rollout_flags) if f]
        rollouts = iter(generate_rollouts(student, tokenizer, prompts, cfg, gen_seed))

    sequences = []
    for ex, on in zip(examples, rollout_flags):
        if on:
            seq_arm, src = "onpolicy", next(rollouts)
        elif arm in ("teacher", "mixed"):
            seq_arm, src = "teacher", teacher_cache[ex.question_id]
        else:  # human, ref_hardlabel: the cleaned dataset solution
            seq_arm, src = arm, None
        sequences.append(data.build_for_arm(seq_arm, tokenizer, ex, src, max_seq_len=max_seq_len))
    return sequences


# ---------------------------------------------------------------------------
# 4. One optimizer step
# ---------------------------------------------------------------------------

def train_step(sequences, arm, student, teacher, tokenizer, optimizer, scaler, cfg):
    """Forward + backward over the step's sequences, then one optimizer update.

    Returns (loss, n_response_tokens). The loss is the token average over the
    whole step, so it is comparable between arms with long and short answers.
    """
    kind = "hard_ce" if arm == "ref_hardlabel" else cfg["train"]["loss"]
    s_dev = model_device(student)
    pad = pad_token_id(tokenizer)
    n_step_tokens = sum(s.n_response for s in sequences)

    optimizer.zero_grad(set_to_none=True)
    student.train()
    step_loss = 0.0
    for seq in sequences:  # micro-batch of one sequence (see DECISIONS)
        batch = data.collate_sequences([seq], pad, device=s_dev)

        teacher_logits = None
        if kind != "hard_ce":
            t_dev = model_device(teacher)
            with torch.no_grad(), autocast(t_dev, cfg):
                teacher_logits = teacher(
                    input_ids=batch["input_ids"].to(t_dev),
                    attention_mask=batch["attention_mask"].to(t_dev),
                ).logits

        with autocast(s_dev, cfg):
            student_logits = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits

        loss, _ = distillation_loss(
            kind, student_logits, batch["input_ids"], batch["response_mask"],
            teacher_logits=teacher_logits,
            temperature=cfg["train"]["temperature"],
            normalizer=n_step_tokens,
        )
        scaler.scale(loss).backward()
        step_loss += loss.item()
        del batch, teacher_logits, student_logits, loss

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["train"]["grad_clip"])
    scaler.step(optimizer)
    scaler.update()
    return step_loss, n_step_tokens


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fresh_scaler(old, enabled):
    """A new GradScaler that keeps the old one's loss scale.

    After an out-of-memory error between scaler.unscale_() and scaler.update()
    the old scaler still marks the optimizer as unscaled, and the retry would
    fail with "unscale_() has already been called on this optimizer".
    """
    return torch.amp.GradScaler("cuda", init_scale=old.get_scale() if enabled else 65536.0,
                                enabled=enabled)


def check_vocab(student, teacher):
    """Compare the output vocabulary sizes once. Returns the sliced size, or None if equal.

    kd.losses slices both logits to the shared size (briefing pack 2.4 allows
    it); this makes a mismatch visible in the console and in the run summary.
    """
    if teacher is None or not hasattr(student, "get_output_embeddings"):
        return None
    vs = student.get_output_embeddings().weight.shape[0]
    vt = teacher.get_output_embeddings().weight.shape[0]
    if vs == vt:
        return None
    shared = min(vs, vt)
    print(f"[train] warning: student logits have {vs} entries, teacher {vt}; "
          f"the loss uses the first {shared} (recorded as vocab_sliced_to)")
    return shared


# ---------------------------------------------------------------------------
# 5. The loop (models passed in, so tests can use tiny ones)
# ---------------------------------------------------------------------------

def train_loop(cfg, arm, seed, out_dir, student, teacher, tokenizer, pool, teacher_cache=None):
    """Train `student` in place for one arm and seed. Returns a summary dict."""
    if arm not in ARMS + STRETCH_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS + STRETCH_ARMS}")
    if arm != "ref_hardlabel" and teacher is None:
        raise ValueError(f"arm {arm!r} needs the teacher model")
    if arm in ("teacher", "mixed") and teacher_cache is None:
        raise ValueError(f"arm {arm!r} needs the teacher cache (scripts/10_cache_teacher_sequences.py)")

    tcfg = cfg["train"]
    n_rollouts = None
    if arm == "mixed":
        n_rollouts = check_mixed_lambda(tcfg.get("mixed_lambda"), tcfg["questions_per_step"])
    vocab_sliced_to = check_vocab(student, teacher)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_id = {ex.question_id: ex for ex in pool}
    order = question_order(by_id, seed, tcfg["epochs"])
    steps = step_batches(order, tcfg["questions_per_step"], tcfg.get("max_steps"))
    if not steps:
        raise ValueError("no training steps: pool smaller than questions_per_step?")
    max_seq_len = (max(len(data.prompt_ids(tokenizer, ex.question)) for ex in pool)
                   + cfg["generation"]["max_new_tokens"])

    # TODO(T1): AdamW defaults (weight decay 0.01) and a constant LR, as in v1.
    optimizer = torch.optim.AdamW(student.parameters(), lr=tcfg["lr"])
    s_dev = model_device(student)
    use_scaler = s_dev.type == "cuda" and tcfg["autocast_dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"[train] arm={arm} seed={seed} steps={len(steps)} "
          f"questions/step={tcfg['questions_per_step']} max_seq_len={max_seq_len}")

    log_every = tcfg.get("log_every", 10)
    t_start = time.time()
    totals = {"oom_retries": 0, "unfinished": 0, "truncated": 0, "tokens": 0}
    recent_losses = []

    with open(out_dir / "train_log.jsonl", "w", encoding="utf-8") as log_file:
        for step, qids in enumerate(steps, start=1):
            t_step = time.time()
            examples = [by_id[q] for q in qids]

            flags = (onpolicy_positions(seed, step, len(examples), n_rollouts)
                     if arm == "mixed" else None)

            for attempt in range(tcfg["oom_retries"] + 1):
                try:
                    t_gen = time.time()
                    seqs = build_step_sequences(
                        arm, examples, tokenizer, cfg, max_seq_len,
                        teacher_cache=teacher_cache, student=student,
                        gen_seed=sampling_seed(seed, step), rollout_flags=flags,
                    )
                    gen_seconds = time.time() - t_gen
                    loss, n_tokens = train_step(seqs, arm, student, teacher, tokenizer,
                                                optimizer, scaler, cfg)
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    scaler = fresh_scaler(scaler, use_scaler)
                    free_memory()
                    if attempt == tcfg["oom_retries"]:
                        raise RuntimeError(f"step {step}: out of memory {attempt + 1} times. "
                                           "Stopping instead of skipping the step (equal updates).")
                    totals["oom_retries"] += 1
                    print(f"[train] step {step}: out of memory, retrying the same questions")

            unfinished = sum(not s.finished for s in seqs)
            truncated = sum(s.truncated for s in seqs)
            totals["unfinished"] += unfinished
            totals["truncated"] += truncated
            totals["tokens"] += n_tokens
            recent_losses = (recent_losses + [loss])[-10:]

            row = {
                "step": step,
                "loss": round(loss, 6),
                "tokens": n_tokens,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - t_start, 2),
                "step_time_s": round(time.time() - t_step, 2),
                "generation_s": round(gen_seconds, 2) if arm in ("onpolicy", "mixed") else 0.0,
                "question_ids_hash": question_ids_hash(qids),
                "n_sequences": len(seqs),
                "n_rollouts": sum(s.arm == "onpolicy" for s in seqs),
                "n_unfinished": unfinished,
                "n_truncated": truncated,
            }
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            if step == 1 or step % log_every == 0 or step == len(steps):
                print(f"[train] step {step}/{len(steps)} loss={loss:.4f} tokens={n_tokens} "
                      f"step_time={row['step_time_s']}s")

    return {
        "steps": len(steps),
        "final_loss_mean_last_10": round(sum(recent_losses) / len(recent_losses), 6),
        "tokens": totals["tokens"],
        "oom_retries": totals["oom_retries"],
        "unfinished_sequences": totals["unfinished"],
        "truncated_sequences": totals["truncated"],
        "train_seconds": round(time.time() - t_start, 1),
        "mixed_lambda": tcfg.get("mixed_lambda") if arm == "mixed" else None,
        "vocab_sliced_to": vocab_sliced_to,
    }


# ---------------------------------------------------------------------------
# 6. Entry point used by scripts/20_train.py
# ---------------------------------------------------------------------------

def load_models(cfg, arm):
    """Student (trainable), teacher (frozen; not loaded for ref_hardlabel) and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy: tests need no download

    s_dev = resolve_device(cfg["devices"]["student"])
    t_dev = resolve_device(cfg["devices"]["teacher"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["student"])
    student = AutoModelForCausalLM.from_pretrained(
        cfg["student"], dtype=getattr(torch, cfg["train"]["student_dtype"])).to(s_dev)

    teacher = None
    if arm != "ref_hardlabel":
        data.assert_tokenizers_compatible(tokenizer, AutoTokenizer.from_pretrained(cfg["teacher"]))
        t_dtype = getattr(torch, cfg["train"]["autocast_dtype"]) if t_dev.type == "cuda" else torch.float32
        teacher = AutoModelForCausalLM.from_pretrained(cfg["teacher"], dtype=t_dtype).to(t_dev)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)  # the teacher never receives gradients
    return student, teacher, tokenizer


def run_training(cfg, arm, seed, out_dir):
    """Train one arm for one seed and save the student. See the module docstring."""
    out_dir = Path(out_dir)
    student, teacher, tokenizer = load_models(cfg, arm)
    pool = data.load_pool(cfg["data"]["pool_size"], cfg["data"]["pool_seed"])

    teacher_cache = None
    if arm in ("teacher", "mixed"):
        teacher_cache = data.load_teacher_cache(
            cfg["data"]["teacher_sequences"], expected_ids=[ex.question_id for ex in pool])

    summary = train_loop(cfg, arm, seed, out_dir, student, teacher, tokenizer, pool, teacher_cache)

    checkpoint = out_dir / "checkpoint"
    student.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    print(f"[train] saved the student to {checkpoint}")
    return summary
