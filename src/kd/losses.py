"""Distillation losses. Owner: T1. Reviewer: E1.

STATUS: DRAFT for T1 to review. It runs and is tested (tests/test_losses.py),
but T1 owns it: read it, change what you disagree with, and keep the tests
passing. Lines marked "TODO(T1)" are open points.

Ported from token_kl / distill_loss in notebooks/archive/
distillation-on-gsm8k-with-qwens_v1.ipynb, with three changes:
  1. one function serves every arm (kind = forward_kl | reverse_kl | hard_ce);
  2. the loss is computed only on response tokens, picked with the mask
     from data.py (prompt and padding never count);
  3. the caller can pass `normalizer`, so a step made of several small
     batches is averaged over ALL its response tokens (briefing pack 2.4).

Shapes: logits are (batch, seq_len, vocab); input_ids and response_mask are
(batch, seq_len), exactly as data.collate_sequences returns them.

Alignment (the most common silent bug): the logits at position t predict the
token at position t+1. So logits[:, :-1] are scored against input_ids[:, 1:]
and the mask that says "this target is a response token" is response_mask[:, 1:].
"""

import torch
import torch.nn.functional as F

LOSS_KINDS = ("forward_kl", "reverse_kl", "hard_ce")


def response_positions(logits, input_ids, response_mask):
    """Keep only the positions whose NEXT token is a response token.

    Returns (logits_at_response, target_ids): shapes (n_tokens, vocab) and (n_tokens,).
    """
    mask = response_mask[:, 1:].bool()
    return logits[:, :-1][mask], input_ids[:, 1:][mask]


def forward_kl(student_logits, teacher_logits, temperature=1.0):
    """KL(teacher || student) per token, the classic distillation target (Hinton).

    Mass-covering: the student is pushed to give probability to everything
    the teacher finds likely. Inputs are (n_tokens, vocab); returns (n_tokens,).
    """
    log_p_s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)


def reverse_kl(student_logits, teacher_logits, temperature=1.0):
    """KL(student || teacher) per token (MiniLLM, Thinking Machines). Stretch run only.

    Mode-seeking: the student is punished for putting mass where the teacher
    puts little. Inputs are (n_tokens, vocab); returns (n_tokens,).
    """
    log_p_s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return (log_p_s.exp() * (log_p_s - log_p_t)).sum(dim=-1)


def hard_label_ce(student_logits, target_ids):
    """Plain cross-entropy on the correct next token: the ref_hardlabel arm.

    No teacher involved. Inputs (n_tokens, vocab) and (n_tokens,); returns (n_tokens,).
    """
    return F.cross_entropy(student_logits.float(), target_ids, reduction="none")


def distillation_loss(kind, student_logits, input_ids, response_mask,
                      teacher_logits=None, temperature=1.0, normalizer=None):
    """The loss for one batch. Returns (loss, n_response_tokens).

    kind:        "forward_kl" | "reverse_kl" | "hard_ce" (ref_hardlabel)
    normalizer:  divide the summed per-token loss by this number. Pass the
                 number of response tokens in the whole optimizer step, so a
                 step split into micro-batches gives a token average over the
                 step. None = average over this batch's own tokens.
    """
    if kind not in LOSS_KINDS:
        raise ValueError(f"unknown loss {kind!r}; expected one of {LOSS_KINDS}")

    s_logits, targets = response_positions(student_logits, input_ids, response_mask)
    n_tokens = int(targets.numel())
    if n_tokens == 0:
        raise ValueError("batch has no response tokens")

    if kind == "hard_ce":
        per_token = hard_label_ce(s_logits, targets)
    else:
        if teacher_logits is None:
            raise ValueError(f"{kind} needs teacher_logits")
        t_logits, _ = response_positions(teacher_logits.to(s_logits.device), input_ids, response_mask)
        # Qwen3 checkpoints can pad the vocabulary to different sizes; compare
        # only the shared part. Both models use the same tokenizer (data.py checks).
        vocab = min(s_logits.shape[-1], t_logits.shape[-1])
        s_logits, t_logits = s_logits[:, :vocab], t_logits[:, :vocab]
        kl = forward_kl if kind == "forward_kl" else reverse_kl
        # T^2 keeps gradient size comparable across temperatures (Hinton et al. 2015).
        per_token = kl(s_logits, t_logits, temperature) * temperature ** 2

    denom = normalizer if normalizer is not None else n_tokens
    return per_token.sum() / denom, n_tokens
