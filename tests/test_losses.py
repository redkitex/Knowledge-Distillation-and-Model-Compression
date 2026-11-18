"""Tests for kd.losses (owner: T1). Run: pytest tests/test_losses.py

These are the checks from the plan (section 5): KL is 0 for identical logits,
it matches torch's own kl_div, and positions outside the response never
change the loss. All offline, random tensors only.
"""

import pytest
import torch
import torch.nn.functional as F

from kd.losses import distillation_loss, forward_kl, hard_label_ce, response_positions, reverse_kl

B, L, V = 2, 7, 11


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(B, L, V, generator=g)
    teacher = torch.randn(B, L, V, generator=g)
    input_ids = torch.randint(0, V, (B, L), generator=g)
    # prompt = first 3 tokens, then response; second row has 2 padding tokens at the end
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1],
                         [0, 0, 0, 1, 1, 0, 0]])
    return student, teacher, input_ids, mask


def test_kl_is_zero_for_identical_logits():
    s, _, ids, mask = make_batch()
    for kind in ("forward_kl", "reverse_kl"):
        loss, _ = distillation_loss(kind, s, ids, mask, teacher_logits=s.clone())
        assert abs(loss.item()) < 1e-6


def test_forward_kl_matches_torch_kl_div():
    s, t, _, _ = make_batch()
    s, t = s.reshape(-1, V), t.reshape(-1, V)
    ours = forward_kl(s, t)
    torch_kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                        reduction="none", log_target=True).sum(-1)
    assert torch.allclose(ours, torch_kl, atol=1e-6)


def test_reverse_kl_swaps_the_roles():
    s, t, _, _ = make_batch()
    s, t = s.reshape(-1, V), t.reshape(-1, V)
    assert torch.allclose(reverse_kl(s, t), forward_kl(t, s), atol=1e-6)


def test_hard_label_ce_matches_cross_entropy():
    s, _, ids, _ = make_batch()
    s, ids = s.reshape(-1, V), ids.reshape(-1)
    assert torch.allclose(hard_label_ce(s, ids), F.cross_entropy(s, ids, reduction="none"))


def test_alignment_uses_logits_t_for_token_t_plus_1():
    s, _, ids, mask = make_batch()
    logits, targets = response_positions(s, ids, mask)
    # row 0: targets are positions 3..6, predicted by logits at 2..5
    assert torch.equal(targets[:4], ids[0, 3:7])
    assert torch.equal(logits[0], s[0, 2])
    assert logits.shape[0] == int(mask[:, 1:].sum())


@pytest.mark.parametrize("kind", ["forward_kl", "reverse_kl", "hard_ce"])
def test_masked_positions_do_not_change_the_loss(kind):
    s, t, ids, mask = make_batch()
    loss, n = distillation_loss(kind, s, ids, mask, teacher_logits=t)
    # Scramble everything that is not used: logits that predict prompt or padding tokens
    keep = mask[:, 1:].bool()
    s2, t2 = s.clone(), t.clone()
    s2[:, :-1][~keep] = 100 * torch.randn_like(s2[:, :-1][~keep])
    t2[:, :-1][~keep] = 100 * torch.randn_like(t2[:, :-1][~keep])
    s2[:, -1], t2[:, -1] = 99.0, -99.0  # the last position predicts nothing
    loss2, n2 = distillation_loss(kind, s2, ids, mask, teacher_logits=t2)
    assert n == n2 == int(mask[:, 1:].sum())
    assert torch.allclose(loss, loss2, atol=1e-6)


def test_normalizer_gives_a_token_average_over_the_whole_step():
    s, t, ids, mask = make_batch()
    full, n_full = distillation_loss("forward_kl", s, ids, mask, teacher_logits=t)
    # the same batch as two micro-batches of one sequence, normalised by the step's tokens
    parts = [distillation_loss("forward_kl", s[i:i + 1], ids[i:i + 1], mask[i:i + 1],
                               teacher_logits=t[i:i + 1], normalizer=n_full)[0] for i in range(B)]
    assert torch.allclose(sum(parts), full, atol=1e-6)


def test_teacher_with_a_larger_vocabulary_is_sliced():
    s, t, ids, mask = make_batch()
    t_big = torch.cat([t, torch.full((B, L, 3), -1e4)], dim=-1)
    a, _ = distillation_loss("forward_kl", s, ids, mask, teacher_logits=t)
    b, _ = distillation_loss("forward_kl", s, ids, mask, teacher_logits=t_big)
    assert torch.allclose(a, b, atol=1e-5)


def test_temperature_scaling():
    s, t, ids, mask = make_batch()
    at_1, _ = distillation_loss("forward_kl", s, ids, mask, teacher_logits=t, temperature=1.0)
    at_2, _ = distillation_loss("forward_kl", s, ids, mask, teacher_logits=t, temperature=2.0)
    assert at_1.item() > 0 and at_2.item() > 0
    assert not torch.isclose(at_1, at_2)


def test_errors():
    s, t, ids, mask = make_batch()
    with pytest.raises(ValueError, match="unknown loss"):
        distillation_loss("jsd", s, ids, mask, teacher_logits=t)
    with pytest.raises(ValueError, match="needs teacher_logits"):
        distillation_loss("forward_kl", s, ids, mask)
    with pytest.raises(ValueError, match="no response tokens"):
        distillation_loss("hard_ce", s, ids, torch.zeros_like(mask))
