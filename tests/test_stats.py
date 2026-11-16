"""Tests for kd.stats. Run: pytest tests/test_stats.py

No GPU and no internet needed. Expected values are either computed by hand
(closed-form Wilson cases) or taken from the headroom notebook output.
"""

import json
import math

import pytest

from kd import stats
from kd.stats import (
    accuracy_by_length,
    agreement,
    align,
    arm_table,
    compare_arms,
    load_runs,
    paired_bootstrap_diff_ci,
    parse_run_id,
    results_table,
    seed_summary,
    summarise_run,
    wilson_ci,
)


def rec(q, correct, answer=None, tokens=100, truncated=False):
    """A fake predictions.jsonl row in the format evaluate.py writes today."""
    return {"question": q, "response": "...", "predicted": answer, "expected": 1.0,
            "correct": correct, "response_tokens": tokens, "truncated": truncated}


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------

def test_wilson_matches_headroom_notebook():
    # headroom test: teacher 249/300 = 83.0% (78.3-86.8), student 170/300 = 56.7% (51.0-62.2)
    low, high = wilson_ci(249, 300)
    assert round(100 * low, 1) == 78.3 and round(100 * high, 1) == 86.8
    low, high = wilson_ci(170, 300)
    assert round(100 * low, 1) == 51.0 and round(100 * high, 1) == 62.2


def test_wilson_closed_form_edges():
    # with 0 successes the upper bound is z^2 / (n + z^2), with n successes the lower one is n / (n + z^2)
    z2 = 1.96**2
    low, high = wilson_ci(0, 10)
    assert low == 0.0
    assert high == pytest.approx(z2 / (10 + z2))
    low, high = wilson_ci(10, 10)
    assert low == pytest.approx(10 / (10 + z2))
    assert high == 1.0


def test_wilson_half_by_hand():
    # p = 0.5, n = 100: centre stays 0.5, half-width = z*sqrt(0.0025 + z^2/40000) / (1 + z^2/100)
    z = 1.96
    half = z * math.sqrt(0.25 / 100 + z**2 / 40000) / (1 + z**2 / 100)
    low, high = wilson_ci(50, 100)
    assert low == pytest.approx(0.5 - half)
    assert high == pytest.approx(0.5 + half)
    assert round(low, 4) == 0.4038 and round(high, 4) == 0.5962


def test_wilson_bad_input():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_identical_runs_give_zero():
    a = [1, 0, 1, 1, 0, 1]
    diff, (low, high) = paired_bootstrap_diff_ci(a, a, n_boot=500)
    assert diff == 0 and low == 0 and high == 0


def test_bootstrap_always_right_vs_always_wrong():
    diff, (low, high) = paired_bootstrap_diff_ci([1] * 20, [0] * 20, n_boot=500)
    assert diff == 1 and low == 1 and high == 1


def test_bootstrap_is_reproducible_and_contains_the_observed_gap():
    a = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1] * 10
    b = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0] * 10
    first = paired_bootstrap_diff_ci(a, b, n_boot=2000)
    second = paired_bootstrap_diff_ci(a, b, n_boot=2000)
    assert first == second                      # fixed seed -> same interval every time
    diff, (low, high) = first
    assert diff == pytest.approx(0.3)
    assert low < diff < high
    assert low > 0                              # 30 extra correct, 0 extra wrong: clearly positive


def test_bootstrap_needs_paired_input():
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci([1, 0], [1, 0, 1])
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci([], [])


# --------------------------------------------------------------------------
# Seeds, run ids, alignment
# --------------------------------------------------------------------------

def test_seed_summary():
    s = seed_summary([0.50, 0.60, 0.70])
    assert s["n_seeds"] == 3
    assert s["mean"] == pytest.approx(0.60)
    assert s["min"] == 0.50 and s["max"] == 0.70
    assert s["per_seed"] == [0.50, 0.60, 0.70]
    with pytest.raises(ValueError):
        seed_summary([])


@pytest.mark.parametrize("run_id, expected", [
    ("onpolicy_s2", ("onpolicy", 2)),
    ("ref_hardlabel_s3", ("ref_hardlabel", 3)),
    ("pilot_teacher_s3", ("pilot_teacher", 3)),
    ("ref_teacher", ("ref_teacher", None)),
    ("ref_student0", ("ref_student0", None)),
])
def test_parse_run_id(run_id, expected):
    assert parse_run_id(run_id) == expected


def test_align_reorders_by_question():
    a = [rec("q1", True), rec("q2", False), rec("q3", True)]
    b = [rec("q3", False), rec("q1", True), rec("q2", True)]
    a2, b2 = align(a, b)
    assert [r["question"] for r in b2] == ["q1", "q2", "q3"]


def test_align_prefers_question_id():
    a = [{"question_id": 7, "question": "x", "correct": True}]
    b = [{"question_id": 7, "question": "x (reformatted)", "correct": False}]
    a2, b2 = align(a, b)
    assert b2[0]["correct"] is False


def test_align_refuses_different_questions():
    with pytest.raises(ValueError):
        align([rec("q1", True)], [rec("q2", True)])
    with pytest.raises(ValueError):
        align([rec("q1", True), rec("q1", False)], [rec("q1", True), rec("q2", True)])


# --------------------------------------------------------------------------
# Per-run metrics
# --------------------------------------------------------------------------

def test_summarise_run_by_hand():
    records = [
        rec("q1", True, 5, tokens=100),
        rec("q2", True, 3, tokens=200),
        rec("q3", False, None, tokens=512, truncated=True),
        rec("q4", False, 8, tokens=300),
    ]
    s = summarise_run(records)
    assert s["n"] == 4 and s["correct"] == 2
    assert s["accuracy"] == 0.5
    assert (s["ci_low"], s["ci_high"]) == wilson_ci(2, 4)
    assert s["mean_tokens"] == pytest.approx(278.0)      # (100+200+512+300)/4
    assert s["median_tokens"] == pytest.approx(250.0)    # (200+300)/2
    assert s["truncated_rate"] == 0.25
    assert s["no_answer_rate"] == 0.25


def test_summarise_run_accepts_briefing_pack_field_names():
    records = [{"question_id": 0, "extracted_answer": 4.0, "correct": True,
                "n_response_tokens": 50, "truncated": False}]
    s = summarise_run(records)
    assert s["mean_tokens"] == 50 and s["no_answer_rate"] == 0


def test_agreement_by_hand():
    teacher = [rec("q1", True, 5), rec("q2", False, 7), rec("q3", False, 9), rec("q4", True, 2)]
    student = [rec("q1", True, 5),      # agrees, both right
               rec("q2", False, 7),     # agrees on a wrong answer -> copied mistake
               rec("q3", False, 1),     # both wrong, different numbers
               rec("q4", False, None)]  # no number -> does not agree
    out = agreement(student, teacher)
    assert out["n"] == 4
    assert out["agreement"] == 0.5
    assert out["n_teacher_wrong"] == 2
    assert out["agreement_when_teacher_wrong"] == 0.5


def test_agreement_treats_12_and_12_point_0_as_equal():
    out = agreement([rec("q", True, 12)], [rec("q", True, 12.0)])
    assert out["agreement"] == 1.0
    assert out["agreement_when_teacher_wrong"] is None


def test_accuracy_by_length_bins():
    records = [rec("a", True, tokens=50), rec("b", False, tokens=120),
               rec("c", True, tokens=130), rec("d", False, tokens=600)]
    bins = accuracy_by_length(records, edges=(0, 128, 512))
    assert [(b["from_tokens"], b["to_tokens"], b["n"]) for b in bins] == [
        (0, 128, 2), (128, 512, 1), (512, None, 1)]
    assert [b["accuracy"] for b in bins] == [0.5, 1.0, 0.0]


# --------------------------------------------------------------------------
# Tables, using real files on disk
# --------------------------------------------------------------------------

def write_run(runs_dir, run_id, records):
    folder = runs_dir / run_id
    folder.mkdir(parents=True)
    with open(folder / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def fake_results(tmp_path):
    qs = [f"q{i}" for i in range(40)]
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "ref_teacher", [rec(q, True, 1) for q in qs])
    for seed in (1, 2, 3):
        # onpolicy gets the first 30 + seed right, teacher arm the first 20
        write_run(runs_dir, f"onpolicy_s{seed}",
                  [rec(q, i < 30 + seed, 1 if i < 30 + seed else 0) for i, q in enumerate(qs)])
        write_run(runs_dir, f"teacher_s{seed}",
                  [rec(q, i < 20, 1 if i < 20 else 0) for i, q in enumerate(qs)])
    (runs_dir / "empty_run").mkdir()     # a folder without predictions is ignored
    return runs_dir


def test_load_runs_and_results_table(fake_results):
    runs = load_runs(fake_results)
    assert sorted(runs) == ["onpolicy_s1", "onpolicy_s2", "onpolicy_s3",
                            "ref_teacher", "teacher_s1", "teacher_s2", "teacher_s3"]
    rows = {r["run_id"]: r for r in results_table(runs)}
    assert rows["onpolicy_s2"]["accuracy"] == 32 / 40
    assert rows["onpolicy_s2"]["agreement_with_teacher"] == 32 / 40
    assert rows["ref_teacher"]["agreement_with_teacher"] is None

    arms = {r["arm"]: r for r in arm_table(list(rows.values()))}
    assert arms["onpolicy"]["seeds"] == [1, 2, 3]
    assert arms["onpolicy"]["mean"] == pytest.approx(32 / 40)
    assert arms["onpolicy"]["min"] == 31 / 40 and arms["onpolicy"]["max"] == 33 / 40
    assert arms["ref_teacher"]["seeds"] == [None]


def test_compare_arms(fake_results):
    runs = load_runs(fake_results)
    out = compare_arms(runs, "onpolicy", "teacher")
    assert [r["seed"] for r in out["per_seed"]] == [1, 2, 3]
    first = out["per_seed"][0]
    assert first["diff"] == pytest.approx(11 / 40)
    assert first["only_a_correct"] == 11 and first["only_b_correct"] == 0
    assert first["significant"]
    assert out["across_seeds"]["mean"] == pytest.approx(12 / 40)


def test_compare_arms_skips_missing_seeds(fake_results):
    runs = load_runs(fake_results)
    del runs["teacher_s3"]
    out = compare_arms(runs, "onpolicy", "teacher")
    assert [r["seed"] for r in out["per_seed"]] == [1, 2]


def test_compare_arms_raises_instead_of_returning_nothing(fake_results):
    runs = load_runs(fake_results)
    with pytest.raises(ValueError, match="human"):
        compare_arms(runs, "human", "teacher")


def test_compare_arms_against_an_unseeded_reference(fake_results):
    # "did distillation beat the untrained student?": every seed vs the same ref_student0
    qs = [f"q{i}" for i in range(40)]
    write_run(fake_results, "ref_student0", [rec(q, i < 25, 1 if i < 25 else 0) for i, q in enumerate(qs)])
    runs = load_runs(fake_results)
    out = compare_arms(runs, "onpolicy", "ref_student0")
    assert [(r["run_a"], r["run_b"]) for r in out["per_seed"]] == [
        ("onpolicy_s1", "ref_student0"), ("onpolicy_s2", "ref_student0"), ("onpolicy_s3", "ref_student0")]
    assert [r["diff"] for r in out["per_seed"]] == pytest.approx([6 / 40, 7 / 40, 8 / 40])
    assert all(r["n"] == 40 for r in out["per_seed"])


def test_compare_two_references_only_once(fake_results):
    qs = [f"q{i}" for i in range(40)]
    write_run(fake_results, "ref_student0", [rec(q, i < 25, 1 if i < 25 else 0) for i, q in enumerate(qs)])
    runs = load_runs(fake_results)
    out = compare_arms(runs, "ref_teacher", "ref_student0")
    assert len(out["per_seed"]) == 1
    assert out["per_seed"][0]["seed"] is None
    assert out["per_seed"][0]["diff"] == pytest.approx(15 / 40)
    assert out["across_seeds"]["n_seeds"] == 1


def test_empty_predictions_file_is_skipped(fake_results, capsys):
    (fake_results / "human_s1").mkdir()
    (fake_results / "human_s1" / "predictions.jsonl").write_text("", encoding="utf-8")
    runs = load_runs(fake_results)
    assert "human_s1" not in runs
    assert "skipping human_s1" in capsys.readouterr().out


def test_half_written_file_gives_a_clear_error(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(rec("q1", True, 1)) + "\n" + '{"question": "q2", "corr', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        stats.load_predictions(path)


def test_pilot_run_on_fewer_questions_does_not_crash(fake_results):
    # pilot / smoke runs only cover part of the test set, ref_teacher covers all of it
    qs = [f"q{i}" for i in range(10)]
    write_run(fake_results, "pilot_onpolicy_s3", [rec(q, i < 5, 1 if i < 5 else 0) for i, q in enumerate(qs)])
    runs = load_runs(fake_results)
    rows = {r["run_id"]: r for r in results_table(runs)}
    pilot = rows["pilot_onpolicy_s3"]
    assert pilot["n"] == 10
    assert pilot["agreement_n"] == 10                 # only the 10 shared questions
    assert pilot["agreement_with_teacher"] == 0.5
    assert rows["onpolicy_s1"]["agreement_n"] == 40


def test_agreement_with_no_shared_questions():
    out = agreement([rec("a", True, 1)], [rec("b", True, 1)])
    assert out["n"] == 0 and out["agreement"] is None


def test_align_shared_only_keeps_common_questions_in_order():
    a = [rec("q3", True), rec("q1", False), rec("q9", True)]
    b = [rec("q1", True), rec("q2", True), rec("q3", False)]
    a2, b2 = align(a, b, shared_only=True)
    assert [r["question"] for r in a2] == ["q3", "q1"]
    assert [r["question"] for r in b2] == ["q3", "q1"]
    assert [r["correct"] for r in b2] == [False, True]


def test_command_line_prints_and_writes_csv(fake_results, tmp_path, capsys, monkeypatch):
    out_csv = tmp_path / "tables" / "results.csv"
    monkeypatch.setattr("sys.argv", ["stats", "--runs-dir", str(fake_results), "--out", str(out_csv)])
    stats.main()
    printed = capsys.readouterr().out
    assert "onpolicy_s1" in printed
    assert out_csv.exists()
    assert out_csv.read_text(encoding="utf-8").splitlines()[0].startswith("run_id,arm,seed")
