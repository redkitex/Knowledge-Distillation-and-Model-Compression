"""Statistics and summary metrics for the evaluation (owner: E1, reviewer: E2).

Everything in this file works on the predictions.jsonl files that
scripts/30_evaluate.py writes into results/runs/<run_id>/, so it needs no GPU
and can be rerun at any time.

What is in here:
  - wilson_ci: confidence interval for one accuracy
  - paired_bootstrap_diff_ci: confidence interval for the gap between two runs
    that answered the same questions
  - seed_summary: mean and range of a number measured on several seeds
  - summarise_run: accuracy, answer length, truncation for one run
  - agreement: how often the student's final answer matches the teacher's
  - accuracy_by_length: accuracy for short vs long answers
  - results_table / compare_arms: the tables used in the note and the slides
  - fairness_check: the pilot's "It's fair" check on the training logs

From the repo root:
    python -m kd.stats                                   # table of all evaluated runs
    python -m kd.stats --out results/tables/results_table.csv
    python -m kd.stats --fair pilot_human pilot_teacher pilot_onpolicy pilot_ref_hardlabel
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

Z_95 = 1.96          # two-sided 95% interval
N_BOOT = 10_000      # bootstrap resamples, as in the briefing pack
BOOT_SEED = 42       # same seed as the headroom test, so its numbers can be reproduced
SAME_NUMBER_TOL = 1e-6


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------
# evaluate.py currently writes "predicted" and "response_tokens", while the
# briefing pack (section 2.6) names them "extracted_answer" and
# "n_response_tokens". The helpers below accept both, so nothing here breaks
# if the field names change later.

def question_key(rec):
    """The id used to pair the same question across runs."""
    if rec.get("question_id") is not None:
        return rec["question_id"]
    return rec["question"]


def final_answer(rec):
    if "extracted_answer" in rec:
        return rec["extracted_answer"]
    return rec.get("predicted")


def n_tokens(rec):
    if "n_response_tokens" in rec:
        return rec["n_response_tokens"]
    return rec.get("response_tokens")


def read_jsonl(path):
    """Read a .jsonl file (one JSON object per line)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                # usually a Kaggle session that died while writing the file
                raise ValueError(f"{path}, line {line_no}: not valid JSON ({e.msg}). "
                                 "Was the run interrupted?") from e
    return records


def load_predictions(path):
    """Read one predictions.jsonl file."""
    return read_jsonl(path)


def load_runs(runs_dir="results/runs"):
    """Return {run_id: records} for every run folder that has predictions.

    Folders without predictions.jsonl, or with an empty one (a run that is
    still going), are skipped with a message instead of breaking the table.
    """
    runs = {}
    for pred_file in sorted(Path(runs_dir).glob("*/predictions.jsonl")):
        records = load_predictions(pred_file)
        if records:
            runs[pred_file.parent.name] = records
        else:
            print(f"skipping {pred_file.parent.name}: predictions.jsonl is empty")
    return runs


def parse_run_id(run_id):
    """'onpolicy_s2' -> ('onpolicy', 2); 'ref_teacher' -> ('ref_teacher', None)."""
    m = re.fullmatch(r"(.+)_s(\d+)", run_id)
    if m:
        return m.group(1), int(m.group(2))
    return run_id, None


def align(records_a, records_b, shared_only=False):
    """Put two runs in the same question order so they can be compared.

    By default both runs must contain exactly the same questions, otherwise
    we stop instead of guessing. With shared_only=True only the questions
    that appear in both runs are kept (in the order of run A). That is
    needed when a pilot or smoke run on 300 questions is compared with a
    reference evaluated on the full test set. Callers report how many
    questions were actually used.
    """
    by_key_b = {question_key(r): r for r in records_b}
    keys_a = [question_key(r) for r in records_a]
    if len(set(keys_a)) != len(keys_a) or len(by_key_b) != len(records_b):
        raise ValueError("a run contains the same question twice")
    if shared_only:
        kept = [(r, by_key_b[k]) for r, k in zip(records_a, keys_a) if k in by_key_b]
        return [a for a, _ in kept], [b for _, b in kept]
    if set(keys_a) != set(by_key_b):
        raise ValueError("the two runs do not contain the same questions")
    return list(records_a), [by_key_b[k] for k in keys_a]


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def wilson_ci(successes, n, z=Z_95):
    """Wilson score interval for a proportion (Wilson 1927).

    Better than the usual p +- z*sqrt(p(1-p)/n): it never leaves [0, 1] and
    it still behaves when the accuracy is close to 0 or 1.
    Returns (lower, upper). For n == 0 there is nothing to estimate, so it
    returns (0.0, 0.0), as the headroom notebook did.
    """
    if n == 0:
        return (0.0, 0.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be between 0 and n, got {successes} of {n}")
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    # max/min only remove floating point noise at the edges (e.g. -1e-17)
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def paired_bootstrap_diff_ci(correct_a, correct_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Accuracy of A minus accuracy of B, with a 95% paired bootstrap interval.

    correct_a and correct_b are 0/1 lists for the SAME questions in the SAME
    order. Each resample draws questions with replacement and keeps both
    answers to a question together. Easy and hard questions then affect
    both runs at once, which gives a much tighter interval than comparing
    two separate Wilson intervals (Miller 2024, "analyse paired differences").

    Ported from the headroom notebook. The loop and the seed are kept the
    same on purpose, so it gives exactly the same numbers.
    Returns (observed_diff, (lower, upper)).
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("correct_a and correct_b must have the same length (paired)")
    if len(a) == 0:
        raise ValueError("cannot compare two empty runs")

    rng = np.random.default_rng(seed)
    n = len(a)
    observed = a.mean() - b.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(observed), (float(lower), float(upper))


def seed_summary(values):
    """Per-seed values plus their mean and range.

    With only three seeds a standard deviation says very little, so we
    report every seed and the min-max range instead.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("no values to summarise")
    return {
        "n_seeds": len(vals),
        "per_seed": vals,
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def summarise_run(records):
    """Accuracy (with Wilson CI), answer length and truncation for one run."""
    n = len(records)
    if n == 0:
        raise ValueError("run has no predictions")
    correct = sum(bool(r["correct"]) for r in records)
    lengths = [n_tokens(r) for r in records if n_tokens(r) is not None]
    low, high = wilson_ci(correct, n)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "ci_low": low,
        "ci_high": high,
        "mean_tokens": float(np.mean(lengths)) if lengths else None,
        "median_tokens": float(np.median(lengths)) if lengths else None,
        "truncated_rate": sum(bool(r.get("truncated")) for r in records) / n,
        # answers where no number could be found at all
        "no_answer_rate": sum(final_answer(r) is None for r in records) / n,
    }


def same_number(x, y):
    """True if both answers exist and are the same number."""
    if x is None or y is None:
        return False
    return abs(float(x) - float(y)) < SAME_NUMBER_TOL


def agreement(student_records, teacher_records):
    """How often the student's final answer equals the teacher's.

    A simple fidelity measure in the sense of Stanton et al. (2021): accuracy
    tells us whether the student is right, agreement tells us whether it
    behaves like its teacher. The rate on the questions the teacher got
    WRONG is the interesting part: agreeing there means the student copied
    the teacher's mistake, which plain accuracy cannot show.

    If either model gave no number, the pair counts as "not agreeing".
    Only the questions both runs answered are used (e.g. a 300-question
    pilot run against the full-set teacher); "n" says how many that was.
    """
    student, teacher = align(student_records, teacher_records, shared_only=True)
    agree = [same_number(final_answer(s), final_answer(t)) for s, t in zip(student, teacher)]
    teacher_wrong = [i for i, t in enumerate(teacher) if not t["correct"]]

    n = len(agree)
    if n == 0:
        return {"n": 0, "agreement": None, "ci_low": None, "ci_high": None,
                "n_teacher_wrong": 0, "agreement_when_teacher_wrong": None}
    k = sum(agree)
    low, high = wilson_ci(k, n)
    out = {"n": n, "agreement": k / n, "ci_low": low, "ci_high": high,
           "n_teacher_wrong": len(teacher_wrong)}
    if teacher_wrong:
        k_wrong = sum(agree[i] for i in teacher_wrong)
        out["agreement_when_teacher_wrong"] = k_wrong / len(teacher_wrong)
    else:
        out["agreement_when_teacher_wrong"] = None
    return out


def accuracy_by_length(records, edges=(0, 128, 256, 384, 512)):
    """Accuracy for answers grouped by length in tokens.

    edges=(0, 128, 256) gives the bins [0, 128), [128, 256) and [256, inf).
    Used to check whether an arm is better only because it writes longer or
    shorter answers. Long bins are dominated by truncated answers, so read
    them together with the truncation rate.
    """
    bins = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        in_bin = [r for r in records
                  if n_tokens(r) is not None and n_tokens(r) >= lo
                  and (hi is None or n_tokens(r) < hi)]
        k = sum(bool(r["correct"]) for r in in_bin)
        bins.append({
            "from_tokens": lo,
            "to_tokens": hi,
            "n": len(in_bin),
            "accuracy": k / len(in_bin) if in_bin else None,
        })
    return bins


# ---------------------------------------------------------------------------
# Tables for the note and the slides
# ---------------------------------------------------------------------------

def results_table(runs, teacher_run="ref_teacher"):
    """One row per run: accuracy with CI, length, truncation, agreement.

    runs is the dict returned by load_runs. Agreement is only filled in if
    the teacher's own predictions (ref_teacher) are there.
    """
    teacher = runs.get(teacher_run)
    rows = []
    for run_id in sorted(runs):
        arm, seed = parse_run_id(run_id)
        row = {"run_id": run_id, "arm": arm, "seed": seed}
        row.update(summarise_run(runs[run_id]))
        if teacher is not None and run_id != teacher_run:
            agr = agreement(runs[run_id], teacher)
            row["agreement_with_teacher"] = agr["agreement"]
            row["agreement_n"] = agr["n"]
        else:
            row["agreement_with_teacher"] = None
            row["agreement_n"] = None
        rows.append(row)
    return rows


def arm_table(rows, metric="accuracy"):
    """Collapse the per-run rows into one row per arm (all seeds together)."""
    by_arm = {}
    for row in rows:
        if row.get(metric) is not None:
            by_arm.setdefault(row["arm"], []).append(row)
    table = []
    for arm, arm_rows in sorted(by_arm.items()):
        arm_rows = sorted(arm_rows, key=lambda r: (r["seed"] is None, r["seed"]))
        s = seed_summary(r[metric] for r in arm_rows)
        table.append({"arm": arm, "metric": metric,
                      "seeds": [r["seed"] for r in arm_rows], **s})
    return table


def find_run(runs, arm, seed):
    """Run id for an arm and seed: 'onpolicy_s2' for trained arms, or the
    plain name ('ref_student0', 'ref_teacher') for references, which are
    evaluated once and have no seed. None if neither exists."""
    if f"{arm}_s{seed}" in runs:
        return f"{arm}_s{seed}"
    if arm in runs:
        return arm
    return None


def compare_arms(runs, arm_a, arm_b, seeds=(1, 2, 3)):
    """Paired difference arm_a - arm_b, seed by seed, plus mean and range.

    Examples: compare_arms(runs, "onpolicy", "teacher") or, against a
    reference, compare_arms(runs, "onpolicy", "ref_student0"); a reference
    has no seed, so every seed of arm_a is compared with the same run.
    Each seed gets its own bootstrap interval. The bootstrap only covers the
    choice of test questions, not training noise, which is why the per-seed
    spread is reported next to it (never only the best seed).

    Raises ValueError if not a single seed could be compared, so a typo in
    an arm name does not silently return an empty result.
    """
    per_seed = []
    done = set()
    for seed in seeds:
        id_a, id_b = find_run(runs, arm_a, seed), find_run(runs, arm_b, seed)
        if id_a is None or id_b is None or id_a == id_b:
            continue
        if (id_a, id_b) in done:
            # two references (no seeds): compare them once, not once per seed
            continue
        done.add((id_a, id_b))
        if id_a == arm_a and id_b == arm_b:
            seed = None
        a, b = align(runs[id_a], runs[id_b], shared_only=True)
        if not a:
            continue
        diff, (low, high) = paired_bootstrap_diff_ci(
            [r["correct"] for r in a], [r["correct"] for r in b])
        per_seed.append({
            "seed": seed, "run_a": id_a, "run_b": id_b, "n": len(a),
            "diff": diff, "ci_low": low, "ci_high": high,
            # questions only one of the two runs got right
            "only_a_correct": sum(x["correct"] and not y["correct"] for x, y in zip(a, b)),
            "only_b_correct": sum(y["correct"] and not x["correct"] for x, y in zip(a, b)),
            "significant": low > 0 or high < 0,
        })
    if not per_seed:
        raise ValueError(f"no run pairs found for {arm_a!r} vs {arm_b!r}; "
                         f"available runs: {', '.join(sorted(runs))}")
    summary = seed_summary(r["diff"] for r in per_seed)
    return {"arm_a": arm_a, "arm_b": arm_b, "per_seed": per_seed, "across_seeds": summary}


# ---------------------------------------------------------------------------
# "It's fair" check (pilot gate)
# ---------------------------------------------------------------------------

def fairness_check(runs_dir, run_ids):
    """Check that several training runs saw the same questions in the same order.

    Reads results/runs/<run_id>/train_log.jsonl for each run (one row per
    optimizer step with "step" and "question_ids_hash", as T1 logs them) and
    checks three things:
      1. no step is missing or repeated in any run (no skipped updates),
      2. all runs have the same number of steps,
      3. at every step, all runs have the same question_ids_hash.
    Tokens per step are allowed to differ (answers have different lengths
    per arm); they are only reported.

    Example for the pilot:
        fairness_check("results/runs", ["pilot_human", "pilot_teacher",
                                         "pilot_onpolicy", "pilot_ref_hardlabel"])
    Returns {"fair": bool, "problems": [...], "runs": {run_id: {...}}}.
    """
    if len(run_ids) < 2:
        raise ValueError("give at least two runs to compare")

    problems = []
    hashes = {}     # run_id -> {step: hash}
    info = {}
    for run_id in run_ids:
        path = Path(runs_dir) / run_id / "train_log.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        rows = read_jsonl(path)
        if not rows:
            problems.append(f"{run_id}: train_log.jsonl is empty")
            hashes[run_id] = {}
            info[run_id] = {"n_steps": 0}
            continue

        steps = [r["step"] for r in rows]
        expected = list(range(steps[0], steps[0] + len(steps)))
        if steps != expected:
            problems.append(f"{run_id}: steps are not consecutive "
                            f"(missing or repeated steps, e.g. a skipped update)")
        if any("question_ids_hash" not in r for r in rows):
            problems.append(f"{run_id}: some steps have no question_ids_hash")

        hashes[run_id] = {r["step"]: r.get("question_ids_hash") for r in rows}
        info[run_id] = {
            "n_steps": len(rows),
            "first_step": steps[0],
            "last_step": steps[-1],
            "total_tokens": sum(r.get("tokens") or 0 for r in rows),
            "elapsed_s": rows[-1].get("elapsed_s"),
        }

    n_steps = {run_id: i["n_steps"] for run_id, i in info.items()}
    if len(set(n_steps.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in n_steps.items())
        problems.append(f"different number of steps: {detail}")

    all_steps = sorted(set().union(*(h.keys() for h in hashes.values())))
    bad_steps = [s for s in all_steps
                 if len({hashes[r].get(s) for r in run_ids}) > 1]
    if bad_steps:
        problems.append(f"question order differs at {len(bad_steps)} step(s), "
                        f"first at step {bad_steps[0]}")

    return {"fair": not problems, "problems": problems, "runs": info,
            "mismatched_steps": bad_steps}


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x, pct=True):
    if x is None:
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.0f}"


def main():
    parser = argparse.ArgumentParser(description="Summarise every evaluated run.")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--out", help="optional CSV path for the per-run table")
    parser.add_argument("--fair", nargs="+", metavar="RUN_ID",
                        help="instead of the table, check that these training runs "
                             "saw the same questions in the same order")
    args = parser.parse_args()

    if args.fair:
        report = fairness_check(args.runs_dir, args.fair)
        for run_id, i in report["runs"].items():
            print(f"{run_id:<24} steps={i['n_steps']:<6} tokens={i.get('total_tokens', '-')}")
        if report["fair"]:
            print("\nFAIR: same number of steps and the same questions at every step.")
        else:
            print("\nNOT FAIR:")
            for p in report["problems"]:
                print(f"  - {p}")
            raise SystemExit(1)
        return

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No predictions.jsonl found under {args.runs_dir}/")
        return
    rows = results_table(runs)
    print(f"{'run':<22}{'n':>6}{'acc %':>8}{'95% CI':>15}{'tokens':>8}{'trunc %':>9}{'agree %':>9}")
    for r in rows:
        ci = f"{_fmt(r['ci_low'])}-{_fmt(r['ci_high'])}"
        print(f"{r['run_id']:<22}{r['n']:>6}{_fmt(r['accuracy']):>8}{ci:>15}"
              f"{_fmt(r['mean_tokens'], pct=False):>8}{_fmt(r['truncated_rate']):>9}"
              f"{_fmt(r['agreement_with_teacher']):>9}")
    if args.out:
        write_csv(rows, args.out)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
