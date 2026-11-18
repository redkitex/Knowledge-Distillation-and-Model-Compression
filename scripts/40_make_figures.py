#!/usr/bin/env python3
"""Regenerate all figures and tables strictly from the results/ directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from kd.stats import paired_bootstrap_diff_ci, wilson_ci  # type: ignore
except (ImportError, AttributeError):

    def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
        if n <= 0:
            return (0.0, 0.0)
        p = k / n
        denom = 1.0 + (z**2) / n
        center = (p + (z**2) / (2.0 * n)) / denom
        half = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    def paired_bootstrap_diff_ci(
        correct_a: List[int],
        correct_b: List[int],
        n_boot: int = 2000,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        arr_a = np.asarray(correct_a, dtype=float)
        arr_b = np.asarray(correct_b, dtype=float)
        if len(arr_a) == 0 or len(arr_a) != len(arr_b):
            return (0.0, 0.0, 0.0)
        diffs = arr_a - arr_b
        mean_diff = float(np.mean(diffs))
        rng = np.random.default_rng(seed)
        n = len(diffs)
        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means[i] = np.mean(diffs[idx])
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        return (mean_diff, float(ci_low), float(ci_high))


ARMS_ORDER = ["human", "teacher", "onpolicy", "ref_hardlabel"]
ARM_LABELS = {
    "human": "Human (KD)",
    "teacher": "Teacher (KD)",
    "onpolicy": "On-Policy (KD)",
    "ref_hardlabel": "Hard Labels (CE)",
    "onpolicy_rkl": "On-Policy (Rev KL)",
}
ARM_COLORS = {
    "human": "#1f77b4",
    "teacher": "#ff7f0e",
    "onpolicy": "#2ca02c",
    "ref_hardlabel": "#7f7f7f",
    "onpolicy_rkl": "#9467bd",
}
MODEL_PARAMS_B = {
    "ref_teacher": 4.0,
    "ref_student0": 0.6,
    "human": 0.6,
    "teacher": 0.6,
    "onpolicy": 0.6,
    "ref_hardlabel": 0.6,
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_run_log(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    runs: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = (row.get("run_id") or row.get("run") or "").strip()
            if run_id:
                runs[run_id] = row
    return runs


def parse_arm_and_seed(run_id: str) -> Tuple[str, Optional[int]]:
    if run_id in ("ref_teacher", "ref_student0"):
        return (run_id, None)
    if run_id.startswith("pilot_"):
        return (run_id.replace("pilot_", ""), 0)
    if "_s" in run_id:
        arm, seed_str = run_id.rsplit("_s", 1)
        if seed_str.isdigit():
            return (arm, int(seed_str))
    return (run_id, None)


def collect_runs(results_dir: Path, max_new_tokens: int = 512) -> Dict[str, Dict[str, Any]]:
    run_log = read_run_log(results_dir / "run_log.csv")
    runs_dir = results_dir / "runs"
    all_run_ids = set(run_log.keys())
    if runs_dir.exists():
        for p in runs_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                all_run_ids.add(p.name)

    data: Dict[str, Dict[str, Any]] = {}
    for run_id in sorted(all_run_ids):
        arm, seed = parse_arm_and_seed(run_id)
        run_folder = runs_dir / run_id
        preds = read_jsonl(run_folder / "predictions.jsonl")
        train_log = read_jsonl(run_folder / "train_log.jsonl")
        log_row = run_log.get(run_id, {})

        if preds:
            preds.sort(key=lambda x: x.get("question_id", x.get("id", x.get("idx", 0))))

        correct_flags: List[int] = []
        extracted_answers: Dict[Any, Any] = {}
        lengths: List[int] = []
        truncated_flags: List[int] = []

        for idx, p in enumerate(preds):
            qid = p.get("question_id", p.get("id", p.get("idx", idx)))
            is_corr = int(bool(p.get("correct", p.get("is_correct", False))))
            correct_flags.append(is_corr)
            extracted_answers[qid] = p.get(
                "pred_answer", p.get("extracted_answer", p.get("prediction"))
            )
            tok_len = p.get("num_tokens", p.get("completion_tokens", p.get("length")))
            if tok_len is None:
                text = str(p.get("generated_text", p.get("response", p.get("output", ""))))
                tok_len = len(text.split())
            tok_len = int(tok_len)
            lengths.append(tok_len)

            is_trunc = p.get("truncated")
            if is_trunc is None:
                is_trunc = tok_len >= max_new_tokens
            truncated_flags.append(int(bool(is_trunc)))

        n_eval = len(correct_flags)
        k_corr = sum(correct_flags)
        if n_eval > 0:
            acc = k_corr / n_eval
            ci_low, ci_high = wilson_ci(k_corr, n_eval)
        elif "accuracy" in log_row and log_row["accuracy"] not in ("", None):
            acc = float(log_row["accuracy"])
            n_eval = int(log_row.get("n_eval") or 1319)
            k_corr = int(round(acc * n_eval))
            ci_low, ci_high = wilson_ci(k_corr, n_eval)
        else:
            continue

        eval_sec = float(log_row.get("eval_time_s") or log_row.get("eval_seconds") or 0.0)
        step_sec = float(log_row.get("sec_per_step") or log_row.get("step_time_s") or 0.0)
        if step_sec == 0.0 and train_log:
            step_times = [
                float(r["step_time_s"])
                for r in train_log
                if "step_time_s" in r and r["step_time_s"] is not None
            ]
            if step_times:
                step_sec = float(np.mean(step_times))

        data[run_id] = {
            "run_id": run_id,
            "arm": arm,
            "seed": seed,
            "n": n_eval,
            "k": k_corr,
            "accuracy": acc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "correct_flags": correct_flags,
            "extracted_answers": extracted_answers,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "truncation_rate": float(np.mean(truncated_flags)) if truncated_flags else 0.0,
            "train_log": train_log,
            "eval_time_s": eval_sec,
            "step_time_s": step_sec,
        }

    teacher_ans = data.get("ref_teacher", {}).get("extracted_answers", {})
    for info in data.values():
        if teacher_ans and info["extracted_answers"]:
            shared = [q for q in info["extracted_answers"] if q in teacher_ans]
            if shared:
                matches = sum(
                    1
                    for q in shared
                    if info["extracted_answers"][q] is not None
                    and info["extracted_answers"][q] == teacher_ans[q]
                )
                info["teacher_agreement"] = matches / len(shared)
            else:
                info["teacher_agreement"] = None
        else:
            info["teacher_agreement"] = None

    return data


def write_summary_tables(data: Dict[str, Dict[str, Any]], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    by_arm: Dict[str, List[Dict[str, Any]]] = {}

    for info in data.values():
        summary_rows.append(
            {
                "run_id": info["run_id"],
                "arm": info["arm"],
                "seed": info["seed"] if info["seed"] is not None else "-",
                "n": info["n"],
                "accuracy_pct": f"{info['accuracy'] * 100:.2f}",
                "ci_95_pct": f"[{info['ci_low'] * 100:.2f}, {info['ci_high'] * 100:.2f}]",
                "mean_length_tok": f"{info['mean_length']:.1f}",
                "truncation_pct": f"{info['truncation_rate'] * 100:.2f}",
                "teacher_agreement_pct": (
                    f"{info['teacher_agreement'] * 100:.2f}"
                    if info["teacher_agreement"] is not None
                    else "-"
                ),
            }
        )
        if info["seed"] in (1, 2, 3) or info["arm"] in ("ref_teacher", "ref_student0"):
            by_arm.setdefault(info["arm"], []).append(info)

    for arm, runs in by_arm.items():
        if len(runs) > 1:
            accs = [r["accuracy"] * 100 for r in runs]
            lens = [r["mean_length"] for r in runs]
            truncs = [r["truncation_rate"] * 100 for r in runs]
            agrees = [
                r["teacher_agreement"] * 100
                for r in runs
                if r["teacher_agreement"] is not None
            ]
            summary_rows.append(
                {
                    "run_id": f"{arm}_mean",
                    "arm": arm,
                    "seed": "mean (range)",
                    "n": runs[0]["n"],
                    "accuracy_pct": f"{np.mean(accs):.2f} ({np.min(accs):.2f}-{np.max(accs):.2f})",
                    "ci_95_pct": "-",
                    "mean_length_tok": f"{np.mean(lens):.1f}",
                    "truncation_pct": f"{np.mean(truncs):.2f}",
                    "teacher_agreement_pct": f"{np.mean(agrees):.2f}" if agrees else "-",
                }
            )

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with (tables_dir / "accuracy_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        with (tables_dir / "accuracy_summary.md").open("w", encoding="utf-8") as f:
            f.write("| " + " | ".join(fieldnames) + " |\n")
            f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
            for r in summary_rows:
                f.write("| " + " | ".join(str(r[k]) for k in fieldnames) + " |\n")

    comp_rows: List[Dict[str, Any]] = []
    for arm in ["ref_teacher", "ref_student0"] + ARMS_ORDER:
        runs = by_arm.get(arm, []) or [v for v in data.values() if v["arm"] == arm]
        if not runs:
            continue
        accs = [r["accuracy"] * 100 for r in runs]
        step_secs = [r["step_time_s"] for r in runs if r["step_time_s"] > 0]
        eval_secs = [r["eval_time_s"] for r in runs if r["eval_time_s"] > 0]
        comp_rows.append(
            {
                "model_or_arm": arm,
                "params_billions": MODEL_PARAMS_B.get(arm, 0.6),
                "compression_ratio": "1.0x" if arm == "ref_teacher" else "6.7x (0.6B vs 4B)",
                "mean_accuracy_pct": f"{np.mean(accs):.2f}",
                "train_sec_per_step": f"{np.mean(step_secs):.2f}" if step_secs else "-",
                "eval_time_sec": f"{np.mean(eval_secs):.1f}" if eval_secs else "-",
            }
        )

    if comp_rows:
        c_fields = list(comp_rows[0].keys())
        with (tables_dir / "size_speed_accuracy.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=c_fields)
            writer.writeheader()
            writer.writerows(comp_rows)
        with (tables_dir / "size_speed_accuracy.md").open("w", encoding="utf-8") as f:
            f.write("| " + " | ".join(c_fields) + " |\n")
            f.write("| " + " | ".join(["---"] * len(c_fields)) + " |\n")
            for r in comp_rows:
                f.write("| " + " | ".join(str(r[k]) for k in c_fields) + " |\n")


def plot_fig1_accuracy(data: Dict[str, Dict[str, Any]], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    arms_present = [a for a in ARMS_ORDER if any(v["arm"] == a for v in data.values())] or ARMS_ORDER

    for x_idx, arm in enumerate(arms_present):
        arm_runs = [v for v in data.values() if v["arm"] == arm]
        color = ARM_COLORS.get(arm, "#333333")
        offsets = np.linspace(-0.12, 0.12, max(len(arm_runs), 1))
        for r, dx in zip(arm_runs, offsets):
            acc_pct = r["accuracy"] * 100
            low_pct = r["ci_low"] * 100
            high_pct = r["ci_high"] * 100
            ax.errorbar(
                x_idx + dx,
                acc_pct,
                yerr=[[acc_pct - low_pct], [high_pct - acc_pct]],
                fmt="o",
                color=color,
                alpha=0.65,
                capsize=3,
                markersize=6,
            )
        if arm_runs:
            mean_acc = float(np.mean([r["accuracy"] * 100 for r in arm_runs]))
            ax.scatter(
                [x_idx],
                [mean_acc],
                marker="D",
                s=70,
                color=color,
                edgecolor="black",
                zorder=5,
            )

    if "ref_teacher" in data:
        t_acc = data["ref_teacher"]["accuracy"] * 100
        ax.axhline(t_acc, color="#d62728", linestyle="--", linewidth=1.5, label=f"Teacher (4B: {t_acc:.1f}%)")
    if "ref_student0" in data:
        s_acc = data["ref_student0"]["accuracy"] * 100
        ax.axhline(s_acc, color="#111111", linestyle=":", linewidth=1.5, label=f"Untrained Student (0.6B: {s_acc:.1f}%)")

    ax.set_xticks(range(len(arms_present)))
    ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms_present])
    ax.set_ylabel("GSM8K Test Accuracy (%)")
    ax.set_title("GSM8K Accuracy by Training Text Source (All Seeds & 95% Wilson CIs)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_accuracy_per_arm.pdf")
    fig.savefig(figures_dir / "fig1_accuracy_per_arm.png", dpi=200)
    plt.close(fig)


def plot_fig2_paired_diffs(data: Dict[str, Dict[str, Any]], figures_dir: Path) -> None:
    pairs = [
        ("onpolicy", "teacher", "On-Policy - Teacher"),
        ("teacher", "human", "Teacher - Human"),
        ("onpolicy", "human", "On-Policy - Human"),
        ("human", "ref_hardlabel", "Human (Soft KD) - Hard Labels"),
    ]
    results_list: List[Tuple[str, float, float, float]] = []
    for arm_a, arm_b, label in pairs:
        pooled_a: List[int] = []
        pooled_b: List[int] = []
        runs_a = {v["seed"]: v for v in data.values() if v["arm"] == arm_a}
        runs_b = {v["seed"]: v for v in data.values() if v["arm"] == arm_b}
        for s in sorted(set(runs_a.keys()) & set(runs_b.keys())):
            ca, cb = runs_a[s]["correct_flags"], runs_b[s]["correct_flags"]
            if ca and len(ca) == len(cb):
                pooled_a.extend(ca)
                pooled_b.extend(cb)
        if pooled_a:
            diff, low, high = paired_bootstrap_diff_ci(pooled_a, pooled_b)
            results_list.append((label, diff * 100, low * 100, high * 100))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    if results_list:
        y_pos = np.arange(len(results_list))
        for y, (_, diff, low, high) in zip(y_pos, results_list):
            ax.errorbar(diff, y, xerr=[[diff - low], [high - diff]], fmt="o", color="#1f77b4", capsize=4, markersize=7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r[0] for r in results_list])
    else:
        ax.text(0.5, 0.5, "No paired prediction files found yet in results/runs/", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Paired Accuracy Difference (percentage points, 95% Bootstrap CI)")
    ax.set_title("Paired Comparisons Across Training Arms")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_paired_differences.pdf")
    fig.savefig(figures_dir / "fig2_paired_differences.png", dpi=200)
    plt.close(fig)


def plot_fig3_training_loss(data: Dict[str, Dict[str, Any]], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = False
    for info in data.values():
        tlog = info["train_log"]
        if not tlog:
            continue
        steps = [int(r.get("step", i + 1)) for i, r in enumerate(tlog)]
        losses = [float(r.get("loss", r.get("train_loss", 0.0))) for r in tlog]
        ax.plot(steps, losses, color=ARM_COLORS.get(info["arm"], "#333333"), alpha=0.75, linewidth=1.5, label=info["run_id"])
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No train_log.jsonl files found yet in results/runs/", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss (KL / Cross-Entropy)")
    ax.set_title("Training Loss Curves Across Arms and Seeds")
    ax.grid(True, linestyle="--", alpha=0.4)
    if plotted:
        ax.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_training_loss.pdf")
    fig.savefig(figures_dir / "fig3_training_loss.png", dpi=200)
    plt.close(fig)


def plot_fig4_length_truncation(data: Dict[str, Dict[str, Any]], figures_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    arms_order = ["ref_teacher", "ref_student0"] + ARMS_ORDER
    arms_present = [a for a in arms_order if any(v["arm"] == a and v["n"] > 0 for v in data.values())]

    if arms_present:
        x = np.arange(len(arms_present))
        mean_lens = [float(np.mean([r["mean_length"] for r in data.values() if r["arm"] == a and r["n"] > 0])) for a in arms_present]
        mean_truncs = [float(np.mean([r["truncation_rate"] * 100 for r in data.values() if r["arm"] == a and r["n"] > 0])) for a in arms_present]
        colors = [ARM_COLORS.get(a, "#555555") for a in arms_present]

        ax1.bar(x, mean_lens, color=colors, alpha=0.85, edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels([ARM_LABELS.get(a, a) for a in arms_present], rotation=25, ha="right")
        ax1.set_ylabel("Mean Generated Length (tokens)")
        ax1.set_title("Evaluation Answer Length")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        ax2.bar(x, mean_truncs, color=colors, alpha=0.85, edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels([ARM_LABELS.get(a, a) for a in arms_present], rotation=25, ha="right")
        ax2.set_ylabel("Truncated at Cap (%)")
        ax2.set_title("Answer Truncation Rate (512-Token Cap)")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No predictions.jsonl files found yet", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_answer_length_truncation.pdf")
    fig.savefig(figures_dir / "fig4_answer_length_truncation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all B3 figures and tables from results/.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    figures_dir = args.results_dir / "figures"
    tables_dir = args.results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = collect_runs(args.results_dir, max_new_tokens=args.max_new_tokens)
    write_summary_tables(data, tables_dir)
    plot_fig1_accuracy(data, figures_dir)
    plot_fig2_paired_diffs(data, figures_dir)
    plot_fig3_training_loss(data, figures_dir)
    plot_fig4_length_truncation(data, figures_dir)
    print(f"Generated figures in {figures_dir} and tables in {tables_dir} ({len(data)} runs processed).")


if __name__ == "__main__":
    main()
