"""Train one arm for one seed: the run driver. Owner: T3 (training loop: T1).

Usage (from the repo root):
    python scripts/20_train.py --config configs/default.yaml --arm onpolicy --seed 1
    python scripts/20_train.py --config configs/smoke.yaml --arm human --seed 1
    python scripts/20_train.py --config configs/default.yaml --arm human --seed 3 \
        --run-id pilot_human_s3 --account T3

This script does the bookkeeping around a run, so every run is recorded the
same way:
  1. reads the config and records its hash and the git commit;
  2. marks the run as "running" in results/run_log.csv;
  3. calls the training loop in src/kd/train.py (T1);
  4. marks the run "trained" (or "failed") with its end time and GPU-hours.
Evaluation is a separate step: scripts/30_evaluate.py --run <run_id>.

Interface expected from T1 (src/kd/train.py):

    def run_training(cfg: dict, arm: str, seed: int, out_dir: pathlib.Path) -> dict:
        '''Train the student for one arm and seed.

        Must write  out_dir / "train_log.jsonl"   (one row per optimizer step)
        and save    out_dir / "checkpoint"        (save_pretrained of the student
                                                   and its tokenizer; this is the
                                                   path 30_evaluate.py loads).
        Returns a small summary dict, e.g. {"steps": 500, "oom_retries": 0}.
        '''
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ARMS = ("human", "teacher", "onpolicy", "ref_hardlabel")
RUN_LOG = Path("results/run_log.csv")
RUN_LOG_COLUMNS = [
    "run_id", "arm", "seed", "account", "started_utc", "ended_utc", "gpu_hours",
    "status", "git_commit", "config_path", "config_hash", "checkpoint_location",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    """Commit the run was launched from, with a '+dirty' mark if files were changed."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def update_run_log(run_id: str, **fields) -> None:
    """Set fields on one row of results/run_log.csv, adding the row if it is new."""
    rows = []
    if RUN_LOG.exists():
        with open(RUN_LOG, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    for row in rows:
        if row["run_id"] == run_id:
            row.update({k: str(v) for k, v in fields.items()})
            break
    else:
        new = dict.fromkeys(RUN_LOG_COLUMNS, "")
        new.update({"run_id": run_id, **{k: str(v) for k, v in fields.items()}})
        rows.append(new)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one arm for one seed")
    parser.add_argument("--config", required=True, help="configs/default.yaml or configs/smoke.yaml")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id", help="default: <arm>_s<seed>; use pilot_<arm>_s<seed> for the pilot")
    parser.add_argument("--account", default="", help="whose Kaggle account runs this, e.g. T3")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = args.run_id or f"{args.arm}_s{args.seed}"

    # Imported here so that --help works before T1's training loop exists.
    try:
        from kd.train import run_training
    except ImportError as err:
        sys.exit(f"src/kd/train.py does not provide run_training yet ({err}). "
                 "See the interface at the top of this script.")

    out_dir = Path("results/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "run_id": run_id,
        "arm": args.arm,
        "seed": args.seed,
        "account": args.account,
        "started_utc": utc_now(),
        "git_commit": git_commit(),
        "config_path": config_path.as_posix(),
        "config_hash": file_hash(config_path),
    }
    update_run_log(run_id, status="running", **{k: v for k, v in info.items() if k != "run_id"})
    print(f"Run {run_id}: arm={args.arm} seed={args.seed} config={info['config_path']} "
          f"({info['config_hash']}) commit={info['git_commit']}")

    t0 = time.time()
    status, summary = "failed", {}
    try:
        summary = run_training(cfg, arm=args.arm, seed=args.seed, out_dir=out_dir) or {}
        status = "trained"
    finally:
        gpu_hours = round((time.time() - t0) / 3600, 3)  # wall-clock hours of the session
        checkpoint = out_dir / "checkpoint"
        update_run_log(
            run_id,
            status=status,
            ended_utc=utc_now(),
            gpu_hours=gpu_hours,
            checkpoint_location=checkpoint.as_posix() if checkpoint.exists() else "",
        )
        info.update(status=status, ended_utc=utc_now(), gpu_hours=gpu_hours, summary=summary)
        # A copy of the run-log row next to the outputs, so a run folder
        # downloaded from Kaggle explains itself even if run_log.csv is not merged yet.
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"Run {run_id}: {status} after {gpu_hours} h")

    print(f"Next: python scripts/30_evaluate.py --config {info['config_path']} --run {run_id}")


if __name__ == "__main__":
    main()
