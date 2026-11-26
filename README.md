# Knowledge-Distillation-and-Model-Compression
AAI2 Practical Project B3

## Statistics and results tables (E1)

`src/kd/stats.py` turns the evaluated runs into the numbers we report: accuracy with 95% Wilson intervals, paired bootstrap comparisons between arms, the spread across seeds, answer length and truncation, and agreement with the teacher. It only reads `results/runs/<run_id>/predictions.jsonl` and `train_log.jsonl`, so it needs no GPU and runs in a few seconds on any laptop.

Run everything from the repository root, after `pip install -r requirements.txt && pip install -e .`

**Quick look at every evaluated run**

```bash
python -m kd.stats
```

```text
run                        n   acc %         95% CI  tokens  trunc %  agree %
...
onpolicy_s1             1319    65.4      62.8-67.9     272      6.5     65.0
...
ref_student0            1319    62.0      59.4-64.6     232      1.7     61.0
ref_teacher             1319    90.3      88.6-91.8     270      1.9        -
teacher_s1              1319    63.5      60.9-66.1     284      8.6     63.3
...
```

**All tables for the note and the slides**

```bash
python -m kd.stats --tables results/tables
```

This rewrites every file in `results/tables/`. Never edit them by hand; rerun the command instead.

| File | Contents |
|---|---|
| `results.md` | Everything below in readable form |
| `results_table.csv` | One row per run (pilots included): accuracy, CI, length, truncation, agreement |
| `arm_summary.csv` | Accuracy per arm: each seed, mean and range |
| `paired_comparisons.csv` | Paired differences between arms per seed, with bootstrap intervals |
| `agreement.csv` | How often each student's final answer matches the teacher's |
| `note_table.md` | The accuracy table that `report/note.qmd` includes |

**Fairness check** (all arms of one seed saw the same questions in the same order)

```bash
python -m kd.stats --fair human_s1 teacher_s1 onpolicy_s1 ref_hardlabel_s1
```

It prints `FAIR` if every run has the same number of steps, no skipped step and the same `question_ids_hash` at every step. Otherwise it prints `NOT FAIR`, lists the problems and exits with code 1.

**Settings.** 95% intervals (z = 1.96), 10,000 bootstrap resamples with seed 42. Runs are paired by `question_id`. A missing final number never counts as agreeing with the teacher. The reasoning behind these choices is in `notes/theory/e1-fair-evaluation-statistics.md`.

**Tests**

```bash
pytest tests/test_stats.py
```
