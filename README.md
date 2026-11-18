# Project B3: Knowledge Distillation and Model Compression

When a small model learns from a bigger one, does it matter whose text it practises on? We train **Qwen3-0.6B** (student) to match the full next-token probabilities of **Qwen3-4B** (teacher) on **GSM8K** maths problems using forward KL divergence ($T=1$), changing only where the training text comes from:
- **`human`**: The dataset's own human-written solutions.
- **`teacher`**: Answers written by the teacher (`Qwen3-4B`), generated once and saved.
- **`onpolicy`**: Answers the student (`Qwen3-0.6B`) writes itself, regenerated as it learns.
- **`ref_hardlabel`**: Standard cross-entropy on the same `human` solutions instead of teacher probabilities.
- **References (no training)**: `ref_teacher` (teacher accuracy ceiling) and `ref_student0` (untrained student floor).

## Repository Structure

```text
README.md                        # What it is, how to run it, where data comes from
requirements.txt                 # Pinned versions, tested on Kaggle
pyproject.toml                   # Lets every script and notebook `import kd`
configs/
  default.yaml                   # All settings for the real runs (locked Nov 20)
  smoke.yaml                     # Tiny version: all arms in about 15 minutes
src/kd/                          # Reusable code, six files
  data.py                        # GSM8K, training pool, prompts, text per arm
  answer_extraction.py           # Grading: the last number in an answer
  losses.py                      # Forward and reverse KL, hard-label loss
  train.py                       # Seeding, on-policy answers, training loop
  evaluate.py                    # Batched greedy evaluation
  stats.py                       # Confidence intervals, paired bootstrap
scripts/                         # Run in this order
  10_cache_teacher_sequences.py  # Writes data/teacher_answers.jsonl (~3 MB)
  20_train.py                    # Reads config, writes results/runs/<run_id>/
  30_evaluate.py                 # Evaluates a run and writes predictions.jsonl
  40_make_figures.py             # Reads only results/ to make figures and tables
notebooks/
  kaggle_run.ipynb               # Clones the repo, runs scripts 20 and 30
  archive/                       # v1 and headroom notebooks, unchanged
tests/                           # pytest for the risky parts, five files
results/
  run_log.csv                    # One row per run
  runs/<run_id>/                 # predictions.jsonl, train_log.jsonl
  figures/                       # Made only by script 40
  tables/                        # Made only by script 40
report/
  note.qmd                       # The scientific note (3 to 4 pages)
  references.bib                 # Checked references
  slides/                        # Slide source
notes/                           # Personal notes; notes/theory/ has one file each
Sources/README.md                # Literature table
```

## Where the Data and Models Come From

GSM8K and both models download automatically from Hugging Face:
- **Dataset**: `openai/gsm8k` (fixed pool of 2,000 training questions and all 1,319 test questions for evaluation) is downloaded from Hugging Face via `datasets` in `src/kd/data.py`.
- **Models**: `Qwen/Qwen3-4B` (teacher) and `Qwen/Qwen3-0.6B` (student) are downloaded from Hugging Face Hub via `transformers`.
- **Teacher Answers**: `data/teacher_answers.jsonl` (~3 MB) is committed to the repository so nobody has to regenerate it.

## How to Run

Set `CFG=configs/smoke.yaml` for a 15-minute check of the whole pipeline, or `CFG=configs/default.yaml` for the full experiment runs:

```bash
pip install -r requirements.txt && pip install -e .

CFG=configs/default.yaml
python scripts/10_cache_teacher_sequences.py --config $CFG
python scripts/20_train.py --config $CFG --arm onpolicy --seed 1
python scripts/30_evaluate.py --config $CFG --run onpolicy_s1
python scripts/40_make_figures.py
```

Run names follow `<arm>_s<seed>` for training runs (`human_s1`, `ref_hardlabel_s3`), `ref_teacher` and `ref_student0` for the references, `pilot_<arm>` for the pilot, and `onpolicy_rkl_s<seed>` for the stretch run.

To run the test suite:

```bash
pytest
```
