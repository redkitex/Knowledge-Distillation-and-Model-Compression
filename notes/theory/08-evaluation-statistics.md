# Fair evaluation: confidence intervals, paired comparisons, seeds and fidelity

Owner: E1 · Reviewer: T1 · Feeds: slides 8 (evaluation method) and 9 (results); note sections 4 (evaluation protocol) and 5 (results)

This note covers topics 08 (evaluation statistics) and 09 (fidelity vs accuracy) of the old skeleton, since the Nov 15 guide gives one topic per person. The code is in `src/kd/stats.py`.

## Definitions

- **Accuracy.** The share of the 1,319 GSM8K test questions where the last number in the answer equals the gold answer. One run gives one accuracy.
- **Accuracy is an estimate.** The test set is only a sample of all the grade-school problems that could have been written. Another sample of 1,319 questions would give a slightly different number, so every accuracy needs an error bar.
- **Confidence interval (CI).** A range that would contain the true accuracy in 95% of repeated experiments. We report 95% intervals everywhere.
- **Wilson interval.** The CI we use for a single accuracy. Unlike the textbook `p ± 1.96·sqrt(p(1-p)/n)`, it never goes below 0 or above 1 and still works near 0% or 100%.
- **Paired comparison.** All runs answer the same questions. So instead of comparing two separate CIs, we look at the difference question by question. Hard questions are hard for both runs, and pairing removes that shared noise.
- **Paired bootstrap.** We resample the questions 10,000 times with replacement, keeping both runs' answers to each question together, and recompute the difference each time. The 2.5% and 97.5% percentiles give the CI of the difference. If it does not contain 0, the difference is significant.
- **Seed variation.** Training the same arm with another seed (data order, sampling, dropout) gives a different student. The bootstrap only covers the choice of test questions, not this, so we report every seed separately.
- **Fidelity.** How closely the student behaves like its teacher, as opposed to how often it is right (*generalisation* or accuracy). We measure it as **agreement**: the share of questions where the student's final number equals the teacher's.

## Formula

Wilson interval for `k` correct answers out of `n`, with `p = k/n` and `z = 1.96`:

```
          p + z²/(2n)  ±  z · sqrt( p(1-p)/n + z²/(4n²) )
CI  =  -----------------------------------------------------
                         1 + z²/n
```

Paired difference between runs A and B on the same `n` questions, with `a_i, b_i ∈ {0, 1}`:

```
d = (1/n) Σ (a_i − b_i)
```

Bootstrap: draw `n` question indices with replacement, compute `d*` on them, repeat 10,000 times (seed 42), take the 2.5th and 97.5th percentiles of `d*`.

Only the questions where the two runs disagree move `d`. That is why `compare_arms` also reports "only A correct" and "only B correct".

Across seeds: per-seed values, their mean and their range (min to max). With three seeds a standard deviation would not mean much.

Agreement, with `ŷ` the extracted final number:

```
agreement = (1/n) Σ 1[ ŷ_student,i = ŷ_teacher,i ]
```

A missing number never counts as agreeing. We also report agreement only on the questions the teacher got wrong. Agreeing there means the student copied a teacher mistake, which accuracy cannot show.

## What we rely on (Established)

- [Established] An eval score is a sample estimate and should be reported with its standard error or CI. Comparisons between models on the same questions should use paired differences, because that removes the variance coming from question difficulty (`miller2024errorbars`).
- [Established] Miller's other advice either does not apply or is already covered. Clustered errors are only needed when questions share a context, which GSM8K questions do not. Resampling answers per question is only needed with sampling, and our evaluation uses greedy decoding (`miller2024errorbars`).
- [Established] The Wilson interval keeps good coverage for small `n` and for proportions near 0 or 1, where the normal approximation fails (`wilson1927probable`).
- [Established] The bootstrap estimates the sampling distribution of a statistic by resampling the observed data with replacement (`efron1993bootstrap`).
- [Established] Students often reach low fidelity with their teacher even when they could match it, and higher fidelity does not always mean higher accuracy. So fidelity and accuracy should be measured separately (`stanton2021does`). That study used image classifiers and measured top-1 agreement and the KL between predictive distributions. For text we use agreement on the final answer, which is the closest equivalent we can afford.

## What we found (Ours)

- [Ours] Headroom test, 300 test questions: teacher 83.0% (Wilson 95% CI 78.3 to 86.8), student 56.7% (51.0 to 62.2), paired gap 26.3 points (bootstrap CI 21.0 to 31.7). `stats.py` gives the same Wilson intervals (checked in `tests/test_stats.py`). Source: `notebooks/archive/headroom-test.ipynb`.
- [Ours] Planning number: on the full test set (n = 1,319), a run at about 60% accuracy has a Wilson half-width of about 2.6 points. Differences between arms smaller than that need the paired bootstrap to be detected at all. Computed with `wilson_ci`, not measured.
- [Ours] TODO after Nov 24: accuracy per arm and seed, paired differences, agreement with the teacher. Fill in from `results/tables/`, never by hand.

## Open questions

- `evaluate.py` does not write a `question_id` yet, so runs are paired by question text. This works because every run uses the same test set, but an id would be safer (to discuss with E2).
- Do we also want agreement between arms (e.g. onpolicy vs teacher), not only with the teacher?
- If one seed behaves very differently from the other two, we report it as it is. We should agree on how to discuss that before the results come in, not after.
