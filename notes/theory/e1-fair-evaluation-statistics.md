# Theory Note: e1-fair-evaluation-statistics

**Fair evaluation: confidence intervals, paired comparisons, seeds and fidelity**

Owner: E1 · Reviewer: T1 · Feeds: slides 8 (evaluation method) and 9 (results); note sections 4 (evaluation protocol) and 5 (results)

This note covers the old topics 08 (evaluation statistics) and 09 (fidelity vs accuracy), merged into one note per person as in the Nov 15 guide. The code is in `src/kd/stats.py`.

## Definitions

- **Accuracy.** The share of the 1,319 GSM8K test questions where the last number in the answer equals the gold answer. One run gives one accuracy.
- **Accuracy is an estimate.** The test set is only a sample of all the grade-school problems that could have been written. Another sample of 1,319 questions would give a slightly different number, so every accuracy needs an error bar.
- **Confidence interval (CI).** A range computed by a method that, if we repeated the experiment with new samples of questions many times, would contain the true accuracy 95% of the time. We report 95% intervals everywhere.
- **Wilson interval.** The CI we use for a single accuracy. Unlike the textbook `p ± 1.96·sqrt(p(1-p)/n)`, it never goes below 0 or above 1 and still works near 0% or 100%.
- **Paired comparison.** All runs answer the same questions. So instead of comparing two separate CIs, we look at the difference question by question. Hard questions are hard for both runs, and pairing removes that shared noise.
- **Paired bootstrap.** We resample the questions 10,000 times with replacement, keeping both runs' answers to each question together, and recompute the difference each time. The 2.5% and 97.5% percentiles give the CI of the difference. If it does not contain 0, the difference is significant.
- **Seed variation.** Training the same arm with another seed (a different order of the training questions and, for onpolicy, different sampled answers) gives a different student. The bootstrap only covers the choice of test questions, not this, so we report every seed separately.
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
- [Established] The score interval for a proportion was introduced by Wilson (`wilson1927probable`). The usual normal-approximation (Wald) interval can have coverage well below 95% even for large `n` and near 0 or 1, while the Wilson interval stays close to the nominal level and is recommended instead (`brown2001interval`).
- [Established] The bootstrap estimates the sampling distribution of a statistic by resampling the observed data with replacement (`efron1993bootstrap`).
- [Established] Students often reach low fidelity with their teacher even when they could match it, and higher fidelity does not always mean higher accuracy. So fidelity and accuracy should be measured separately (`stanton2021does`). That study used image classifiers and measured top-1 agreement and the KL between predictive distributions. For text we use agreement on the final answer, which is the closest equivalent we can afford.

## What we found (Ours)

- [Ours] Headroom test, 300 test questions: teacher 83.0% (Wilson 95% CI 78.3 to 86.8), student 56.7% (51.0 to 62.2), paired gap 26.3 points (bootstrap CI 21.0 to 31.7). `stats.py` gives the same Wilson intervals (checked in `tests/test_stats.py`). Source: `notebooks/archive/headroom-test.ipynb`.
- [Ours] Planning number: on the full test set (n = 1,319), a run at about 60% accuracy has a Wilson half-width of about 2.6 points. Looking at whether two separate intervals overlap would therefore hide differences of a few points. The paired bootstrap is more sensitive, because it removes the variation that comes from question difficulty. Computed with `wilson_ci`, not measured.
- [Ours] Final runs, full test set: the teacher reaches 90.3% (CI 88.6 to 91.8) and the untrained student 62.0% (59.4 to 64.6), so the real gap is 28.3 points. The 300-question headroom test gave lower numbers for both (83.0% vs 56.7%), on a different subset and with a 384-token cap. Source: `results/tables/results.md`.
- [Ours] Pairing is what makes the comparisons readable. With separate intervals of about ±2.6 points, on-policy (65.4 to 69.1%) and teacher text (60.2 to 63.5%) would look close; the paired bootstrap shows the gap clearly in seeds 2 and 3 (+8.1 [+5.7, +10.5] and +6.7 [+4.2, +9.2]). Source: `results/tables/paired_comparisons.csv`.
- [Ours] Seeds matter as much as the test sample. The same arm moves by 2.7 to 3.9 points between seeds, more than the half-width of one interval. Seed 1 alone would have shown no significant difference between on-policy and teacher text (+1.9 [-0.6, +4.4]); seeds 2 and 3 alone would have suggested a clear one. Reporting all seeds is what keeps this honest. Source: `results/tables/results.md`.
- [Ours] Fidelity and accuracy moved together here. On-policy students agree with the teacher's final answer on 65.0 to 68.8% of questions (61.0% before training), and on the 128 questions the teacher gets wrong they repeat its answer 17 to 23% of the time (11% before training). Hard-label students agree least (39.3 to 41.3%). Source: `results/tables/agreement.csv`.
- [Ours] The fairness check passes for all three seeds: in every seed the four arms ran 500 steps with the same question order and the same config hash. Source: `python -m kd.stats --fair <arm>_s<seed> ...` on the `train_log.jsonl` files.

## Open questions

- The bootstrap intervals cover the choice of test questions, not training noise. With three seeds we can show the spread but not estimate it well; more seeds would be the first thing to add with more compute.
- The learning rate was chosen on the on-policy arm and on the test set. A held-out slice of the training data would have been the cleaner choice; we name it as a limitation.
- Agreement between arms (for example on-policy vs teacher text) is not reported; it could show whether different sources teach the same answers.
