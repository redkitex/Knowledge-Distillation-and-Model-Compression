# Theory Note: t2-soft-targets-temperature

**Topic 1: Soft targets, dark knowledge, temperature, hard vs soft labels**
Owner: T2 · Reviewer: E2 · Feeds: slides "KD basics" (8 min); note section "Theory"

---

## Definitions

- **Logits** $z_i$ : the raw scores a network outputs, one per class (for an LLM, one per vocabulary token)
- **Hard label (hard target):** the single correct output as a one-hot vector (for an LLM it is the actual next token of the training text) 
- **Soft targets:** the teacher's full probability distribution, used as the student's training target (Hinton et al., 2015, §1). For an LLM this is the teacher's next-token distribution over the whole vocabulary at every position of the sequence. 
- **Temperature $T$:** divides the logits before the softmax (Hinton et al., 2015, eq. 1). $T = 1$ is the ordinary softmax. Raising $T$ flattens the distribution; $T \to 0$ collapses it onto the argmax (one-hot); $T \to \infty$ makes it uniform. The ranking of outputs never changes, only how far apart their probabilities are. 
- **Dark knowledge:** the information carried by the teacher's probabilities for the *wrong* outputs, i.e. which wrong answers it considers close. Hinton's example: an image of a 2 might get probability $10^{-6}$ of being a 3 and $10^{-9}$ of being a 7. These ratios encode similarity between classes but are so small they barely move a $T = 1$ cross-entropy (Hinton et al., 2015, §1). The phrase itself does not appear in the 2015 paper, Furlanello et al. (2018, §3.2) use it for this idea.
- **Transfer set:** the inputs on which the teacher's targets are computed (Hinton et al., 2015, §2).  In our project the transfer set is what differs between arms: human solutions (`human`), cached teacher answers (`teacher`), the student's own rollouts (`onpolicy`).
- **Distillation temperature vs sampling temperature:** the same formula used for two different jobs. The distillation temperature is applied inside the loss and changes the *targets*. A sampling temperature is used when generating text and changes *which tokens get picked*. Our project has three: loss $T = 1$; student rollouts sampled at 0.7 (top-p 0.8, top-k 20); teacher answers generated greedily (the $T \to 0$ limit).

## Formula

**1. Temperature softmax** (Hinton et al., 2015, eq. 1) 

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}, \qquad \log\frac{p_i}{p_j} = \frac{z_i - z_j}{T}$$

The second form shows the mechanism: higher $T$ shrinks every log-ratio, so small probabilities move closer to large ones.

*Toy example* (logits $z = [5, 3, 2, 0]$, illustrative only, computed with numpy):

| $T$ | probabilities | entropy (nats) | $p_{\text{top}} / p_{\text{last}}$ |
|---|---|---|---|
| 0.5 | 0.980, 0.018, 0.002, 0.000 | 0.11 | 22,026 |
| 1 | 0.839, 0.114, 0.042, 0.006 | 0.56 | 148 |
| 2 | 0.598, 0.220, 0.133, 0.049 | 1.06 | 12 |
| 5 | 0.387, 0.259, 0.212, 0.142 | 1.32 | 3 |

(The maximum entropy for 4 outputs is $\ln 4 = 1.39$.)

**2. Hinton's distillation objective** (§2; the $\alpha$ notation is ours) 

$$\mathcal{L} = \alpha\, T^2\, H\!\big(p_T^{(T)},\, p_S^{(T)}\big) + (1-\alpha)\, H\!\big(y,\, p_S^{(1)}\big)$$

This is a weighted average of cross-entropy with the soft targets (teacher and student both at temperature $T$) and cross-entropy with the hard label $y$ at $T = 1$. Hinton et al. recommend a considerably lower weight on the hard term; their speech experiments set it to 0.5. The soft-target gradients shrink as $1/T^2$, so they are multiplied by $T^2$ to keep their contribution stable when $T$ changes.

**3. Gradient and the link to logit matching** (§2.1) 

$$\frac{\partial C}{\partial z_i} = \frac{1}{T}\,(q_i - p_i) \;\xrightarrow{\;T \to \infty\;}\; \frac{1}{N T^2}\,(z_i - v_i)$$

Here $q$ is the student, $p$ the teacher, $v$ the teacher's logits, and $N$ the number of classes; the limit assumes zero-mean logits. At high temperature distillation is the same as matching logits, so logit matching (Buciluă et al., 2006; Ba & Caruana, 2014) is a special case. At lower temperatures it pays less attention to very negative logits.

**4. Our loss** (briefing pack §2.4; `losses.forward_kl`) **[O, design]**

$$\mathcal{L} = \frac{1}{|M|}\sum_{t \in M}\sum_{v} p_T(v \mid c_t)\,\log\frac{p_T(v \mid c_t)}{p_S(v \mid c_t)}$$

This is per-token forward KL over the full vocabulary, averaged over response positions $M$. It uses $T = 1$ and has no hard-label term ($\alpha = 1$ in eq. 2). The code still multiplies by $T^2$, which equals 1 here. It is the word-level KD of Kim & Rush (2016) applied to every response token. **[E for the lineage]**

**5. Why KL, cross-entropy and hard labels are one family** (standard identities, checked numerically) 

$$\mathrm{KL}(p_T \,\|\, p_S) = H(p_T, p_S) - H(p_T), \qquad \frac{\partial\,\mathrm{KL}}{\partial z^S} = p_S - p_T \;\;(T = 1)$$

The teacher's entropy $H(p_T)$ does not depend on the student, so KL and soft cross-entropy give identical gradients. If $p_T$ is replaced by a one-hot vector on the true token $y$, the KL becomes $-\log p_S(y)$, which is exactly hard-label cross-entropy. So `human` and `ref_hardlabel` differ in one thing only: on the same text with the same updates, the target at each position is either the teacher's full distribution or a one-hot vector on the human's token. **[O, design]**

Furlanello et al. (2018, §3.2) split the gradient $p_S - p_T$ into a true-class part, $(q_* - p_*)$, which acts as the hard-label gradient re-weighted by the teacher's confidence, and a dark-knowledge part, $\sum_{i \ne *}(q_i - p_i)$. 

**6. Why $T = 1$ here** (goes into the limitations) 

- It is the default in the briefing pack and the locked config.
- At $T = 1$ the loss is the KL between the distributions the models actually use, in nats per token. The same function therefore doubles as E1's KL (fidelity) metric.
- $T$ is a hyperparameter that Hinton et al. tuned per task (20 on MNIST, with intermediate values best for small students). Sweeping it would multiply our 12 core runs, and we had no budget for that.
- *Hypothesis, untested:* LLM next-token distributions at $T = 1$ are already far from one-hot at many positions, so soft targets carry non-argmax information without heating. This could be checked by logging the teacher's per-token entropy.
- Consequence: we can say nothing about the effect of temperature itself.

## What we rely on (Established)

1. **Soft targets carry more information per example than hard labels.** When they have high entropy, they also vary less in gradient between examples, so a student can often learn from far less data and at a higher learning rate (Hinton et al., 2015, §1).
2. **Raising $T$ exposes the information in small probabilities.** At $T = 1$ the tiny probabilities of wrong classes hardly affect the cross-entropy; raising $T$ makes them count (Hinton et al., 2015, §1–2).
3. **Distillation works on MNIST.** The large regularised net makes 67 test errors and the small net 146; the small net distilled at $T = 20$ makes 74. When every 3 was removed from the transfer set, the distilled student still got 98.6% of test 3s right after its bias for 3 was corrected. Knowledge about a class it never saw arrived through soft targets alone (Hinton et al., 2015, §3).
4. **Distillation works on speech.** On frame accuracy / WER: baseline 58.9% / 10.9%, 10-model ensemble 61.1% / 10.7%, distilled single model 60.8% / 10.7%. More than 80% of the ensemble's accuracy gain transfers (Hinton et al., 2015, §4, Table 1).
5. **Soft targets act as a regulariser.** This is the closest result to our `human` vs `ref_hardlabel` comparison. With only 3% of the speech data, hard targets overfit (44.5% frame accuracy, even with early stopping). Soft targets reached 57.0% without early stopping, against 58.9% for hard targets on the full data (Hinton et al., 2015, §6, Table 5).
6. **Both parts of the KD gradient matter.** On CIFAR-100 Furlanello et al. tested two ablations. CWTM keeps only the confidence weighting: it weights hard-label loss by the teacher's top probability. DKPP shuffles the teacher's non-argmax probabilities, which breaks the class-similarity structure. DKPP beat the teacher on all four DenseNets but both ablations were worse than full distillation, so the non-argmax structure adds something beyond confidence weighting (Furlanello et al., 2018, §3.2, Table 2).
7. **Distillation without softening can help.** Born-Again Networks are students the same size as their teacher, trained on the teacher's outputs with no temperature described (effectively $T = 1$). They outperform their teachers, e.g. DenseNet-90-60 on CIFAR-10 goes from 3.81% to 3.5% error. On Penn Treebank the LSTM language models only worked when the teacher loss was combined with the label loss (Furlanello et al., 2018).
8. **Token-level KD for sequence models** applies the soft-target loss at every position of the sequence (Kim & Rush, 2016). Sequence-level KD is covered in note 02.

**Limit of this evidence:** all of the results above are for classification: MNIST, speech frames, CIFAR, and word-level PTB. Carrying them over to 4B → 0.6B distillation over a vocabulary of about 150k tokens, with long reasoning chains, is an assumption our experiment tests only partly.

## What we found (Ours)

*(To be edited once all runs are evaluated.)* Planned entries:

- `human` vs `ref_hardlabel`: the accuracy difference with a paired 95% CI over 3 seeds. This is our direct measure of what the teacher's full distribution is worth over hard labels on the same text.
- Optional: the teacher's mean per-token entropy on the training pool, which tests the $T = 1$ hypothesis in Formula §6.

---

### References

- Ba, J. & Caruana, R. (2014). Do Deep Nets Really Need to be Deep? *NeurIPS 2014*.
- Buciluă, C., Caruana, R. & Niculescu-Mizil, A. (2006). Model Compression. *KDD 2006*.
- Furlanello, T., Lipton, Z. C., Tschannen, M., Itti, L. & Anandkumar, A. (2018). Born Again Neural Networks. *ICML 2018*, PMLR 80:1607–1616. arXiv:1805.04770.
- Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS 2014 Deep Learning Workshop*. arXiv:1503.02531.
- Kim, Y. & Rush, A. M. (2016). Sequence-Level Knowledge Distillation. *EMNLP 2016*.
- Project B3 LLM Briefing Pack §2.4 (loss); Simplified Plan, "Settings to lock" (temperature 1, sampling settings).
