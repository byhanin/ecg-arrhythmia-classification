# HEART

### Hybrid ECG Arrhythmia Recognition via Twin Branches

**Can a model recognize an abnormal heartbeat by understanding both what it looks like and when it happens?**

An ECG beat tells two related stories. Its **morphology** describes the waveform itself; its **rhythm** describes how it relates to neighboring beats. Either can be informative, but neither tells the whole story in every patient.

**HEART** investigates what happens when a model learns both views in parallel and allows them to exchange information before making a prediction.

Developed as a **2025-2026 Computer Science graduation research project at King Saud University**, HEART (*Hybrid ECG Arrhythmia Recognition via Twin Branches*) is a dual-branch deep learning architecture for beat-level classification of three ECG categories: **Normal (N)**, **Supraventricular ectopic (S)**, and **Ventricular ectopic (V)**.

The central challenge is not simply to classify familiar ECG recordings. It is to **generalize to patients absent from training**, and to understand where that generalization breaks down.

> **Research only.** HEART was evaluated retrospectively using public ECG datasets. The reported performance is not evidence of prospective clinical validation or readiness for diagnostic deployment.

## 01 The research problem

ECG arrhythmia classification is difficult for several reasons:

- **Patient variation:** The same arrhythmia category can look different across individuals.
- **Class imbalance:** Normal beats substantially outnumber some abnormal beat types; accuracy alone can conceal missed arrhythmias.
- **Overlapping patterns:** Supraventricular beats can resemble normal beats morphologically, even when their timing is unusual.
- **Domain shift:** Recordings, leads, and patient populations vary from one dataset to another.

These challenges motivated three questions:

1. Can **rhythm and morphology** be modeled separately, then fused more intelligently than by simple concatenation?
2. Which architectural choices actually contribute to performance on **unseen patients**?
3. Do the learned representations remain useful on a **different dataset**?

## 02 The idea behind HEART

The framework processes ECG signals through a preprocessing and feature-engineering pipeline, then sends the resulting representations to two complementary branches.

| Stage | Design | Purpose |
| --- | --- | --- |
| **ECG preprocessing** | Filtering, adaptive edge handling, beat segmentation | Prepare recordings and reduce noise-related artifacts |
| **Feature representation** | **360-dimensional** vector per beat | Combine waveform, wavelet, morphological, and rhythm-related information |
| **Rhythm branch** | Bidirectional **GRU**, **7-beat** context | Model local timing and rhythm dependencies |
| **Morphology branch** | **Transformer**, **13-beat** context | Model waveform and contextual morphological relationships |
| **Cross-branch interaction** | **Residual Gated Cross-Attention (RGCA)** | Pass gated information bidirectionally between representations |
| **Feature conditioning** | **HybridQRS** gating mechanism, where applicable | Reduce the influence of selected misleading rhythm-sensitive features |
| **Prediction** | Fusion and classification head | Output N, S, or V |

### Why two branches?

A rhythm irregularity can flag a potentially abnormal beat without identifying its morphology. Conversely, a distinctive waveform may provide strong evidence for a ventricular event even where local timing is less decisive. HEART was designed to exploit these complementary patterns.

### Why RGCA?

The branches are **not simply independent models averaged at the end**. RGCA creates bidirectional interaction through learned messages, gates, and residual corrections. Each branch can incorporate relevant context from the other before the final classification decision.

The important question is whether this interaction helps in practice. The ablation study below tests exactly that.

## 03 Data and evaluation design

Three public ECG resources contributed to the experimental program:

| Dataset | Role |
| --- | --- |
| **MIT-BIH Arrhythmia Database** | Main benchmark; patient-disjoint and intra-patient evaluation |
| **MIT-BIH Supraventricular Arrhythmia Database (SVDB)** | Supplementary training data in the main inter-patient MIT-BIH experiment |
| **Icentia11k** | Evaluation on a larger, different ECG distribution and source dataset for transfer learning |

**Inter-patient evaluation** keeps test patients separate from training patients. **Intra-patient evaluation** allows beats from the same patients across partitions; its higher scores therefore answer a different, easier question. The two should not be compared as though they were the same test.

The study prioritizes **macro-F1** and **Matthews correlation coefficient (MCC)** in addition to accuracy. Macro-F1 gives each class equal weight, rather than allowing the majority Normal class to dominate the summary.

## 04 Three experiments, three questions

### Study 1: Can HEART generalize to unseen MIT-BIH patients?

The principal experiment trained using the **DS1** partition (with SVDB supplementation) and evaluated on the patient-disjoint **DS2** test set: **22 records and 47,912 beats**. Model selection and decision thresholds were determined using validation data, not the test set.

| Overall metric | Result |
| --- | ---: |
| Accuracy | **98.50%** |
| Macro-F1 | **93.70%** |
| Macro-F2 | 92.22% |
| Weighted-F1 | 98.46% |
| Micro-F1 | 98.50% |
| MCC | **91.99%** |

**Performance by class**

| Class | Precision | Recall | F1 | F2 | PR-AUC | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal (N) | 98.77% | 99.64% | 99.20% | 99.46% | 98.97% | 42,911 |
| Supraventricular (S) | 92.69% | 81.13% | 86.52% | 83.20% | 74.48% | 1,828 |
| Ventricular (V) | 97.75% | 93.10% | 95.37% | 93.99% | 98.47% | 3,173 |

**What stands out:** Despite high overall accuracy, **S-class recall (81.13%)** is notably lower than N- and V-class recall. This is why class-level metrics matter: a single aggregate number does not describe the model's most difficult decisions.

### Study 2: What happens on Icentia11k?

The second experiment tested HEART on a larger, distinct dataset. The held-out evaluation included **31,694 beats from 20 unseen patients**. The reported final configuration used tuned decision thresholds (**S = 0.50, V = 0.90**).

| Metric | Argmax | Tuned thresholds |
| --- | ---: | ---: |
| Accuracy | 90.00% | **91.19%** |
| Macro-F1 | 88.98% | **90.24%** |
| Macro-F2 | 90.37% | 90.90% |
| Weighted-F1 | 91.00% | 91.28% |
| MCC | Not reported | **83.98%** |

**Performance by class: tuned thresholds**

| Class | Precision | Recall | F1 | ROC-AUC | PR-AUC | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal (N) | 95.48% | 91.73% | 93.57% | 97.52% | 98.44% | 19,729 |
| Supraventricular (S) | 81.06% | 87.53% | 84.17% | 96.11% | 91.68% | 7,441 |
| Ventricular (V) | 91.20% | 94.85% | 92.99% | 97.62% | 94.15% | 4,524 |

**What stands out:** Threshold calibration improved macro-F1 from **88.98% to 90.24%**. The S class remained the most difficult to distinguish, with more confusion between Normal and Supraventricular beats than between Supraventricular and Ventricular beats. The **HybridQRS gate was not used in the final Icentia11k configuration**, because it did not offer a measurable benefit in that setting.

### Study 3: Can knowledge transfer between ECG datasets?

The third experiment asked a different question: can representations learned on Icentia11k be reused for MIT-BIH? HEART was **pretrained on Icentia11k**, **fine-tuned on MIT-BIH DS1**, then tested on the **unseen DS2 patients** (47,912 beats), with thresholds calibrated on MIT-BIH validation data (**S = 0.68, V = 0.80**).

| Overall metric | Result |
| --- | ---: |
| Accuracy | **98.18%** |
| Macro-F1 | **92.61%** |
| Macro-F2 | 93.41% |
| Weighted-F1 | 98.21% |
| MCC | **90.69%** |

**Performance by class**

| Class | Precision | Recall | F1 | F2 |
| --- | ---: | ---: | ---: | ---: |
| Normal (N) | 99.23% | 98.82% | 99.02% | 98.90% |
| Supraventricular (S) | 77.99% | 87.42% | 82.43% | 85.35% |
| Ventricular (V) | 97.03% | 95.71% | 96.37% | 95.97% |

**What stands out:** Compared with Study 1 on the same MIT-BIH DS2 patients, transfer learning increased **S recall from 81.13% to 87.42%** and **V recall from 93.10% to 95.71%**. But the improvement in S recall came with a substantial drop in **S precision from 92.69% to 77.99%**. Better detection does not automatically mean fewer false alarms.

### Results at a glance

| Evaluation | Accuracy | Macro-F1 | MCC |
| --- | ---: | ---: | ---: |
| MIT-BIH **inter-patient** (Study 1) | **98.50%** | **93.70%** | **91.99%** |
| Icentia11k, tuned thresholds (Study 2) | 91.19% | 90.24% | 83.98% |
| Icentia11k → MIT-BIH transfer (Study 3) | 98.18% | 92.61% | 90.69% |
| MIT-BIH **intra-patient** (additional evaluation) | 99.68% | 99.68% | 99.52% |

> These rows represent **different experimental protocols**. They show how HEART behaves in different conditions, not a single apples-to-apples leaderboard.

## 05 Did every component earn its place?

An architecture can look compelling on paper. **Ablation experiments** test whether its components genuinely affect measured performance.

For Study 1, the research compared full HEART against seven modified configurations on the MIT-BIH inter-patient setting.

| Configuration | Parameters | Accuracy | Macro-F1 | MCC | S precision | S recall | S F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full HEART** | 2,888,416 | **98.50%** | **93.70%** | **91.99%** | **92.69%** | 81.13% | **86.52%** |
| Without Transformer | 993,368 | 96.60% | 87.78% | 83.57% | 64.45% | 86.87% | 74.00% |
| Without GRU | 1,979,156 | 98.17% | 93.03% | 90.64% | 80.34% | **91.90%** | 85.74% |
| Without RGCA | 2,770,650 | 96.86% | 88.75% | 84.80% | 66.16% | 89.93% | 76.23% |
| GRU replaced by LSTM | 3,112,672 | 98.05% | 91.73% | 89.72% | 83.42% | 82.55% | 82.98% |
| Without SVDB supplementation | 2,888,416 | 97.44% | 88.56% | 86.46% | 78.73% | 73.52% | 76.04% |
| Without noise reduction | 2,888,416 | 98.42% | 93.12% | 91.74% | 80.91% | 87.92% | 84.27% |

**Ventricular-class outcomes in the same ablation study**

| Configuration | V precision | V recall |
| --- | ---: | ---: |
| Full HEART | 97.75% | 93.10% |
| Without Transformer | 90.55% | 91.84% |
| Without GRU | 95.84% | 92.91% |
| Without RGCA | 91.77% | 91.68% |
| GRU replaced by LSTM | 94.78% | 91.62% |
| Without SVDB supplementation | 91.55% | 90.14% |
| Without noise reduction | 96.95% | **94.82%** |

### The most revealing comparisons

- **Without RGCA:** Macro-F1 fell from **93.70% to 88.75%**, and S precision from **92.69% to 66.16%**. The interaction mechanism mattered in this evaluation.
- **Without Transformer:** S recall increased, but **S precision collapsed**. Rhythm alone could detect more candidate abnormal beats while mistaking more Normal beats for S.
- **Without GRU:** S recall increased to **91.90%**, while S precision declined to **80.34%**. The complete model produced a better overall balance of errors.
- **Without SVDB:** S recall decreased from **81.13% to 73.52%**, underscoring the contribution of supplementary minority-class data.

The full model achieved the highest **macro-F1 and MCC** of the tested configurations. Crucially, though, it did **not** maximize every individual metric; the ablations expose real precision and recall trade-offs.

## 06 The intra-patient result and why it needs context

Under an additional **MIT-BIH intra-patient** evaluation, with SVDB omitted, HEART achieved:

| Metric | Inter-patient | Intra-patient |
| --- | ---: | ---: |
| Accuracy | 98.50% | **99.68%** |
| Macro-F1 | 93.70% | **99.68%** |
| MCC | 91.99% | **99.52%** |
| Normal F1 | 99.20% | 99.72% |
| Supraventricular F1 | 86.52% | 99.60% |
| Supraventricular recall | 81.13% | 99.45% |
| Ventricular F1 | 95.37% | 99.70% |
| Ventricular recall | 93.10% | 99.82% |

This result is useful, but it should not be the project's headline. When a patient's beats appear across training and evaluation partitions, the model can encounter similar patient-specific signatures. **Patient-disjoint testing is the more meaningful evidence for generalization to new individuals.**

## 07 What the experiments taught us

**1. Accuracy alone can be misleading.** The Normal class dominates portions of the data, making class-specific precision and recall essential for interpreting arrhythmia recognition.

**2. The fusion mechanism deserves scrutiny, not just a diagram.** Removing RGCA substantially degraded performance, especially S-class precision. This provides experimental support for cross-branch interaction under the benchmark conditions.

**3. Generalization is not one number.** Performance shifted across MIT-BIH, Icentia11k, and transfer learning. A design choice that helped in one distribution, such as HybridQRS, was not necessarily beneficial in another.

**4. The S class remains the hardest case.** In transfer learning, higher S recall was accompanied by more false-positive S predictions. Evaluation needs to capture both sides of that trade-off.

**5. Aggregate scores can hide patient-level concentration.** In the MIT-BIH DS2 evaluation, **record 232 contributed approximately 75.3% of S-class beats**. As a result, the aggregate S-class recall is disproportionately influenced by one patient's recording. That is a limitation of how the benchmark is distributed, not simply a property of the model.

## 08 Limitations and future research

HEART remains a **research prototype**, with important boundaries identified in the graduation report:

- No prospective clinical or wearable-device validation.
- Differences in available signal features across single- and dual-lead datasets.
- Exclusion of paced-beat records and incomplete coverage of certain additional beat categories.
- Potential need to recalibrate class thresholds under distribution shift.
- Rhythm-based features that require preceding beats to build patient-specific timing statistics.
- Meaningful variability across individual patient records even when aggregate metrics are strong.

Future research could evaluate more diverse recording conditions, investigate lead-agnostic representations, extend the class coverage, and assess patient-level calibration and interpretability.

## Research details and attribution

**Full title:** *A Dual-Branch GRU-Transformer Architecture for Inter-Patient ECG Arrhythmia Classification*  
**Project:** HEART: Hybrid ECG Arrhythmia Recognition via Twin Branches  
**Institution:** King Saud University · College of Computer and Information Sciences · Computer Science Department  
**Courses:** CSC 496: Graduation Project I & CSC 497: Graduation Project II  
**Year:** 2025-2026

**Research team:** Raghad Baselm, Samar Alfallaj, Albatoul Alharbi, Dimah Althunayan, and Hanin Alshaibani  
**Supervisor:** Dr. Sarah Alotaibi  
**Co-supervisor:** Dr. Areej Almazroa

### About this repository

This is an **individual portfolio write-up of a collaborative graduation research project**. The architecture, experiments, and results described here come from the team's shared research report and are credited to the full team. The repository is presented as a **single Markdown document**, without separate figures or code files.

**The complete model implementation is not included here**, and the results have not been independently reproduced from this repository. Refer to the original research report for full methodology, configuration details, and discussion.

---

*HEART began with a question about rhythm and shape. Its experiments led to a broader one: how can ECG models make reliable decisions when the next patient, and the next dataset, looks different from anything they have seen before?*
