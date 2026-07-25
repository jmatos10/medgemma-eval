# Preregistration

**Study:** Does medical pretraining earn its keep? A controlled evaluation of MedGemma 1.5 4B on two MedMNIST+ classification tasks.

**Author:** Juan A Matos Muñoz

**Date registered:** 2026-07-25 02:23 UTC (commit 41a958a)

**Status:** Frozen at first commit. No edits after Phase 4 begins. Any change after that date goes in `DEVIATIONS.md` with a reason and a timestamp.

---

## A. Study information

### A1. Research questions

**RQ1.** When a vision language model is evaluated zero-shot on a medical image classification task, what fraction of its measured error comes from producing an unparseable answer rather than from choosing the wrong class?

**RQ2.** At an identical LoRA budget on an identical task, does MedGemma 1.5 4B outperform Gemma 3 4B IT, the general-purpose model it was built from?

**RQ3.** Does either 4B model outperform a fine-tuned ResNet-18, which has roughly 400 times fewer parameters?

### A2. Hypotheses

Predictions are directional and stated before any model has been run against either dataset.

**H1 (format artifact).** Measured accuracy under strict parsing will understate zero-shot capability. Constrained decoding will raise macro-F1 by at least 10 percentage points over strict parsing on both datasets, with the strict-parsing condition producing unparseable output on more than 20 percent of items.

**H2 (medical pretraining).** After LoRA fine-tuning, MedGemma 1.5 4B will achieve higher macro-F1 than Gemma 3 4B IT on both datasets. Predicted advantage is small, between 1 and 5 percentage points.

**H3 (modality interaction, primary test).** The MedGemma advantage over Gemma will be larger on DermaMNIST than on BloodMNIST. Dermatology appears in MedGemma's stated image-encoder pretraining domains; blood cell microscopy does not. Predicted difference in the advantage is at least 2 percentage points of macro-F1.

**H4 (parameter efficiency).** ResNet-18 will achieve macro-F1 within 5 percentage points of the better 4B model on both datasets, and will exceed it on at least one.

### A3. What would falsify each hypothesis

| Hypothesis | Falsified if |
|---|---|
| H1 | Constrained decoding improves macro-F1 by under 10 points, or unparseable rate under strict parsing is below 20 percent |
| H2 | Gemma 3 equals or exceeds MedGemma on either dataset after fine-tuning |
| H3 | The advantage is equal or larger on BloodMNIST than on DermaMNIST |
| H4 | ResNet-18 trails the better 4B model by more than 5 points on both datasets |

H3 is the primary test. H2 can hold while H3 fails, which would indicate that MedGemma's advantage does not depend on the modality having been seen during pretraining.

---

## B. Design

### B1. Study type

Fully crossed factorial comparison. Two datasets by six arms. No human subjects. No randomized assignment, since conditions are deterministic model configurations.

### B2. Conditions

| Arm | Model | Treatment |
|---|---|---|
| A1 | MedGemma 1.5 4B | Zero-shot, strict parsing (exact label match only) |
| A2 | MedGemma 1.5 4B | Zero-shot, lenient parsing (substring and alias match) |
| A3 | MedGemma 1.5 4B | Zero-shot, constrained decoding over label set |
| A4 | MedGemma 1.5 4B | LoRA fine-tuned |
| A5 | Gemma 3 4B IT | LoRA fine-tuned |
| A6 | ResNet-18 (ImageNet init) | Fine-tuned |

A1 reproduces the evaluation method used in widely circulated public tutorials. It is included as a deliberate negative control, not as a good-faith baseline.

### B3. Held constant across A4 and A5

LoRA rank, alpha, dropout, target modules, learning rate, epochs, batch size, gradient accumulation, random seed, and the exact training subset. Starting weights are the only difference.

---

## C. Sampling

### C1. Data status

Both datasets are public and downloaded. Class distributions and split sizes have been inspected. No model has produced a prediction on either dataset. This registration is therefore prior to any outcome observation, and after inspection of input characteristics only.

### C2. Datasets

| | DermaMNIST | BloodMNIST |
|---|---|---|
| Modality | Dermatoscopic | Blood cell microscopy |
| In MedGemma pretraining domains | Yes | No |
| Classes | 7 | 8 |
| Train / val / test | 7,007 / 1,003 / 2,005 | 11,959 / official split |
| Majority class share (train) | 67.0% | 19.5% |
| Resolution | 224 x 224 RGB | 224 x 224 RGB |
| Source licence | CC BY-NC 4.0 | CC BY 4.0 |

Official MedMNIST+ splits are used without modification.

### C3. Training subsample

Training is subsampled to 5,000 images per dataset using seed 42, drawn once and reused identically across A4, A5, and A6. Rationale is compute budget, not statistical power. Validation and test splits are used whole.

### C4. Stopping rule

All six arms run to completion on both datasets. No interim analysis determines whether further arms are run. Total budget is capped at 14 GPU-hours; if exceeded, the third dataset extension is dropped and the two primary datasets still complete.

---

## D. Variables

### D1. Manipulated

**Model initialization:** MedGemma 1.5 4B, Gemma 3 4B IT, or ResNet-18.

**Output handling:** strict parsing, lenient parsing, or constrained decoding. Applied to zero-shot arms only.

**Adaptation:** none (zero-shot) or LoRA fine-tuning.

### D2. Primary outcome

Macro-averaged F1 on the held-out test split.

Macro-F1 is primary rather than accuracy because DermaMNIST is severely imbalanced. A constant predictor emitting the majority class scores 67.0 percent accuracy on DermaMNIST while carrying no diagnostic information. This choice is fixed here, before results exist, to remove the option of selecting the metric that flatters the outcome.

### D3. Secondary outcomes

Accuracy. Per-class F1. Confusion matrix. Unparseable-response rate. Wall-clock training time. GPU-seconds per arm. Trainable parameter count.

### D4. Parsing definitions

**Strict:** model output equals a canonical class label after whitespace and case normalization. Anything else scores as incorrect.

**Lenient:** a canonical label or a registered alias appears as a substring of the output. Aliases are fixed before running and listed in `src/eval_harness.py`. BloodMNIST class 3, `immature granulocytes(myelocytes, metamyelocytes and promyelocytes)`, receives explicit alias handling; its length and internal punctuation make exact emission unlikely.

**Constrained:** decoding is restricted to the label set, so the model selects among valid options and cannot produce an unparseable answer.

---

## E. Analysis

### E1. Test split discipline

The test split is evaluated once, after all six arms are trained and all development is complete. Prompt design, alias lists, and hyperparameters are fixed using the validation split only.

### E2. Uncertainty

Every reported metric carries a bootstrap 95 percent confidence interval from 1,000 resamples of the test set. Between-arm differences are reported as a difference with its own bootstrap interval.

### E3. Decision rule

A difference is treated as real when the bootstrap 95 percent interval for that difference excludes zero. Differences whose intervals include zero are reported as inconclusive at this sample size, not as null findings.

### E4. Seeds

Single seed (42) for all primary arms, a compute-budget concession that limits inference about run-to-run variance. If budget allows, A4 and A5 repeat at seeds 43 and 44 on DermaMNIST, and the spread is reported.

---

## F. Known threats to validity

**F1. Contamination.** MedMNIST+ is public and derived from public sources. MedGemma may have encountered these images or their source datasets during pretraining. Google's own model card raises this risk and advises validating on non-public data. That is not available here. Any MedGemma advantage may reflect memorization rather than transferable medical representation, and this study cannot separate the two.

**F2. Imbalance confound.** The two datasets differ on the variable of interest (modality in or out of the pretraining domains) and also on class balance (67.0 percent versus 19.5 percent majority share). H3 compares a between-model difference within each dataset, which cancels much of the dataset-level difficulty, though not all of it. Imbalance may change how much fine-tuning helps, and that effect is not separable in this design.

**F3. Single architecture family.** MedGemma 1.5 4B and Gemma 3 4B IT share an architecture, which is what makes the comparison clean. It also means findings may not generalize to other medically adapted models.

**F4. Prompt sensitivity.** One prompt template per dataset, fixed on validation data. Results may vary under alternative phrasings. No prompt search is performed.

**F5. Subsampling.** Training on 5,000 images per dataset means absolute performance will fall below what full-data fine-tuning would reach. Comparisons between arms remain valid since all arms see the same subset.

---

## G. Reporting commitment

All six arms are reported for both datasets regardless of outcome. Arms are not dropped for underperforming. If H2 or H3 fails, the failure is the headline finding.

Raw per-run records are appended to `results/results.jsonl` at run time and are not edited afterward.
