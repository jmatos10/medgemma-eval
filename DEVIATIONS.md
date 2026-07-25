# Deviations and decision log

Companion to `HYPOTHESES.md`, which was frozen at commit `41a958a` on
2026-07-25 02:23 UTC.

Anything in this file happened **after** that freeze. Entries are appended,
never edited. Each says what changed, when, why, and what it does to the
interpretation of the results.

Two kinds of entry:

- **Deviation.** Something that differs from what was preregistered, or adds
  to it. Results affected by a deviation are labeled exploratory.
- **Decision.** A choice the preregistration explicitly left open, resolved
  as specified. Not a deviation.

---

## D-001. Deviation: exploratory arm A2b, answer extraction

**Date:** 2026-07-25
**Status:** Exploratory. Not preregistered. Reported separately from A1 to A6.

### What was added

A fourth parsing treatment applied to the same zero-shot generations used by
A1 and A2. No additional inference.

The rule, in full:

1. Take the text inside the first markdown bold span, `**...**`.
2. If no bold span exists, take the first non-empty line.
3. Run the frozen `parse_lenient` on that span alone, using the same alias
   table and the same ambiguity rule.
4. An empty or ambiguous span scores unparseable.

Implemented as `parse_extracted` in `src/eval_harness.py`, registered under
`EXPLORATORY_PARSERS` rather than `PARSERS` to keep the preregistered set
separate.

### Why

A 50-image pilot on DermaMNIST validation showed that MedGemma states its
answer plainly and then enumerates the remaining classes in order to reject
them. Example, lightly trimmed:

> Based on the image, the best description is **melanocytic nevi**.
> The other options are less likely:
> Actinic keratoses..., Basal cell carcinoma..., Dermatofibroma...

The frozen lenient rule finds several distinct classes in that response,
applies the ambiguity rule from section D4, and returns unparseable. A human
reader extracts the answer without difficulty.

So A2 as defined measures whether the model avoided listing alternatives,
not whether it communicated an answer. Those are different quantities, and
only the second is what RQ1 asks about.

Measured on the 50-image pilot:

| Parser | Unparseable | Correct |
|---|---|---|
| A1 strict | 50/50 | 0 |
| A2 lenient | 45/50 | 1 |
| A2b extraction | 0/50 | 29 |

### What was considered and rejected

**Amending the lenient definition in section D4.** Rejected. D4 is frozen and
Phase 4 had begun. Redefining a preregistered parser after seeing outputs is
the failure mode this study exists to document. The frozen definition stands
and A2 is reported as specified.

**Changing nothing.** Viable, and A2 would have been reported as a
conservative lower bound. Rejected because the A2 to A2b gap is itself
informative: it separates apparent failure caused by enumeration from
failure caused by medical error.

### Effect on interpretation

A1 and A2 remain the preregistered basis for H1. A2b is reported alongside
them, labeled exploratory, and is not used to test any preregistered
hypothesis.

The extraction rule was written and committed before it was scored on
anything beyond five hand-inspected outputs, so it could not be tuned to
improve the result.

**Known risk.** If a model bolds something other than its answer, extraction
silently takes the wrong span. Observed on 50 of 50 DermaMNIST outputs
without failure. Must be re-checked on BloodMNIST and on Gemma 3, whose
formatting habits may differ.

---

## D-002. Decision: production generation cap set to 512

**Date:** 2026-07-25
**Status:** Decision under prespecified latitude. Not a deviation.

The Phase 4 spec deliberately left `max_new_tokens` unset, to be determined
by pilot measurement rather than guessed. Setting it too low would truncate
answers, score them unparseable, and manufacture the very artifact H1 exists
to measure.

Pilot: 50 DermaMNIST validation images, cap opened to 512.

| Percentile | Tokens |
|---|---|
| p50 | 239 |
| p90 | 268 |
| p95 | 291 |
| p99 | 398 |
| max | 401 |

Truncated at cap: 0 of 50.

The distribution is bimodal. Short responses run 39 to 62 tokens and give a
one-sentence justification. Long responses run about 237 to 240 tokens and
enumerate every class.

**Cap fixed at 512**, which is 111 tokens above the longest observed output.
Applies to all zero-shot generation arms on both datasets.

---

## D-003. Correction: token counting bug in the pilot script

**Date:** 2026-07-25
**Status:** Implementation bug. No effect on any reported result.

The first pilot run counted generated tokens by filtering
`tokenizer.pad_token_id` (0). In batched generation the framework fills
finished sequences to match the longest in the batch, and the fill token is
not that id. Gemma stops on `<end_of_turn>` (106), not `<eos>` (1).

Every sequence therefore reported the batch maximum. All four rows in the
first batch reported 239 tokens despite output text ranging from roughly 180
to 1,000 characters.

Fixed by scanning each row for the first stop token and counting up to it,
which does not depend on knowing the fill token. Generated text was
unaffected and is byte-identical across both runs. Only the length
measurement changed.

---

## D-004. Observation: majority-class collapse on the pilot

**Date:** 2026-07-25
**Status:** Observation. Not a result. Sample too small and not class-balanced.

On the 50-image pilot, scored with A2b:

| Metric | Value |
|---|---|
| Accuracy | 0.580 |
| Majority-class baseline accuracy | 0.540 |
| Macro-F1 | 0.200 |
| Per-class F1 | 0.00, 0.00, 0.35, 0.00, 0.29, 0.76, 0.00 |

Predicted class 5 for 36 of 50 images. True count is 27.

Four of seven classes score zero. Two of those (classes 1 and 3) have no
true instances in this draw, so their zero is an artifact of the sample.
Averaged over the five classes present, macro-F1 is 0.28.

Accuracy sits four points above a constant predictor emitting the majority
class. This is why section D2 fixed macro-F1 as the primary outcome before
any results existed.

**Open question, logged not concluded.** Melanocytic nevi is the dominant
class in HAM10000, the source dataset for DermaMNIST. A model that absorbed
that base rate during pretraining would over-predict nevi without reading
the image carefully. That is consistent with threat F1 (contamination) and
equally consistent with ordinary majority-class bias. This pilot cannot
distinguish the two. Revisit when A4 and A5 are available.

---

## D-005. Correction to D-004: no majority-class collapse at full scale

**Date:** 2026-07-25
**Status:** Correction. Supersedes the collapse claim in D-004.

D-004 reported majority-class collapse from the 50-image pilot: 36 of 50
predictions were class 5 against a true count of 27.

The full DermaMNIST validation split, 1,003 images, does not reproduce it.

| | Pilot, n=50 | Full split, n=1,003 |
|---|---|---|
| Class 5 predicted | 36 | 662 |
| Class 5 true | 27 | 671 |
| Over-prediction | +33% | -1% |
| Macro-F1 (A2b) | 0.200 | 0.365 |
| Classes with F1 = 0 | 4 of 7 | 0 of 7 |

The pilot draw contained 27 nevi out of 50, and two classes had no instances
at all, so their F1 was zero by construction. The collapse was an artifact of
the sample, not a property of the model.

At full scale the model predicts nevi almost exactly as often as nevi occur,
and posts non-zero F1 on all seven classes.

**The D-004 open question about contamination is unaffected.** Calibrated
base-rate matching is equally consistent with a model that learned HAM10000's
prior and one that reads images competently. Nothing here separates them.
Revisit when A4 and A5 are available.

**Method note.** A 50-image pilot was adequate for its stated purpose,
measuring output length to set the generation cap. It was not adequate for
estimating per-class performance, and D-004 should have labeled the collapse
claim as underpowered rather than as an observation. Recorded here so the
same error is not repeated on BloodMNIST.

---

## D-006. Result: H1 confirmed on DermaMNIST validation

**Date:** 2026-07-25
**Status:** Preregistered result. Validation split only. Test remains untouched.

Full DermaMNIST validation, 1,003 images, MedGemma 1.5 4B zero-shot, greedy
decoding, cap 512, zero truncated.

| Arm | Macro-F1 | 95% CI | Accuracy | Unparseable |
|---|---|---|---|---|
| A1 strict | 0.000 | [0.000, 0.000] | 0.000 | 100.0% |
| A2 lenient | 0.137 | [0.080, 0.184] | 0.036 | 92.4% |
| A2b extraction | 0.365 | [0.295, 0.432] | 0.643 | 0.0% |

A2b minus A1 macro-F1: **+0.365**, 95% CI [+0.295, +0.432], excludes zero.

H1 predicted at least +0.10 macro-F1 from constrained decoding over strict,
and above 20 percent unparseable under strict. Measured: +0.365 and 100
percent. The hypothesis holds by a wide margin.

The model produced a bare canonical label zero times in 1,003 attempts.

All three rows come from one generation pass. Identical weights, prompt,
decoding, and output text. The only thing that differs is how the text is
read.

**Secondary observation, primary metric doing its job.** A2b accuracy of
0.643 sits below the majority baseline of 0.669. Reported on accuracy alone
the model would appear worse than a constant predictor. A constant predictor
scores 0.115 macro-F1 against A2b's 0.365, with non-zero F1 on all seven
classes. Section D2 fixed macro-F1 as primary before any result existed, and
this is the case it was fixed for.

---

## D-007. Result: H1 fails on BloodMNIST validation

**Date:** 2026-07-25
**Status:** Preregistered result, reported per section G. Validation only.

Full BloodMNIST validation, 1,712 images, MedGemma 1.5 4B zero-shot, greedy
decoding, cap 512.

| Arm | Macro-F1 | 95% CI | Accuracy | Unparseable |
|---|---|---|---|---|
| A1 strict | 0.000 | [0.000, 0.000] | 0.000 | 100.0% |
| A2 lenient | 0.038 | [0.034, 0.043] | 0.157 | 4.8% |
| A2b extraction | 0.037 | [0.033, 0.043] | 0.160 | 0.0% |

A2b minus A1: +0.037, 95% CI [+0.033, +0.043].

H1 predicted at least +0.10 macro-F1 on **both** datasets. BloodMNIST returns
+0.037. **H1 is falsified on this dataset.**

### Why the recovery is small

The model predicted class 3, immature granulocytes, for 1,669 of 1,712
images (97.5 percent). Six of eight classes score exactly zero F1. A constant
predictor emitting class 3 scores approximately 0.033 macro-F1 against
A2b's 0.037.

There is no latent capability for better parsing to recover. Format handling
cannot rescue a model that answers the same class regardless of input.

Generated text shows the model producing a plausible morphological
description of what it was shown, then concluding immature granulocyte
anyway. Example, an erythroblast: it described a large dark purple nucleus
with scant cytoplasm, which is accurate, and answered class 3.

### Interpretation

H1 holds where the model has competence (DermaMNIST, +0.365) and fails where
it does not (BloodMNIST, +0.037). The corrected claim is narrower and
sharper than the original: format artifacts inflate measured error only when
there is real capability being obscured. They cannot manufacture capability
that is absent.

Reported per section G. Arms are not dropped for failing.

### Secondary: the A2 enumeration risk did not generalize

D-001 flagged that extraction was validated only on DermaMNIST and had to be
re-checked elsewhere. It has been.

Lenient parsing left 4.8 percent unparseable on BloodMNIST against 92.4
percent on DermaMNIST. The enumerate-and-reject behavior that broke A2 is
dermatology-specific under this prompt. A2 and A2b are statistically
indistinguishable here (0.038 against 0.037), which is the correct behavior
for an exploratory arm: it adds nothing when there is nothing to add.

---

## D-008. Result: A3 on DermaMNIST, and the prespecified primary underperforms

**Date:** 2026-07-25
**Status:** Preregistered result. Validation only.

Full DermaMNIST validation, 1,003 images, constrained decoding over the
label set. No text generated, so parse failure is impossible.

| Variant | Macro-F1 | 95% CI | Accuracy | Status |
|---|---|---|---|---|
| mean log-prob per token | 0.198 | [0.176, 0.220] | 0.529 | **primary, prespecified** |
| raw log-prob sum | 0.334 | [0.279, 0.377] | 0.596 | robustness |
| PMI | 0.369 | [0.309, 0.425] | 0.610 | robustness |

A1 scores 0.000 with a CI of [0.000, 0.000], so the A3 minus A1 difference
equals A3's own value: **+0.198, 95% CI [0.176, 0.220]**, excluding zero.

H1 predicted at least +0.10 on this dataset. It holds.

### The primary is the worst of the three

Mean log-probability underperforms PMI by 17 points of macro-F1. The
prespecification in the Phase 4 spec fixed `mean` as primary before any
image was scored, on the theoretical argument that dividing by token count
cancels the length penalty.

That argument was incomplete. Long labels with internally predictable
continuations carry per-token log-probabilities near zero, which inflates
their average. Dividing by length does not cancel the length effect on such
labels, it can invert it.

Had the primary been left open, PMI at 0.369 could have been reported as the
headline with `mean` described as an alternative also considered. No reader
could have detected the substitution. That option did not exist because the
choice was committed in advance.

All three are reported, primary first, and the primary's underperformance is
stated rather than buried.

### Normalization changes the answer 55 percent of the time

The three scorings agree on 454 of 1,003 images. The 50-image pilot showed
agreement on 20 of 50, a similar rate at a much smaller sample.

This belongs in limitations. A constrained-decoding result on this task is
substantially a function of the normalization chosen, and papers reporting a
single constrained-decoding number without stating which normalization was
used, and whether it was chosen in advance, are underspecified.

### Convergent estimates

PMI at 0.369 and A2b extraction at 0.365 agree within noise despite sharing
no machinery: one scores candidate labels without generating, the other
reads generated text. Together with the pilot, three independent estimates
cluster near 0.36 for MedGemma 1.5 zero-shot capability on DermaMNIST.

---

## D-009. Phase 4 close: H1 verdict on both datasets

**Date:** 2026-07-25
**Status:** Preregistered result. Validation split only. Test untouched.

All six zero-shot treatments, both datasets, full validation splits.
MedGemma 1.5 4B, revision `91850547`, greedy decoding, cap 512, seed 42.

| Arm | Treatment | DermaMNIST (n=1,003) | BloodMNIST (n=1,712) |
|---|---|---|---|
| A1 | strict parsing | 0.000 | 0.000 |
| A2 | lenient parsing | 0.137 | 0.038 |
| A2b | extraction (exploratory) | 0.365 | 0.037 |
| **A3** | **constrained, mean logp (primary)** | **0.198** | **0.036** |
| A3-sum | constrained, raw sum | 0.334 | 0.021 |
| A3-pmi | constrained, PMI | 0.369 | 0.019 |
| | majority-class baseline | 0.115 | 0.041 |

Macro-F1. Confidence intervals in D-006 through D-008.

### H1 verdict

H1 predicted at least +0.10 macro-F1 from constrained decoding over strict
parsing, and above 20 percent unparseable under strict, **on both datasets**.

| | Unparseable under A1 | A3 minus A1 | Verdict |
|---|---|---|---|
| DermaMNIST | 100% | +0.198 [0.176, 0.220] | **holds** |
| BloodMNIST | 100% | +0.036 [0.033, 0.039] | **falsified** |

The unparseable clause holds everywhere and by a wide margin. MedGemma
produced a bare canonical label zero times in 2,715 attempts across both
datasets. The magnitude clause holds on one dataset and fails on the other.

**Revised claim, stated for the writeup:** format artifacts inflate measured
error only where genuine capability is being obscured. They cannot
manufacture capability that is absent. H1 as written did not distinguish
these cases, and should have.

### BloodMNIST is degenerate under every treatment

No arm exceeds the 0.041 majority baseline. A3 under the primary
normalization predicts class 3 for all 1,712 images, which makes it a
constant predictor by construction: its 0.036 is arithmetically identical to
what always answering class 3 would score.

### Normalization sensitivity scales inversely with signal

Agreement among the three A3 scorings:

| Dataset | Agreement | Model has signal? |
|---|---|---|
| DermaMNIST | 454/1,003 (45%) | yes |
| BloodMNIST | 5/1,712 (0.3%) | no |

Where the image carries information, the normalizations mostly agree and
differ at the margin. Where it does not, the normalization produces the
ranking entirely, and the three variants select three different constants.
Near-total disagreement among normalizations is therefore a usable signal
that a constrained-decoding result is measuring label token statistics
rather than the image.

### Convergent estimates on DermaMNIST

Three methods sharing no machinery: A2b extraction 0.365, A3-pmi 0.369, and
the 50-image pilot 0.200 under mean normalization matching the full-split
0.198. The clustering near 0.36 for the two strongest methods supports
treating that as MedGemma's real zero-shot ceiling here, against a 0.115
baseline.

### Cost

Phase 4 consumed approximately 3.1 GPU-hours, about $3.40. Four full passes
plus pilots. Running total for the project is roughly $11.

### What this sets up

H2 and H3 concern the fine-tuned gap between MedGemma and Gemma 3, which
Phase 4 does not address. One thing is now known that bears on H3: MedGemma's
zero-shot advantage on the in-domain modality (DermaMNIST) over the
out-of-domain one (BloodMNIST) is large before any fine-tuning. Whether that
gap survives LoRA on both models is the actual test.