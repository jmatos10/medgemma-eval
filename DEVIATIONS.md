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
