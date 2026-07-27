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

---

## D-010. Phase 5 setup: subsample, training target, and throughput decisions

**Date:** 2026-07-25
**Status:** Decisions under prespecified latitude, plus one logging note.

### D-010a. Epochs fixed at 1

Section B3 requires epochs be held constant across A4 and A5 but never fixed
the value. Set to **1**.

Rationale is compute budget, consistent with F5, which already states that
subsampling to 5,000 images puts absolute performance below full-data
fine-tuning while leaving between-arm comparisons valid. The same argument
covers epochs: a smaller budget lowers both arms equally, and H2 and H3
compare arms rather than absolute performance.

### D-010b. Training subsample is simple random, not stratified

Section C3 fixed 5,000 images at seed 42 and did not specify the sampling
scheme. Simple random was chosen.

Stratifying to balance classes would train on one distribution and evaluate
on another, so the model's learned prior would not match the distribution it
is scored against. Simple random preserves the natural imbalance.

Drawn indices are frozen in `subsamples/{dataset}_train_5000.json` with a
SHA-256 of the sorted index list. `load_subsample()` re-derives the hash and
refuses to proceed on a mismatch, so A4, A5, and A6 provably see identical
data rather than assertedly.

Realized class proportions track the parent split within 0.86 percentage
points on both datasets.

**Known cost.** DermaMNIST class 3 (dermatofibroma) gets 56 training
examples and class 6 (vascular lesions) gets 68. At one epoch each is seen
once. Both will likely score near-zero F1 in A4 and A5, so 2 of 7 classes
contribute nothing to macro-F1 for either model.

This does not bias H2 or H3, since both arms face it identically. It reduces
sensitivity: the predicted MedGemma advantage is 1 to 5 points of macro-F1,
and diluting the metric across two dead classes makes a small real
difference harder to resolve against the bootstrap interval. Logged now
rather than raised after seeing results. The primary metric is not changing.

### D-010c. Training target is the bare canonical label

Target is `melanocytic nevi`, not a formatted sentence. Loss is computed on
target tokens only; prompt, image, and padding tokens are set to -100.
Verified by decoding the first batch: 7 of 342 tokens supervised, decoding
to the bare label plus the turn-end token.

Phase 4 established that MedGemma emits a bare label zero times in 2,715
zero-shot attempts. Training on formatted prose would carry that format
penalty into A4 and A5. Bare labels mean the four parsers should converge on
the fine-tuned arms, and that convergence is the answer to Q1.

Q1 then decomposes without a further experiment: A1 to A4 is the total jump,
A3 to A4 is the medical component alone since A3 carries no format penalty,
and the difference is the format component.

**Limitation.** Fine-tuning on bare labels makes the model better at
classification and worse at explaining itself. This study measures
classification, not clinical utility.

### D-010d. Vision encoder frozen

LoRA is applied to language-model projections only, 238 modules spanning
layers 0 to 33. Target modules are discovered by scanning `named_modules`
and excluding anything under the vision tower, because SigLIP also exposes
`q_proj` and a bare suffix match would adapt it.

The script counts trainable parameters under the vision tower and exits if
the count is not zero.

This is required for H3 to mean anything. H3 asks what MedGemma's image
encoder absorbed during medical pretraining. Fine-tuning that encoder would
let both models learn the modality during training and wash out the
difference being tested.

Trainable: 29,802,496 of 4,329,881,968 parameters, 0.688 percent.

### D-010e. Gradient checkpointing disabled, measured not assumed

64-example smoke test, A4 on DermaMNIST, identical seed and effective batch:

| Configuration | Samples/sec | Train loss |
|---|---|---|
| batch 2, accum 8, checkpointing on | 1.048 | 1.4993 |
| batch 2, accum 8, checkpointing off | 2.185 | 1.5010 |

Checkpointing trades compute for activation memory. Disabling it more than
doubles throughput. The near-identical loss confirms the optimization is
untouched.

Batch 4 was tried and ran out of memory inside `cross_entropy`, whose logits
tensor is `[batch x sequence, 262k vocabulary]` in float32, about 1.4 GB at
batch 4. Batch stays at 2 with accumulation 8, effective batch 16.

The script refuses to run if `batch x accum` does not equal 16, so
throughput can be retuned without the optimization drifting between arms.
Defaults were changed in the committed file rather than passed as flags, so
B3 compliance does not depend on remembering three arguments four times.

Revised Phase 5 estimate: 4 runs, about 38 minutes each, 2.5 GPU-hours,
roughly $2.75. This brings the projected project total to about 13.5
GPU-hours, within the 14-hour cap in section C4.

### D-010f. Two smoke-test records in results.jsonl

Two 64-example runs appended records to `results/results.jsonl` during setup.
Hard rule 5 makes that file append-only, so they remain.

They are identifiable by `n_train: 64`. The script was subsequently patched
to write `"smoke_test": true` on any run using `--limit`, but the two
existing records predate that field.

**Analysis must filter Phase 5 training records to `n_train == 5000`.**

---

## D-011. Correction to D-010f: the smoke_test patch was never applied

**Date:** 2026-07-26
**Status:** Correction. Supersedes the patch claim in D-010f.

D-010f states that `train_lora.py` "was subsequently patched to write
`"smoke_test": true` on any run using `--limit`." That patch was written but
never uploaded to the instance. The committed script does not contain it, and
**none of the six Phase 5 training records carry the field.**

Verified: `grep -c smoke_test scripts/train_lora.py` returns 0, and every
record in `results.jsonl` with `stage == "train"` lacks the key.

The patch is not being applied retroactively. The committed script is the one
that produced all four adapters, and substituting a version that never ran
would be worse than the missing field.

**Operative rule is unchanged and now the only safeguard: analysis must
filter Phase 5 training records to `n_train == 5000`.** Two records with
`n_train == 64` are 64-example smoke tests from setup and are not arm A4.

### Why this happened

The deviation entry was written while the first training run was in flight,
describing an intended change rather than a verified one. The upload was
deferred and never completed.

Entries asserting a code change should be written after the change is
verified present, not before. The four result-bearing entries in this file
(D-006 through D-009) were all written from pasted terminal output and are
unaffected.

---

## D-012. Phase 5 complete: four adapters trained

**Date:** 2026-07-26
**Status:** Training complete. No evaluation yet. These are not results.

| Arm | Model | Dataset | Train loss | Minutes |
|---|---|---|---|---|
| A4 | MedGemma 1.5 4B | DermaMNIST | 0.1360 | 38.1 |
| A5 | Gemma 3 4B IT | DermaMNIST | 0.1666 | 38.2 |
| A4 | MedGemma 1.5 4B | BloodMNIST | 0.1481 | 38.2 |
| A5 | Gemma 3 4B IT | BloodMNIST | 0.1509 | 38.4 |

Identical hyperparameters, identical 5,000-image subsample verified by hash,
seed 42, vision encoder frozen, 29,802,496 trainable parameters (0.688
percent) in all four runs. Wall clock within 0.3 minutes across all four, so
both arms received equal compute on both datasets.

Adapters are 114 MB each and are **not committed**. GitHub rejects files over
100 MB, and they are reproducible from the committed subsample indices,
hyperparameters, and seed. `adapters/` is in `.gitignore`.

### Training-loss gap, a hint and not a test

| Dataset | A4 | A5 | Gap favoring MedGemma |
|---|---|---|---|
| DermaMNIST (in-domain) | 0.1360 | 0.1666 | +0.0306 |
| BloodMNIST (out-of-domain) | 0.1481 | 0.1509 | +0.0027 |

The gap is roughly 11 times larger on the modality inside MedGemma's stated
pretraining domains, which is the shape H3 predicts.

**This is not evidence for H3.** H2 and H3 are stated on macro-F1 over a
held-out split with a bootstrap confidence interval. Training loss is neither
held out nor interval-bounded, and a lower training loss is equally
consistent with better learning and with more memorization of those 5,000
examples, which is precisely threat F1. Recorded because it was observed, and
labeled so it cannot later be presented as confirmation.

### Infrastructure: three mid-run instance shutdowns

The A5 BloodMNIST run was killed three times before completing. Kernel logs
from the failed boots show clean, graceful shutdowns: Docker containers
stopped in order, filesystems unmounted, then SIGTERM. No GPU fault, no
kernel OOM, no full disk.

Diagnosis: **Workbench idle detection appears to track Jupyter kernel
activity, not terminal activity or GPU utilization.** Runs launched with
`nohup` from a terminal were stopped mid-training while the GPU sat at 100
percent, and disabling the idle-shutdown setting did not prevent it. The run
completed only when launched from a notebook cell, which keeps a kernel busy
for the duration.

Cost: roughly 30 minutes of GPU, about $0.55. Adapters save only on
completion, so a killed run produces a 4 KB output directory that looks
present in `ls` but contains no weights. Check for
`adapter_model.safetensors` at roughly 114 MB, not for the directory.

**Implication for Phase 7.** The test pass has longer runs. Either launch
from a notebook cell, or add step-level checkpointing with
`--resume-from-checkpoint` so a shutdown costs minutes rather than the run.

---

## D-013. Measured run-to-run variance under a fixed seed

**Date:** 2026-07-26
**Status:** Measurement. Bears on how confidently small differences can be
reported.

### Why it was measured

`train_resnet.py` originally saved no checkpoint. Evaluating A6 on the test
split would therefore have required retraining, and the test number would
have come from a different fitted model than the validation number,
confounding generalization with run-to-run variance.

The script now saves `checkpoints/A6_{dataset}/resnet18.pt` after training
and accepts `--eval-only` to load it. A6 was retrained once so a checkpoint
exists, which produced an incidental replication.

### The measurement

Identical seed (42), identical hash-verified subsample, identical code and
hyperparameters. Only GPU scheduling nondeterminism differs.

| Dataset | First run | Replication | Delta |
|---|---|---|---|
| DermaMNIST | 0.731 | 0.739 | +0.008 |
| BloodMNIST | 0.991 | 0.989 | -0.002 |

**Run-to-run variance is roughly 0.008 macro-F1 for ResNet-18 training on
this data.**

### What it implies for the reported results

Section E4 acknowledged that a single seed limits inference about run-to-run
variance. This is the first empirical estimate.

| Result | Magnitude | Multiple of observed variance |
|---|---|---|
| H3 interaction | +0.118 | ~15x |
| H2 DermaMNIST | +0.141 | ~18x |
| H1 DermaMNIST | +0.198 | ~25x |
| H4 DermaMNIST | +0.156 | ~20x |
| H4 BloodMNIST | +0.039 | ~5x |
| **H2 BloodMNIST** | **+0.023** | **~3x** |

The primary result and the large effects are far above this floor. **The
BloodMNIST H2 advantage of +0.023 is only about three times the observed
variance and should be reported with that caveat**, even though its bootstrap
interval [+0.0094, +0.0364] excludes zero. The bootstrap quantifies sampling
uncertainty over the evaluation set. It does not quantify variance from
retraining, and those are different sources.

**Limitation of this estimate.** It comes from ResNet-18 training, not LoRA
fine-tuning of a 4B model. Different architecture, different optimizer,
different batch composition. Suggestive for the LoRA arms rather than
directly applicable. Measuring LoRA variance properly would require the
seed-43 and seed-44 repeats section E4 left conditional on budget.

### Consequence for committed files

The A6 raw prediction files were overwritten by the replication, so the
committed A6 predictions and the checkpoint now correspond to the same
model. Validation macro-F1 for A6 is therefore 0.739 and 0.989, not the
0.731 and 0.991 quoted in earlier working notes.

H4 shifted accordingly, from +0.1477 to +0.1556 on DermaMNIST and from
+0.0407 to +0.0389 on BloodMNIST. The H3 interaction was byte-identical at
+0.1178, as expected, since it is computed only from A4 and A5.

---

## D-014. Correction: the pretraining modality list was from MedGemma v1

**Date:** 2026-07-27
**Status:** Correction to a premise behind H3. Affects interpretation, not
arithmetic.

### What was wrong

The working framing around H3 described MedGemma's image encoder as
pretrained on "chest X-ray, dermatology, ophthalmology, and histopathology."

That is the MedGemma v1 list. The model evaluated here is MedGemma 1.5,
whose Technical Report (arXiv:2604.05081) describes a 400M MedSigLIP
encoder trained on:

1. 3D radiology (CT and MRI volumes)
2. Whole-slide histopathology
3. Chest X-ray with anatomical bounding-box localization
4. Multi-timepoint radiology
5. Dermatology

Ophthalmology is not called out for 1.5.

The error came from working off the earlier report without checking that it
covered the checkpoint actually being run. The wrong technical report was
also cited (arXiv:2507.05201, which covers MedGemma 4B and 27B).

### H3's registered premise survives

H3 states: "Dermatology appears in MedGemma's stated image-encoder
pretraining domains; blood cell microscopy does not."

Both halves remain true under the 1.5 list. Dermatology is named. Blood cell
microscopy is not. No registered claim requires revision.

### The substantive problem this surfaces

**Whole-slide histopathology is brightfield microscopy of stained cells and
tissue. A peripheral blood smear is also brightfield microscopy of stained
cells.** They differ in preparation and in what is being counted, but they
share imaging physics, magnification regime, and staining conventions.

BloodMNIST is therefore **not the maximally out-of-domain control the
preregistration implies.** It is a modality not named in the pretraining
list, which shares substantial low-level visual structure with one that is.

This should have been stated in section F as a threat to validity. It was
not, because the modality list being used was wrong.

### Direction of the bias

Partial overlap between BloodMNIST and the pretraining distribution would be
expected to **raise** MedGemma's out-of-domain performance, which
**shrinks** the measured interaction. A genuinely distant modality, one
sharing neither imaging physics nor stain, would plausibly show a larger
gap.

So the reported interaction is more likely an underestimate than an
overestimate. That is stated as a direction of bias, not as a defence of the
result: the contrast is weaker than registered, and a weaker contrast is a
weaker test regardless of which way the bias runs.

### Anchor from the 1.5 report

Reported zero-shot: DermMCQA 73.5, PathMCQA 70.0. No blood or
peripheral-smear microscopy results are reported, so there is no published
figure to check the BloodMNIST arms against.

### Actions

1. The modality list used in project notes corrected to the 1.5 list.
2. The brightfield overlap is added to the reported limitations as a named
   threat to validity.
3. Citation corrected to arXiv:2604.05081. The earlier report is retained
   only as lineage, labeled as the MedGemma 4B and 27B release.
4. `HYPOTHESES.md` is **not** edited. It is frozen, its claim is true as
   written, and this entry is the correction of record.

### Related citation errors found in the same review

- **arXiv:2507.05201 cited for MedGemma 1.5.** Wrong report. Corrected.
- **MILM attributed to "Wu, Y."** Fabricated. The name was carried over from
  the GNMT length-normalization reference. Corrected on 2026-07-27 by
  reading the source: Chung, H.-H., Li, S., Wald, Y., Han, X., Saria, S.,
  Ghosh, J., arXiv:2605.13711v1, 13 May 2026. Nothing in this project should
  cite an author name that has not been read from the source.
- **arXiv:2604.23801 understated.** It is direct prior work: Gemma 3 4B
  against MedGemma 4B on MedQA-USMLE, finding +6.8 points from domain
  fine-tuning. That is Q2 in text-only form and must be cited as such, with
  this study's three differences named (vision rather than text, matched
  LoRA budget rather than quantized off-the-shelf inference, and the
  modality-inside versus modality-outside contrast, which they do not run).
- **MedMNIST at size=224 comes from MedMNIST+**, a separate Zenodo release,
  not the 2023 *Scientific Data* v2 paper. Both are cited.

Full verification record in `REFERENCES.md`.

---

## D-015. Twelve mislabeled records in results.jsonl

**Date:** 2026-07-27
**Status:** Logging defect. No effect on any reported number.

### What happened

`scripts/score_arms.py` detected file type by the presence of an
`output_text` field and, finding it, scored the file as the zero-shot arms
A1, A2, and A2b. The fine-tuned generation files `gen_A4_*.jsonl` and
`gen_A5_*.jsonl` also contain `output_text`.

So four fine-tuned files were logged under three zero-shot arm ids each,
producing **12 records in `results/results.jsonl` whose `arm` field does not
match the arm recorded inside their `source_file`.**

### Why no reported number is wrong

Records are appended in filename order, and the true zero-shot files
(`gen_dermamnist_test.jsonl`, `gen_bloodmnist_test.jsonl`) sort after the
`gen_A4_*` and `gen_A5_*` files. Reading the latest record per arm therefore
returns the correct zero-shot values, which is what was reported.

The A1 zero-shot figure of 0.000 could not have come from a fine-tuned file:
strict parsing of A4's output scores 0.617 on DermaMNIST. The independent
independent recomputation described below confirms every reported number
re-derives from its stated source file, with 0 mismatches.

### Fix

`score_arms.py` now reads the `arm` field from the file it is scoring and
uses it when present, falling back to A1/A2/A2b only for files that carry no
arm id. The four files were rescored, adding correctly labeled records for
A4, A4-lenient, A4-extract, A5, A5-lenient, and A5-extract.

The 12 incorrect records remain. `results.jsonl` is append-only under hard
rule 5, and deleting them would be a worse precedent than leaving a
documented defect in place.

### Rule for reading results.jsonl

**Filter on `source_file`, not on `arm` alone.** A record's `source_file`
field is authoritative about which raw file produced it. Reading by `arm`
alone will return a fine-tuned model's score under a zero-shot arm label,
and specifically will suggest that zero-shot strict parsing scored 0.947 on
BloodMNIST when the true value is 0.000.

The correct read is the latest record for each `(arm, source_file)` pair.
The zero-shot arms come from `gen_{dataset}_{split}.jsonl`; the fine-tuned
arms come from `gen_A4_*` and `gen_A5_*`.

### How this was caught

An audit script written after the test pass recomputed every logged
macro-F1 from its stated `source_file` and separately compared each record's
`arm` field against the arm recorded inside that file. The recomputation
passed on all ten arm-dataset pairs with zero mismatches, which establishes
that no reported number is affected. The label comparison failed on 12
records, which is this entry.

That script is not committed. Its findings are, here.

---

## D-016. Q1 answered on the test split: format versus learned capability

**Date:** 2026-07-27
**Status:** Preregistered result. Test split, single evaluation.

All three parsers return **identical** macro-F1 to four decimal places on the
fine-tuned arms, on both datasets:

| Arm | DermaMNIST | BloodMNIST |
|---|---|---|
| A4 strict | 0.6165 [0.5726, 0.6572] | 0.9469 [0.9391, 0.9548] |
| A4 lenient | 0.6165 | 0.9469 |
| A4 extraction | 0.6165 | 0.9469 |
| A5 strict | 0.4220 [0.3918, 0.4513] | 0.9346 [0.9257, 0.9437] |
| A5 lenient | 0.4220 | 0.9346 |
| A5 extraction | 0.4220 | 0.9346 |

Zero-shot MedGemma produced a bare canonical label **zero times in 2,715
attempts**. After fine-tuning on bare labels, strict parsing, the method that
scored 0.000, matches the most permissive parser exactly. The format penalty
is eliminated, not reduced.

### Decomposition

| | DermaMNIST | BloodMNIST |
|---|---|---|
| A1 strict, zero-shot | 0.000 | 0.000 |
| A3 constrained, format removed | 0.197 | 0.036 |
| A4 fine-tuned | 0.617 | 0.947 |
| Format component (A1 to A3) | +0.197 (32%) | +0.036 (4%) |
| Learned component (A3 to A4) | +0.420 (68%) | +0.911 (96%) |

**Wording that matters.** A3 to A4 is what fine-tuning added beyond format
handling. It bundles medical discrimination with general task adaptation, and
this design cannot separate them. It is the *learned* component, not the
*medical* component. Calling it medical would overclaim, since A5 gained
substantially on the same axis without medical pretraining.

On DermaMNIST roughly a third of the apparent zero-shot-to-fine-tuned jump
was the measurement rather than the model. On BloodMNIST almost none of it
was, because there was no latent capability for better parsing to reveal.

---

## D-017. Corrections from an independent review of the repository

**Date:** 2026-07-27
**Status:** Corrections to the README. No result changes.

An independent review of the published repository found seven errors in the
README. Six were real and are corrected. One was not.

### Corrected

**1. Compute figure was roughly double the logged value.** The README stated
"roughly 12 GPU-hours, about $24." Summing `gpu_seconds` across every logged
run gives **8.76 GPU-hours and $9.48** at the $1.082 hourly rate. The larger
figure was an estimate of total instance billing, which includes idle time,
model loading, and three training runs killed by instance shutdowns, and it
should not have been presented as compute. Now states the logged figure and
says explicitly what it excludes.

**2. Attempt count belonged to the wrong split.** The claim "zero times in
2,715 attempts" appeared inside a paragraph reporting test-split results.
2,715 is the validation total (1,003 + 1,712); test is 5,426 (2,005 +
3,421). The underlying claim holds on both splits, since strict parsing
scores 100 percent unparseable everywhere, so only the number was wrong.

**3. Preregistration timestamp was five minutes early.** Both `HYPOTHESES.md`
and the README recorded the freeze as 02:23 UTC. That was commit `b2efe04`,
which was amended to correct an author identity. The commit that exists,
`41a958a`, is timestamped **02:28:40 UTC**. The README now carries the
correct time and a note naming `8f5e151`, the single later commit touching
`HYPOTHESES.md`, which replaced a placeholder on the `Date registered:` line
and changed nothing substantive.

`HYPOTHESES.md` itself is not edited; it is frozen and this entry is the
correction of record.

**4. Figure caption overclaimed.** `figures/fig1_arms_*.png` shows A1, A3,
A4, A5, and A6, with a footer reading "All arms shown." The A3 robustness
variants are absent. The README no longer claims every arm appears, and
states where the omitted variants can be found.

**5. `score_arms.py` was missing from the reproduction instructions.** It is
the only script producing the A1, A2, and A2b rows, so following the README
end to end would not have reproduced three arms of the results table.

**6. Deviation count was wrong.** The README said 16 entries against 14
present. Now 17.

### Not corrected, because the review was mistaken

The review stated that arXiv:2509.18015 contains no Gemma 3 control and that
the README mischaracterized it. The review was working from the abstract.

Section 4.5 of that paper, titled "Comparing MedGemma to Gemma," evaluates
Gemma 3 27B against MedGemma 27B and reports zero-shot average hit rates of
17.6 versus 17.7 percent, CoT 25.0 versus 22.5, and few-shot 32.0 versus
25.5. The README description is accurate and stands.

**This strengthens rather than weakens the present study.** That paper found
no MedGemma advantage zero-shot on chest radiograph localization. This study
finds a substantial advantage after matched LoRA fine-tuning on
dermatoscopic images, and none to speak of on blood microscopy. Domain
advantage appearing under some conditions and not others is the shape H3
predicts.

### Method note

Verifying a criticism before acting on it belongs in the same category as
verifying a result. Five of seven items here were caught by someone reading
more carefully than the author; one would have introduced an error if
accepted uncritically.