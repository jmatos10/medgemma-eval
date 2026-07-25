"""
eval_harness.py

Scoring for the MedGemma 1.5 evaluation study.
Built in Phase 3, before any model has been run, so that parsing rules
cannot be adjusted after seeing results.

Frozen alongside HYPOTHESES.md. Changes after Phase 4 begins go in
DEVIATIONS.md with a reason and a timestamp.

Three parsing modes, matching preregistration section D4:
  strict      output must equal a canonical label exactly
  lenient     a canonical label or registered alias appears as a whole word
  constrained decoding restricted to the label set, parse failure impossible

Unparseable predictions are recorded as -1, never silently resolved.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

UNPARSEABLE = -1

# ----------------------------------------------------------------------
# Canonical labels, verbatim from medmnist info['label']
# ----------------------------------------------------------------------

CANONICAL = {
    "dermamnist": [
        "actinic keratoses and intraepithelial carcinoma",
        "basal cell carcinoma",
        "benign keratosis-like lesions",
        "dermatofibroma",
        "melanoma",
        "melanocytic nevi",
        "vascular lesions",
    ],
    "bloodmnist": [
        "basophil",
        "eosinophil",
        "erythroblast",
        "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet",
    ],
}

# ----------------------------------------------------------------------
# Alias table, frozen before Phase 4.
#
# Deliberately absent because they match more than one class:
#   keratosis, lesions, carcinoma, granulocyte, ig
# Excluding them means an ambiguous response scores as a parse failure
# rather than being resolved to a guess.
# ----------------------------------------------------------------------

ALIASES = {
    "dermamnist": {
        0: ["actinic keratosis", "actinic keratoses", "akiec",
            "bowen disease", "bowens disease", "bowen's disease",
            "intraepithelial carcinoma", "intraepithelial carcinomas"],
        1: ["bcc", "basal cell", "basal cell carcinomas"],
        2: ["benign keratosis", "benign keratoses", "bkl",
            "benign keratosis-like lesion",
            "seborrheic keratosis", "seborrheic keratoses",
            "seborrhoeic keratosis", "seborrhoeic keratoses",
            "solar lentigo", "solar lentigines"],
        3: ["df", "dermatofibromas"],
        4: ["mel", "melanomas", "malignant melanoma", "malignant melanomas"],
        5: ["melanocytic nevus", "melanocytic naevus", "melanocytic naevi",
            "nevus", "nevi", "naevus", "naevi", "nv", "mole", "moles"],
        6: ["vasc", "vascular lesion",
            "angioma", "angiomas", "hemangioma", "hemangiomas",
            "haemangioma", "haemangiomas",
            "pyogenic granuloma", "pyogenic granulomas"],
    },
    "bloodmnist": {
        0: ["basophils", "baso"],
        1: ["eosinophils", "eos"],
        2: ["erythroblasts", "normoblast", "normoblasts",
            "nucleated red blood cell", "nucleated red blood cells",
            "nrbc", "nrbcs"],
        3: ["immature granulocyte", "immature granulocytes",
            "myelocyte", "myelocytes",
            "metamyelocyte", "metamyelocytes",
            "promyelocyte", "promyelocytes"],
        4: ["lymphocytes", "lymph"],
        5: ["monocytes", "mono"],
        6: ["neutrophils", "neut",
            "segmented neutrophil", "segmented neutrophils",
            "band neutrophil", "band neutrophils"],
        7: ["platelets", "thrombocyte", "thrombocytes"],
    },
}


def prompt_for(dataset: str) -> str:
    """The single prompt template, identical across every arm."""
    subject = {
        "dermamnist": "skin lesion",
        "bloodmnist": "blood cell",
    }[dataset]
    options = "\n".join(CANONICAL[dataset])
    return (
        f"Which of the following best describes this {subject} image?\n\n"
        f"Options:\n{options}\n\n"
        f"Answer with exactly one option from the list above."
    )


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation.

    Applied identically to model output and to canonical labels so that
    trivial differences in spacing or a trailing period do not count as
    medical errors.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n.:;!\"'`*-")
    return text


def _whole_word_spans(haystack: str, needle: str):
    """Character spans where `needle` appears in `haystack` as whole words.

    Word boundaries matter here. Plain substring matching would find "mel"
    inside "melanocytic" and misroute a correct nevus answer to melanoma.
    """
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return [m.span() for m in re.finditer(pattern, haystack)]


# ----------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------

def parse_strict(output: str, dataset: str) -> int:
    """Exact match against a canonical label, after normalization only.

    This reproduces the evaluation method used in public tutorials and is
    included as a negative control, not as a good-faith baseline.
    """
    norm = normalize(output)
    for idx, label in enumerate(CANONICAL[dataset]):
        if norm == normalize(label):
            return idx
    return UNPARSEABLE


def parse_lenient(output: str, dataset: str) -> int:
    """Whole-word match against canonical labels and registered aliases.

    Rules, fixed in Phase 3:
      1. Longest match wins, so a short alias nested inside a longer match
         does not shadow it.
      2. If surviving matches point at more than one class, the response is
         unparseable rather than wrong. Resolving it to a guess would hide
         the format artifact that H1 exists to measure.
    """
    norm = normalize(output)

    # Collect every candidate match as (start, end, class_index).
    hits = []
    for idx, label in enumerate(CANONICAL[dataset]):
        for term in [normalize(label)] + ALIASES[dataset].get(idx, []):
            for start, end in _whole_word_spans(norm, normalize(term)):
                hits.append((start, end, idx))

    if not hits:
        return UNPARSEABLE

    # Rule 1: discard any match fully contained inside a longer one.
    surviving = []
    for start, end, idx in hits:
        contained = any(
            (o_start <= start and end <= o_end) and (o_end - o_start) > (end - start)
            for o_start, o_end, _ in hits
        )
        if not contained:
            surviving.append(idx)

    classes = set(surviving)
    if len(classes) == 1:
        return classes.pop()

    # Rule 2: zero or several distinct classes both count as a parse failure.
    return UNPARSEABLE


def parse_constrained(selected_index: int, dataset: str) -> int:
    """Validate an index chosen by constrained decoding.

    Decoding is restricted to the label set, so the model picks among valid
    options and a parse failure cannot occur. This function only guards
    against an out-of-range index, which would signal a bug in the decoder
    rather than a model error.
    """
    n = len(CANONICAL[dataset])
    if not isinstance(selected_index, (int, np.integer)):
        raise TypeError("constrained decoding must return an integer index")
    if not 0 <= int(selected_index) < n:
        raise ValueError(f"index {selected_index} outside 0..{n - 1} for {dataset}")
    return int(selected_index)


PARSERS = {
    "strict": parse_strict,
    "lenient": parse_lenient,
}


# ----------------------------------------------------------------------
# Exploratory, added Phase 4. NOT preregistered. See DEVIATIONS.md.
#
# The 50-image pilot showed MedGemma states its answer plainly and then
# enumerates the remaining classes in order to reject them. The frozen
# lenient rule sees several distinct classes and returns UNPARSEABLE, so a
# response a human reads without difficulty scores as a parse failure.
#
# A2b measures what the model communicated rather than whether it avoided
# listing alternatives. The frozen definitions in D4 are untouched.
# ----------------------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def extract_answer_span(output: str) -> str:
    """Isolate the span where the model states its conclusion.

    Rule 1: the first markdown bold span. The enumerated bullets are also
    bold, but they follow the answer sentence, so the first one is the
    model's own stated conclusion.

    Rule 2: no bold present, fall back to the first non-empty line.

    Fixed before this function was ever scored, so it cannot be tuned to
    flatter the result.
    """
    match = _BOLD.search(output)
    if match:
        return match.group(1)
    for line in output.splitlines():
        if line.strip():
            return line
    return ""


def parse_extracted(output: str, dataset: str) -> int:
    """Lenient matching applied to the extracted answer span only.

    Same alias table and same ambiguity rule as parse_lenient. The only
    difference is how much of the response is considered.
    """
    span = extract_answer_span(output)
    if not span.strip():
        return UNPARSEABLE
    return parse_lenient(span, dataset)


EXPLORATORY_PARSERS = {
    "extracted": parse_extracted,
}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_metrics(y_true, y_pred, dataset: str) -> dict:
    """Primary and secondary outcomes from preregistration sections D2 and D3.

    Unparseable predictions stay as -1 and are counted as errors. They are
    never dropped, because dropping them would let a model that answers
    almost nothing score well on the few items it did answer.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n_classes = len(CANONICAL[dataset])
    labels = list(range(n_classes))

    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": [
            float(v) for v in
            f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        ],
        "unparseable_rate": float((y_pred == UNPARSEABLE).mean()),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels + [UNPARSEABLE]
        ).tolist(),
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, y_pred, dataset: str, metric: str = "macro_f1",
                 n_boot: int = 1000, seed: int = 42):
    """Percentile bootstrap interval for a single arm.

    Resamples the test set with replacement `n_boot` times and reports the
    2.5th and 97.5th percentiles. This answers "how much would this number
    move if we had drawn a different test set of the same size."
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rng = np.random.default_rng(seed)
    n = len(y_true)

    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws.append(compute_metrics(y_true[idx], y_pred[idx], dataset)[metric])

    point = compute_metrics(y_true, y_pred, dataset)[metric]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_difference(y_true, y_pred_a, y_pred_b, dataset: str,
                         metric: str = "macro_f1", n_boot: int = 1000,
                         seed: int = 42):
    """Paired bootstrap for the gap between two arms.

    Both arms are scored on the same resampled indices each iteration. The
    pairing matters: two arms evaluated on independent resamples would show
    a wider interval than the real uncertainty in their difference.

    Decision rule from preregistration section E3: the difference is treated
    as real when the interval excludes zero.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_a = np.asarray(y_pred_a).ravel()
    y_pred_b = np.asarray(y_pred_b).ravel()
    rng = np.random.default_rng(seed)
    n = len(y_true)

    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = compute_metrics(y_true[idx], y_pred_a[idx], dataset)[metric]
        b = compute_metrics(y_true[idx], y_pred_b[idx], dataset)[metric]
        draws.append(a - b)

    point = (compute_metrics(y_true, y_pred_a, dataset)[metric]
             - compute_metrics(y_true, y_pred_b, dataset)[metric])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "difference": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def append_result(record: dict, path: str = "results/results.jsonl") -> None:
    """Append one run to the log. Never rewrite an existing line.

    Required keys, per CLAUDE.md hard rule 4: arm, model, revision, dataset,
    split, seed, gpu_seconds, plus the metrics dict.
    """
    record = dict(record)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
