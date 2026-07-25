"""
test_eval_harness.py

Unit tests for the parsers in src/eval_harness.py.

These are hand-written model outputs paired with the class each one should
resolve to. No model is involved. The point is to confirm the parsing logic
behaves as specified before any real output is scored, because a scoring bug
would corrupt every arm at once and would not announce itself in the numbers.

Run from the repo root:
    python tests/test_eval_harness.py

Caught during Phase 3: BloodMNIST class 3 aliases were singular only, so
"immature granulocytes" scored unparseable. That would have shown up as a
near-zero F1 for class 3 and read as a hematology finding rather than an
alias-table defect.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_harness import UNPARSEABLE as U
from eval_harness import parse_lenient, parse_strict

# (dataset, model_output, expected_strict, expected_lenient, note)
CASES = [
    # clean answers, both parsers agree
    ("dermamnist", "melanoma", 4, 4, "bare canonical label"),
    ("dermamnist", "Melanoma.", 4, 4, "capital and period stripped"),
    ("dermamnist", "**melanoma**", 4, 4, "markdown bold stripped"),
    ("dermamnist", "basal cell carcinoma", 1, 1, "multiword canonical"),
    ("dermamnist", "benign keratosis-like lesions", 2, 2, "hyphenated canonical"),
    ("bloodmnist", "neutrophil", 6, 6, "bare canonical"),
    ("bloodmnist", "platelet", 7, 7, "bare canonical"),
    ("bloodmnist",
     "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
     3, 3, "66-character canonical, exact"),

    # strict fails, lenient recovers. these are the H1 cases.
    ("dermamnist", "The answer is melanoma", U, 4, "prose wrapper"),
    ("dermamnist", "nv", U, 5, "HAM10000 class code"),
    ("dermamnist", "This appears to be a dermatofibroma.", U, 3, "full sentence"),
    ("bloodmnist", "neutrophils", U, 6, "plural via alias"),
    ("bloodmnist", "immature granulocytes", U, 3, "plural of alias"),
    ("bloodmnist", "myelocyte", U, 3, "singular alias"),
    ("bloodmnist", "metamyelocytes", U, 3, "plural alias"),

    # word boundary must not fire on a nested prefix
    ("dermamnist", "melanocytic nevi", 5, 5, "'mel' must not match inside"),

    # ambiguity rule: more than one class named means unparseable
    ("dermamnist", "a melanocytic nevus, not melanoma", U, U, "two classes named"),
    ("dermamnist", "keratosis", U, U, "excluded ambiguous alias"),
    ("dermamnist", "carcinoma", U, U, "excluded ambiguous alias"),
    ("bloodmnist", "granulocyte", U, U, "excluded ambiguous alias"),

    # genuine refusal
    ("dermamnist", "I cannot determine this from the image.", U, U, "refusal"),

    # plural and spelling sweep
    ("dermamnist", "vascular lesion", U, 6, "singular of plural canonical"),
    ("dermamnist", "seborrheic keratoses", U, 2, "plural alias"),
    ("dermamnist", "naevi", U, 5, "British spelling"),
    ("dermamnist", "haemangioma", U, 6, "British spelling"),
    ("dermamnist", "malignant melanoma", U, 4, "containment, inner match same class"),
    ("dermamnist", "melanomas", U, 4, "plural alias"),
    ("bloodmnist", "thrombocytes", U, 7, "plural alias"),
    ("bloodmnist", "band neutrophils", U, 6, "plural compound alias"),
    ("bloodmnist", "nucleated red blood cells", U, 2, "plural multiword alias"),
    ("bloodmnist", "promyelocytes", U, 3, "plural alias"),
]


def main() -> int:
    failures = []

    header = f"{'dataset':11}{'output':46}{'strict':>14}{'lenient':>14}  note"
    print(header)
    print("-" * len(header))

    for dataset, output, exp_strict, exp_lenient, note in CASES:
        got_strict = parse_strict(output, dataset)
        got_lenient = parse_lenient(output, dataset)
        ok = (got_strict == exp_strict) and (got_lenient == exp_lenient)
        if not ok:
            failures.append((dataset, output, exp_strict, got_strict,
                             exp_lenient, got_lenient))
        flag = "" if ok else "  FAIL"
        print(f"{dataset:11}{output[:44]:46}"
              f"{exp_strict}->{got_strict:<10}"
              f"{exp_lenient}->{got_lenient:<10}{flag}  {note}")

    print("-" * len(header))
    print(f"{len(CASES) - len(failures)}/{len(CASES)} cases passed")

    if failures:
        print("\nFailures:")
        for dataset, output, es, gs, el, gl in failures:
            print(f"  [{dataset}] {output!r}")
            print(f"    strict  expected {es}, got {gs}")
            print(f"    lenient expected {el}, got {gl}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
