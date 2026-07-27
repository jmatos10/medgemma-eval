"""
score_arms.py

Turns a raw output file into logged metrics. Reads `results/raw/*.jsonl`,
applies every parser, computes the preregistered outcomes with bootstrap
confidence intervals, and appends one record per arm to
`results/results.jsonl`.

No GPU. No model. This runs entirely off saved text, which is the point of
hard rule 4: re-scoring must never require re-running inference.

Handles two file shapes, detected automatically:

  generation files  have `output_text`   -> arms A1, A2, A2b
  scoring files     have `logp_mean`     -> arm A3 and its robustness variants

Usage, from the repo root:
    python scripts/score_arms.py results/raw/gen_dermamnist_val.jsonl
    python scripts/score_arms.py results/raw/A3_dermamnist_val.jsonl

Add --gpu-seconds to record what the inference run cost, since this script
cannot know it.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_harness import (  # noqa: E402
    UNPARSEABLE,
    append_result,
    bootstrap_ci,
    bootstrap_difference,
    compute_metrics,
    parse_extracted,
    parse_lenient,
    parse_strict,
)

L4_HOURLY_USD = 1.082

# arm id -> (parser, preregistered?)
GENERATION_ARMS = [
    ("A1", "strict", parse_strict, True),
    ("A2", "lenient", parse_lenient, True),
    ("A2b", "extraction", parse_extracted, False),
]

# arm id -> (record key, scoring name, primary?)
SCORING_ARMS = [
    ("A3", "pred_mean", "mean logp per token", True),
    ("A3-sum", "pred_sum", "raw logp sum", False),
    ("A3-pmi", "pred_pmi", "pointwise mutual information", False),
]


def load(path: Path):
    records = [json.loads(line) for line in path.open() if line.strip()]
    if not records:
        raise SystemExit(f"{path} is empty")
    return records


def majority_baseline(y):
    """Accuracy and macro-F1 of always answering the most common true class.

    Any model must beat this to have demonstrated anything. On DermaMNIST a
    constant predictor scores 67 percent accuracy while carrying no
    diagnostic information, which is why macro-F1 is the primary outcome.
    """
    counts = Counter(y)
    top = counts.most_common(1)[0][0]
    return top, counts[top] / len(y)


def report(arm, label, y_true, y_pred, dataset, meta, preregistered,
           n_boot, seed):
    m = compute_metrics(y_true, y_pred, dataset)
    ci = bootstrap_ci(y_true, y_pred, dataset, "macro_f1",
                      n_boot=n_boot, seed=seed)

    flag = "" if preregistered else "  [exploratory]"
    print(f"\n{arm}  {label}{flag}")
    print(f"  macro-F1     {m['macro_f1']:.3f}  "
          f"95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
    print(f"  accuracy     {m['accuracy']:.3f}")
    print(f"  unparseable  {m['unparseable_rate']:.3f}")
    print(f"  per-class F1 {[round(v, 2) for v in m['per_class_f1']]}")
    print(f"  predictions  {dict(sorted(Counter(y_pred).items()))}")

    record = {
        "arm": arm,
        "treatment": label,
        "preregistered": preregistered,
        **meta,
        **m,
        "macro_f1_ci_low": ci["ci_low"],
        "macro_f1_ci_high": ci["ci_high"],
        "n_bootstrap": n_boot,
    }
    append_result(record, str(REPO / "results" / "results.jsonl"))
    return m, y_pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a file under results/raw/")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-seconds", type=float, default=None,
                    help="inference cost of the run that produced this file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print metrics without appending to results.jsonl")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    records = load(path)
    first = records[0]
    dataset = first["dataset"]
    split = first["split"]
    y_true = [r["true_label"] for r in records]

    if "output_text" in first:
        kind = "generation"          # A1, A2, A2b share one generation pass
    elif "pred_class" in first:
        kind = "direct"              # A6, a classifier emits a class index
    else:
        kind = "scoring"             # A3 and its normalization variants

    print(f"file      {path.name}")
    print(f"kind      {kind}")
    print(f"dataset   {dataset} [{split}]")
    print(f"n         {len(records)}")
    print(f"model     {first.get('model')}")
    print(f"revision  {first.get('revision')}")

    top_class, top_share = majority_baseline(y_true)
    print(f"\ntruth spread {dict(sorted(Counter(y_true).items()))}")
    print(f"majority baseline: class {top_class}, accuracy {top_share:.3f}")

    if split == "test":
        print("\n  Scoring the TEST split. Hard rule 1 allows this exactly")
        print("  once, after all arms are complete. If development is still")
        print("  in progress, stop.")

    meta = {
        "dataset": dataset,
        "split": split,
        "model": first.get("model"),
        "revision": first.get("revision"),
        "n": len(records),
        "seed": args.seed,
        "source_file": path.name,
    }
    if args.gpu_seconds is not None:
        meta["gpu_seconds"] = args.gpu_seconds
        meta["est_usd"] = round(args.gpu_seconds / 3600 * L4_HOURLY_USD, 4)

    if args.dry_run:
        global append_result

        def append_result(*_a, **_k):  # noqa: F811
            return None
        print("\n(dry run, nothing will be written to results.jsonl)")

    preds = {}

    if kind == "generation":
        # A fine-tuned file carries its own arm id (A4, A5). Zero-shot files
        # written by pilot_lengths.py do not, and are the A1/A2/A2b source.
        #
        # Without this check every file containing output_text would be
        # logged as A1, A2, and A2b, so A4 and A5 would overwrite the
        # zero-shot records or be overwritten by them depending on filename
        # order. Both outcomes silently corrupt results.jsonl.
        arm_id = first.get("arm")

        if arm_id:
            variants = [
                (arm_id, "strict parsing", parse_strict),
                (f"{arm_id}-lenient", "lenient parsing", parse_lenient),
                (f"{arm_id}-extract", "answer extraction", parse_extracted),
            ]
            for name, label, fn in variants:
                y_pred = [fn(r["output_text"], dataset) for r in records]
                report(name, label, y_true, y_pred, dataset, meta, True,
                       args.n_boot, args.seed)
                preds[name] = y_pred

            base = preds.get(arm_id)
            same = all(preds[v[0]] == base for v in variants)
            print(f"\n  All three parsers agree on every item: {same}")
            print("  Convergence means the format penalty is gone and every")
            print("  remaining difference between arms is medical. That is")
            print("  the answer to Q1.")
        else:
            for arm, label, fn, prereg in GENERATION_ARMS:
                y_pred = [fn(r["output_text"], dataset) for r in records]
                report(arm, label, y_true, y_pred, dataset, meta, prereg,
                       args.n_boot, args.seed)
                preds[arm] = y_pred

        # H1 is about exactly this gap. Only meaningful for the zero-shot
        # file, where A1 and A2b both exist.
        if "A1" in preds and "A2b" in preds:
            d = bootstrap_difference(y_true, preds["A2b"], preds["A1"],
                                     dataset, "macro_f1",
                                     n_boot=args.n_boot, seed=args.seed)
            print(f"\nA2b minus A1 macro-F1: {d['difference']:+.3f}  "
                  f"95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
            print(f"  excludes zero: {d['excludes_zero']}")
            print("  H1 predicted at least +0.10 from format handling alone.")
    elif kind == "direct":
        # A classifier produces a class index directly. No parsing, no
        # normalization, no possibility of an unparseable answer. Routed
        # through this script rather than scored inline so that A6 carries
        # the same bootstrap intervals as every other arm in the table.
        arm = first.get("arm", "A6")
        y_pred = [r["pred_class"] for r in records]
        report(arm, "direct classification", y_true, y_pred, dataset, meta,
               True, args.n_boot, args.seed)
        preds[arm] = y_pred
    else:
        for arm, key, label, primary in SCORING_ARMS:
            if key not in first:
                continue
            y_pred = [r[key] for r in records]
            star = " (primary, prespecified)" if primary else ""
            report(arm, label + star, y_true, y_pred, dataset, meta, True,
                   args.n_boot, args.seed)
            preds[arm] = y_pred

        keys = [k for k in ("A3", "A3-sum", "A3-pmi") if k in preds]
        if len(keys) == 3:
            agree = sum(
                1 for i in range(len(records))
                if preds["A3"][i] == preds["A3-sum"][i] == preds["A3-pmi"][i]
            )
            print(f"\nAll three scorings agree on {agree}/{len(records)}")
            print("  Disagreement means the normalization choice changes the")
            print("  answer. Report as a robustness finding, and note that")
            print("  the primary was fixed before any run.")

    unparseable_arms = [
        a for a, p in preds.items()
        if sum(1 for v in p if v == UNPARSEABLE) == len(p)
    ]
    if unparseable_arms:
        print(f"\n  {unparseable_arms} scored 100 percent unparseable.")
        print("  Expected for A1. Investigate for any other arm.")

    if not args.dry_run:
        print(f"\nAppended {len(preds)} records to results/results.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())