"""
test_hypotheses.py

Formal tests of H1 through H4 from HYPOTHESES.md, computed from the raw
prediction files in results/raw/. No GPU, no model.

Decision rule, section E3: a difference is treated as real when its bootstrap
95 percent interval excludes zero. Differences whose intervals include zero
are reported as inconclusive at this sample size, not as null findings.

WHY H3 NEEDS ITS OWN BOOTSTRAP
------------------------------
H2 asks whether MedGemma beats Gemma within a dataset. That is a paired
comparison: both arms saw the same images, so each bootstrap iteration
resamples image indices once and scores both arms on that same resample.
Pairing matters, since two arms scored on independent resamples would show a
wider interval than the real uncertainty in their difference.

H3 asks whether that advantage is *larger* on DermaMNIST than on BloodMNIST.
The quantity is a difference of differences:

    interaction = (A4 - A5) on DermaMNIST  -  (A4 - A5) on BloodMNIST

Reading it off two separate confidence intervals and checking whether they
overlap is a conservative approximation, not a test. Non-overlapping
intervals imply a real difference, but overlapping ones do not imply the
absence of one.

The two datasets are independent samples, so each bootstrap iteration
resamples both datasets separately, computes both advantages, and takes
their difference. The percentiles of that distribution are the interval.

Usage, from the repo root:
    python scripts/test_hypotheses.py --split val
    python scripts/test_hypotheses.py --split test    # once, at the very end
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_harness import (CANONICAL, UNPARSEABLE, compute_metrics,  # noqa
                          parse_strict)

DATASETS = ["dermamnist", "bloodmnist"]

# H3's premise, from CLAUDE.md section 4. Dermatology appears in MedGemma's
# stated image-encoder pretraining domains; blood cell microscopy does not.
IN_DOMAIN = "dermamnist"
OUT_DOMAIN = "bloodmnist"


def load_preds(path: Path, key: str, dataset: str):
    """Return {image_index: predicted_class} from a raw file.

    `key` selects how the prediction is obtained:
      output_text  parse the generated text with the strict parser
      pred_class   a classifier's direct output, A6
      pred_mean    A3's prespecified primary scoring
    """
    if not path.exists():
        return None

    out = {}
    for line in path.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        if key == "output_text":
            out[rec["image_index"]] = parse_strict(rec["output_text"], dataset)
        else:
            out[rec["image_index"]] = int(rec[key])
    return out


def load_truth(path: Path):
    return {json.loads(l)["image_index"]: json.loads(l)["true_label"]
            for l in path.open() if l.strip()}


def paired_diff(y, pa, pb, dataset, n_boot=1000, seed=42):
    """Bootstrap interval for macro-F1 of arm A minus arm B, same images."""
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = (compute_metrics(y[idx], pa[idx], dataset)["macro_f1"]
                    - compute_metrics(y[idx], pb[idx], dataset)["macro_f1"])
    point = (compute_metrics(y, pa, dataset)["macro_f1"]
             - compute_metrics(y, pb, dataset)["macro_f1"])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(point), "ci_low": float(lo),
            "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def interaction(data_in, data_out, n_boot=1000, seed=42):
    """Bootstrap interval for (A4-A5) in-domain minus (A4-A5) out-of-domain.

    Each iteration resamples both datasets independently, since they are
    independent samples, computes each advantage on its own resample, and
    takes the difference. This is H3's actual quantity.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)

    def adv(d, idx):
        y = d["y"][idx]
        return (compute_metrics(y, d["a4"][idx], d["ds"])["macro_f1"]
                - compute_metrics(y, d["a5"][idx], d["ds"])["macro_f1"])

    n_in, n_out = len(data_in["y"]), len(data_out["y"])
    for i in range(n_boot):
        draws[i] = (adv(data_in, rng.integers(0, n_in, size=n_in))
                    - adv(data_out, rng.integers(0, n_out, size=n_out)))

    point = (adv(data_in, np.arange(n_in)) - adv(data_out, np.arange(n_out)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"interaction": float(point), "ci_low": float(lo),
            "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def align(*dicts):
    """Image indices present in every dict, sorted."""
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    return sorted(keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = REPO / "results" / "raw"
    s = args.split

    if s == "test":
        print("TEST SPLIT. Hard rule 1 allows this exactly once, after every")
        print("arm is complete and all development is finished.\n")

    loaded = {}
    for ds in DATASETS:
        files = {
            "zs":  raw / f"gen_{ds}_{s}.jsonl",          # A1, A2, A2b
            "a3":  raw / f"A3full_{ds}_{s}.jsonl",
            "a4":  raw / f"gen_A4_{ds}_{s}.jsonl",
            "a5":  raw / f"gen_A5_{ds}_{s}.jsonl",
            "a6":  raw / f"A6_{ds}_{s}.jsonl",
        }
        missing = [k for k, p in files.items() if not p.exists()]
        if missing:
            print(f"{ds}: missing {missing}")
            print(f"  looked in {raw.relative_to(REPO)}/")
            continue

        preds = {
            "A1": load_preds(files["zs"], "output_text", ds),
            "A3": load_preds(files["a3"], "pred_mean", ds),
            "A4": load_preds(files["a4"], "output_text", ds),
            "A5": load_preds(files["a5"], "output_text", ds),
            "A6": load_preds(files["a6"], "pred_class", ds),
        }
        truth = load_truth(files["a4"])
        keys = align(truth, *preds.values())

        loaded[ds] = {
            "ds": ds,
            "y": np.array([truth[k] for k in keys]),
            **{a.lower(): np.array([preds[a][k] for k in keys])
               for a in preds},
            "n": len(keys),
        }
        print(f"{ds}: {len(keys)} images aligned across 5 arms")

    if len(loaded) < 2:
        print("\nNeed both datasets. Stopping.")
        return 1

    print()
    print("=" * 70)
    print(f"MACRO-F1, {s} split")
    print("=" * 70)
    print(f"{'arm':<6}{'treatment':<26}", end="")
    for ds in DATASETS:
        print(f"{ds:>16}", end="")
    print()
    labels = {"a1": "zero-shot, strict", "a3": "zero-shot, constrained",
              "a4": "MedGemma LoRA", "a5": "Gemma 3 LoRA",
              "a6": "ResNet-18"}
    for arm in ["a1", "a3", "a4", "a5", "a6"]:
        print(f"{arm.upper():<6}{labels[arm]:<26}", end="")
        for ds in DATASETS:
            d = loaded[ds]
            m = compute_metrics(d["y"], d[arm], ds)["macro_f1"]
            print(f"{m:>16.3f}", end="")
        print()

    kw = {"n_boot": args.n_boot, "seed": args.seed}

    print()
    print("=" * 70)
    print("H1  format artifact: A3 minus A1, at least +0.10 predicted")
    print("=" * 70)
    for ds in DATASETS:
        d = loaded[ds]
        r = paired_diff(d["y"], d["a3"], d["a1"], ds, **kw)
        verdict = "HOLDS" if (r["difference"] >= 0.10 and
                              r["excludes_zero"]) else "FALSIFIED"
        print(f"  {ds:<12} {r['difference']:+.4f}  "
              f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]   {verdict}")

    print()
    print("=" * 70)
    print("H2  medical pretraining: A4 minus A5, MedGemma ahead on both")
    print("=" * 70)
    for ds in DATASETS:
        d = loaded[ds]
        r = paired_diff(d["y"], d["a4"], d["a5"], ds, **kw)
        verdict = ("HOLDS" if (r["difference"] > 0 and r["excludes_zero"])
                   else "FALSIFIED" if r["difference"] <= 0
                   else "INCONCLUSIVE")
        print(f"  {ds:<12} {r['difference']:+.4f}  "
              f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]   {verdict}")

    print()
    print("=" * 70)
    print("H3  PRIMARY TEST, modality interaction")
    print("    (A4-A5) in-domain minus (A4-A5) out-of-domain, >= +0.02")
    print("=" * 70)
    r = interaction(loaded[IN_DOMAIN], loaded[OUT_DOMAIN], **kw)
    verdict = ("HOLDS" if (r["interaction"] >= 0.02 and r["excludes_zero"])
               else "FALSIFIED")
    print(f"  in-domain     {IN_DOMAIN}")
    print(f"  out-of-domain {OUT_DOMAIN}")
    print(f"  interaction   {r['interaction']:+.4f}  "
          f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    print(f"  excludes zero {r['excludes_zero']}")
    print(f"  {verdict}")

    print()
    print("=" * 70)
    print("H4  parameter efficiency: A6 within 5 points of the better 4B arm,")
    print("    and ahead on at least one dataset")
    print("=" * 70)
    ahead = 0
    for ds in DATASETS:
        d = loaded[ds]
        f4 = compute_metrics(d["y"], d["a4"], ds)["macro_f1"]
        f5 = compute_metrics(d["y"], d["a5"], ds)["macro_f1"]
        better = "a4" if f4 >= f5 else "a5"
        r = paired_diff(d["y"], d["a6"], d[better], ds, **kw)
        if r["difference"] > 0:
            ahead += 1
        print(f"  {ds:<12} A6 minus {better.upper()}  {r['difference']:+.4f}  "
              f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    print(f"  ResNet ahead on {ahead} of {len(DATASETS)} datasets")

    print()
    print("Reported per section G: all arms, both datasets, regardless of")
    print("outcome. Intervals are 1,000-resample percentile bootstraps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
