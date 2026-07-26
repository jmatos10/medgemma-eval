"""
subsample.py

Draws the fixed training subsample required by HYPOTHESES.md section C3:
5,000 images per dataset, seed 42, drawn once and reused identically across
A4 (MedGemma LoRA), A5 (Gemma 3 LoRA), and A6 (ResNet-18).

Writes `subsamples/{dataset}_train_5000.json`, which holds the selected
indices and a SHA-256 of them. That file is committed. It is the evidence
behind section B3's claim that the three arms saw the same data, rather than
something a reader has to take on trust.

Training scripts call `load_subsample()`, which re-derives the hash and
refuses to proceed if it does not match. A silently altered index file
cannot slip into a run.

WHY SIMPLE RANDOM AND NOT STRATIFIED
------------------------------------
DermaMNIST train is 67 percent melanocytic nevi, and validation and test
carry the same imbalance. Stratifying the training subsample to balance
classes would train on one distribution and evaluate on another, so the
model's learned prior would no longer match the distribution it is scored
against. That is a confound in every downstream result.

Simple random preserves the natural distribution. The cost is that rare
classes stay rare: DermaMNIST class 3 has 80 instances in 7,007, so roughly
57 land in the sample. Those classes will train poorly. That is a property
of the task, not an artifact of the sampling.

Usage, from the repo root:
    python scripts/subsample.py                 # draw both datasets
    python scripts/subsample.py --verify        # check existing files only

No GPU. Run with the instance stopped.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_harness import CANONICAL  # noqa: E402

DATASET_CLASSES = {"dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}

N_SUBSAMPLE = 5000
SEED = 42
SUBSAMPLE_DIR = REPO / "subsamples"


def index_hash(indices) -> str:
    """SHA-256 of the sorted index list.

    A hash rather than a length check, because two different 5,000-image
    draws both have 5,000 entries. Only the hash detects a different draw.
    """
    payload = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha256(payload.encode()).hexdigest()


def subsample_path(dataset: str) -> Path:
    return SUBSAMPLE_DIR / f"{dataset}_train_{N_SUBSAMPLE}.json"


def draw(dataset: str, n: int = N_SUBSAMPLE, seed: int = SEED) -> dict:
    """Draw the subsample and return the record to be written."""
    import medmnist

    cls = getattr(medmnist, DATASET_CLASSES[dataset])
    ds = cls(split="train", size=224, download=False)

    total = len(ds.imgs)
    if n > total:
        raise ValueError(f"{dataset} train has {total} images, cannot draw {n}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=False)
    idx.sort()

    labels = [int(ds.labels[i][0]) for i in idx]
    full_labels = [int(ds.labels[i][0]) for i in range(total)]

    return {
        "dataset": dataset,
        "split": "train",
        "n_requested": n,
        "n_selected": int(len(idx)),
        "seed": seed,
        "full_split_size": total,
        "sha256": index_hash(idx),
        "class_counts": {str(k): v for k, v in sorted(Counter(labels).items())},
        "full_class_counts": {
            str(k): v for k, v in sorted(Counter(full_labels).items())
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "indices": [int(i) for i in idx],
    }


def load_subsample(dataset: str) -> list:
    """Return the frozen index list, verifying it has not been altered.

    Training scripts call this rather than re-drawing. Re-drawing would work
    today and would silently diverge if the seed, the library version, or
    the draw order ever changed.
    """
    path = subsample_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python scripts/subsample.py"
        )

    rec = json.loads(path.read_text())
    indices = rec["indices"]

    actual = index_hash(indices)
    if actual != rec["sha256"]:
        raise ValueError(
            f"{path.name} hash mismatch.\n"
            f"  recorded {rec['sha256']}\n"
            f"  actual   {actual}\n"
            "The index list was modified after it was written. Do not train "
            "on this. Restore the committed version from git."
        )

    if len(set(indices)) != len(indices):
        raise ValueError(f"{path.name} contains duplicate indices")

    return indices


def describe(rec: dict) -> None:
    """Print the subsample against the full split so skew is visible."""
    dataset = rec["dataset"]
    names = CANONICAL[dataset]
    sub = rec["class_counts"]
    full = rec["full_class_counts"]
    n_sub = rec["n_selected"]
    n_full = rec["full_split_size"]

    print(f"\n{dataset}: {n_sub} of {n_full} train images")
    print(f"  sha256 {rec['sha256'][:16]}...")
    print(f"\n  {'class':>5}  {'subsample':>18}  {'full split':>18}  label")
    print("  " + "-" * 78)
    for k in range(len(names)):
        s = sub.get(str(k), 0)
        f = full.get(str(k), 0)
        print(f"  {k:>5}  {s:>7} ({100*s/n_sub:5.1f}%)  "
              f"{f:>7} ({100*f/n_full:5.1f}%)  {names[k][:38]}")

    # Largest deviation between subsample and full split proportions.
    drift = max(
        abs(sub.get(str(k), 0) / n_sub - full.get(str(k), 0) / n_full)
        for k in range(len(names))
    )
    print(f"\n  max class-proportion drift: {100*drift:.2f} percentage points")
    if drift > 0.02:
        print("  Above 2 points. Unusual for a 5,000 draw. Investigate.")

    rare = [k for k in range(len(names)) if sub.get(str(k), 0) < 100]
    if rare:
        print(f"  classes with under 100 training examples: {rare}")
        print("  Expect poor per-class F1 on these. Preserved deliberately,")
        print("  see the module docstring on stratification.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check existing files without redrawing")
    ap.add_argument("--force", action="store_true",
                    help="redraw even if a file already exists")
    args = ap.parse_args()

    SUBSAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        ok = True
        for dataset in DATASET_CLASSES:
            path = subsample_path(dataset)
            try:
                indices = load_subsample(dataset)
                rec = json.loads(path.read_text())
                print(f"{path.name}: OK, {len(indices)} indices, "
                      f"sha256 {rec['sha256'][:16]}...")
            except (FileNotFoundError, ValueError) as exc:
                print(f"{path.name}: FAILED\n  {exc}")
                ok = False
        return 0 if ok else 1

    for dataset in DATASET_CLASSES:
        path = subsample_path(dataset)

        if path.exists() and not args.force:
            print(f"\n{path.name} already exists. Not redrawing.")
            print("  This file is frozen once training has begun. Use --force")
            print("  only before any arm has trained, and log it in")
            print("  DEVIATIONS.md if any has.")
            rec = json.loads(path.read_text())
            describe(rec)
            continue

        rec = draw(dataset)
        path.write_text(json.dumps(rec, indent=2))
        describe(rec)
        print(f"\n  wrote {path.relative_to(REPO)}")

    print("\nVerifying what was written...")
    for dataset in DATASET_CLASSES:
        indices = load_subsample(dataset)
        print(f"  {dataset}: {len(indices)} indices, hash verified")

    print("\nCommit these files. They are the evidence that A4, A5, and A6")
    print("trained on identical data, which section B3 requires.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
