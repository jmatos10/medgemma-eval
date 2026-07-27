"""
make_figures.py

Two figures for the writeup, generated from the raw prediction files.

WHY THIS WAS WRITTEN BEFORE THE TEST NUMBERS EXISTED
----------------------------------------------------
Chart choices are as tunable as metric choices, and less visible to a reader.
A y-axis that starts at 0.4 instead of 0 doubles the apparent size of every
gap. Dropping error bars makes an inconclusive difference look decisive.
Showing four arms instead of six lets the weakest comparison disappear.

None of those are detectable from the finished image. So the choices are
fixed here, before any test result has been seen:

  1. **Y-axis starts at zero.** Always. No truncated axes.
  2. **Every arm appears.** No arm is dropped for underperforming, per
     section G.
  3. **Bootstrap 95 percent intervals on every bar.** 1,000 resamples, the
     same procedure as section E2.
  4. **The majority-class baseline is drawn on every panel.** A number
     without its floor is not interpretable.
  5. **Identical scale across datasets** so the two panels are comparable
     by eye.

Figures are computed from results/raw/, the same files test_hypotheses.py
reads, so a figure and a reported number cannot disagree.

Usage, from the repo root:
    python scripts/make_figures.py --split val
    python scripts/make_figures.py --split test
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_harness import CANONICAL, compute_metrics, parse_strict  # noqa

DATASETS = ["dermamnist", "bloodmnist"]
DATASET_LABELS = {
    "dermamnist": "DermaMNIST\n(in MedGemma pretraining domains)",
    "bloodmnist": "BloodMNIST\n(outside pretraining domains)",
}

# Order is fixed: zero-shot first, then fine-tuned, then the CNN. This is
# the narrative order of the study and does not change with the results.
ARMS = [
    ("A1", "zs",  "output_text", "Zero-shot\nstrict"),
    ("A3", "a3",  "pred_mean",   "Zero-shot\nconstrained"),
    ("A4", "a4",  "output_text", "MedGemma\nLoRA"),
    ("A5", "a5",  "output_text", "Gemma 3\nLoRA"),
    ("A6", "a6",  "pred_class",  "ResNet-18\n11M params"),
]

# Colorblind-safe. Grey for zero-shot, blue for the medical model, orange
# for its control, green for the CNN.
COLORS = {
    "A1": "#999999",
    "A3": "#666666",
    "A4": "#0072B2",
    "A5": "#E69F00",
    "A6": "#009E73",
}


def file_for(raw: Path, key: str, dataset: str, split: str) -> Path:
    return {
        "zs": raw / f"gen_{dataset}_{split}.jsonl",
        "a3": raw / f"A3full_{dataset}_{split}.jsonl",
        "a4": raw / f"gen_A4_{dataset}_{split}.jsonl",
        "a5": raw / f"gen_A5_{dataset}_{split}.jsonl",
        "a6": raw / f"A6_{dataset}_{split}.jsonl",
    }[key]


def load(path: Path, field: str, dataset: str):
    """Return {image_index: (true_label, predicted_class)}."""
    out = {}
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        pred = (parse_strict(r["output_text"], dataset)
                if field == "output_text" else int(r[field]))
        out[r["image_index"]] = (r["true_label"], pred)
    return out


def boot_ci(y, p, dataset, n_boot=1000, seed=42):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = compute_metrics(y[idx], p[idx], dataset)["macro_f1"]
    point = compute_metrics(y, p, dataset)["macro_f1"]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def majority_macro_f1(y, dataset):
    """Macro-F1 of always answering the most common class."""
    top = Counter(y).most_common(1)[0][0]
    return compute_metrics(y, [top] * len(y), dataset)["macro_f1"]


def gather(raw: Path, split: str, n_boot: int):
    """Point estimates and intervals for every arm on every dataset."""
    data = {}
    for ds in DATASETS:
        paths = {k: file_for(raw, k, ds, split) for _, k, _, _ in ARMS}
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            print(f"{ds}: missing {missing}, skipping")
            continue

        loaded = {arm: load(paths[k], f, ds) for arm, k, f, _ in ARMS}
        keys = set.intersection(*(set(d) for d in loaded.values()))
        keys = sorted(keys)
        y = [loaded["A4"][k][0] for k in keys]

        entry = {"n": len(keys), "baseline": majority_macro_f1(y, ds)}
        for arm, _, _, _ in ARMS:
            p = [loaded[arm][k][1] for k in keys]
            entry[arm] = boot_ci(y, p, ds, n_boot=n_boot)
        # kept for the interaction panel
        entry["_y"] = y
        entry["_A4"] = [loaded["A4"][k][1] for k in keys]
        entry["_A5"] = [loaded["A5"][k][1] for k in keys]
        data[ds] = entry
        print(f"{ds}: {len(keys)} images, baseline macro-F1 "
              f"{entry['baseline']:.3f}")
    return data


def figure_arms(data, split, out_dir):
    """Every arm, both datasets, with intervals and the baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(data), figsize=(12, 5.2), sharey=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (ds, entry) in zip(axes, data.items()):
        xs = np.arange(len(ARMS))
        for i, (arm, _, _, label) in enumerate(ARMS):
            point, lo, hi = entry[arm]
            ax.bar(i, point, color=COLORS[arm], width=0.68)
            ax.errorbar(i, point, yerr=[[point - lo], [hi - point]],
                        fmt="none", ecolor="black", capsize=4, linewidth=1.2)
            ax.text(i, point + (hi - point) + 0.03, f"{point:.3f}",
                    ha="center", fontsize=9)

        ax.axhline(entry["baseline"], color="crimson", linestyle="--",
                   linewidth=1.2)
        ax.text(len(ARMS) - 0.5, entry["baseline"] + 0.015,
                f"majority baseline {entry['baseline']:.3f}",
                ha="right", fontsize=8, color="crimson")

        ax.set_xticks(xs)
        ax.set_xticklabels([a[3] for a in ARMS], fontsize=9)
        ax.set_title(f"{DATASET_LABELS[ds]}\nn = {entry['n']}", fontsize=10)
        ax.set_ylim(0, 1.05)          # never truncated
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Macro-F1", fontsize=11)
    fig.suptitle(
        f"Does medical pretraining earn its keep?  ({split} split)",
        fontsize=13, y=0.99,
    )
    fig.text(0.5, 0.005,
             "Error bars are 1,000-resample bootstrap 95% intervals. "
             "All arms shown, per the preregistered reporting commitment.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = out_dir / f"fig1_arms_{split}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def figure_interaction(data, split, out_dir, n_boot):
    """H3: the MedGemma advantage, in-domain versus out-of-domain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    bars = []
    for ds, entry in data.items():
        y = np.asarray(entry["_y"])
        a4, a5 = np.asarray(entry["_A4"]), np.asarray(entry["_A5"])
        n = len(y)
        draws = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[i] = (compute_metrics(y[idx], a4[idx], ds)["macro_f1"]
                        - compute_metrics(y[idx], a5[idx], ds)["macro_f1"])
        point = (compute_metrics(y, a4, ds)["macro_f1"]
                 - compute_metrics(y, a5, ds)["macro_f1"])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        bars.append((ds, point, lo, hi))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (ds, point, lo, hi) in enumerate(bars):
        color = "#0072B2" if ds == "dermamnist" else "#999999"
        ax.bar(i, point, color=color, width=0.55)
        ax.errorbar(i, point, yerr=[[point - lo], [hi - point]], fmt="none",
                    ecolor="black", capsize=5, linewidth=1.3)
        ax.text(i, hi + 0.008, f"{point:+.3f}\n[{lo:+.3f}, {hi:+.3f}]",
                ha="center", fontsize=9)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([DATASET_LABELS[b[0]] for b in bars], fontsize=9)
    ax.set_ylabel("MedGemma advantage over Gemma 3\n(macro-F1, A4 minus A5)",
                  fontsize=10)
    ax.set_title("H3: is the advantage larger on a modality\n"
                 f"inside MedGemma's pretraining domains?  ({split} split)",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.01,
             "A bar above zero means MedGemma beats its non-medical parent. "
             "H3 predicts the left bar exceeds the right by at least 0.02.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = out_dir / f"fig2_h3_interaction_{split}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    raw = REPO / "results" / "raw"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = gather(raw, args.split, args.n_boot)
    if len(data) < 2:
        print("Need both datasets. Stopping.")
        return 1

    p1 = figure_arms(data, args.split, out_dir)
    p2 = figure_interaction(data, args.split, out_dir, args.n_boot)
    print(f"\nwrote {p1.relative_to(REPO)}")
    print(f"wrote {p2.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
