"""
train_resnet.py

Arm A6. ResNet-18, ImageNet-initialized, fine-tuned on the same 5,000-image
subsample that A4 and A5 used. This is the parameter-efficiency reference
for RQ3 and H4.

WHY THE RECIPE DIFFERS FROM A4 AND A5
-------------------------------------
Section B3 constrains A4 against A5 only. A6 is unconstrained, and matching
its epoch count to theirs would be symmetric in name and unfair in
substance. The 4B models arrive with enormous pretrained knowledge and need
little adaptation; ResNet-18 has 11M parameters and needs more passes to
converge. One epoch would guarantee H4 fails for a reason unrelated to
parameter efficiency.

The data is held constant. The recipe is not. Each method gets a competent
standard configuration, which means H4 compares methods as practiced rather
than architectures under an artificially matched protocol. That is a real
limitation and belongs in the writeup.

WHAT IS NOT DONE
----------------
No early stopping and no model selection on validation. Epochs are fixed at
20 in advance and the final model is reported. Validation loss is logged so
convergence is visible, but nothing is chosen from it. The LoRA arms had no
early stopping either, so this keeps that symmetric.

Augmentation is horizontal and vertical flips only. Neither dermatoscopy nor
blood microscopy has a canonical orientation, so flips are standard and
withholding them would handicap this arm artificially. It also keeps the
setup comparable to the published MedMNIST ResNet-18 baselines, which is a
free external check on whether this implementation is sane.

Usage, from the repo root:
    python scripts/train_resnet.py --dataset dermamnist
    python scripts/train_resnet.py --dataset bloodmnist

About 8 minutes per dataset on an L4.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from eval_harness import CANONICAL, append_result, compute_metrics  # noqa: E402
from subsample import load_subsample  # noqa: E402

# Fixed in advance. No search, no early stopping, no selection on validation.
HP = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "lr_scheduler": "cosine",
    "augmentation": "random horizontal and vertical flip",
    "init": "ImageNet",
    "seed": 42,
}

# ImageNet statistics. The pretrained weights expect inputs normalized this
# way; using dataset statistics instead would put the input distribution
# somewhere the pretrained filters were never calibrated for.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

L4_HOURLY_USD = 1.082
DATASET_CLASSES = {"dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}


class MedMNISTTensor(Dataset):
    """MedMNIST images as normalized tensors, with optional flips."""

    def __init__(self, dataset: str, split: str, indices=None, train=False):
        import medmnist

        cls = getattr(medmnist, DATASET_CLASSES[dataset])
        ds = cls(split=split, size=224, download=False)
        self.imgs = ds.imgs
        self.labels = ds.labels
        self.indices = list(range(len(ds.imgs))) if indices is None \
            else list(indices)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = self.imgs[j].astype(np.float32) / 255.0     # HWC, 0 to 1

        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]     # horizontal flip
            if np.random.rand() < 0.5:
                img = img[::-1, :, :]     # vertical flip

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1))   # CHW

        return (torch.from_numpy(img),
                int(self.labels[j][0]),
                int(j))


def build_model(n_classes: int):
    """ResNet-18 with ImageNet weights and a fresh classification head."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    """Return predictions, truths, image indices, and mean loss."""
    model.eval()
    preds, truths, idxs = [], [], []
    total_loss, n = 0.0, 0

    for x, y, j in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(criterion(logits, y)) * x.shape[0]
        n += x.shape[0]
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        truths.extend(y.cpu().tolist())
        idxs.extend(j.tolist())

    return preds, truths, idxs, total_loss / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CANONICAL))
    ap.add_argument("--split", default="val",
                    help="split to evaluate on. Hard rule 1: not test until "
                         "every arm is complete.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override, for smoke tests only")
    args = ap.parse_args()

    if args.epochs is not None:
        HP["epochs"] = args.epochs

    if not torch.cuda.is_available():
        print("No CUDA device. Start the instance and check nvidia-smi.")
        return 1

    device = torch.device("cuda")
    torch.manual_seed(HP["seed"])
    np.random.seed(HP["seed"])

    n_classes = len(CANONICAL[args.dataset])
    print(f"arm       A6, ResNet-18")
    print(f"dataset   {args.dataset}, {n_classes} classes")
    print(f"eval on   {args.split}")
    print(f"hyperparameters {json.dumps(HP)}")

    if args.split == "test":
        print("\n  Evaluating on TEST. Hard rule 1 allows this exactly once,")
        print("  after every arm is complete. If development continues, stop.")

    indices = load_subsample(args.dataset)
    print(f"\n  {len(indices)} training images, hash verified, "
          "identical to A4 and A5")

    train_ds = MedMNISTTensor(args.dataset, "train", indices, train=True)
    eval_ds = MedMNISTTensor(args.dataset, args.split, None, train=False)
    print(f"  {len(eval_ds)} {args.split} images")

    train_loader = DataLoader(train_ds, batch_size=HP["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=False)
    eval_loader = DataLoader(eval_ds, batch_size=HP["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(n_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  parameters {total_params:,}")
    print(f"  versus 4,329,881,968 for the 4B arms, "
          f"a factor of {4_329_881_968 / total_params:.0f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=HP["learning_rate"],
                                  weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP["epochs"]
    )

    print(f"\nTraining {HP['epochs']} epochs...")
    t0 = time.time()
    history = []

    for epoch in range(1, HP["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss) * x.shape[0]
            n += x.shape[0]
        scheduler.step()
        train_loss = running / n

        # Logged so convergence is visible. Nothing is selected on it.
        _, _, _, eval_loss = evaluate(model, eval_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "eval_loss": eval_loss})
        print(f"  epoch {epoch:>2}  train {train_loss:.4f}  "
              f"{args.split} {eval_loss:.4f}", flush=True)

    elapsed = time.time() - t0

    preds, truths, idxs, final_eval_loss = evaluate(
        model, eval_loader, device, criterion
    )

    # Raw predictions to disk before scoring, per hard rule 4.
    out_path = REPO / "results" / "raw" / f"A6_{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for p, t, j in zip(preds, truths, idxs):
            fh.write(json.dumps({
                "dataset": args.dataset,
                "split": args.split,
                "image_index": int(j),
                "true_label": int(t),
                "model": "resnet18",
                "revision": "torchvision IMAGENET1K_V1",
                "arm": "A6",
                "pred_class": int(p),
            }) + "\n")

    metrics = compute_metrics(truths, preds, args.dataset)

    print(f"\n=== A6 RESNET-18, {args.dataset} [{args.split}] ===")
    print(f"  macro-F1     {metrics['macro_f1']:.3f}")
    print(f"  accuracy     {metrics['accuracy']:.3f}")
    print(f"  per-class F1 {[round(v, 2) for v in metrics['per_class_f1']]}")
    print(f"  {args.split} loss {final_eval_loss:.4f}")

    append_result({
        "arm": "A6",
        "phase": 6,
        "stage": "train_and_eval",
        "model": "resnet18",
        "revision": "torchvision IMAGENET1K_V1",
        "dataset": args.dataset,
        "split": args.split,
        "n_train": len(indices),
        "smoke_test": args.epochs is not None,
        "seed": HP["seed"],
        "hyperparameters": HP,
        "total_params": total_params,
        "trainable_params": total_params,
        "final_train_loss": history[-1]["train_loss"],
        "final_eval_loss": final_eval_loss,
        "history": history,
        "gpu_seconds": round(elapsed, 1),
        "est_usd": round(elapsed / 3600 * L4_HOURLY_USD, 4),
        **metrics,
    }, str(REPO / "results" / "results.jsonl"))

    print(f"\n  time      {elapsed / 60:.1f} min  "
          f"~${elapsed / 3600 * L4_HOURLY_USD:.2f}")
    print(f"  raw       {out_path.relative_to(REPO)}")
    print(f"  logged to results/results.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
