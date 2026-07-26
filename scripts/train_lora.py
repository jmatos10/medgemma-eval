"""
train_lora.py

Arms A4 (MedGemma 1.5 4B) and A5 (Gemma 3 4B IT). One script, one set of
hyperparameters, two starting weights. Section B3 requires that everything
except the initialization be held constant, and running both arms through
the same code path is how that is enforced rather than asserted.

WHAT IS TRAINED
---------------
LoRA adapters on the language model only. The vision encoder is frozen.

That is deliberate and it matters for H3. The sharp hypothesis is about what
MedGemma's image encoder absorbed during medical pretraining. If the encoder
were fine-tuned, both models could learn the modality during training and the
difference H3 is trying to detect would wash out. Freezing it means H3 tests
the pretrained representation, which is the actual question.

TRAINING TARGET
---------------
The bare canonical label, nothing else. Not "Based on the image, the best
description is **melanoma**." Just "melanoma".

Phase 4 showed MedGemma never emits a bare label zero-shot: 0 times in 2,715
attempts. Training on formatted prose would carry that format penalty into
A4 and A5, and the fine-tuned numbers would inherit the artifact this study
exists to document. Bare labels mean strict, lenient, extraction, and
constrained decoding should all converge on the fine-tuned arms, and that
convergence is itself the answer to Q1.

Q1 decomposes for free:
    A1 -> A4   total jump, format plus medicine
    A3 -> A4   medicine alone, since A3 has no format penalty
    difference format component

LOSS MASKING
------------
Loss is computed on the target tokens only. Prompt tokens, image tokens, and
padding are set to -100, which the loss ignores. Without this the model
spends capacity learning to predict your own prompt back.

The script decodes and prints exactly what it is training on for the first
batch. Verify that before letting a run proceed.

Usage, from the repo root:
    python scripts/train_lora.py --arm A4 --dataset dermamnist
    python scripts/train_lora.py --arm A5 --dataset dermamnist

Four runs total: {A4, A5} x {dermamnist, bloodmnist}.
Roughly 30 to 45 minutes each at one epoch.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from eval_harness import CANONICAL, prompt_for  # noqa: E402
from subsample import load_subsample  # noqa: E402

# Section B3: identical across A4 and A5. Changing any value here means
# rerunning both arms, not one.
HP = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,   # effective batch 16
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "seed": 42,
    "max_grad_norm": 1.0,
    # Measured on a 64-example smoke test, A4 on DermaMNIST:
    #   checkpointing on   1.048 samples/sec, loss 1.4993
    #   checkpointing off  2.185 samples/sec, loss 1.5010
    # Checkpointing trades compute for activation memory. Turning it off
    # more than doubles throughput and leaves the optimization untouched,
    # which the near-identical loss confirms. Batch stays at 2: batch 4
    # runs out of memory in cross_entropy, whose logits tensor is
    # [batch x seq, 262k vocab] in float32.
    "gradient_checkpointing": False,
}

# Effective batch is per_device_batch x grad_accum. It is part of the
# optimization spec and must not change between A4 and A5. Throughput
# knobs may be retuned as long as this product stays fixed.
EFFECTIVE_BATCH = HP["per_device_train_batch_size"] * HP["gradient_accumulation_steps"]

ARM_MODELS = {
    "A4": "google/medgemma-1.5-4b-it",
    "A5": "google/gemma-3-4b-it",
}

REVISIONS = {
    "google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
}

# Projection names to adapt. Attention plus MLP, which is consistently
# stronger than attention alone at the same rank.
TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Substrings that identify the vision tower. Any module whose qualified name
# contains one of these is excluded, because SigLIP also has q_proj and
# friends and a bare suffix match would adapt the encoder we mean to freeze.
VISION_MARKERS = ("vision_tower", "vision_model", "visual", "image_encoder")

L4_HOURLY_USD = 1.082
DATASET_CLASSES = {"dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}


def get_hf_token(project_number: str = "609652923595") -> str:
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_number}/secrets/hf-token/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode()


class LabelDataset(Dataset):
    """Images paired with their bare canonical label."""

    def __init__(self, dataset: str, indices):
        import medmnist

        cls = getattr(medmnist, DATASET_CLASSES[dataset])
        ds = cls(split="train", size=224, download=False)
        self.imgs = ds.imgs
        self.labels = ds.labels
        self.indices = list(indices)
        self.names = CANONICAL[dataset]
        self.prompt = prompt_for(dataset)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return {
            "image": Image.fromarray(self.imgs[j]),
            "target": self.names[int(self.labels[j][0])],
        }


def find_target_modules(model) -> list:
    """Qualified names of language-model projections to adapt.

    Built by scanning the model rather than hardcoded, because the attribute
    layout differs across transformers versions and model families. Passing
    bare suffixes like "q_proj" to LoraConfig would also match the vision
    tower, which must stay frozen for H3 to mean anything.
    """
    import torch.nn as nn

    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(marker in name for marker in VISION_MARKERS):
            continue
        if name.split(".")[-1] in TARGET_SUFFIXES:
            names.append(name)

    if not names:
        print("\nNo target modules found. Linear layers present:")
        seen = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                leaf = name.split(".")[-1]
                if leaf not in seen:
                    seen.add(leaf)
                    print(f"  {leaf}   e.g. {name}")
        raise SystemExit("Adjust TARGET_SUFFIXES or VISION_MARKERS.")

    return names


def build_collator(processor, prompt: str):
    """Batch images and targets, masking loss to the target tokens only."""
    tok = processor.tokenizer

    # Right padding for training. Gemma pads left by default because that is
    # correct for generation, but with left padding the real sequence does
    # not start at index 0 and masking the prompt by index would mask the
    # wrong tokens.
    tok.padding_side = "right"

    def collate(batch):
        user_turns = [
            [{"role": "user", "content": [
                {"type": "image", "image": ex["image"]},
                {"type": "text", "text": prompt},
            ]}]
            for ex in batch
        ]
        full_turns = [
            turn + [{"role": "assistant", "content": [
                {"type": "text", "text": ex["target"]},
            ]}]
            for turn, ex in zip(user_turns, batch)
        ]

        enc = processor.apply_chat_template(
            full_turns,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        # Per-example prompt length, so the mask boundary is exact.
        prompt_lens = []
        for turn in user_turns:
            p = processor.apply_chat_template(
                [turn],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_lens.append(p["input_ids"].shape[1])

        labels = enc["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        enc["labels"] = labels
        return enc

    return collate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_MODELS))
    ap.add_argument("--dataset", required=True, choices=list(CANONICAL))
    ap.add_argument("--limit", type=int, default=0,
                    help="train on N examples only, for a smoke test")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything, show one batch, do not train")
    ap.add_argument("--bs", type=int, default=None,
                    help="per-device batch size, throughput only")
    ap.add_argument("--accum", type=int, default=None,
                    help="gradient accumulation, throughput only")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing, faster, more memory")
    args = ap.parse_args()

    # Throughput overrides. These change wall-clock time, not the
    # optimization, provided the effective batch is preserved. Whatever
    # is used here must be used for both A4 and A5.
    if args.bs is not None:
        HP["per_device_train_batch_size"] = args.bs
    if args.accum is not None:
        HP["gradient_accumulation_steps"] = args.accum
    if args.no_ckpt:
        HP["gradient_checkpointing"] = False

    effective = (HP["per_device_train_batch_size"]
                 * HP["gradient_accumulation_steps"])
    if effective != EFFECTIVE_BATCH:
        print(f"Effective batch is {effective}, spec requires "
              f"{EFFECTIVE_BATCH}.")
        print("bs x accum must equal the spec value. Refusing to run.")
        return 1

    model_id = ARM_MODELS[args.arm]
    revision = REVISIONS[model_id]

    if not torch.cuda.is_available():
        print("No CUDA device. Start the instance and check nvidia-smi.")
        return 1

    torch.manual_seed(HP["seed"])
    np.random.seed(HP["seed"])

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              Trainer, TrainingArguments)

    print(f"arm       {args.arm}")
    print(f"model     {model_id}")
    print(f"revision  {revision}")
    print(f"dataset   {args.dataset}")
    print(f"hyperparameters {json.dumps(HP)}")
    print()

    indices = load_subsample(args.dataset)
    if args.limit:
        indices = indices[:args.limit]
        print(f"  LIMIT {args.limit}, smoke test only, do not report results")
    print(f"  {len(indices)} training images, hash verified")

    token = get_hf_token()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision,
                                              token=token)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, token=token,
        dtype=torch.bfloat16, device_map="auto",
    )

    targets = find_target_modules(model)
    print(f"\n  {len(targets)} LoRA target modules, vision tower excluded")
    print(f"  first  {targets[0]}")
    print(f"  last   {targets[-1]}")

    peft_config = LoraConfig(
        r=HP["lora_r"],
        lora_alpha=HP["lora_alpha"],
        lora_dropout=HP["lora_dropout"],
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  trainable {trainable:,} of {total:,} "
          f"({100 * trainable / total:.3f}%)")

    # Confirm the vision tower really is frozen.
    vision_trainable = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and any(m in n for m in VISION_MARKERS)
    )
    print(f"  vision tower trainable params: {vision_trainable:,}")
    if vision_trainable:
        print("  Vision tower is NOT frozen. H3 would be confounded. Stopping.")
        return 1

    train_ds = LabelDataset(args.dataset, indices)
    collate = build_collator(processor, prompt_for(args.dataset))

    # Decode one batch and show exactly what the loss sees. This is the
    # single most valuable check in the script: a masking bug produces a
    # model that trains without error and learns the wrong thing.
    print("\n=== FIRST BATCH, WHAT THE LOSS SEES ===")
    sample = collate([train_ds[0], train_ds[1]])
    for i in range(sample["labels"].shape[0]):
        kept = sample["labels"][i]
        kept = kept[kept != -100]
        text = processor.tokenizer.decode(kept)
        n_total = int((sample["input_ids"][i] !=
                       processor.tokenizer.pad_token_id).sum())
        print(f"  [{i}] {len(kept)} of {n_total} tokens supervised")
        print(f"      target text: {text!r}")
    print("  Expect the bare label plus a turn-end token. Anything else,")
    print("  especially prompt text, means the mask boundary is wrong.")

    if args.dry_run:
        print("\nDry run, stopping before training.")
        return 0

    out_dir = REPO / "adapters" / f"{args.arm}_{args.dataset}"
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=HP["num_train_epochs"],
        per_device_train_batch_size=HP["per_device_train_batch_size"],
        gradient_accumulation_steps=HP["gradient_accumulation_steps"],
        learning_rate=HP["learning_rate"],
        lr_scheduler_type=HP["lr_scheduler_type"],
        warmup_ratio=HP["warmup_ratio"],
        max_grad_norm=HP["max_grad_norm"],
        seed=HP["seed"],
        bf16=True,
        gradient_checkpointing=HP["gradient_checkpointing"],
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    if targs.gradient_checkpointing:
        model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collate,
    )

    print(f"\nTraining {args.arm} on {args.dataset}...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    if args.limit:
        print(f"\n  --limit set, NOT saving adapter. A {args.limit}-example "
              f"run is not arm {args.arm}.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        print(f"\nAdapter saved to {out_dir.relative_to(REPO)}")

    record = {
        "arm": args.arm,
        "phase": 5,
        "stage": "train",
        "model": model_id,
        "revision": revision,
        "dataset": args.dataset,
        "split": "train",
        "n_train": len(indices),
        "seed": HP["seed"],
        "hyperparameters": HP,
        "n_target_modules": len(targets),
        "trainable_params": trainable,
        "total_params": total,
        "train_loss": float(result.training_loss),
        "gpu_seconds": round(elapsed, 1),
        "est_usd": round(elapsed / 3600 * L4_HOURLY_USD, 4),
    }

    # Smoke tests must not enter the results log. A record with n_train=64
    # is honest but would be picked up by any analysis filtering on arm,
    # and results.jsonl is append-only so it could not be removed cleanly.
    if args.limit:
        print("\n  --limit set, NOT logging to results.jsonl")
        print(f"  would have been: {json.dumps(record)[:120]}...")
    else:
        from eval_harness import append_result
        append_result(record, str(REPO / "results" / "results.jsonl"))

    print(f"\n=== DONE ===")
    print(f"  train loss  {result.training_loss:.4f}")
    print(f"  time        {elapsed / 60:.1f} min  ~${record['est_usd']:.2f}")
    print(f"  logged to   results/results.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())