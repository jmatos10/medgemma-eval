"""
score_labels.py

Arm A3, constrained decoding. The model never writes free text. Instead we
ask it, for each candidate class label, how likely it considers that exact
sequence of tokens given the image and prompt. Highest score wins.

A parse failure is impossible by construction, which is the point. A1 and A2
measure how often the model's answer survives a parser. A3 measures what the
model actually believes.

THE MATH
--------
A language model assigns a probability to each next token. The probability of
a whole label is the product of its token probabilities, each conditioned on
everything before it:

    P(label | image, prompt) = P(t1) * P(t2 | t1) * P(t3 | t1,t2) * ...

Probabilities multiply to very small numbers, so we work in logs, where
multiplication becomes addition:

    log P(label) = sum_i log P(t_i | image, prompt, t_1..t_{i-1})

THE LENGTH TRAP
---------------
Every term in that sum is negative, because probabilities are below 1. So
longer labels score worse purely for being longer. BloodMNIST class 3 is
"immature granulocytes(myelocytes, metamyelocytes and promyelocytes)", about
25 tokens, against "platelet" at about 3. At an identical per-token
confidence of -0.5, class 3 scores -12.5 and platelet scores -1.5. Class 3
would be structurally unable to win, and its F1 would read as a hematology
failure rather than as arithmetic.

Three scores are computed for every image and label. Primary was fixed in
the Phase 4 spec before any run, so the flattering variant cannot be chosen
afterward.

    sum   raw sum of log probabilities, length-penalized, logged only
    mean  sum divided by token count. PRIMARY. length cancels.
    pmi   log P(label | image, prompt) - log P(label | prompt only)
          corrects for some labels being common in English regardless of
          the image. Robustness check.

Usage, from the repo root:
    python scripts/score_labels.py --dataset dermamnist --n 50

Start with --n 50 to confirm it runs before spending on the full split.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_harness import CANONICAL, prompt_for  # noqa: E402

REVISIONS = {
    "google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
}

L4_HOURLY_USD = 1.082

DATASET_CLASSES = {"dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}

# Keys returned by the processor that run along the sequence dimension and
# therefore must be extended when candidate label tokens are appended.
SEQ_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def get_hf_token(project_number: str = "609652923595") -> str:
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_number}/secrets/hf-token/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode()


def load_split(dataset: str, split: str, n: int, seed: int = 42):
    import medmnist

    cls = getattr(medmnist, DATASET_CLASSES[dataset])
    ds = cls(split=split, size=224, download=False)
    rng = np.random.default_rng(seed)
    k = len(ds.imgs) if n <= 0 else min(n, len(ds.imgs))
    idx = rng.choice(len(ds.imgs), size=k, replace=False)
    idx.sort()
    images = [Image.fromarray(ds.imgs[i]) for i in idx]
    labels = [int(ds.labels[i][0]) for i in idx]
    return idx.tolist(), images, labels


@torch.inference_mode()
def score_candidates(model, processor, prompt: str, image, label_token_ids):
    """Log probability of each candidate label, given prompt and optional image.

    Returns two lists: summed log probability, and token count, one entry per
    candidate. Pass image=None for the prompt-only term used by PMI.

    All candidates are scored in a single batched forward pass. Each row is
    the prompt followed by one candidate's tokens, right-padded. Causal
    attention means trailing pad positions cannot influence earlier ones.
    """
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [[{"role": "user", "content": content}]]

    enc = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_len = enc["input_ids"].shape[1]
    n_cand = len(label_token_ids)
    lengths = [len(ids) for ids in label_token_ids]
    max_len = max(lengths)
    device = model.device
    pad_id = processor.tokenizer.pad_token_id or 0

    batch = {}
    for key, val in enc.items():
        if key in SEQ_KEYS:
            rows = []
            for ids in label_token_ids:
                tail = list(ids) + [pad_id] * (max_len - len(ids))
                if key == "input_ids":
                    ext = tail
                elif key == "attention_mask":
                    ext = [1] * len(ids) + [0] * (max_len - len(ids))
                else:  # token_type_ids, label tokens are text
                    ext = [0] * max_len
                rows.append(torch.cat([
                    val[0],
                    torch.tensor(ext, dtype=val.dtype, device=device),
                ]))
            batch[key] = torch.stack(rows)
        else:
            # pixel_values and anything else: one copy per candidate
            reps = [1] * val.dim()
            reps[0] = n_cand
            batch[key] = val.repeat(*reps)

    out = model(**batch)

    # float32 for numerical stability. bf16 log_softmax loses precision.
    logits = out.logits.float()
    logprobs = F.log_softmax(logits, dim=-1)

    sums = []
    for k, ids in enumerate(label_token_ids):
        total = 0.0
        for i, tok in enumerate(ids):
            # the logit at position p predicts the token at position p+1
            pos = prompt_len + i - 1
            total += float(logprobs[k, pos, tok])
        sums.append(total)

    return sums, lengths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dermamnist", choices=list(CANONICAL))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50, help="0 for the whole split")
    ap.add_argument("--model", default="google/medgemma-1.5-4b-it")
    args = ap.parse_args()

    if args.model not in REVISIONS:
        print(f"Refusing to load unpinned model {args.model!r}.")
        return 1
    revision = REVISIONS[args.model]

    if not torch.cuda.is_available():
        print("No CUDA device. Start the instance and check nvidia-smi.")
        return 1

    from transformers import AutoModelForImageTextToText, AutoProcessor

    labels_text = CANONICAL[args.dataset]
    prompt = prompt_for(args.dataset)

    print(f"model     {args.model}")
    print(f"revision  {revision}")
    print(f"dataset   {args.dataset} [{args.split}], n={args.n or 'all'}")
    print(f"candidates {len(labels_text)}")
    print()

    token = get_hf_token()

    indices, images, truths = load_split(args.dataset, args.split, args.n)
    print(f"Loaded {len(images)} images")

    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision, token=token
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=revision, token=token,
        dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"Model on {model.device}, dtype {model.dtype}")

    # Tokenize each candidate once. add_special_tokens=False because these
    # continue the prompt rather than starting a new sequence.
    tok = processor.tokenizer
    label_token_ids = [
        tok(lbl, add_special_tokens=False)["input_ids"] for lbl in labels_text
    ]
    print("candidate token counts:",
          {i: len(v) for i, v in enumerate(label_token_ids)})

    # PMI denominator: prompt-only, no image. Identical for every image, so
    # compute it once rather than per image.
    print("Computing prompt-only baseline for PMI...")
    prior_sums, lengths = score_candidates(
        model, processor, prompt, None, label_token_ids
    )
    print("  done")

    out_path = REPO / "results" / "raw" / f"A3_{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    records = []
    t0 = time.time()

    for n_done, (idx, img, truth) in enumerate(zip(indices, images, truths), 1):
        sums, lens = score_candidates(
            model, processor, prompt, img, label_token_ids
        )
        means = [s / L for s, L in zip(sums, lens)]
        pmis = [s - p for s, p in zip(sums, prior_sums)]

        records.append({
            "dataset": args.dataset,
            "split": args.split,
            "image_index": int(idx),
            "true_label": int(truth),
            "model": args.model,
            "revision": revision,
            "arm": "A3",
            "logp_sum": sums,
            "logp_mean": means,
            "logp_pmi": pmis,
            "token_counts": lens,
            "pred_sum": int(np.argmax(sums)),
            "pred_mean": int(np.argmax(means)),   # primary
            "pred_pmi": int(np.argmax(pmis)),
        })

        if n_done % 10 == 0 or n_done == len(images):
            rate = (time.time() - t0) / n_done
            print(f"  {n_done}/{len(images)}  {rate:.2f}s/image", end="\r",
                  flush=True)

    elapsed = time.time() - t0
    print()

    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # ---------------- report ----------------
    from collections import Counter

    from eval_harness import compute_metrics

    y = [r["true_label"] for r in records]
    print("\n=== A3 CONSTRAINED DECODING ===")
    print(f"{'scoring':10}{'accuracy':>10}{'macro_f1':>10}  prediction spread")
    print("-" * 62)
    for name, key in [("mean*", "pred_mean"), ("sum", "pred_sum"),
                      ("pmi", "pred_pmi")]:
        p = [r[key] for r in records]
        m = compute_metrics(y, p, args.dataset)
        spread = dict(sorted(Counter(p).items()))
        print(f"{name:10}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}  {spread}")
    print("-" * 62)
    print("* primary, prespecified before any run")

    majority = max(Counter(y).values()) / len(y)
    print(f"\ntruth spread {dict(sorted(Counter(y).items()))}")
    print(f"majority baseline accuracy {majority:.3f}")

    agree = sum(1 for r in records
                if r["pred_mean"] == r["pred_sum"] == r["pred_pmi"])
    print(f"all three scorings agree on {agree}/{len(records)} images")

    cost = elapsed / 3600 * L4_HOURLY_USD
    print(f"\n=== COST ===")
    print(f"  {elapsed:.0f}s  ~${cost:.3f}  ({elapsed/len(records):.2f}s/image)")
    print(f"  raw scores -> {out_path.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
