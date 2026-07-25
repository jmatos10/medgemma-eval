"""
pilot_lengths.py

Phase 4, step 4. Measures how many tokens MedGemma actually emits before it
stops, so the production `max_new_tokens` cap is set from data rather than a
guess.

Why this exists: H1 predicts that strict parsing understates zero-shot
capability because models wrap answers in prose. If the token cap is set too
low, the model gets truncated mid-answer, scores unparseable, and we
manufacture the exact artifact we set out to measure. That result would look
identical to a real finding in the numbers.

So this run sets the cap wide open at 512 and reports the distribution.

Nothing is parsed or scored here. Raw text is written to disk first. Scoring
happens later from the saved file, which means an alias bug costs 2 seconds
to re-score instead of an hour of GPU.

Usage, from the repo root:
    python scripts/pilot_lengths.py --dataset dermamnist --n 50

Expected cost: about 3 to 5 minutes of L4 time, well under $0.15.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_harness import CANONICAL, prompt_for  # noqa: E402

# Pinned per CLAUDE.md hard rule 2. See SETUP.md for how these were obtained.
REVISIONS = {
    "google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
}

L4_HOURLY_USD = 1.082

DATASET_CLASSES = {
    "dermamnist": "DermaMNIST",
    "bloodmnist": "BloodMNIST",
}


def get_hf_token(project_number: str = "609652923595") -> str:
    """Read the Hugging Face token from Secret Manager.

    The token is never written to a file or pasted into a cell. The instance
    service account was granted Secret Manager Secret Accessor in Phase 1.
    """
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_number}/secrets/hf-token/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode()


def load_split(dataset: str, split: str, n: int, seed: int = 42):
    """Load `n` images from a MedMNIST split, sampled with a fixed seed."""
    import medmnist

    cls = getattr(medmnist, DATASET_CLASSES[dataset])
    ds = cls(split=split, size=224, download=False)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds.imgs), size=min(n, len(ds.imgs)), replace=False)
    idx.sort()

    images = [Image.fromarray(ds.imgs[i]) for i in idx]
    labels = [int(ds.labels[i][0]) for i in idx]
    return idx.tolist(), images, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dermamnist", choices=list(CANONICAL))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--model", default="google/medgemma-1.5-4b-it")
    args = ap.parse_args()

    if args.model not in REVISIONS:
        print(f"Refusing to load unpinned model {args.model!r}.")
        print(f"Known: {list(REVISIONS)}")
        return 1

    revision = REVISIONS[args.model]

    if not torch.cuda.is_available():
        print("No CUDA device. Start the instance and check nvidia-smi.")
        return 1

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"model     {args.model}")
    print(f"revision  {revision}")
    print(f"dataset   {args.dataset} [{args.split}], n={args.n}")
    print(f"cap       max_new_tokens={args.max_new_tokens} (deliberately high)")
    print()

    token = get_hf_token()

    print("Loading data...")
    indices, images, labels = load_split(args.dataset, args.split, args.n)
    counts = {c: labels.count(c) for c in sorted(set(labels))}
    print(f"  {len(images)} images, class counts {counts}")

    print("Loading processor and model (first run downloads ~8GB)...")
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision, token=token
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=revision,
        token=token,
        dtype=torch.bfloat16,      # v5 renamed this from torch_dtype
        device_map="auto",
    )
    model.eval()
    print(f"  loaded in {time.time() - t_load:.0f}s")
    print(f"  device {model.device}, dtype {model.dtype}")

    prompt = prompt_for(args.dataset)
    print(f"\n--- prompt ---\n{prompt}\n--------------\n")

    out_path = REPO / "results" / "raw" / f"pilot_{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    records = []
    t_gen = time.time()

    for start in range(0, len(images), args.batch_size):
        batch_imgs = images[start:start + args.batch_size]
        batch_idx = indices[start:start + args.batch_size]
        batch_lbl = labels[start:start + args.batch_size]

        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            for img in batch_imgs
        ]

        inputs = processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,           # greedy, deterministic
            )

        new_tokens = out[:, input_len:]
        texts = processor.batch_decode(new_tokens, skip_special_tokens=True)

        # Count real tokens by finding the first stop token in each row.
        #
        # Do not count by filtering the pad id. In batched generation the
        # framework fills finished sequences so every row matches the longest
        # one, and the fill token is not necessarily tokenizer.pad_token_id.
        # Gemma stops on <end_of_turn> (106), not <eos> (1), so a pad-based
        # filter silently reports the batch maximum for every sequence.
        tok = processor.tokenizer
        stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")}
        stop_ids.discard(None)

        for j, text in enumerate(texts):
            row = new_tokens[j].tolist()
            n_new = len(row)
            finish = "length"
            for k, tid in enumerate(row):
                if tid in stop_ids:
                    n_new = k + 1
                    finish = "eos"
                    break
            rec = {
                "dataset": args.dataset,
                "split": args.split,
                "image_index": int(batch_idx[j]),
                "true_label": int(batch_lbl[j]),
                "model": args.model,
                "revision": revision,
                "max_new_tokens": args.max_new_tokens,
                "n_new_tokens": n_new,
                "finish_reason": finish,
                "output_text": text,
            }
            records.append(rec)

        done = min(start + args.batch_size, len(images))
        print(f"  {done}/{len(images)}", end="\r", flush=True)

    gen_seconds = time.time() - t_gen
    print(f"  {len(records)}/{len(images)} done in {gen_seconds:.0f}s")

    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # ---------------- report ----------------
    lengths = np.array([r["n_new_tokens"] for r in records])
    truncated = sum(1 for r in records if r["finish_reason"] == "length")

    print("\n=== TOKEN LENGTH DISTRIBUTION ===")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p:<3} {np.percentile(lengths, p):6.0f}")
    print(f"  mean {lengths.mean():6.1f}")
    print(f"  truncated at cap: {truncated}/{len(records)}")

    if truncated:
        print("\n  Truncation detected. The cap is too low and any H1 number")
        print("  from this configuration would be contaminated. Raise it.")
    else:
        suggested = int(np.percentile(lengths, 99) * 1.5)
        suggested = max(32, min(args.max_new_tokens, suggested))
        print(f"\n  Suggested production cap: {suggested}")
        print("  (99th percentile with 50 percent headroom)")

    print("\n=== FIRST 5 RAW OUTPUTS ===")
    for rec in records[:5]:
        truth = CANONICAL[args.dataset][rec["true_label"]]
        print(f"\n  [idx {rec['image_index']}] truth: {truth}")
        print(f"  tokens: {rec['n_new_tokens']}  finish: {rec['finish_reason']}")
        print(f"  output: {rec['output_text']!r}")

    cost = gen_seconds / 3600 * L4_HOURLY_USD
    print(f"\n=== COST ===")
    print(f"  generation {gen_seconds:.0f}s  ~${cost:.3f}")
    print(f"  raw outputs -> {out_path.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
