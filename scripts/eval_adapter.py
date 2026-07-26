"""
eval_adapter.py

Generation from a LoRA fine-tuned model. Arms A4 (MedGemma) and A5 (Gemma 3).

Same generation loop and same prompt as the zero-shot arms, with one
difference: a trained adapter is loaded on top of the pinned base weights.
Holding the prompt and decoding fixed is what makes A1 through A5 comparable.

WHAT TO EXPECT, AND WHY IT MATTERS
----------------------------------
Zero-shot MedGemma emitted about 230 tokens per image, wrapping its answer in
prose and enumerating rejected classes. It produced a bare canonical label
zero times in 2,715 attempts.

A4 and A5 were fine-tuned on bare labels, so they should emit roughly 5
tokens. If they do, three things follow:

  1. Strict, lenient, extraction, and constrained decoding should converge,
     and that convergence is the answer to Q1.
  2. Inference becomes roughly forty times cheaper than the zero-shot arms.
  3. The A1-to-A4 jump decomposes: A3-to-A4 is the medical component, since
     A3 already carries no format penalty, and the remainder is format.

If they do not, that is a finding rather than a bug, and the cap protects
against truncating it.

The cap starts wide at 512 and is set from measurement, following D-002.
Truncation would fabricate exactly the artifact this study documents.

Usage, from the repo root:
    python scripts/eval_adapter.py --arm A4 --dataset dermamnist --n 50
    python scripts/eval_adapter.py --arm A4 --dataset dermamnist --n 0 --tag gen

Raw text is written to disk before anything is parsed, per hard rule 4.
Score it afterward with scripts/score_arms.py.
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

ARM_MODELS = {
    "A4": "google/medgemma-1.5-4b-it",
    "A5": "google/gemma-3-4b-it",
}

REVISIONS = {
    "google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
}

L4_HOURLY_USD = 1.082
DATASET_CLASSES = {"dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_MODELS))
    ap.add_argument("--dataset", required=True, choices=list(CANONICAL))
    ap.add_argument("--split", default="val",
                    help="hard rule 1: not test until every arm is complete")
    ap.add_argument("--n", type=int, default=50, help="0 for the whole split")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model_id = ARM_MODELS[args.arm]
    revision = REVISIONS[model_id]
    adapter_dir = REPO / "adapters" / f"{args.arm}_{args.dataset}"

    if not adapter_dir.exists():
        print(f"No adapter at {adapter_dir}. Train it first with:")
        print(f"  python scripts/train_lora.py --arm {args.arm} "
              f"--dataset {args.dataset}")
        return 1

    weights = list(adapter_dir.glob("adapter_model.safetensors"))
    if not weights:
        print(f"{adapter_dir} exists but holds no adapter_model.safetensors.")
        print("A killed training run leaves an empty output directory that")
        print("looks present in ls. Retrain this arm.")
        return 1

    if not torch.cuda.is_available():
        print("No CUDA device. Start the instance and check nvidia-smi.")
        return 1

    if args.split == "test":
        print("  Generating on the TEST split. Hard rule 1 allows this once,")
        print("  after every arm is complete. If development is ongoing, stop.")

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"arm       {args.arm}")
    print(f"base      {model_id}")
    print(f"revision  {revision}")
    print(f"adapter   {adapter_dir.relative_to(REPO)}")
    print(f"dataset   {args.dataset} [{args.split}], n={args.n or 'all'}")
    print(f"cap       max_new_tokens={args.max_new_tokens}")
    print()

    token = get_hf_token()

    indices, images, labels = load_split(args.dataset, args.split, args.n)
    print(f"Loaded {len(images)} images")

    processor = AutoProcessor.from_pretrained(model_id, revision=revision,
                                              token=token)
    base = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, token=token,
        dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    print(f"Adapter loaded onto {model.device}, dtype {base.dtype}")

    # Left padding for batched generation. Decoder-only models generate from
    # the rightmost position, so right padding would have the model continue
    # from pad tokens. Training used right padding, which is correct there
    # and wrong here, so it is set explicitly rather than inherited.
    processor.tokenizer.padding_side = "left"

    prompt = prompt_for(args.dataset)

    out_path = (REPO / "results" / "raw" /
                f"{args.tag}_{args.arm}_{args.dataset}_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_indices = set()
    if args.resume and out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done_indices.add(json.loads(line)["image_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_indices)} already done")
    elif out_path.exists():
        out_path.unlink()

    work = [(i, im, lb) for i, im, lb in zip(indices, images, labels)
            if i not in done_indices]
    print(f"{len(work)} images to generate")

    tok = processor.tokenizer
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")}
    stop_ids.discard(None)

    t0 = time.time()
    n_written = 0

    with out_path.open("a") as out_fh:
        for start in range(0, len(work), args.batch_size):
            chunk = work[start:start + args.batch_size]
            batch_idx = [c[0] for c in chunk]
            batch_imgs = [c[1] for c in chunk]
            batch_lbl = [c[2] for c in chunk]

            messages = [
                [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}]
                for img in batch_imgs
            ]

            inputs = processor.apply_chat_template(
                messages, padding=True, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out = model.generate(**inputs,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False)

            new_tokens = out[:, input_len:]
            texts = processor.batch_decode(new_tokens,
                                           skip_special_tokens=True)

            for j, text in enumerate(texts):
                row = new_tokens[j].tolist()
                n_new = len(row)
                finish = "length"
                for k, tid in enumerate(row):
                    if tid in stop_ids:
                        n_new = k + 1
                        finish = "eos"
                        break
                out_fh.write(json.dumps({
                    "dataset": args.dataset,
                    "split": args.split,
                    "image_index": int(batch_idx[j]),
                    "true_label": int(batch_lbl[j]),
                    "model": model_id,
                    "revision": revision,
                    "arm": args.arm,
                    "adapter": str(adapter_dir.relative_to(REPO)),
                    "max_new_tokens": args.max_new_tokens,
                    "n_new_tokens": n_new,
                    "finish_reason": finish,
                    "output_text": text,
                }) + "\n")
                n_written += 1

            out_fh.flush()
            print(f"  {n_written}/{len(work)}", end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"  {n_written}/{len(work)} generated in {elapsed:.0f}s")

    records = [json.loads(line) for line in out_path.open()]
    lengths = np.array([r["n_new_tokens"] for r in records])
    truncated = sum(1 for r in records if r["finish_reason"] == "length")

    print("\n=== TOKEN LENGTH DISTRIBUTION ===")
    for p in [50, 90, 99, 100]:
        print(f"  p{p:<3} {np.percentile(lengths, p):6.0f}")
    print(f"  mean {lengths.mean():6.1f}")
    print(f"  truncated at cap: {truncated}/{len(records)}")
    if truncated:
        print("  Truncation detected. Any parsing result from this run is")
        print("  contaminated. Raise the cap and rerun.")

    print("\n=== FIRST 5 RAW OUTPUTS ===")
    for rec in records[:5]:
        truth = CANONICAL[args.dataset][rec["true_label"]]
        print(f"  [idx {rec['image_index']}] truth: {truth}")
        print(f"    {rec['n_new_tokens']} tokens, {rec['finish_reason']}: "
              f"{rec['output_text']!r}")

    cost = elapsed / 3600 * L4_HOURLY_USD
    print(f"\n=== COST ===")
    print(f"  {elapsed:.0f}s  ~${cost:.3f}  "
          f"({elapsed / max(n_written, 1):.2f}s/image)")
    print(f"  raw -> {out_path.relative_to(REPO)}")
    print(f"\nScore with:")
    print(f"  python scripts/score_arms.py {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
