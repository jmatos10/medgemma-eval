# SETUP

Environment record for `medgemma-eval`.

All infrastructure was configured through the Google Cloud Console web UI.
No `gcloud` CLI was used.

---

## 1. Google Cloud

| Item | Value |
|---|---|
| Project name | ML Project |
| Project ID | `ml-project-503500` |
| Project number | `609652923595` |
| Region / zone | `us-central1` / `us-central1-a` |
| Service | Vertex AI Workbench, Instances |
| Machine type | `g2-standard-8` (8 vCPU, 32 GB RAM) |
| GPU | 1x NVIDIA L4, 23,034 MiB |
| Boot disk | 150 GB Balanced Persistent Disk |
| Data disk | 200 GB Balanced Persistent Disk, mounted at `/home/jupyter` |
| Provisioning | On demand, not Spot. Workbench does not offer Spot. |
| Idle shutdown | 30 minutes |
| On-demand rate | $1.082 per hour |

### APIs enabled

Compute Engine, Vertex AI, Notebooks, Secret Manager.

### Quotas

| Quota | Scope | Value |
|---|---|---|
| `NVIDIA_L4_GPUS` | `us-central1` | 1 |
| `GPUS_ALL_REGIONS` | global | 1, granted on request |
| `CPUS` | `us-central1` | 8 |

`GPUS_ALL_REGIONS` defaults to 0 on a new project and must be requested
separately from the regional GPU quota. Regional quota alone is not enough.

### Budget

Budget `medgemma-eval-cap`, scoped to this project only, $50.
Alert thresholds at 50, 90, and 100 percent actual, plus 100 percent forecast.
Budgets notify, they do not cap spending.

### Secrets

Hugging Face read token stored in Secret Manager as `hf-token`.
Read access granted to `609652923595-compute@developer.gserviceaccount.com`
with the role Secret Manager Secret Accessor. The token never appears in a
notebook cell or a committed file.

---

## 2. Verified environment

Output of `nvidia-smi` on first boot:

```
NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: 13.0
NVIDIA L4    0MiB / 23034MiB
```

```
torch                2.12.1+cu130
torch.cuda.is_available()      True
torch.cuda.get_device_capability()   (8, 9)
transformers         5.14.1
huggingface_hub      1.24.0
medmnist             3.0.2
scikit-learn         1.9.0
```

Compute capability 8.9 confirms bf16 support. bf16 requires 8.0 or higher,
so a T4 (7.5) would not work for this project.

### Known issue in the base image

The base image shipped a scikit-learn build that failed to import against its
installed NumPy, raising `ImportError: cannot import name 'ComplexWarning'
from 'numpy.core.numeric'`. This surfaced as a generic
"Please install the required packages first" message from `medmnist`, whose
`__init__.py` catches import errors. Resolved on retry with scikit-learn
1.9.0. Worth checking first if `import medmnist` prints that message.

---

## 3. Pinned model revisions

Every model load pins a revision. Google has
pushed mid-life fixes to this model family, including an end-of-image token
correction documented on the MedGemma card.

| Repo | Revision |
|---|---|
| `google/medgemma-1.5-4b-it` | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |

Both are gated under the Health AI Developer Foundations terms and the Gemma
terms respectively. Access must be accepted on Hugging Face before download.
Status is visible at `huggingface.co/settings/gated-repos`.

MedGemma 1.5 exists only as a 4B multimodal instruction-tuned variant.

### transformers 5.x API notes

Version 5 changed two things this project depends on:

- `from_pretrained(..., dtype=torch.bfloat16)`. The `torch_dtype` argument
  was renamed.
- `processor.apply_chat_template(messages, padding=True,
  add_generation_prompt=True, tokenize=True, return_dict=True,
  return_tensors="pt")` accepts images inline and returns model-ready
  tensors. This replaces the older `processor(text=..., images=...)` call.

Gemma models pad on the left by default, which is required for correct
batched generation in decoder-only models.

---

## 4. Data

MedMNIST+ via `pip install medmnist`, loaded at `size=224`.
Cached under `/home/jupyter/.medmnist`, roughly 2.6 GB for both datasets.

| | DermaMNIST | BloodMNIST |
|---|---|---|
| Download | 1.09 GB | 1.54 GB |
| Train / val / test | 7,007 / 1,003 / 2,005 | 11,959 / official / official |
| Classes | 7 | 8 |
| Shape | (N, 224, 224, 3) | (N, 224, 224, 3) |
| Majority class share (train) | 67.0% | 19.5% |
| Licence | CC BY-NC 4.0 | CC BY 4.0 |

Images are already 3-channel, so `as_rgb=True` is not needed.

DermaMNIST class counts (train): {0: 228, 1: 359, 2: 769, 3: 80, 4: 779,
5: 4693, 6: 99}

BloodMNIST class counts (train): {0: 852, 1: 2181, 2: 1085, 3: 2026,
4: 849, 5: 993, 6: 2330, 7: 1643}

---

## 5. Rebuilding from scratch

Persistent disks bill about $1.15 per day whether the instance runs or not.
Deleting the instance between working sessions and rebuilding is cheaper than
leaving it stopped for more than a few days.

1. Console search `Workbench`, Instances, Create New
2. Region `us-central1`, zone `us-central1-a`
3. Advanced options, Machine type, GPU type NVIDIA L4, 1 GPU, `g2-standard-8`
4. Disks: boot 150 GB, data 200 GB, both Balanced
5. Idle shutdown 30 minutes
6. Open JupyterLab, then:

```
pip install medmnist huggingface_hub google-cloud-secret-manager transformers accelerate
git clone https://github.com/jmatos10/medgemma-eval.git
```

7. Re-download MedMNIST at `size=224`, about 3 minutes
8. Verify with `nvidia-smi` and `python tests/test_eval_harness.py`

The repo is the source of truth. Nothing on the instance is irreplaceable.

---

## 6. Stockout note

On 2026-07-25 a Start request failed with
`STOCKOUT ... zone us-central1-a does not have enough resources`.
This is zone-level GPU capacity, unrelated to quota or configuration.
Retrying after a delay worked. If it persists, `us-central1-b`, `-c`, and
`-f` draw on the same regional quota and are valid alternatives.