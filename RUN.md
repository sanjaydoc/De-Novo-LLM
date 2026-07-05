# RUN.md — Running De-Novo-LLM on Windows, macOS & Linux

Complete, platform-by-platform setup and run guide. Author: **Dr. Sanjay Anbu**.

Target hardware for local runs: **NVIDIA RTX 3000, 6 GB VRAM, 16 GB RAM, i7**.
The defaults in `configs/` are tuned for exactly this budget.

---

## 0. Prerequisites

- **Python 3.9–3.12**
- **Git**
- For GPU training: an **NVIDIA GPU + recent driver** (CUDA 12.x). Check with:
  ```bash
  nvidia-smi
  ```
- ~5 GB free disk for model weights and caches.

Clone the repository first:

```bash
git clone https://github.com/sanjaydoc/De-Novo-LLM.git
cd De-Novo-LLM
```

---

## 1. Create a virtual environment (Python 3.12)

> Install **Python 3.12** first if you don't have it (python.org). Verify with
> `py -3.12 --version` on Windows, or `python3.12 --version` elsewhere.

### Linux
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### macOS
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Windows — PowerShell
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked once:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows — Command Prompt (cmd)
```bat
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
```

---

## 2. Install PyTorch (do this BEFORE the package)

> **Use Python 3.12.** It has wheels for the entire stack (CUDA PyTorch,
> RDKit, every dependency). Python 3.13/3.14 are **missing CUDA and RDKit
> wheels** — you'll hit "No matching distribution" (GPU torch) and a blocked
> RDKit DLL. Install Python 3.12 from python.org and build the venv with it
> (`py -3.12 -m venv .venv` on Windows, `python3.12 -m venv .venv` elsewhere).

### CPU — Linux / Windows / macOS
```bash
pip install torch
```
The simplest option; plenty for every smoke test in this guide. On macOS Apple
Silicon it also enables the MPS backend automatically.

### NVIDIA GPU (RTX 3000) — Linux / Windows  (needs Python 3.12)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify what you got:
```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```
`...+cpu` → CPU build (fine for demos). `...+cu128 ... True` → GPU ready.

---

## 3. Install De-Novo-LLM

```bash
pip install -e ".[chem]"        # core + RDKit + SELFIES (recommended)
```

Optional extras:
```bash
pip install -e ".[chem,quant,track,dev]"
# quant  -> bitsandbytes (4-bit QLoRA, needed for ProtGPT2/BioMistral)
# track  -> tensorboard
# dev    -> pytest
```

> **Windows + bitsandbytes:** recent `bitsandbytes` wheels support Windows.
> If 4-bit loading fails, prefer a model that fits a full fine-tune
> (ProGen2-small, GPT2-ZINC) or run QLoRA on Linux/WSL2.

---

## 4. Validate your setup (offline smoke tests)

These run on CPU, need no network, and prove each track works end to end. They
use tiny models on toy data — they don't produce real molecules, they prove the
plumbing. If the `denovo` / `denovo-mol` commands aren't found (a pip warning
about `PATH`), use the `python -m` form shown beside each.

```bash
# SE(3)-equivariant flow-matching 3D generator
denovo-mol pipeline -c configs/mol_flow_smoke.yaml
python -m denovo.structure.cli pipeline -c configs/mol_flow_smoke.yaml   # module form

# Sequence-LLM pipeline (builds a tiny local model first — no Hugging Face needed)
python scripts/make_tiny_local_model.py
denovo pipeline -c configs/smoke_local.yaml
python -m denovo.cli pipeline -c configs/smoke_local.yaml                # module form

# Closed-loop Bayesian optimization demo (writes docs/assets/closed_loop.png)
python scripts/closed_loop_demo.py

# Test suite (21 passing)
python -m pytest
```

---

## 5. Real run — fine-tune ProGen2 on your GPU

ProGen2-small (151M) is the recommended de novo protein model and fits 6 GB.

```bash
# (Optional) clean your own data into train/eval splits
denovo prepare  -c configs/progen2_protein.yaml -i data/your_proteins.txt

# Fine-tune
denovo train    -c configs/progen2_protein.yaml

# Generate novel sequences
denovo generate -c configs/progen2_protein.yaml -n 500 -o generated/prot.txt

# Score them (validity / uniqueness / novelty / diversity)
denovo evaluate -c configs/progen2_protein.yaml -i generated/prot.txt
```

Other ready configs:

| Goal | Command |
|------|---------|
| Small molecules (SMILES) | `denovo train -c configs/small_molecule.yaml` |
| Protein via QLoRA (ProtGPT2) | `denovo train -c configs/protein.yaml` |
| DNA (char-level) | `denovo train -c configs/nucleic_acid.yaml` |
| Biomedical text (QLoRA) | `denovo train -c configs/biomistral_7b.yaml` |

Inspect what a config resolves to:
```bash
denovo info -c configs/progen2_protein.yaml
```

---

## 6. Bayesian hyperparameter optimization

Tune decoding hyperparameters (temperature, top-p, top-k) against the quality
metrics using Optuna's TPE (Bayesian) sampler — no retraining needed:

```bash
denovo optimize -c configs/progen2_protein.yaml \
    --mode sampling -m outputs/progen2_small --trials 25 \
    -o docs/results/bo_study.json
```

Render the website graphs from the study:
```bash
python scripts/make_figures.py --study docs/results/bo_study.json
```

Optimize *training* hyperparameters instead (expensive — retrains each trial):
```bash
denovo optimize -c configs/progen2_protein.yaml --mode training --trials 8
```

---

## 7. Benchmarking multiple checkpoints

```bash
python scripts/benchmark.py \
    -c configs/progen2_protein.yaml \
    -m ProGen2-small outputs/progen2_small \
    -m ProtGPT2      outputs/protgpt2_qlora \
    -n 1000
# -> docs/results/benchmark.json + benchmark.md
python scripts/make_figures.py        # refresh docs/assets/*.png
```

---

## 8. Fitting 6 GB — knobs that matter

If you hit **CUDA out of memory**, apply these in order (edit your config):

1. Lower `train.batch_size` (try `1`) and raise `train.grad_accum` to compensate.
2. Set `train.gradient_checkpointing: true`.
3. Keep `train.fp16: true` (NVIDIA) — **on macOS set `fp16: false`**.
4. Reduce `data.max_length` (256 → 128).
5. For models >1B params, set `model.load_in_4bit: true` and `lora.use_lora: true` (QLoRA).
6. Close other GPU apps; the "shared" 6 GB is system RAM and much slower.

Rule of thumb: ≤200M params → full fine-tune · 200M–1B → LoRA · >1B → QLoRA.

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not find a version that satisfies the requirement torch` (GPU/cu index) | The CUDA index has no wheel for your Python. Use **Python 3.12**; or `pip install torch` (CPU). |
| `CUDA available: False` after `pip install torch` | The default wheel is CPU-only on Windows. Install the GPU build: `pip install torch --index-url https://download.pytorch.org/whl/cu128` (Python 3.12). |
| `denovo` / `denovo-mol` : command not found | The console scripts aren't on `PATH`. Activate your venv, or run the module form: `python -m denovo.cli ...` / `python -m denovo.structure.cli ...`. |
| `ModuleNotFoundError: No module named 'matplotlib'` | Reinstall the package (`pip install -e .`) — matplotlib is now a bundled dependency. |
| `ImportError: DLL load failed while importing rdBase: An Application Control policy has blocked this file` | Windows **Smart App Control / Application Control** is blocking RDKit's DLL. Either turn Smart App Control off (Windows Security → App & browser control — one-way change), or just proceed without RDKit (only SMILES-validity metrics are lost; stability metrics still work). |
| `OSError: Can't load ... from huggingface.co` | No internet / blocked network. Models download from the HF Hub on first use. Use the offline `configs/smoke_local.yaml` + `configs/mol_flow_smoke.yaml` to test without downloads. |
| `trust_remote_code` prompt (ProGen2, HyenaDNA) | The config sets `trust_remote_code: true`; that is expected. |
| `bitsandbytes` import error on Windows | Use a full-fine-tune model, or run QLoRA under WSL2/Linux. |
| Out of memory | See section 8. |
| RDKit metrics show as `0` / `-` | RDKit not importable (see the DLL row above) — install/enable RDKit, or accept stability-only metrics. |

---

## 10. Website & docs

- Project website (GitHub Pages): enable Pages from the `/docs` folder in repo
  settings → served at `https://sanjaydoc.github.io/De-Novo-LLM/`.
- Model selection & 6 GB feasibility: [`docs/MODELS.md`](docs/MODELS.md).
