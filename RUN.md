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

## 1. Create a virtual environment

### Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows — PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked once:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows — Command Prompt (cmd)
```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

---

## 2. Install PyTorch (do this BEFORE the package)

Pick the line matching your machine.

### NVIDIA GPU (RTX 3000) — Linux / Windows
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### CPU only — Linux / Windows
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### macOS (Apple Silicon or Intel)
```bash
pip install torch          # Apple Silicon uses the MPS backend automatically
```

Verify the GPU is visible:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

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

## 4. Validate your setup (CPU smoke test, ~1 minute)

Runs the entire pipeline (prepare → train → generate → evaluate) with a tiny
random model. It does **not** produce real molecules — it proves your install
works.

```bash
denovo pipeline -c configs/smoke.yaml
```

Run the unit tests too (no ML needed):
```bash
pytest -q
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
| `CUDA available: False` | Reinstall torch with the `cu121` index; update your NVIDIA driver. |
| `OSError: Can't load ... from huggingface.co` | No internet / blocked network. Models download from the HF Hub on first use. |
| `trust_remote_code` prompt (ProGen2, HyenaDNA) | The config sets `trust_remote_code: true`; that is expected. |
| `bitsandbytes` import error on Windows | Use a full-fine-tune model, or run QLoRA under WSL2/Linux. |
| Out of memory | See section 8. |
| RDKit metrics show as `-` | `pip install rdkit selfies`. |

---

## 10. Website & docs

- Project website (GitHub Pages): enable Pages from the `/docs` folder in repo
  settings → served at `https://sanjaydoc.github.io/De-Novo-LLM/`.
- Model selection & 6 GB feasibility: [`docs/MODELS.md`](docs/MODELS.md).
