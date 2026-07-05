# De-Novo-LLM

> Fine-tune language models to generate **de novo biomolecules** — small
> molecules (SMILES/SELFIES), proteins/peptides, and nucleic acids (DNA/RNA) —
> from one modular, config-driven pipeline.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers%20%7C%20PEFT-ee4c2c)
![Tuning](https://img.shields.io/badge/fine--tuning-LoRA%20%7C%20QLoRA-7c3aed)
![BayesOpt](https://img.shields.io/badge/Bayesian%20Opt-Optuna%20TPE-059669)
![License](https://img.shields.io/badge/license-MIT-green)
![Platforms](https://img.shields.io/badge/OS-Windows%20%7C%20macOS%20%7C%20Linux-informational)

**Author:** Dr. Sanjay Anbu · **Website:** https://sanjaydoc.github.io/De-Novo-LLM/ ·
**Run guide:** [RUN.md](RUN.md) · **Model guide:** [docs/MODELS.md](docs/MODELS.md)

Built to run on modest hardware (developed against an **RTX 3000, 6GB VRAM**):
tiny models fine-tune fully, larger ones via **LoRA / 4-bit QLoRA**. Ships a
**Bayesian hyperparameter optimizer** (Optuna/TPE) and a **benchmarking +
figure pipeline** for the project website.

Cross-platform commands (Windows / macOS / Linux) are in **[RUN.md](RUN.md)**.

## Why a registry?

Every biomolecule type is described once in a **modality registry**
(`src/denovo/modalities.py`): how to validate a string, how to canonicalise it,
and a sensible default model. The `prepare → train → generate → evaluate`
machinery is completely modality-agnostic — so adding a new biomolecule type is
a few lines, not a new pipeline.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core
pip install rdkit selfies        # chemistry validity + metrics (recommended)
pip install bitsandbytes         # only for 4-bit QLoRA (protein/large models)
```

For CUDA, install a matching torch build first, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 60-second smoke test (CPU, no GPU needed)

Proves the whole pipeline works end-to-end with a tiny random model:

```bash
denovo pipeline -c configs/smoke.yaml
```

## Real runs

```bash
# 1. Clean your raw data into train/eval splits (validates + canonicalises)
denovo prepare  -c configs/small_molecule.yaml -i data/your_smiles.txt

# 2. Fine-tune
denovo train    -c configs/small_molecule.yaml

# 3. Generate novel molecules
denovo generate -c configs/small_molecule.yaml -n 1000 -o generated/mols.txt

# 4. Score them (validity / uniqueness / novelty / diversity)
denovo evaluate -c configs/small_molecule.yaml -i generated/mols.txt

# ...or all four at once:
denovo pipeline -c configs/small_molecule.yaml -i data/your_smiles.txt
```

### Bayesian hyperparameter optimization

Tune decoding hyperparameters against the quality metrics with Optuna's TPE
(Bayesian) sampler — no retraining needed:

```bash
denovo optimize -c configs/progen2_protein.yaml \
    --mode sampling -m outputs/progen2_small --trials 25 \
    -o docs/results/bo_study.json
python scripts/make_figures.py --study docs/results/bo_study.json
```

Inspect what a config resolves to:

```bash
denovo info -c configs/progen2_protein.yaml
```

## Which model should I use?

Short answer for a 6GB laptop and **de novo** quality: start with
**ProGen2-small** (`configs/progen2_protein.yaml`). Full comparison of
Evo 2 / ESM-3 / BioMistral-7B / ProGen2 / GPT2-ZINC and what fits your GPU is in
**[docs/MODELS.md](docs/MODELS.md)**.

| Modality | Default config | Model | Fits 6GB |
|----------|----------------|-------|----------|
| Small molecules (SMILES) | `configs/small_molecule.yaml` | `entropy/gpt2_zinc_87m` | ✅ full fine-tune |
| Protein (recommended) | `configs/progen2_protein.yaml` | `hugohrban/progen2-small` | ✅ full fine-tune |
| Protein (larger) | `configs/protein.yaml` | `nferruz/ProtGPT2` | ✅ QLoRA |
| DNA / RNA | `configs/nucleic_acid.yaml` | `gpt2` char-level / HyenaDNA | ✅ |
| Biomedical text | `configs/biomistral_7b.yaml` | `BioMistral/BioMistral-7B` | ⚠️ QLoRA, tight |

## Configuration

A run is one YAML file with five sections — `data`, `model`, `lora`, `train`,
`generate`. Only override what you need; everything else uses 6GB-friendly
defaults. See `src/denovo/config.py` for every field and its default.

```yaml
data:
  modality: smiles                 # smiles | selfies | protein | dna | rna
  train_file: data/my_smiles.txt
model:
  pretrained_model: entropy/gpt2_zinc_87m   # any HF causal LM; blank = modality default
lora:
  use_lora: false                  # true = LoRA; add model.load_in_4bit for QLoRA
train:
  output_dir: outputs/run
  fp16: true
generate:
  num_samples: 1000
```

## Metrics

`denovo evaluate` reports the field-standard de novo metrics:

- **Validity** — fraction of generations that parse (RDKit for molecules;
  alphabet checks for sequences).
- **Uniqueness** — distinct valid / valid.
- **Novelty** — valid-unique not present in the training set (canonicalised).
- **Diversity** — mean pairwise Morgan-fingerprint distance (molecules; needs RDKit).

## Project layout

```
configs/            ready-to-run YAML configs (start with smoke.yaml)
data/samples/       tiny example datasets (SMILES / protein / DNA)
docs/MODELS.md      model comparison + 6GB feasibility guide
src/denovo/
  modalities.py     the registry: validators, canonicalisers, defaults
  config.py         typed YAML config
  data.py           read / clean / tokenise
  model.py          load model+tokenizer, LoRA / QLoRA
  train.py          HF Trainer fine-tuning loop
  generate.py       sampling
  evaluate.py       validity / uniqueness / novelty / diversity
  cli.py            `denovo` command-line entry point
tests/              unit tests for the modality-agnostic core
```

## Roadmap

- [x] Modular modality registry (molecules / proteins / nucleic acids)
- [x] Fine-tune (full / LoRA / QLoRA), generate, evaluate
- [x] Bayesian hyperparameter optimization (Optuna / TPE)
- [x] Benchmarking + figure pipeline and GitHub Pages website
- [ ] Property-conditioned generation (logP, QED, target binding)
- [ ] Adapters for SDK-based giants (ESM-3, Evo 2) + NVIDIA NIM inference
- [ ] SE(3)-equivariant structure track (RFdiffusion → ProteinMPNN) for 3D de novo design
- [ ] Scaffold / target-constrained decoding

## Author & citation

**Dr. Sanjay Anbu** — [github.com/sanjaydoc/De-Novo-LLM](https://github.com/sanjaydoc/De-Novo-LLM)

```bibtex
@software{anbu_denovo_llm_2026,
  author = {Sanjay Anbu},
  title  = {De-Novo-LLM: Fine-tuning language models for de novo biomolecule generation},
  year   = {2026},
  url    = {https://github.com/sanjaydoc/De-Novo-LLM}
}
```

## License

Released under the [MIT License](LICENSE) © 2026 Dr. Sanjay Anbu.
