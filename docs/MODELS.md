# Model guide: what fits a 6GB laptop, and what needs the cloud

Your hardware: **RTX 3000, 6GB VRAM, 16GB RAM, i7**. That budget is the single
biggest constraint on model choice. This page is the honest reality-check.

## The models you named

| Model | Modality | Smallest size | Standard HF? | Fine-tune on 6GB? | Notes |
|-------|----------|---------------|--------------|-------------------|-------|
| **Evo 2** | DNA / genomic | 1B (also 7B, 40B) | ❌ needs `evo2`/vortex, StripedHyena-2, FP8 | ❌ **No** | Designed for H100-class FP8 GPUs. Even 1B inference is impractical at 6GB. Use via **NVIDIA NIM / BioNeMo** in the cloud. |
| **ESM-3** | Protein | 1.4B (`esm3-sm-open-v1`) | ❌ own `esm` SDK, license-gated | ❌ **No** (6GB) | Not an `AutoModelForCausalLM`; needs a dedicated adapter. QLoRA needs ~12GB+. Great as a **cloud/NIM** generator. |
| **BioMistral-7B** | Biomedical **text** | 7B | ✅ Mistral | ⚠️ **Barely** (QLoRA) | Generates biomedical *prose*, not raw sequences. 4-bit + batch 1 + grad-ckpt + seq≤256 fits 6GB but is slow/tight. |

**Key point:** Evo 2 and ESM-3 are the real *biomolecule sequence* generators,
but neither fits your GPU. BioMistral-7B fits (barely) but is a *text* model.

## NVIDIA NIM — where it fits

NVIDIA **NIM** (and BioNeMo) host Evo 2, ESM, and other bio-foundation models as
**inference** microservices/APIs (see build.nvidia.com). That is the practical
way to *use* Evo 2 / ESM-3 without an H100 — but NIM is for **inference**, not
for fine-tuning on your laptop. Fine-tuning those giants is done with the
**NVIDIA BioNeMo Framework** on cloud/DGX, not locally. So:

- Want to **generate** with Evo 2 / ESM-3 today → call them via **NIM** (cloud API).
- Want to **fine-tune locally on 6GB** → use the small siblings below.

## Recommended: models that ARE good at de novo AND fit 6GB

These are drop-in with this repo's pipeline (`AutoModelForCausalLM`), so the
same `prepare → train → generate → evaluate` flow just works.

| Modality | Recommended model | Params | Why | Config |
|----------|-------------------|--------|-----|--------|
| **Protein / peptide** ⭐ | **ProGen2-small** (`hugohrban/progen2-small`) | 151M | Purpose-built **autoregressive** protein LM — literally trained for de novo generation. Fits 6GB with room to spare. | `configs/progen2_protein.yaml` |
| Protein (alt) | ProtGPT2 (`nferruz/ProtGPT2`) | 738M | Popular de novo protein LM; needs 4-bit QLoRA at 6GB. | `configs/protein.yaml` |
| Small molecules | GPT2-ZINC (`entropy/gpt2_zinc_87m`) | 87M | GPT-2 pretrained on ZINC SMILES; full fine-tune fits easily. | `configs/small_molecule.yaml` |
| Small molecules (alt) | ChemGPT-4.7M (`ncfrey/ChemGPT-4.7M`) | 4.7M | Tiny SELFIES model; trains in minutes. | set `modality: selfies` |
| DNA / RNA | GPT-2 char-level, or HyenaDNA-tiny | 124M / 1.6M | Local, standard, fits 6GB. Evo 2 stand-in for experimentation. | `configs/nucleic_acid.yaml` |
| Biomedical text | BioGPT (`microsoft/biogpt`, 347M) | 347M | If you want a *generative biomedical text* model that actually fits, this beats BioMistral-7B on your GPU. | — |

## My recommendation (one model, good at de novo)

For **de novo biomolecule generation on your laptop, start with
`ProGen2-small`** (`configs/progen2_protein.yaml`):

- It is an autoregressive model *designed* for de novo protein design.
- 151M params → comfortable full fine-tune on 6GB (no QLoRA gymnastics).
- Standard sampling → the repo's validity/uniqueness/novelty metrics apply directly.

Then, when you want the heavyweight quality of **Evo 2 (DNA)** or **ESM-3
(protein)**, run those via **NVIDIA NIM / BioNeMo in the cloud** for generation,
or fine-tune them on a rented A100 using the same config format (just point
`pretrained_model` at them on the bigger machine).

## Cheat-sheet: which knobs to turn at 6GB

- Prefer **full fine-tune** for ≤200M models; **LoRA** for ~200M–1B; **QLoRA
  (`load_in_4bit`)** for >1B.
- Keep `batch_size` small (1–4) and use `grad_accum` to reach an effective batch.
- Turn on `gradient_checkpointing: true` and `fp16: true` to save VRAM.
- Shorten `max_length` (128–256) — attention memory grows with sequence length.
