# llm-bias-eval

Bachelor thesis (UCPH, supervised by Anders Søgaard) — investigating whether instruction-tuning genuinely reduces stereotypical bias in LLMs or merely suppresses it at the output level while leaving internal representations biased.

**Hypothesis:** Alignment significantly suppresses *expressed* bias but the underlying latent representations retain or even concentrate historical bias levels.

## Approach

| Layer | What we measure | Benchmarks / Method |
|---|---|---|
| Expressed | Output-level bias via logits | CrowS-Pairs, BBQ, StereoSet, LLM-IAT |
| Encoded | Internal representational bias | Linear probes on residual-stream activations |

Compared across **24 base ↔ instruct pairs (48 checkpoints)** spanning Llama, Qwen, Gemma, Mistral, OLMo.

## Setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
brew install uv

# 2. Install dependencies
uv sync --extra dev

# 3. Configure secrets
cp .env.example .env
# Edit .env to add your HF_TOKEN and WANDB_API_KEY
```

You need:
- A **HuggingFace token** with access to gated repos (Llama, Gemma). Accept licenses at the model pages on huggingface.co before running.
- A **Weights & Biases account** (free academic tier at wandb.ai).

## Running

Compute target: 1× NVIDIA A100 80GB (Google Colab Pro+). Locally, only the smoke test on a tiny model is expected to run.

```bash
# Smoke test (CPU/MPS, tiny model, ~1 min)
uv run pytest tests/ -m integration

# Stage 1: logit benchmarks on all 48 models (~30–40 GPU-h)
uv run python scripts/run_logit_benchmarks.py --config configs/models.yaml

# Stage 2: probing on the 14-model subset (~10–15 GPU-h)
uv run python scripts/run_probing.py --config configs/models.yaml

# Stage 3: aggregate + plot
uv run python scripts/run_analysis.py
```

All scripts checkpoint per model and resume on restart.

## Layout

```
src/biaseval/        # Library code
configs/             # Model registry + benchmark configs
scripts/             # Stage 1/2/3 run scripts + download
notebooks/           # Exploratory notebooks
tests/               # Unit + integration tests
results/             # Outputs (git-ignored)
figures/             # Generated thesis figures
docs/                # Project description, notes
```

See `docs/project_description.md` for the full thesis brief.
