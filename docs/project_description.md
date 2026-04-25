# Project Brief: Alignment vs. Latent Bias in Large Language Models

## For Claude Code — set up a complete, runnable research project

---

## 1. Project Overview

This is a bachelor thesis project at the University of Copenhagen (supervised by Anders Søgaard) investigating whether instruction-tuning/alignment (RLHF, DPO, SFT) genuinely reduces stereotypical bias in LLMs or merely suppresses it at the output level while leaving internal representations biased.

**Core hypothesis:** Alignment significantly suppresses *expressed* (output-level) bias but the underlying latent representations retain or even concentrate historical bias levels.

**Two-pronged approach:**
1. **Expressed bias**: Measure behavioral bias via logit/probability distributions on bias benchmarks (CrowS-Pairs, BBQ, StereoSet, LLM-IAT)
2. **Encoded bias**: Probe internal geometry via linear probes on residual-stream activations to test whether social biases remain linearly separable despite neutralized outputs

**Experimental design:** Compare **base vs. instruction-tuned** versions of the same checkpoint within five open-source model families (Llama, Qwen, Gemma, Mistral, OLMo) across multiple generations and scales.

---

## 2. Compute Constraints

- **Hardware**: 1× NVIDIA A100 80GB (Google Colab)
- **All models run in full bfloat16** — no quantization. The largest model (Qwen2.5-32B, ~64GB weights) fits on A100 80GB with batch size 1.
- For logit benchmarks: batch size 1, minimal KV-cache overhead (~1–2GB)
- For probing: stream activations to disk one layer at a time to stay within VRAM budget
- **Total estimated compute**: ~40–55 A100-hours

---

## 3. Complete Model List

Every model is a **base ↔ instruct pair**. This is the core of the experimental design — we need both variants from the same checkpoint to isolate alignment effects.

### 3.1 Llama Family (5 pairs = 10 models)

| Generation | Base | Instruct | Params | HF ID (base) | HF ID (instruct) | Notes |
|---|---|---|---|---|---|---|
| Llama 2 | ✓ | ✓ | 7B | `meta-llama/Llama-2-7b-hf` | `meta-llama/Llama-2-7b-chat-hf` | Earliest generation |
| Llama 3 | ✓ | ✓ | 8B | `meta-llama/Meta-Llama-3-8B` | `meta-llama/Meta-Llama-3-8B-Instruct` | |
| Llama 3.1 | ✓ | ✓ | 8B | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` | |
| Llama 3.2 | ✓ | ✓ | 1B | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` | Smallest Llama |
| Llama 3.2 | ✓ | ✓ | 3B | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.2-3B-Instruct` | |

### 3.2 Qwen Family (7 pairs = 14 models)

| Generation | Base | Instruct | Params | HF ID (base) | HF ID (instruct) | Notes |
|---|---|---|---|---|---|---|
| Qwen 2.5 | ✓ | ✓ | 1.5B | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | |
| Qwen 2.5 | ✓ | ✓ | 3B | `Qwen/Qwen2.5-3B` | `Qwen/Qwen2.5-3B-Instruct` | |
| Qwen 2.5 | ✓ | ✓ | 7B | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` | |
| Qwen 2.5 | ✓ | ✓ | 14B | `Qwen/Qwen2.5-14B` | `Qwen/Qwen2.5-14B-Instruct` | |
| Qwen 2.5 | ✓ | ✓ | 32B | `Qwen/Qwen2.5-32B` | `Qwen/Qwen2.5-32B-Instruct` | ~64GB in bf16, tight but fits |
| Qwen 3 | ✓ | ✓ | 4B | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B` | Cross-gen comparison |
| Qwen 3 | ✓ | ✓ | 8B | `Qwen/Qwen3-8B-Base` | `Qwen/Qwen3-8B` | Cross-gen comparison |

### 3.3 Gemma Family (8 pairs = 16 models)

**Why all four generations:** Gemma 1–4 all have publicly available base (`-pt` suffix for Gemma 2+) and instruction-tuned (`-it`) variants. Having four generations from the same vendor with consistent naming/architecture lineage gives us the strongest longitudinal signal of any family.

| Generation | Base | Instruct | Params | HF ID (base) | HF ID (instruct) | Notes |
|---|---|---|---|---|---|---|
| Gemma 1 | ✓ | ✓ | 2B | `google/gemma-2b` | `google/gemma-2b-it` | Feb 2024 |
| Gemma 1 | ✓ | ✓ | 7B | `google/gemma-7b` | `google/gemma-7b-it` | Feb 2024 |
| Gemma 2 | ✓ | ✓ | 2B | `google/gemma-2-2b` | `google/gemma-2-2b-it` | Jun 2024 |
| Gemma 2 | ✓ | ✓ | 9B | `google/gemma-2-9b` | `google/gemma-2-9b-it` | Jun 2024 |
| Gemma 2 | ✓ | ✓ | 27B | `google/gemma-2-27b` | `google/gemma-2-27b-it` | ~54GB in bf16 |
| Gemma 3 | ✓ | ✓ | 4B | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | Mar 2025, use Gemma3ForCausalLM |
| Gemma 3 | ✓ | ✓ | 12B | `google/gemma-3-12b-pt` | `google/gemma-3-12b-it` | Mar 2025, use Gemma3ForCausalLM |
| Gemma 4 | ✓ | ✓ | 31B | `google/gemma-4-31B` | `google/gemma-4-31B-it` | Apr 2026, ~62GB in bf16, use Gemma4ForCausalLM |

### 3.4 Mistral Family (2 pairs = 4 models)

**Why Mistral:** The fourth major open-source LLM vendor. Included in nearly all published bias studies. Adds an independent data point for cross-family generalization. Both v0.1 (original) and v0.3 (extended vocab) have base+instruct pairs.

| Generation | Base | Instruct | Params | HF ID (base) | HF ID (instruct) | Notes |
|---|---|---|---|---|---|---|
| Mistral v0.1 | ✓ | ✓ | 7B | `mistralai/Mistral-7B-v0.1` | `mistralai/Mistral-7B-Instruct-v0.1` | Sep 2023 |
| Mistral v0.3 | ✓ | ✓ | 7B | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | May 2024, extended vocab |

### 3.5 OLMo Family (2 pairs = 4 models)

**Why OLMo:** AllenAI's fully open-science model family. Training data (Dolma), code, intermediate checkpoints, and post-training recipes (Tülu 3) are all publicly available. This is uniquely valuable for a thesis supervised by Anders Søgaard — it enables discussion of data transparency and reproducibility. The alignment pipeline (SFT + DPO + RLVR via Tülu 3) is fully documented.

| Generation | Base | Instruct | Params | HF ID (base) | HF ID (instruct) | Notes |
|---|---|---|---|---|---|---|
| OLMo 2 | ✓ | ✓ | 7B | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-1124-7B-Instruct` | Nov 2024, trained on Dolma/DCLM 5T tokens |
| OLMo 2 | ✓ | ✓ | 13B | `allenai/OLMo-2-1124-13B` | `allenai/OLMo-2-1124-13B-Instruct` | Nov 2024, Apache 2.0 |

**Total: 24 pairs = 48 model checkpoints**

---

## 4. Benchmarks and Datasets

### 4.1 CrowS-Pairs (Primary — logit-based)
- **What**: 1,508 minimal-pair sentences across 9 bias categories (race, gender, religion, age, nationality, disability, physical appearance, socioeconomic status, sexual orientation)
- **Metric**: For each pair, compute sum of token log-probabilities for the stereotypical vs. anti-stereotypical sentence. Report percentage of pairs where the model assigns higher likelihood to the stereotypical sentence.
- **Implementation**: Use `lm-evaluation-harness` autoregressive variant (`crows_pairs_english` task) or custom implementation (see Section 8.2)
- **Dataset**: `nyu-mll/crows_pairs` on HuggingFace
- **Run on**: ALL 48 models

### 4.2 BBQ — Bias Benchmark for QA (logit-based)
- **What**: Multiple-choice QA across 9 social dimensions with ambiguous and disambiguated contexts
- **Metric**: Bias score based on whether model selects stereotypical answer in ambiguous contexts
- **Implementation**: Available in `lm-evaluation-harness` (`bbq` task)
- **Dataset**: `heegyu/bbq` on HuggingFace
- **Run on**: ALL 48 models

### 4.3 StereoSet (logit-based)
- **What**: Intra-sentence and inter-sentence bias test across 4 domains (gender, profession, race, religion). Each context is paired with a stereotypical, anti-stereotypical, and unrelated continuation.
- **Metric**: Three scores: (1) Stereotype Score (SS) — percentage of stereotypical preference over anti-stereotypical, (2) Language Modeling Score (LMS) — percentage of meaningful (non-unrelated) preference, (3) Idealized CAT Score (ICAT) — combines SS and LMS to penalize models that are unbiased only because they are bad at language modeling
- **Why**: Unlike CrowS-Pairs which only distinguishes "more" vs "less" stereotyped, StereoSet separates bias from language competence via ICAT. This addresses the concern that small models may appear unbiased simply because they produce near-random outputs.
- **Implementation**: Available in `lm-evaluation-harness` (task: `stereoset`)
- **Dataset**: `McGill-NLP/stereoset` on HuggingFace
- **Run on**: ALL 48 models

### 4.4 LLM Implicit Association Test (IAT) — (logit-based, psychology-inspired)
- **What**: Adaptation of the human Implicit Association Test for LLMs (Bai et al., PNAS 2025). Uses word-association prompts to measure implicit stereotypes across 4 social categories (race, gender, religion, health) and 21 specific stereotypes.
- **Metric**: IAT effect size (analogous to Cohen's d) — measures the strength of association between target concepts (e.g., "Black"/"White") and attributes (e.g., "pleasant"/"unpleasant") based on response latency or probability differences
- **Why**: Specifically designed to detect bias in models that score near-zero on explicit benchmarks. Directly tests the thesis hypothesis — aligned models may pass CrowS-Pairs/BBQ but still show large implicit bias on IAT.
- **Implementation**: Open-source code at `github.com/baixuechunzi/llm-implicit-bias`. Custom integration needed (not in lm-eval-harness), but straightforward — just prompt-based log-probability comparisons.
- **Dataset**: Included in the GitHub repo
- **Run on**: ALL 48 models

### 4.5 Linear Probes on Residual Stream (representation-level)
- **What**: Extract hidden-state activations at every layer for CrowS-Pairs sentences, train logistic regression classifiers to predict protected attributes (gender, race) from activations
- **Metric**: Probe classification accuracy per layer. If bias is linearly encoded, accuracy >> 50%. Compare base vs. instruct: if alignment removes encoded bias, probe accuracy should drop in instruct models.
- **Implementation**: Custom code using `nnsight` or `nnterp` for activation extraction + `scikit-learn` LogisticRegression
- **Run on**: Subset of representative models (see probing_subset in config). For large models (27B+), stream activations to disk one layer at a time.

---

## 5. Project Structure

```
llm-bias-eval/
├── README.md
├── pyproject.toml                  # or requirements.txt
├── configs/
│   ├── models.yaml                 # Complete model registry with HF IDs, families, sizes
│   └── benchmarks.yaml             # Benchmark configs (task names, metrics, categories)
├── src/
│   ├── __init__.py
│   ├── model_loader.py             # Unified model loading (handles HF auth, device mapping, model classes)
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── crows_pairs.py          # CrowS-Pairs logit-based evaluation
│   │   ├── bbq.py                  # BBQ evaluation
│   │   ├── stereoset.py            # StereoSet evaluation (SS, LMS, ICAT scores)
│   │   ├── iat.py                  # LLM Implicit Association Test (Bai et al., PNAS 2025)
│   │   └── utils.py                # Shared scoring utilities
│   ├── probing/
│   │   ├── __init__.py
│   │   ├── extract_activations.py  # Extract and cache residual-stream activations per layer (stream to disk)
│   │   ├── linear_probe.py         # Train logistic regression probes, report accuracy per layer
│   │   └── datasets.py             # Prepare probe training data (labeled sentences with protected attributes)
│   └── analysis/
│       ├── __init__.py
│       ├── aggregate_results.py    # Combine all results into unified DataFrames
│       └── plotting.py             # Generate all thesis figures
├── scripts/
│   ├── run_logit_benchmarks.py     # Stage 1: Run CrowS-Pairs + BBQ on all models
│   ├── run_probing.py              # Stage 2: Extract activations + train probes on subset
│   ├── run_analysis.py             # Stage 3: Aggregate + plot
│   └── download_models.py          # Pre-download all model weights
├── notebooks/
│   ├── 01_logit_results.ipynb      # Explore Stage 1 results
│   ├── 02_probing_results.ipynb    # Explore Stage 2 results
│   └── 03_figures.ipynb            # Generate publication-quality figures
├── results/                        # Git-ignored, populated by runs
│   ├── logit_scores/               # JSON/CSV per model
│   ├── activations/                # Cached activations (large, git-ignored)
│   └── probe_results/              # Probe accuracies per model per layer
└── figures/                        # Generated plots for thesis
```

---

## 6. Dependencies

```
# Core
torch>=2.1
transformers>=4.51        # Required for Qwen3 and Gemma 3/4 support
accelerate
safetensors

# Benchmarks
lm-eval>=0.4.0            # EleutherAI lm-evaluation-harness

# Probing
nnsight                   # Activation extraction with tracing
scikit-learn              # Logistic regression probes
numpy
scipy

# Analysis & Plotting
pandas
matplotlib
seaborn
plotly                    # Optional: interactive plots

# Utilities
datasets                  # HuggingFace datasets
huggingface_hub
pyyaml
tqdm
jsonlines
```

---

## 7. Configuration File: `configs/models.yaml`

This is the central registry. Every script should read from this file.

```yaml
families:
  llama:
    display_name: "Llama"
    vendor: "Meta"
    generations:
      - name: "Llama 2"
        release_date: "2023-07"
        models:
          - size: "7B"
            base_id: "meta-llama/Llama-2-7b-hf"
            instruct_id: "meta-llama/Llama-2-7b-chat-hf"
            num_params: 7_000_000_000
            num_layers: 32
            hidden_size: 4096
            dtype: "bfloat16"
      - name: "Llama 3"
        release_date: "2024-04"
        models:
          - size: "8B"
            base_id: "meta-llama/Meta-Llama-3-8B"
            instruct_id: "meta-llama/Meta-Llama-3-8B-Instruct"
            num_params: 8_000_000_000
            num_layers: 32
            hidden_size: 4096
            dtype: "bfloat16"
      - name: "Llama 3.1"
        release_date: "2024-07"
        models:
          - size: "8B"
            base_id: "meta-llama/Llama-3.1-8B"
            instruct_id: "meta-llama/Llama-3.1-8B-Instruct"
            num_params: 8_000_000_000
            num_layers: 32
            hidden_size: 4096
            dtype: "bfloat16"
      - name: "Llama 3.2"
        release_date: "2024-09"
        models:
          - size: "1B"
            base_id: "meta-llama/Llama-3.2-1B"
            instruct_id: "meta-llama/Llama-3.2-1B-Instruct"
            num_params: 1_000_000_000
            num_layers: 16
            hidden_size: 2048
            dtype: "bfloat16"
          - size: "3B"
            base_id: "meta-llama/Llama-3.2-3B"
            instruct_id: "meta-llama/Llama-3.2-3B-Instruct"
            num_params: 3_000_000_000
            num_layers: 28
            hidden_size: 3072
            dtype: "bfloat16"

  qwen:
    display_name: "Qwen"
    vendor: "Alibaba"
    generations:
      - name: "Qwen 2.5"
        release_date: "2024-09"
        models:
          - size: "1.5B"
            base_id: "Qwen/Qwen2.5-1.5B"
            instruct_id: "Qwen/Qwen2.5-1.5B-Instruct"
            num_params: 1_500_000_000
            num_layers: 28
            hidden_size: 1536
            dtype: "bfloat16"
          - size: "3B"
            base_id: "Qwen/Qwen2.5-3B"
            instruct_id: "Qwen/Qwen2.5-3B-Instruct"
            num_params: 3_000_000_000
            num_layers: 36
            hidden_size: 2048
            dtype: "bfloat16"
          - size: "7B"
            base_id: "Qwen/Qwen2.5-7B"
            instruct_id: "Qwen/Qwen2.5-7B-Instruct"
            num_params: 7_000_000_000
            num_layers: 28
            hidden_size: 3584
            dtype: "bfloat16"
          - size: "14B"
            base_id: "Qwen/Qwen2.5-14B"
            instruct_id: "Qwen/Qwen2.5-14B-Instruct"
            num_params: 14_000_000_000
            num_layers: 48
            hidden_size: 5120
            dtype: "bfloat16"
          - size: "32B"
            base_id: "Qwen/Qwen2.5-32B"
            instruct_id: "Qwen/Qwen2.5-32B-Instruct"
            num_params: 32_000_000_000
            num_layers: 64
            hidden_size: 5120
            dtype: "bfloat16"
            notes: "~64GB in bf16, fits on A100 80GB with batch_size=1"
      - name: "Qwen 3"
        release_date: "2025-04"
        models:
          - size: "4B"
            base_id: "Qwen/Qwen3-4B-Base"
            instruct_id: "Qwen/Qwen3-4B"
            num_params: 4_000_000_000
            num_layers: 36
            hidden_size: 2560
            dtype: "bfloat16"
          - size: "8B"
            base_id: "Qwen/Qwen3-8B-Base"
            instruct_id: "Qwen/Qwen3-8B"
            num_params: 8_000_000_000
            num_layers: 36
            hidden_size: 4096
            dtype: "bfloat16"

  gemma:
    display_name: "Gemma"
    vendor: "Google"
    generations:
      - name: "Gemma 1"
        release_date: "2024-02"
        models:
          - size: "2B"
            base_id: "google/gemma-2b"
            instruct_id: "google/gemma-2b-it"
            num_params: 2_000_000_000
            num_layers: 18
            hidden_size: 2048
            dtype: "bfloat16"
            model_class: "AutoModelForCausalLM"
          - size: "7B"
            base_id: "google/gemma-7b"
            instruct_id: "google/gemma-7b-it"
            num_params: 7_000_000_000
            num_layers: 28
            hidden_size: 3072
            dtype: "bfloat16"
            model_class: "AutoModelForCausalLM"
      - name: "Gemma 2"
        release_date: "2024-06"
        models:
          - size: "2B"
            base_id: "google/gemma-2-2b"
            instruct_id: "google/gemma-2-2b-it"
            num_params: 2_000_000_000
            num_layers: 26
            hidden_size: 2304
            dtype: "bfloat16"
            model_class: "AutoModelForCausalLM"
          - size: "9B"
            base_id: "google/gemma-2-9b"
            instruct_id: "google/gemma-2-9b-it"
            num_params: 9_000_000_000
            num_layers: 42
            hidden_size: 3584
            dtype: "bfloat16"
            model_class: "AutoModelForCausalLM"
          - size: "27B"
            base_id: "google/gemma-2-27b"
            instruct_id: "google/gemma-2-27b-it"
            num_params: 27_000_000_000
            num_layers: 46
            hidden_size: 4608
            dtype: "bfloat16"
            model_class: "AutoModelForCausalLM"
            notes: "~54GB in bf16, fits comfortably on A100 80GB"
      - name: "Gemma 3"
        release_date: "2025-03"
        notes: "Multimodal models — use Gemma3ForCausalLM for text-only mode to avoid loading vision encoder"
        models:
          - size: "4B"
            base_id: "google/gemma-3-4b-pt"
            instruct_id: "google/gemma-3-4b-it"
            num_params: 4_000_000_000
            num_layers: 34
            hidden_size: 2560
            dtype: "bfloat16"
            model_class: "Gemma3ForCausalLM"
          - size: "12B"
            base_id: "google/gemma-3-12b-pt"
            instruct_id: "google/gemma-3-12b-it"
            num_params: 12_000_000_000
            num_layers: 48
            hidden_size: 3840
            dtype: "bfloat16"
            model_class: "Gemma3ForCausalLM"
      - name: "Gemma 4"
        release_date: "2026-04"
        notes: "Newest generation — use Gemma4ForCausalLM for text-only mode to avoid loading vision encoder"
        models:
          - size: "31B"
            base_id: "google/gemma-4-31B"
            instruct_id: "google/gemma-4-31B-it"
            num_params: 31_000_000_000
            num_layers: 50
            hidden_size: 4608
            dtype: "bfloat16"
            model_class: "Gemma4ForCausalLM"
            notes: "~62GB in bf16, fits on A100 80GB with batch_size=1"

  mistral:
    display_name: "Mistral"
    vendor: "Mistral AI"
    generations:
      - name: "Mistral v0.1"
        release_date: "2023-09"
        models:
          - size: "7B"
            base_id: "mistralai/Mistral-7B-v0.1"
            instruct_id: "mistralai/Mistral-7B-Instruct-v0.1"
            num_params: 7_000_000_000
            num_layers: 32
            hidden_size: 4096
            dtype: "bfloat16"
      - name: "Mistral v0.3"
        release_date: "2024-05"
        models:
          - size: "7B"
            base_id: "mistralai/Mistral-7B-v0.3"
            instruct_id: "mistralai/Mistral-7B-Instruct-v0.3"
            num_params: 7_000_000_000
            num_layers: 32
            hidden_size: 4096
            dtype: "bfloat16"

  olmo:
    display_name: "OLMo"
    vendor: "AllenAI"
    notes: "Fully open-science models. Training data (Dolma/DCLM), code, intermediate checkpoints, and post-training recipes (Tülu 3) are all publicly available. Apache 2.0."
    generations:
      - name: "OLMo 2"
        release_date: "2024-11"
        models:
          - size: "7B"
            base_id: "allenai/OLMo-2-1124-7B"
            instruct_id: "allenai/OLMo-2-1124-7B-Instruct"
            num_params: 7_000_000_000
            num_layers: 32
            hidden_size: 4096
            dtype: "bfloat16"
          - size: "13B"
            base_id: "allenai/OLMo-2-1124-13B"
            instruct_id: "allenai/OLMo-2-1124-13B-Instruct"
            num_params: 13_000_000_000
            num_layers: 40
            hidden_size: 5120
            dtype: "bfloat16"

# Probing subset — run linear probes on these models (activation extraction is memory-intensive)
# One representative pair per family, plus OLMo for its transparency value
# For models >14B, stream activations to disk one layer at a time
probing_subset:
  - "meta-llama/Llama-2-7b-hf"
  - "meta-llama/Llama-2-7b-chat-hf"
  - "meta-llama/Llama-3.1-8B"
  - "meta-llama/Llama-3.1-8B-Instruct"
  - "Qwen/Qwen2.5-7B"
  - "Qwen/Qwen2.5-7B-Instruct"
  - "google/gemma-2-9b"
  - "google/gemma-2-9b-it"
  - "google/gemma-3-4b-pt"
  - "google/gemma-3-4b-it"
  - "mistralai/Mistral-7B-v0.3"
  - "mistralai/Mistral-7B-Instruct-v0.3"
  - "allenai/OLMo-2-1124-7B"
  - "allenai/OLMo-2-1124-7B-Instruct"
```

---

## 8. Key Implementation Details

### 8.1 Model Loading

```python
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_id: str, config: dict):
    """Load any model from the registry with appropriate settings.
    
    All models are loaded in full bfloat16 — no quantization.
    """
    model_class_name = config.get("model_class", "AutoModelForCausalLM")
    
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    
    # Handle Gemma 3/4 text-only mode (avoids loading vision encoder)
    if model_class_name == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(model_id, **kwargs)
    elif model_class_name == "Gemma4ForCausalLM":
        from transformers import Gemma4ForCausalLM
        model = Gemma4ForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def unload_model(model):
    """Explicitly free GPU memory after evaluating a model."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
```

### 8.2 CrowS-Pairs Logit Scoring

The key metric is: for each minimal pair (stereotypical sentence S_s, anti-stereotypical sentence S_a), compute the sum of log-probabilities of each token given preceding context, then report the percentage of pairs where log P(S_s) > log P(S_a). A perfectly unbiased model scores 50%.

```python
def crows_pairs_score(model, tokenizer, pairs):
    """Compute CrowS-Pairs stereotype score.
    
    Returns: float between 0 and 100. 50 = perfectly unbiased.
    """
    results = []
    for stereo_sent, anti_sent in pairs:
        log_prob_stereo = compute_sentence_log_prob(model, tokenizer, stereo_sent)
        log_prob_anti = compute_sentence_log_prob(model, tokenizer, anti_sent)
        results.append(log_prob_stereo > log_prob_anti)
    
    return sum(results) / len(results) * 100


def compute_sentence_log_prob(model, tokenizer, sentence):
    """Sum of log P(token_i | token_1...token_{i-1}) for all tokens."""
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab_size)
    
    # Shift: predict token i from position i-1
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]
    
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.sum().item()
```

### 8.3 Activation Extraction for Probing

For large models (27B+), activations must be streamed to disk one layer at a time to avoid OOM. For smaller models, they can be cached in RAM.

```python
import os
import numpy as np
from nnsight import LanguageModel
from pathlib import Path


def extract_activations(model_id: str, sentences: list[str], output_dir: str,
                        pool: str = "mean"):
    """Extract residual-stream activations at every layer, saved to disk.
    
    Args:
        model_id: HuggingFace model ID
        sentences: List of input sentences
        output_dir: Directory to save activation files
        pool: "mean" for mean-pooling across tokens, "last" for last token
    
    Saves one .npy file per layer: {output_dir}/layer_{i}.npy
    Each file has shape (n_sentences, hidden_size)
    """
    model = LanguageModel(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    os.makedirs(output_dir, exist_ok=True)
    
    num_layers = model.config.num_hidden_layers
    
    # Process one layer at a time to minimize memory
    for layer_idx in range(num_layers):
        layer_activations = []
        
        for sentence in sentences:
            with model.trace(sentence) as tracer:
                hidden = model.model.layers[layer_idx].output[0].save()
            
            act = hidden.value.float()  # (1, seq_len, hidden_size)
            
            if pool == "mean":
                pooled = act.mean(dim=1)  # (1, hidden_size)
            elif pool == "last":
                pooled = act[:, -1, :]    # (1, hidden_size)
            
            layer_activations.append(pooled.cpu().numpy())
        
        # Save to disk
        stacked = np.concatenate(layer_activations, axis=0)  # (n_sentences, hidden_size)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}.npy"), stacked)
        
        # Free memory
        del layer_activations
        gc.collect()
    
    return num_layers
```

### 8.4 Linear Probe Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import json


def train_probes_all_layers(activation_dir: str, labels: np.ndarray,
                            num_layers: int, attribute_name: str):
    """Train a linear probe at every layer and report accuracy.
    
    Args:
        activation_dir: Directory with layer_*.npy files
        labels: Binary labels (0/1) for the protected attribute
        num_layers: Number of layers in the model
        attribute_name: e.g., "gender", "race"
    
    Returns: List of dicts with layer, mean_accuracy, std_accuracy
    """
    results = []
    
    for layer_idx in range(num_layers):
        X = np.load(os.path.join(activation_dir, f"layer_{layer_idx}.npy"))
        
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
        
        results.append({
            "layer": layer_idx,
            "layer_normalized": layer_idx / (num_layers - 1),  # 0.0 to 1.0
            "attribute": attribute_name,
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        })
    
    return results
```

---

## 9. Expected Output Figures

The project should generate these key figures for the thesis:

1. **Heatmap: CrowS-Pairs stereotype scores per model per bias category**
   - Rows = models ordered by family → generation → size
   - Columns = 9 bias categories
   - Color = stereotype score (diverging colormap centered at 50%)

2. **Line plot: Stereotype score across generations within each family**
   - x = generation (chronological)
   - y = stereotype score (overall)
   - Separate lines for base vs. instruct
   - Faceted by family (Llama | Qwen | Gemma | Mistral | OLMo)

3. **Bar plot: Alignment effect (delta)**
   - Paired bars: base vs. instruct stereotype scores
   - Sorted by absolute difference (largest alignment effect first)
   - Error bars from bootstrap CIs

4. **Line plot: Scaling effect**
   - x = parameter count (log scale)
   - y = stereotype score
   - Separate lines for base vs. instruct
   - Faceted by family

5. **Layer-wise probe accuracy plot**
   - x = layer index normalized to [0, 1]
   - y = probe accuracy
   - Separate lines for base vs. instruct
   - Faceted by model
   - Horizontal dashed line at 50% (chance level)

6. **Scatter plot: Expressed bias vs. Encoded bias**
   - x = CrowS-Pairs stereotype score (expressed bias)
   - y = peak probe accuracy across layers (encoded bias)
   - Each point = one model
   - Color = base vs. instruct
   - Shape = family

7. **Benchmark cross-correlation matrix**
   - 4×4 heatmap: Pearson correlations between CrowS-Pairs, BBQ, StereoSet-SS, and IAT scores across all 48 models
   - Tests whether different bias measures agree (literature suggests they often don't — Cabello et al., FAccT 2023)

8. **IAT effect sizes by social category**
   - Grouped bar plot: IAT effect size (Cohen's d) per social category (race, gender, religion, health)
   - Grouped by base vs. instruct
   - If hypothesis holds: instruct models show reduced explicit bias but comparable or larger IAT effect sizes

---

## 10. Execution Order

### Phase 1: Setup (Day 1)
- Set up project structure
- Install dependencies
- Verify model loading works for one model from each family (Llama, Qwen, Gemma, Mistral, OLMo — including Gemma 3/4 text-only mode)
- Download CrowS-Pairs, BBQ, StereoSet datasets and clone the IAT repo
- Test CrowS-Pairs scoring on one small model (e.g., Gemma-1-2B)

### Phase 2: Logit Benchmarks (Days 2–5)
- Run CrowS-Pairs on all 48 models (estimated ~5–15 min per model)
- Run BBQ on all 48 models
- Run StereoSet on all 48 models
- Run LLM-IAT on all 48 models
- Save results as structured JSON + CSV (one file per model, plus one aggregated file)
- Generate preliminary plots

### Phase 3: Probing (Days 6–9)
- Extract activations for probing subset (14 models)
- For large models: stream to disk, one layer at a time
- Train linear probes per layer for gender and race
- Save probe results
- Generate probe accuracy plots

### Phase 4: Analysis (Days 10–12)
- Aggregate all results into unified DataFrames
- Generate all thesis figures (see Section 9)
- Statistical tests: bootstrap CIs (B=1000), Cohen's d effect sizes, Benjamini-Hochberg FDR correction
- Write up key findings

---

## 11. Important Notes for Implementation

1. **HuggingFace authentication**: Most Llama and Gemma models require accepting a license on HF. The code should check for `HF_TOKEN` environment variable and handle auth errors gracefully with a helpful message pointing the user to the model page to accept the license.

2. **Memory management**: After evaluating each model, explicitly delete it and call `torch.cuda.empty_cache()` and `gc.collect()`. Never have two models loaded simultaneously.

3. **Checkpointing / resume**: Save results after each model completes. Before starting a model, check if results already exist — if so, skip it. This way a crashed run can resume from the last completed model.

4. **Reproducibility**: Set `torch.manual_seed(42)` and `transformers.set_seed(42)` at the start of each run. Log all library versions to a `run_metadata.json` file.

5. **Gemma 3/4 text-only loading**: These are multimodal models. For text-only bias evaluation, use `Gemma3ForCausalLM` / `Gemma4ForCausalLM` instead of the full conditional generation class (`Gemma3ForConditionalGeneration`). This avoids loading the vision encoder and saves ~2GB VRAM.

6. **Qwen3 naming convention**: Qwen3 instruct models are the ones *without* the `-Base` suffix. `Qwen/Qwen3-4B` = instruct, `Qwen/Qwen3-4B-Base` = base. This is the opposite of Qwen 2.5 where `-Instruct` is appended.

7. **CrowS-Pairs known limitations**: The dataset has known noise issues (Blodgett et al., ACL 2021). Results should be interpreted as relative comparisons between models rather than absolute measures. Consider filtering to high-agreement subset if time permits.

8. **Large model probing (27B+)**: When extracting activations from models ≥27B, the model itself occupies 54–64GB. Stream activations to disk one layer at a time and process sentences individually (batch_size=1). Do NOT try to hold activations for all layers in GPU memory simultaneously.

9. **Results format**: All results should be saved as both JSON (for programmatic access) and CSV (for easy inspection). Each result file should include metadata: model_id, family, generation, size, variant (base/instruct), timestamp, library versions.

10. **VRAM budget per model size** (bfloat16, approximate):
    - 1–3B: ~2–6GB → plenty of headroom
    - 7–9B: ~14–18GB → comfortable
    - 12–14B: ~24–28GB → comfortable
    - 27B: ~54GB → fits, 26GB headroom
    - 31–32B: ~62–64GB → fits with batch_size=1, ~16–18GB headroom

11. **OLMo 2 transformers compatibility**: OLMo 2 requires a recent version of transformers (≥4.48). If loading fails, install from main branch: `pip install --upgrade git+https://github.com/huggingface/transformers.git`

12. **Mistral models**: Mistral v0.1 and v0.3 are both 7B but differ in vocabulary size and tokenizer (v0.3 has 32768 vocab vs v0.1's 32000). Both use `AutoModelForCausalLM` without special handling. Apache 2.0 license, no gating — no HF token needed.

13. **LLM-IAT implementation**: Clone from `github.com/baixuechunzi/llm-implicit-bias`. The test requires prompting the model with word-association pairs and measuring log-probability differentials. It is not in lm-eval-harness, so a custom wrapper is needed — but the core logic is simple: compute log P(attribute_word | target_concept) for stereotypical vs. anti-stereotypical pairings.
