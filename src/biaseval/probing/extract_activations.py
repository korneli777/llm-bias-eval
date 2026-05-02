"""Extract residual-stream activations layer-by-layer.

Uses HuggingFace's built-in `output_hidden_states=True` (no nnsight needed).
Saves one .npy per layer at {output_dir}/layer_{i}.npy with shape
(n_sentences, hidden_size).

Memory profile:
- For models ≤14B, we batch sentences and pool in-GPU before moving to CPU.
- For models ≥27B, the model itself takes 54–64GB; we run batch_size=1
  and stream each sentence's per-layer activations directly to the
  pre-allocated CPU array.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pool: str,
    content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool token activations to one vector per sequence.

    hidden_state: (B, T, H), attention_mask: (B, T)
    content_mask: optional (B, T) bool tensor, True = include in pool. Combined
        with attention_mask via AND. Used for the controlled probe (Audit 3) to
        exclude tokens whose surface form gives away the demographic label.
    """
    if content_mask is not None:
        eff_mask = (attention_mask.bool() & content_mask).to(attention_mask.dtype)
    else:
        eff_mask = attention_mask
    mask = eff_mask.unsqueeze(-1).to(hidden_state.dtype)
    if pool == "mean":
        return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if pool == "last":
        # Index of the last non-pad/non-masked token per sequence
        lengths = eff_mask.sum(dim=1).clamp(min=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden_state.size(-1))
        return hidden_state.gather(1, idx).squeeze(1)
    raise ValueError(f"Unknown pool: {pool!r}")


def _build_keyword_mask(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    keyword_set: set[str],
) -> torch.Tensor:
    """Per-token bool mask: True = keep (not a demographic keyword).

    Decodes each token individually, lowercases + strips, and excludes any
    whose surface form is in `keyword_set`. Conservative: only single-token
    matches are excluded (multi-token demographic phrases like
    "African-American" usually tokenise as multiple subwords; if any sub-word
    is itself a keyword, that sub-word is excluded).
    """
    if not keyword_set:
        return torch.ones_like(input_ids, dtype=torch.bool)
    keyword_set_lc = {kw.strip().lower() for kw in keyword_set}
    ids = input_ids.tolist()
    keep = [
        [tokenizer.decode([t]).strip().lower() not in keyword_set_lc for t in seq]
        for seq in ids
    ]
    return torch.tensor(keep, dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def extract_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentences: list[str],
    output_dir: str | Path,
    *,
    pool: str = "last",
    batch_size: int = 4,
    max_length: int = 128,
    mask_keywords: set[str] | None = None,
) -> int:
    """Extract activations and write one .npy file per layer.

    mask_keywords:
        Optional set of surface-form strings whose tokens are excluded from
        the pool (Audit 3 — controls keyword-leakage in linear probes by
        forcing the probe to learn from context, not the demographic word
        itself). When None, behaves identically to the original extraction.

    Returns the number of transformer layers (excluding embedding layer).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(sentences)
    hidden_size = model.config.hidden_size
    # output_hidden_states includes embeddings at index 0 + one per transformer layer.
    num_layers = model.config.num_hidden_layers
    storage = np.zeros((num_layers, n, hidden_size), dtype=np.float32)

    if mask_keywords:
        logger.info("Extraction will mask %d demographic keywords during pooling",
                    len(mask_keywords))

    for start in tqdm(range(0, n, batch_size), desc="Extracting"):
        batch = sentences[start : start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states  # tuple of length num_layers + 1

        content_mask = None
        if mask_keywords:
            content_mask = _build_keyword_mask(tokenizer, enc["input_ids"], mask_keywords)

        # Skip the embedding layer (index 0) and keep the post-block residual states.
        for layer_idx in range(num_layers):
            pooled = _pool(
                hs[layer_idx + 1].float(), enc["attention_mask"], pool,
                content_mask=content_mask,
            )
            storage[layer_idx, start : start + len(batch)] = pooled.cpu().numpy()

        del outputs, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for layer_idx in range(num_layers):
        np.save(output_dir / f"layer_{layer_idx}.npy", storage[layer_idx])

    del storage
    gc.collect()
    return num_layers
