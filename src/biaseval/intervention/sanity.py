"""Sanity checks run alongside every intervention.

Three things we verify before trusting an intervention's downstream bias score:

  1. **Probe nullification**: a fresh logistic regression trained on the
     post-projection activations cannot recover the attribute (accuracy ≤
     `chance_threshold`). If it can, the projection didn't actually erase
     the linear direction — the intervention is invalid.

  2. **LM ability**: token-level perplexity on a small held-out clean text
     sample should not blow up (we flag if post/pre ratio > 1.5×). If it
     does, the projection is too aggressive and the bias score becomes
     uninterpretable (the model is producing word salad).

  3. **Mechanism specificity** (computed in the analysis notebook, not here):
     ablating *gender* should reduce gender-CrowS-pairs more than race-color
     pairs. Confirms the effect is causally targeted, not a generic
     perturbation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from biaseval.intervention.hooks import ProjectionHook

logger = logging.getLogger(__name__)


def verify_nullification(
    X: np.ndarray, y: np.ndarray, projection: np.ndarray,
    *, bias: np.ndarray | None = None, chance_threshold: float = 0.55, seed: int = 42,
) -> dict[str, float]:
    """Train a fresh probe on projected X; return its CV accuracy + a pass flag.

    Row-vector convention (matches the forward hook):
      - Pure projection (INLP):    x' = x @ P
      - Bias-corrected (LEACE):    x' = (x - bias) @ P + bias
    """
    if bias is not None:
        X_proj = (X - bias) @ projection + bias
    else:
        X_proj = X @ projection

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
    scores = cross_val_score(clf, X_proj, y, cv=cv, scoring="accuracy", n_jobs=-1)
    acc = float(scores.mean())
    return {
        "post_intervention_probe_accuracy": acc,
        "chance_threshold": chance_threshold,
        "passed": bool(acc <= chance_threshold),
    }


# Default sentences for the perplexity sanity check. ~30 short, neutral
# Wikipedia-style fragments — same fixed set across runs so the comparison
# is paired (with vs without intervention on identical text).
_DEFAULT_PERPLEXITY_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at one hundred degrees Celsius at standard pressure.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Great Wall of China stretches for thousands of kilometres.",
    "Mount Everest is the highest mountain above sea level.",
    "DNA encodes the genetic instructions of all known living organisms.",
    "The speed of light in vacuum is approximately three hundred thousand kilometres per second.",
    "Shakespeare wrote thirty-seven plays during his lifetime.",
    "The Amazon rainforest covers much of northwestern Brazil.",
    "The human brain contains roughly eighty-six billion neurons.",
    "Antarctica is the coldest continent on Earth.",
    "The Eiffel Tower was completed in eighteen eighty-nine.",
    "Penguins are flightless birds native to the Southern Hemisphere.",
    "The Nile is one of the longest rivers in the world.",
    "Computers process information using sequences of binary digits.",
    "Honey bees communicate the location of food through dances.",
    "The Sahara is the largest hot desert on the planet.",
    "Solar panels convert sunlight directly into electrical energy.",
    "The piano has eighty-eight keys arranged in a standard pattern.",
    "Volcanic eruptions can release significant amounts of ash into the atmosphere.",
    "Bicycles are an efficient mode of human-powered transportation.",
    "The Roman Empire reached its greatest extent in the second century.",
    "Coral reefs support a remarkable diversity of marine life.",
    "Glaciers store a substantial fraction of the world's fresh water.",
    "Trains were a transformative technology of the nineteenth century.",
    "Most species of mushroom grow best in damp, shaded environments.",
    "The Moon completes one orbit around the Earth approximately every month.",
    "Ancient libraries preserved important works of literature and science.",
    "Modern bridges span great distances using steel and reinforced concrete.",
]


@torch.no_grad()
def lm_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str] | None = None,
    *,
    max_length: int = 64,
) -> float:
    """Token-level perplexity over a list of short clean sentences.

    Computed by averaging cross-entropy loss across all token positions in all
    sentences (concatenated weighting), then exponentiating.
    """
    texts = texts or _DEFAULT_PERPLEXITY_TEXTS
    total_loss, total_tokens = 0.0, 0
    for sentence in texts:
        enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(next(model.parameters()).device)
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        n_tok = ids.shape[1] - 1  # next-token prediction loses one
        total_loss += float(out.loss) * n_tok
        total_tokens += n_tok
    if total_tokens == 0:
        return float("nan")
    return float(np.exp(total_loss / total_tokens))


def perplexity_check(
    model: torch.nn.Module,
    tokenizer,
    projection: np.ndarray,
    layer_idx: int,
    *,
    bias: np.ndarray | None = None,
    texts: list[str] | None = None,
    blowup_factor: float = 1.5,
) -> dict[str, float]:
    """Compare perplexity with vs without the projection hook attached.

    Flags `passed=False` if post-perplexity exceeds `blowup_factor` × pre.
    """
    pre = lm_perplexity(model, tokenizer, texts)
    with ProjectionHook(model, projection, layer_idx, bias=bias):
        post = lm_perplexity(model, tokenizer, texts)
    ratio = post / pre if pre > 0 else float("inf")
    return {
        "perplexity_pre": pre,
        "perplexity_post": post,
        "ratio": ratio,
        "blowup_factor": blowup_factor,
        "passed": bool(ratio <= blowup_factor),
    }
