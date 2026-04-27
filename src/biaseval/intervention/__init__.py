"""Causal intervention via linear concept erasure.

Two methods (used as parallel robustness checks):
  - INLP: Iterative Null-space Projection (Ravfogel et al. 2020)
  - LEACE: closed-form orthogonal erasure (Belrose et al. 2023)

Both produce a projection matrix P ∈ R^{H×H} which, applied to the residual
stream at a chosen layer, removes the linear direction that encodes a target
attribute. We then re-run the bias benchmarks with a forward hook applying P,
and ask: does ablating the latent stereotype direction reduce the model's
expressed bias?
"""

from biaseval.intervention.hooks import ProjectionHook
from biaseval.intervention.inlp import (
    LeaceResult,
    NullspaceResult,
    fit_inlp,
    fit_leace,
)
from biaseval.intervention.sanity import (
    lm_perplexity,
    verify_nullification,
)

__all__ = [
    "LeaceResult",
    "NullspaceResult",
    "ProjectionHook",
    "fit_inlp",
    "fit_leace",
    "lm_perplexity",
    "verify_nullification",
]
