"""Forward-hook context manager that applies a projection to the residual stream.

Patches one transformer block's output: ``h' = (h − bias) @ P + bias``. P is
in row-vector convention as produced by ``fit_inlp`` / ``fit_leace``. The
bias term is zero for INLP (mean preserved automatically by orthogonality)
and is the activation mean for LEACE (so the post-projection mean is
unchanged and only the cross-covariance is killed).

The hook is registered on ``model.model.layers[layer_idx]`` for the standard
Llama / Qwen / Mistral layouts; falls back to ``model.transformer.h[layer_idx]``
for GPT-2-style models (used by the smoke tests).
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def _get_layer_module(model: PreTrainedModel, layer_idx: int) -> torch.nn.Module:
    """Return the i-th transformer block, handling the common architecture variants."""
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base.layers[layer_idx]
    base = getattr(model, "transformer", None)
    if base is not None and hasattr(base, "h"):
        return base.h[layer_idx]  # GPT-2 style
    raise AttributeError(
        f"Could not locate transformer block list on {type(model).__name__}; "
        "expected `.model.layers` or `.transformer.h`."
    )


class ProjectionHook(AbstractContextManager):
    """Apply a fixed linear projection to one layer's output during forward pass.

    Parameters
    ----------
    model:
        The transformer the hook will be attached to.
    projection:
        (H, H) projection matrix. Stored on the model's device in its dtype.
    layer_idx:
        Index of the transformer block whose output to patch.
    bias:
        Optional (H,) centring offset (LEACE uses the activation mean; INLP
        leaves this at zero).

    Usage
    -----
        with ProjectionHook(model, P, layer_idx=18, bias=None):
            result = crows_pairs.run(model, tokenizer, spec, ...)
        # hook is removed cleanly on exit, even on exception
    """

    def __init__(
        self,
        model: PreTrainedModel,
        projection: np.ndarray,
        layer_idx: int,
        *,
        bias: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.layer_idx = int(layer_idx)
        self.layer = _get_layer_module(model, self.layer_idx)
        # Match the model's dtype/device so we don't trigger any casts inside attention.
        param = next(model.parameters())
        self._dtype = param.dtype
        self._device = param.device
        self._P = torch.tensor(projection, dtype=self._dtype, device=self._device)
        self._b = (
            torch.tensor(bias, dtype=self._dtype, device=self._device)
            if bias is not None else None
        )
        self._handle: Any = None

    # ------------------------------------------------------------------ hook
    def _hook(self, _module, _inputs, output):
        # Llama-family blocks return a tuple; the first element is the hidden state.
        if isinstance(output, tuple):
            hs = output[0]
            patched = self._patch(hs)
            return (patched, *output[1:])
        return self._patch(output)

    def _patch(self, hs: torch.Tensor) -> torch.Tensor:
        # Row-vector convention: hs has shape (B, T, H); right-multiply by P.
        if self._b is not None:
            return (hs - self._b) @ self._P + self._b
        return hs @ self._P

    # --------------------------------------------------------------- context
    def __enter__(self) -> ProjectionHook:
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
