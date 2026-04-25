"""Unified model loading.

Handles HF auth, dtype, device mapping, and the special model classes for
Gemma 3/4 (text-only mode that skips the vision encoder).
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class ModelSpec:
    """One concrete model checkpoint from configs/models.yaml.

    Holds everything load_model needs plus enough metadata to identify the
    checkpoint in result files.
    """

    model_id: str
    family: str
    generation: str
    size: str
    variant: str  # "base" or "instruct"
    num_params: int
    num_layers: int
    hidden_size: int
    dtype: str = "bfloat16"
    model_class: str = "AutoModelForCausalLM"
    requires_hf_auth: bool = False
    notes: str | None = None

    @property
    def short_name(self) -> str:
        """Filesystem-safe identifier used for results filenames."""
        return self.model_id.replace("/", "__")


def _resolve_model_class(model_class_name: str) -> type[PreTrainedModel]:
    """Resolve the string from the YAML config to an actual class."""
    if model_class_name == "AutoModelForCausalLM":
        return AutoModelForCausalLM
    if model_class_name == "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM

        return Gemma3ForCausalLM
    if model_class_name == "Gemma4ForCausalLM":
        # Gemma 4 may not exist yet in older transformers — surface a clear error.
        try:
            from transformers import Gemma4ForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Gemma4ForCausalLM not available in this transformers version. "
                "Upgrade with: pip install --upgrade git+https://github.com/huggingface/transformers"
            ) from exc
        return Gemma4ForCausalLM
    raise ValueError(f"Unknown model_class: {model_class_name!r}")


def load_model(
    spec: ModelSpec,
    device_map: str | dict[str, Any] = "auto",
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model + tokenizer per the spec.

    Raises
    ------
    PermissionError
        If the model is gated and HF_TOKEN is missing or doesn't grant access.
    """
    if spec.requires_hf_auth and not os.environ.get("HF_TOKEN"):
        raise PermissionError(
            f"{spec.model_id} requires HuggingFace authentication. "
            "Set HF_TOKEN in your environment (see .env.example) and accept the model "
            f"license at https://huggingface.co/{spec.model_id}"
        )

    cls = _resolve_model_class(spec.model_class)
    dtype = DTYPE_MAP[spec.dtype]

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    logger.info("Loading %s (%s, %s)", spec.model_id, spec.dtype, cls.__name__)
    try:
        model = cls.from_pretrained(spec.model_id, **kwargs)
    except OSError as exc:
        # HF raises OSError for both 401 (auth) and 404 (not found)
        msg = str(exc).lower()
        if "401" in msg or "gated" in msg or "access" in msg:
            raise PermissionError(
                f"Access to {spec.model_id} denied. Check HF_TOKEN and accept the "
                f"license at https://huggingface.co/{spec.model_id}"
            ) from exc
        raise

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def unload_model(model: PreTrainedModel | None = None) -> None:
    """Free GPU memory after using a model. Call between checkpoints."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
