"""Shared utilities for logit-based benchmark scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class BenchmarkResult:
    """Output of one benchmark run on one model checkpoint."""

    benchmark: str
    model_id: str
    family: str
    variant: str  # "base" or "instruct"
    summary: dict[str, float]  # e.g. {"overall": 60.4, "race-color": 58.2, ...}
    per_example: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def sentence_log_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
    add_bos: bool = True,
) -> float:
    """Sum of token log-probs P(t_i | t_<i) over the whole sentence.

    Args
    ----
    add_bos:
        Whether to include the BOS token if the tokenizer doesn't add it
        automatically. Required for autoregressive models that don't add
        BOS by default (Llama 3+, Qwen).
    """
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=add_bos)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab)

    # Predict token i from logits at position i-1
    shift_logits = logits[:, :-1, :].float()
    shift_labels = inputs["input_ids"][:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


@torch.no_grad()
def conditional_log_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    context: str,
    continuation: str,
) -> float:
    """log P(continuation | context). Context tokens contribute 0.

    Used for multiple-choice tasks (BBQ) and IAT prompts.
    """
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tokenizer(context + continuation, return_tensors="pt", add_special_tokens=True).input_ids

    ctx_len = ctx_ids.shape[1]
    full_ids = full_ids.to(model.device)

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    # Only the continuation tokens. Tokens at positions [ctx_len-1 .. end-1] in token_log_probs
    # correspond to predictions for positions [ctx_len .. end] in the full sequence.
    return token_log_probs[0, ctx_len - 1 :].sum().item()
