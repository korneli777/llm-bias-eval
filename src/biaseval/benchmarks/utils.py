"""Shared utilities for logit-based benchmark scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

PROMPT_MODES = ("raw", "instruct")
COMPLETION_INSTRUCTION = "You are an assistant. Finish this sentence:"
QA_INSTRUCTION = "Answer the following question."


@dataclass
class BenchmarkResult:
    """Output of one benchmark run on one model checkpoint."""

    benchmark: str
    model_id: str
    family: str
    variant: str  # "base" or "instruct"
    prompt_mode: str  # "raw" or "instruct"
    summary: dict[str, float]
    per_example: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wrap_chat_template(tokenizer: PreTrainedTokenizerBase, user_message: str) -> str:
    """Format `user_message` as a single-turn chat with an open assistant turn.

    For instruct models with a chat template configured, this returns the
    fully-formatted prompt (incl. BOS/special tokens, role markers, and the
    `add_generation_prompt` suffix). For base models without a chat template,
    falls back to `"{user_message}\\n\\n"` so the *prompt structure* remains
    comparable across variants — essential for the base↔instruct contrast.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return f"{user_message}\n\n"


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
    *,
    add_special_tokens: bool = True,
) -> float:
    """log P(continuation | context). Context tokens contribute 0.

    Pass ``add_special_tokens=False`` when ``context`` was produced by
    ``apply_chat_template`` — that string already contains BOS / role markers
    and re-adding them would shift the alignment.
    """
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids
    full_ids = tokenizer(context + continuation, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids

    ctx_len = ctx_ids.shape[1]
    full_ids = full_ids.to(model.device)

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    # Tokens at positions [ctx_len-1 .. end-1] in token_log_probs correspond to
    # predictions for positions [ctx_len .. end] in the full sequence.
    return token_log_probs[0, ctx_len - 1 :].sum().item()
