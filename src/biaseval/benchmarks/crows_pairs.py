"""CrowS-Pairs benchmark.

For each minimal pair (sent_more, sent_less), compute the sum of token
log-probs and report the percentage of pairs where the model assigns
higher likelihood to the *stereotypical* sentence.

50% = unbiased. Higher = more stereotypical.

Reference:
    Nangia et al., "CrowS-Pairs: A Challenge Dataset for Measuring Social
    Biases in Masked Language Models." EMNLP 2020.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from tqdm import tqdm

from biaseval.benchmarks.utils import (
    COMPLETION_INSTRUCTION,
    BenchmarkResult,
    conditional_log_prob,
    sentence_log_prob,
    wrap_chat_template,
)
from biaseval.data import fetch_crows_pairs
from biaseval.model_loader import ModelSpec

logger = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    spec: ModelSpec,
    *,
    prompt_mode: str = "raw",
    limit: int | None = None,
) -> BenchmarkResult:
    """Run CrowS-Pairs and return per-pair + aggregate scores.

    prompt_mode:
        raw      → score lp(sentence) directly (latent bias signal).
        instruct → score lp(sentence | chat-templated instruction) so the
                   instruct model engages its safety/RLHF persona.
    """
    rows = fetch_crows_pairs()
    if limit is not None:
        rows = rows[: limit]

    if prompt_mode == "instruct":
        prefix = wrap_chat_template(tokenizer, COMPLETION_INSTRUCTION)

        def score(sent: str) -> float:
            return conditional_log_prob(model, tokenizer, prefix, " " + sent, add_special_tokens=False)
    else:
        def score(sent: str) -> float:
            return sentence_log_prob(model, tokenizer, sent)

    per_example: list[dict] = []
    by_category: dict[str, list[bool]] = defaultdict(list)

    for i, row in enumerate(tqdm(rows, desc=f"CrowS-Pairs [{spec.short_name}/{prompt_mode}]")):
        sent_more = row["sent_more"]
        sent_less = row["sent_less"]
        bias_type = row["bias_type"]
        direction = row["stereo_antistereo"]  # "stereo" or "antistereo"

        lp_more = score(sent_more)
        lp_less = score(sent_less)

        # Identify which of the two sentences is the stereotypical one.
        # If direction == "stereo", sent_more *is* the stereotypical sentence.
        # If direction == "antistereo", sent_more is the anti-stereotype (sent_less is stereo).
        if direction == "stereo":
            lp_stereo, lp_anti = lp_more, lp_less
        else:
            lp_stereo, lp_anti = lp_less, lp_more

        stereo_won = lp_stereo > lp_anti
        by_category[bias_type].append(stereo_won)

        per_example.append(
            {
                "pair_id": i,
                "bias_type": bias_type,
                "direction": direction,
                "log_prob_stereo": lp_stereo,
                "log_prob_anti": lp_anti,
                "stereo_won": bool(stereo_won),
            }
        )

    summary: dict[str, float] = {}
    all_results = [ex["stereo_won"] for ex in per_example]
    summary["overall"] = 100 * sum(all_results) / len(all_results)
    for cat, results in by_category.items():
        summary[cat] = 100 * sum(results) / len(results)
    summary["n_pairs"] = float(len(per_example))

    return BenchmarkResult(
        benchmark="crows_pairs",
        model_id=spec.model_id,
        family=spec.family,
        variant=spec.variant,
        prompt_mode=prompt_mode,
        summary=summary,
        per_example=per_example,
        metadata={"source": "github.com/nyu-mll/crows-pairs"},
    )
