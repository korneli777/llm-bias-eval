"""Build labeled (sentence, attribute) datasets for probing.

Re-uses CrowS-Pairs sentences but labels each one with the protected
attribute it describes (e.g. gender ∈ {male, female}, race ∈ {Black, White}).
The labels come from CrowS-Pairs' bias_type field plus simple keyword
matching against the sentence text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from biaseval.data import fetch_crows_pairs

logger = logging.getLogger(__name__)


@dataclass
class ProbeDataset:
    sentences: list[str]
    labels: list[int]
    label_names: tuple[str, str]
    attribute: str

    def __len__(self) -> int:
        return len(self.sentences)


# Keyword sets for label assignment. Intentionally narrow — false negatives
# (sentence skipped) are preferable to false positives (mislabelled).
GENDER_KEYWORDS = {
    0: {"man", "men", "boy", "boys", "father", "uncle", "brother", "husband", "son", "he", "his", "him", "male"},
    1: {"woman", "women", "girl", "girls", "mother", "aunt", "sister", "wife", "daughter", "she", "her", "hers", "female"},
}

RACE_KEYWORDS = {
    0: {"white", "european", "caucasian"},
    1: {"black", "african", "african-american", "afro-american", "asian", "hispanic", "latino", "latina", "arab", "middle eastern"},
}


def _label_by_keywords(text: str, keyword_sets: dict[int, set[str]]) -> int | None:
    tokens = set(re.findall(r"[a-z\-]+", text.lower()))
    matches = {label: bool(tokens & kws) for label, kws in keyword_sets.items()}
    hits = [lab for lab, m in matches.items() if m]
    if len(hits) == 1:
        return hits[0]
    return None  # ambiguous or none


def build_probe_dataset(
    attribute: str,
    *,
    max_per_class: int | None = None,
) -> ProbeDataset:
    """Build a balanced binary classification dataset for one attribute.

    Currently supports: 'gender', 'race'.
    """
    if attribute == "gender":
        kws, names, target_bias = GENDER_KEYWORDS, ("male", "female"), "gender"
    elif attribute == "race":
        kws, names, target_bias = RACE_KEYWORDS, ("white", "non_white"), "race-color"
    else:
        raise ValueError(f"Unknown attribute: {attribute}")

    rows = fetch_crows_pairs()

    sents: list[str] = []
    labels: list[int] = []
    counts = {0: 0, 1: 0}

    for row in rows:
        if row.get("bias_type") != target_bias:
            continue
        for text in (row["sent_more"], row["sent_less"]):
            label = _label_by_keywords(text, kws)
            if label is None:
                continue
            if max_per_class is not None and counts[label] >= max_per_class:
                continue
            sents.append(text)
            labels.append(label)
            counts[label] += 1

    logger.info(
        "Built probe dataset for %s: %d sentences (%d/%s, %d/%s)",
        attribute, len(sents), counts[0], names[0], counts[1], names[1],
    )
    return ProbeDataset(sentences=sents, labels=labels, label_names=names, attribute=attribute)
