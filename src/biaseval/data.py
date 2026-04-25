"""Cached dataset loaders.

Centralizes the workarounds for benchmarks whose canonical HF datasets
use loading scripts (no longer supported by `datasets>=3.0`):
- CrowS-Pairs is fetched from the original NYU MLL GitHub CSV.
- BBQ uses the script-free mirror at `oskarvanderwal/bbq`.
- StereoSet's schema needs flattening (gold labels nested in struct).

All downloads cache under ~/.cache/biaseval/.
"""

from __future__ import annotations

import csv
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(os.environ.get("BIASEVAL_CACHE", Path.home() / ".cache" / "biaseval"))

CROWS_PAIRS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
)


def _ensure_cache(name: str) -> Path:
    p = CACHE_ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch_crows_pairs() -> list[dict]:
    """Download (and cache) the CrowS-Pairs CSV; return a list of row dicts."""
    cache = _ensure_cache("crows_pairs.csv")
    if not cache.exists():
        logger.info("Downloading CrowS-Pairs to %s", cache)
        urllib.request.urlretrieve(CROWS_PAIRS_URL, cache)

    with open(cache, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info("CrowS-Pairs: %d rows", len(rows))
    return rows


def load_stereoset_intrasentence(split: str = "validation") -> list[dict]:
    """Load + flatten StereoSet (intrasentence) into one row per sentence triple.

    Each output dict has: id, context, bias_type, sentences=[{sentence, gold_label}, ...].
    """
    from datasets import load_dataset

    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split=split)
    out: list[dict] = []
    for row in ds:
        sents = row["sentences"]
        # struct-of-lists: sents['sentence'][i], sents['gold_label'][i]
        triples = [
            {"sentence": sents["sentence"][i], "gold_label": int(sents["gold_label"][i])}
            for i in range(len(sents["sentence"]))
        ]
        out.append(
            {
                "id": row.get("id"),
                "context": row["context"],
                "bias_type": row["bias_type"],
                "sentences": triples,
            }
        )
    return out


def load_bbq(split: str = "test") -> list[dict]:
    """Load BBQ from the script-free mirror at oskarvanderwal/bbq."""
    from datasets import load_dataset

    ds = load_dataset("oskarvanderwal/bbq", split=split)
    return [dict(row) for row in ds]
