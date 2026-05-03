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

# Bai et al. (PNAS 2025) IAT stimuli — MIT licensed, see LICENSE in upstream repo.
IAT_STIMULI_URL = (
    "https://raw.githubusercontent.com/baixuechunzi/llm-implicit-bias/main/stimuli/iat_stimuli.csv"
)

# Human-readable target labels for each (category, dataset) test. Where stimuli
# are concept words (Pattern 1) the upstream A/B columns already label them;
# where stimuli are paired names (Pattern 2) we provide a conceptual label here.
IAT_TARGET_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("race", "arab/muslim"): ("european_names", "arab_muslim_names"),
    ("race", "asian"): ("white_surnames", "asian_surnames"),
    ("race", "black"): ("white_surnames", "black_surnames"),
    ("race", "hispanic"): ("white_surnames", "hispanic_surnames"),
    ("race", "english"): ("english_proficient_label", "english_learner_label"),
    ("gender", "career"): ("male_names", "female_names"),
    ("gender", "power"): ("male_names", "female_names"),
    ("gender", "science"): ("male_label", "female_label"),
    ("gender", "sexuality"): ("straight_label", "gay_label"),
    ("religion", "buddhism"): ("christian_label", "buddhist_label"),
    ("religion", "islam"): ("christian_label", "muslim_label"),
    ("religion", "judaism"): ("christian_label", "jewish_label"),
    ("age", "age"): ("young_label", "old_label"),
    ("health", "disability"): ("abled_label", "disabled_label"),
    ("health", "eating"): ("thin_label", "fat_label"),
    ("health", "mental illness"): ("physical_illness_label", "mental_illness_label"),
    ("health", "weight"): ("thin_label", "fat_label"),
}

# Attribute side labels per test. attr_a is the stereotype-aligned-with-target_a
# attribute (the "canonical-IAT" pairing). First half of C-column words = attr_a,
# second half = attr_b, in the upstream CSV.
IAT_ATTR_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("race", "racism"): ("pleasant", "unpleasant"),
    ("race", "weapon"): ("harmless", "weapons"),
    ("race", "guilt"): ("acquittal", "conviction"),
    ("race", "skintone"): ("attractive", "unattractive"),
    ("race", "arab/muslim"): ("pleasant", "unpleasant"),
    ("race", "asian"): ("pleasant", "unpleasant"),
    ("race", "black"): ("pleasant", "unpleasant"),
    ("race", "hispanic"): ("pleasant", "unpleasant"),
    ("race", "english"): ("pleasant", "unpleasant"),
    ("gender", "career"): ("career", "family"),
    ("gender", "power"): ("strong", "weak"),
    ("gender", "science"): ("science", "liberal_arts"),
    ("gender", "sexuality"): ("good", "bad"),
    ("religion", "buddhism"): ("pleasant", "unpleasant"),
    ("religion", "islam"): ("pleasant", "unpleasant"),
    ("religion", "judaism"): ("pleasant", "unpleasant"),
    ("age", "age"): ("pleasant", "unpleasant"),
    ("health", "disability"): ("good", "bad"),
    ("health", "eating"): ("good", "bad"),
    ("health", "mental illness"): ("temporary", "permanent"),
    ("health", "weight"): ("good", "bad"),
}


def _ensure_cache(name: str) -> Path:
    p = CACHE_ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch_crows_pairs(language: str = "en") -> list[dict]:
    """Download (and cache) the CrowS-Pairs CSV; return a list of row dicts.

    For English (``language="en"``), uses the original NYU MLL release.
    For non-English, loads from the BigScienceBiasEval/crows_pairs_multilingual
    HF dataset (Néveol et al. 2022) and normalises to the same row shape as
    English so downstream scoring code is language-agnostic.

    Each returned row has at least ``sent_more``, ``sent_less``, ``bias_type``,
    ``stereo_antistereo`` keys — same as the English release.
    """
    if language == "en":
        cache = _ensure_cache("crows_pairs.csv")
        if not cache.exists():
            logger.info("Downloading CrowS-Pairs (en) to %s", cache)
            urllib.request.urlretrieve(CROWS_PAIRS_URL, cache)
        with open(cache, newline="") as f:
            rows = list(csv.DictReader(f))
        logger.info("CrowS-Pairs (en): %d rows", len(rows))
        return rows

    return _fetch_crows_pairs_multilingual(language)


# Multilingual CrowS-Pairs availability is narrow: only English and French
# exist as published, comparable benchmarks. The original BigScience repo
# (`BigScienceBiasEval/crows_pairs_multilingual`) uses a dataset loading
# script that newer `datasets` versions reject; we bypass it by downloading
# the parquet mirror at `jannalu/crows_pairs_multilingual` directly. Spanish,
# German, Italian and Portuguese versions of CrowS-Pairs do *not* exist on
# HuggingFace as of this writing — see the search in `huggingface_hub`.
_LANG_CODE_TO_HF_CONFIG = {
    "fr": "french", "french": "french",
}


def _fetch_crows_pairs_multilingual(language: str) -> list[dict]:
    """Load French CrowS-Pairs from the jannalu parquet mirror.

    Pure `huggingface_hub.hf_hub_download` — no dataset loading script, robust
    to `datasets`-library API changes. The parquet file already uses our
    canonical column names (sent_more, sent_less, stereo_antistereo,
    bias_type), so no normalisation is needed.
    """
    config = _LANG_CODE_TO_HF_CONFIG.get(language.lower())
    if config is None:
        raise ValueError(
            f"Unsupported language {language!r}. Supported multilingual "
            f"CrowS-Pairs: {sorted(set(_LANG_CODE_TO_HF_CONFIG.values()))}. "
            f"(Spanish/German/Italian/Portuguese CrowS-Pairs do not exist as "
            f"published benchmarks.)"
        )

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    fp = hf_hub_download(
        repo_id="jannalu/crows_pairs_multilingual",
        filename=f"{config}/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(fp)
    out: list[dict] = [
        {
            "sent_more": r["sent_more"],
            "sent_less": r["sent_less"],
            "bias_type": r.get("bias_type") or "unknown",
            "stereo_antistereo": r.get("stereo_antistereo") or "stereo",
        }
        for r in table.to_pylist()
        if r.get("sent_more") and r.get("sent_less")
    ]
    logger.info("CrowS-Pairs (%s): %d rows (jannalu parquet mirror)",
                language, len(out))
    return out


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


def load_iat_stimuli() -> list[dict]:
    """Load Bai et al. (PNAS 2025) IAT stimuli into the test-dict format
    that ``benchmarks.iat.run`` consumes.

    The upstream CSV groups rows by ``(category, dataset)`` where ``dataset``
    is the IAT subtype (racism, weapon, career, …). Within each group:

    - Rows where columns ``A`` and ``B`` are non-empty define paired target
      stimuli (Pattern 2 — e.g. career uses Ben/Julia, John/Michelle, …).
      Tests with a single A/B header (Pattern 1 — racism, weapon) use the
      concept words themselves as the only target stimuli.
    - The full ``C`` column lists attribute words. The first half is
      stereotype-aligned with target_a (``attr_a``); the second half with
      target_b (``attr_b``) — this is the canonical IAT design.

    Returns one dict per (category, dataset) test, ready to be passed as
    ``tests=`` to ``iat.run``.
    """
    cache = _ensure_cache("bai_iat_stimuli.csv")
    if not cache.exists():
        logger.info("Downloading Bai et al. IAT stimuli to %s", cache)
        urllib.request.urlretrieve(IAT_STIMULI_URL, cache)

    grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
    with open(cache, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["category"].strip(), row["dataset"].strip())
            g = grouped.setdefault(key, {"A": [], "B": [], "C": []})
            a, b, c = row["A"].strip(), row["B"].strip(), row["C"].strip()
            if a:
                g["A"].append(a)
            if b:
                g["B"].append(b)
            if c:
                g["C"].append(c)

    tests: list[dict] = []
    for (category, subcategory), g in grouped.items():
        if not g["A"] or not g["B"] or not g["C"]:
            logger.warning("Skipping IAT test %s/%s — incomplete columns", category, subcategory)
            continue
        # Split attribute words 50/50 into stereotype-aligned-A vs -B halves.
        half = len(g["C"]) // 2
        if len(g["C"]) % 2:
            logger.warning("IAT test %s/%s has odd # attribute words (%d); truncating",
                           category, subcategory, len(g["C"]))
        attr_a_stim = g["C"][:half]
        attr_b_stim = g["C"][half : 2 * half]

        ta_label, tb_label = IAT_TARGET_LABELS.get((category, subcategory), (g["A"][0], g["B"][0]))
        aa_label, ab_label = IAT_ATTR_LABELS.get((category, subcategory), ("attr_a", "attr_b"))

        tests.append(
            {
                "category": category,
                "subcategory": subcategory,
                "target_a": {"name": ta_label, "stimuli": list(g["A"])},
                "target_b": {"name": tb_label, "stimuli": list(g["B"])},
                "attr_a": {"name": aa_label, "stimuli": attr_a_stim},
                "attr_b": {"name": ab_label, "stimuli": attr_b_stim},
            }
        )
    logger.info("Loaded %d Bai et al. IAT tests across %d categories",
                len(tests), len({t["category"] for t in tests}))
    return tests
