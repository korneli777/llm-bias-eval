"""Pre-download all model weights to the HuggingFace cache.

Useful before kicking off a long Colab run — gets all 48 checkpoints
on disk so the actual benchmark loop never waits on network IO.
Run on a machine with adequate disk (~600GB for the full set).
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from biaseval.registry import filter_specs, load_registry

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--family", default=None)
    p.add_argument("--variant", default=None, choices=["base", "instruct"])
    p.add_argument("--max-size-gb", type=float, default=None,
                   help="Skip models with weights estimated above this size in GB")
    return p.parse_args()


def estimate_bf16_gb(num_params: int) -> float:
    return num_params * 2 / (1024**3)


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    specs = list(filter_specs(load_registry(args.config),
                              family=args.family, variant=args.variant))
    n_done, n_skipped, n_errors = 0, 0, 0
    for spec in specs:
        size_gb = estimate_bf16_gb(spec.num_params)
        if args.max_size_gb and size_gb > args.max_size_gb:
            logger.info("[skip] %s — %.1fGB exceeds --max-size-gb", spec.model_id, size_gb)
            n_skipped += 1
            continue
        try:
            logger.info("Downloading %s (~%.1fGB bf16)", spec.model_id, size_gb)
            snapshot_download(spec.model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"])
            n_done += 1
        except Exception:
            logger.error("[error] %s\n%s", spec.model_id, traceback.format_exc())
            n_errors += 1
    logger.info("Done. downloaded=%d skipped=%d errors=%d", n_done, n_skipped, n_errors)
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
