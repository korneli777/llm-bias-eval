"""Stage 3: aggregate all results to Parquet and render thesis figures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from biaseval.analysis.aggregate_results import (
    aggregate_logit_results,
    aggregate_probe_results,
    write_aggregated,
)
from biaseval.analysis.plotting import generate_all

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-config", default="configs/benchmarks.yaml")
    p.add_argument("--results-root", default="results")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--no-figures", action="store_true",
                   help="Aggregate to Parquet only; skip plotting")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.bench_config) as f:
        bcfg = yaml.safe_load(f)
    out_path = Path(bcfg["outputs"]["aggregated_parquet"])

    counts = write_aggregated(Path(args.results_root), out_path)
    logger.info("Aggregated: %s", counts)

    if args.no_figures or counts["logit_rows"] == 0:
        return 0

    logit_df = aggregate_logit_results(Path(args.results_root))
    probe_df = aggregate_probe_results(Path(args.results_root))
    paths = generate_all(logit_df, probe_df, Path(args.figures_dir))
    for p in paths:
        logger.info("Wrote %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
