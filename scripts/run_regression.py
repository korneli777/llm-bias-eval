"""Stage 4: confound-controlled regression analysis.

Reads the existing per-(model, benchmark, prompt_mode) result
JSONs, fits one OLS (cluster-robust SEs) per benchmark plus a per-example
logistic GEE on CrowS-Pairs, applies Holm-Bonferroni across the variant tests,
and writes:

    {out_dir}/regression_report.md       — formatted tables for the thesis
    {out_dir}/regression_summaries.json  — raw fit objects for further analysis

Usage:
    uv run python scripts/run_regression.py
    uv run python scripts/run_regression.py --results-root results --out-dir results/tables
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from biaseval.analysis.regression import write_regression_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default="results/tables")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    md_path = write_regression_report(Path(args.results_root), Path(args.out_dir))
    print(f"Wrote regression report to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
