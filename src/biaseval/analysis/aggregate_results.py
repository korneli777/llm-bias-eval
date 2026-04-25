"""Walk results/ and consolidate everything into one Parquet for analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_logit_results(results_root: Path) -> pd.DataFrame:
    """Long-format DataFrame: one row per (model, benchmark, metric)."""
    rows: list[dict] = []
    for path in (results_root / "logit_scores").glob("*/*.json"):
        with open(path) as f:
            data = json.load(f)
        spec = data["spec"]
        result = data["result"]
        for metric, value in result["summary"].items():
            rows.append(
                {
                    "model_id": spec["model_id"],
                    "family": spec["family"],
                    "generation": spec["generation"],
                    "size": spec["size"],
                    "variant": spec["variant"],
                    "num_params": spec["num_params"],
                    "benchmark": result["benchmark"],
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def aggregate_probe_results(results_root: Path) -> pd.DataFrame:
    """Long-format DataFrame: one row per (model, attribute, layer)."""
    rows: list[dict] = []
    for path in (results_root / "probe_results").glob("*/*.json"):
        with open(path) as f:
            data = json.load(f)
        spec = data["spec"]
        for layer in data["layers"]:
            rows.append(
                {
                    "model_id": spec["model_id"],
                    "family": spec["family"],
                    "generation": spec["generation"],
                    "size": spec["size"],
                    "variant": spec["variant"],
                    "attribute": data["attribute"],
                    "layer": layer["layer"],
                    "layer_normalized": layer["layer_normalized"],
                    "mean_accuracy": layer["mean_accuracy"],
                    "std_accuracy": layer["std_accuracy"],
                }
            )
    return pd.DataFrame(rows)


def write_aggregated(results_root: Path, output_path: Path) -> dict[str, int]:
    """Write logit + probe DataFrames as separate Parquet sheets and a combined index."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logit_df = aggregate_logit_results(results_root)
    probe_df = aggregate_probe_results(results_root)

    logit_path = output_path.with_name(output_path.stem + "_logit.parquet")
    probe_path = output_path.with_name(output_path.stem + "_probe.parquet")
    if not logit_df.empty:
        logit_df.to_parquet(logit_path, index=False)
    if not probe_df.empty:
        probe_df.to_parquet(probe_path, index=False)

    logger.info("Aggregated %d logit rows, %d probe rows", len(logit_df), len(probe_df))
    return {"logit_rows": len(logit_df), "probe_rows": len(probe_df)}
