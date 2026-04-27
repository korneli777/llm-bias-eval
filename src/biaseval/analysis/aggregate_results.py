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
        prompt_mode = result.get("prompt_mode", "raw")
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
                    "prompt_mode": prompt_mode,
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


def aggregate_intervention_results(results_root: Path) -> pd.DataFrame:
    """Long-format DataFrame of intervened benchmark scores.

    One row per (model, benchmark, attribute, prompt_mode, method, layer, metric).
    Includes ``depth_frac = layer_idx / (num_layers - 1)`` so cross-model
    layer comparisons are meaningful even when models have different depths.
    """
    rows: list[dict] = []
    base = results_root / "intervention_results"
    if not base.exists():
        return pd.DataFrame()
    for path in base.glob("*/*.json"):
        with open(path) as f:
            data = json.load(f)
        spec = data["spec"]
        result = data["result"]
        intv = data.get("intervention", {})
        sanity = intv.get("sanity", {})
        null = sanity.get("nullification", {})
        ppl = sanity.get("perplexity", {})
        layer_idx = intv.get("layer_idx")
        n_layers = spec.get("num_layers")
        depth_frac = (layer_idx / max(n_layers - 1, 1)) if (
            layer_idx is not None and n_layers
        ) else None
        for metric, value in result["summary"].items():
            rows.append(
                {
                    "model_id": spec["model_id"],
                    "family": spec["family"],
                    "generation": spec["generation"],
                    "size": spec["size"],
                    "variant": spec["variant"],
                    "num_params": spec["num_params"],
                    "num_layers": n_layers,
                    "benchmark": result["benchmark"],
                    "prompt_mode": result.get("prompt_mode", "raw"),
                    "attribute": intv.get("attribute"),
                    "method": intv.get("method"),
                    "layer_idx": layer_idx,
                    "depth_frac": depth_frac,
                    "probe_acc_post": null.get("post_intervention_probe_accuracy"),
                    "probe_acc_passed": null.get("passed"),
                    "perplexity_ratio": ppl.get("ratio"),
                    "perplexity_passed": ppl.get("passed"),
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def write_aggregated(results_root: Path, output_path: Path) -> dict[str, int]:
    """Write logit + probe DataFrames as separate Parquet sheets and a combined index."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logit_df = aggregate_logit_results(results_root)
    probe_df = aggregate_probe_results(results_root)
    intv_df = aggregate_intervention_results(results_root)

    logit_path = output_path.with_name(output_path.stem + "_logit.parquet")
    probe_path = output_path.with_name(output_path.stem + "_probe.parquet")
    intv_path = output_path.with_name(output_path.stem + "_intervention.parquet")
    if not logit_df.empty:
        logit_df.to_parquet(logit_path, index=False)
    if not probe_df.empty:
        probe_df.to_parquet(probe_path, index=False)
    if not intv_df.empty:
        intv_df.to_parquet(intv_path, index=False)

    logger.info("Aggregated %d logit rows, %d probe rows, %d intervention rows",
                len(logit_df), len(probe_df), len(intv_df))
    return {
        "logit_rows": len(logit_df), "probe_rows": len(probe_df),
        "intervention_rows": len(intv_df),
    }
