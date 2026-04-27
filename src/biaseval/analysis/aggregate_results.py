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

    One row per (model, benchmark, attribute, prompt_mode, method, metric).
    Mirrors `aggregate_logit_results` so plots/regressions can reuse the same
    code path, with extra columns: ``attribute``, ``method``, ``layer_idx``,
    ``probe_acc_post`` (sanity check), ``perplexity_ratio``.
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
                    "prompt_mode": result.get("prompt_mode", "raw"),
                    "attribute": intv.get("attribute"),
                    "method": intv.get("method"),
                    "layer_idx": intv.get("layer_idx"),
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
