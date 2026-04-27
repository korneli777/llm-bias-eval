"""Synthetic-data tests for the confound-controlled regression module.

Builds a realistic-shape `logit_df` (the format `aggregate_logit_results`
returns) with a known instruct-effect baked in, then verifies that
`fit_summary_model` recovers the right sign and that the markdown / JSON
outputs are well-formed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from biaseval.analysis.regression import (
    HEADLINE_METRIC,
    build_regression_frame,
    coef_table_to_markdown,
    fit_crows_glmm,
    fit_summary_model,
    holm_bonferroni,
    write_regression_report,
)


def _synth_logit_df(seed: int = 0) -> pd.DataFrame:
    """22 base+instruct pairs across 4 families, with a true instruct effect.

    Truth model: bias = 60 + 5*(family_offset) - 8*(variant=instruct) + noise.
    So we expect the variant=instruct coefficient ≈ -8 with tight CI.
    """
    rng = np.random.default_rng(seed)
    families = {
        "llama":   ("Meta",     [("L2-7B", 7e9, 32, 4096), ("L3.1-8B", 8e9, 32, 4096)]),
        "qwen":    ("Alibaba",  [("Q2.5-7B", 7e9, 28, 3584), ("Q3-8B", 8e9, 36, 4096)]),
        "gemma":   ("Google",   [("G2-9B", 9e9, 42, 3584), ("G3-12B", 12e9, 48, 3840)]),
        "mistral": ("Mistral",  [("M-7B-v0.3", 7e9, 32, 4096)]),
    }
    family_offsets = {"llama": 0.0, "qwen": 2.0, "gemma": -3.0, "mistral": 1.0}

    rows = []
    for fam, (_vendor, gens) in families.items():
        for gen, params, _n_layers, _hidden in gens:
            for variant in ("base", "instruct"):
                model_id = f"{fam}/{gen}-{variant}"
                for prompt_mode in ("raw", "instruct"):
                    base_bias = 60 + family_offsets[fam]
                    if variant == "instruct":
                        base_bias -= 8.0
                    if prompt_mode == "instruct":
                        base_bias -= 1.0
                    bias = base_bias + rng.normal(0, 1.5)
                    # Emit one row per benchmark headline metric.
                    for bench, metric in HEADLINE_METRIC.items():
                        offset = {"crows_pairs": 0, "stereoset": 5, "bbq": -55, "iat": -59}[bench]
                        rows.append({
                            "model_id": model_id,
                            "family": fam,
                            "generation": gen,
                            "size": gen,
                            "variant": variant,
                            "num_params": params,
                            "benchmark": bench,
                            "prompt_mode": prompt_mode,
                            "metric": metric,
                            "value": bias + offset,
                        })
    return pd.DataFrame(rows)


def test_build_regression_frame_filters_to_headline_metrics():
    logit_df = _synth_logit_df()
    reg_df = build_regression_frame(logit_df)
    assert not reg_df.empty
    assert set(reg_df["benchmark"].unique()) == set(HEADLINE_METRIC)
    assert "log_params" in reg_df.columns
    # Each (model, prompt_mode, benchmark) should be unique.
    assert reg_df.duplicated(["model_id", "prompt_mode", "benchmark"]).sum() == 0


def test_summary_model_recovers_instruct_effect():
    """The synthetic truth puts -8 on variant=instruct. Fit must recover it."""
    logit_df = _synth_logit_df()
    reg_df = build_regression_frame(logit_df)
    fit = fit_summary_model(reg_df, "crows_pairs")
    assert fit["model_type"] == "ols"
    # Find the variant=instruct coefficient by name.
    variant_term = next(t for t in fit["params"] if "variant" in t and "instruct" in t)
    beta = fit["params"][variant_term]
    lo, hi = fit["ci_lower"][variant_term], fit["ci_upper"][variant_term]
    assert -10 < beta < -6, f"expected ~-8, got {beta}"
    assert lo < beta < hi


def test_summary_model_handles_missing_data():
    fit = fit_summary_model(pd.DataFrame(), "crows_pairs")
    assert fit["n"] == 0
    assert fit["model_type"] == "none"


def test_holm_bonferroni_basic():
    res = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.5})
    assert res["a"]["reject"] is True
    assert res["c"]["reject"] is False
    # Adjusted p must be monotone non-decreasing in raw p order.
    assert res["a"]["p_adj"] <= res["b"]["p_adj"] <= res["c"]["p_adj"]


def test_holm_bonferroni_handles_empty():
    assert holm_bonferroni({}) == {}


def test_coef_table_renders_markdown():
    fit = {
        "params": {"Intercept": 60.0, "C(variant, Treatment('base'))[T.instruct]": -8.0,
                   "log_params": 0.5, "C(family)[T.qwen]": 2.0},
        "ci_lower": {"Intercept": 58, "C(variant, Treatment('base'))[T.instruct]": -10,
                     "log_params": 0.1, "C(family)[T.qwen]": 0.5},
        "ci_upper": {"Intercept": 62, "C(variant, Treatment('base'))[T.instruct]": -6,
                     "log_params": 0.9, "C(family)[T.qwen]": 3.5},
        "pvalues": {"Intercept": 1e-9, "C(variant, Treatment('base'))[T.instruct]": 1e-5,
                    "log_params": 0.02, "C(family)[T.qwen]": 0.04},
    }
    md = coef_table_to_markdown(fit, key_only=True)
    assert "variant" in md
    assert "log_params" in md
    assert "Intercept" not in md  # filtered out by key_only


def test_end_to_end_writes_report(tmp_path):
    """Drive the whole pipeline from synthetic JSONs in a tmp results dir."""
    # Create tmp results/logit_scores/{bench}/{model}__{mode}.json files.
    logit_df = _synth_logit_df()
    summaries: dict[tuple, dict] = {}
    for (mid, bench, mode), grp in logit_df.groupby(["model_id", "benchmark", "prompt_mode"]):
        summaries.setdefault((mid, bench, mode), {})
        for _, row in grp.iterrows():
            summaries[(mid, bench, mode)][row["metric"]] = float(row["value"])

    for (mid, bench, mode), summary in summaries.items():
        sample = logit_df[logit_df["model_id"] == mid].iloc[0]
        spec = {
            "model_id": mid, "family": sample["family"],
            "generation": sample["generation"], "size": sample["size"],
            "variant": sample["variant"], "num_params": int(sample["num_params"]),
            "num_layers": 32, "hidden_size": 4096,
        }
        per_example = []
        if bench == "crows_pairs":
            # Synthesize 100 per-example outcomes consistent with the headline.
            stereo_rate = summary["overall"] / 100
            rng = np.random.default_rng(hash((mid, mode)) & 0xFFFF)
            per_example = [
                {"pair_id": i, "bias_type": "race-color", "direction": "stereo",
                 "log_prob_stereo": -50.0, "log_prob_anti": -51.0,
                 "stereo_won": bool(rng.random() < stereo_rate)}
                for i in range(100)
            ]
        result = {
            "benchmark": bench, "model_id": mid, "family": sample["family"],
            "variant": sample["variant"], "prompt_mode": mode,
            "summary": summary, "per_example": per_example, "metadata": {},
        }
        out = tmp_path / "logit_scores" / bench / f"{mid.replace('/', '__')}__{mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"spec": spec, "result": result, "runtime": {}}))

    # Run the full report.
    md_path = write_regression_report(tmp_path, tmp_path / "tables")
    assert md_path.exists()
    text = md_path.read_text()
    assert "## crows_pairs" in text
    assert "Holm-Bonferroni" in text
    assert "GEE" in text

    # JSON sidecar should also be produced and parse cleanly.
    json_path = tmp_path / "tables" / "regression_summaries.json"
    payload = json.loads(json_path.read_text())
    assert "crows_pairs" in payload
    assert "crows_pairs_glmm" in payload


def test_crows_glmm_handles_empty():
    fit = fit_crows_glmm(pd.DataFrame())
    assert fit["n"] == 0
    assert fit["model_type"] == "none"
