"""Confound-controlled regression analysis.

Per Anders Søgaard's framing of the project:

> "How has bias developed in published language models, **controlling for**
> factors like model size and training data?"

This module estimates the effect of `variant` (base → instruct) on each
benchmark's headline bias metric, after partialling out the obvious confounds
that vary across the 22 base↔instruct pairs:

  * `log10(num_params)` — model size
  * `family`             — vendor / architecture (Meta, Alibaba, Google, Mistral)
  * `generation`         — release-version effects within a family
  * `prompt_mode`        — raw vs chat-templated scoring condition

Two complementary fits are produced:

  1. **Per-benchmark OLS with cluster-robust SEs.** Outcome is the headline
     bias score per (model, prompt_mode); SEs clustered on `model_id` to
     account for the within-checkpoint correlation between the two prompt
     modes. With ~88 obs per benchmark this is well-powered for the half-dozen
     coefficients of interest. Cluster-robust SEs are equivalent to a random
     intercept under the relevant assumptions and are easier to interpret.

  2. **Per-example logistic GEE on CrowS-Pairs** (highest-power benchmark).
     Outcome is the binary `stereo_won` flag at the pair level (~1500 pairs ×
     44 ckpts × 2 modes ≈ 132k obs); GEE with model-level clustering gives
     a robust log-odds estimate for the variant effect.

Both fits report β, 95% CI, and p-values. Multiplicity across the 4 benchmark
fits is handled by Holm-Bonferroni on the variant coefficient (the single
hypothesis of scientific interest).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


# Headline bias metric per benchmark (the outcome variable in each regression).
# Other metrics (per-category, per-bias-type) are still in the parquet for
# downstream analysis; this just picks the one that goes in the thesis table.
HEADLINE_METRIC: dict[str, str] = {
    "crows_pairs": "overall",            # % of pairs where lp(stereo) > lp(anti)
    "stereoset":   "overall_SS",         # stereotype score (50 = unbiased)
    "bbq":         "overall_bias_ambig", # ambiguous-context bias score (0 = unbiased)
    "iat":         "overall_abs_d",      # mean |Cohen's d| across IAT subtests
}

# Coefficients of primary interest in the rendered tables — the rest are
# nuisance controls (family / generation level intercepts) and crowd the view.
KEY_TERMS = ("variant", "prompt_mode", "log_params")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_regression_frame(logit_df: pd.DataFrame) -> pd.DataFrame:
    """Filter the long aggregate to one row per (model, prompt_mode, benchmark).

    Picks `HEADLINE_METRIC[bench]` as the outcome and adds `log_params`.
    """
    keep = pd.concat(
        [logit_df[(logit_df["benchmark"] == b) & (logit_df["metric"] == m)]
         for b, m in HEADLINE_METRIC.items()],
        ignore_index=True,
    )
    if keep.empty:
        return keep
    keep = keep.copy()
    keep["log_params"] = np.log10(keep["num_params"].astype(float))
    keep["variant"] = keep["variant"].astype("category")
    keep["prompt_mode"] = keep["prompt_mode"].astype("category")
    keep["family"] = keep["family"].astype("category")
    keep["generation"] = keep["generation"].astype("category")
    return keep


def load_crows_per_example(results_root: Path) -> pd.DataFrame:
    """Flatten CrowS-Pairs JSONs to one row per (model, prompt_mode, pair_id)."""
    rows: list[dict[str, Any]] = []
    base = Path(results_root) / "logit_scores" / "crows_pairs"
    for fp in base.glob("*.json"):
        with open(fp) as f:
            data = json.load(f)
        spec = data["spec"]
        result = data["result"]
        prompt_mode = result.get("prompt_mode", "raw")
        for ex in result["per_example"]:
            rows.append({
                "model_id": spec["model_id"],
                "family": spec["family"],
                "generation": spec["generation"],
                "size": spec["size"],
                "variant": spec["variant"],
                "num_params": spec["num_params"],
                "prompt_mode": prompt_mode,
                "pair_id": ex["pair_id"],
                "bias_type": ex["bias_type"],
                "stereo_won": int(bool(ex["stereo_won"])),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


_SUMMARY_FORMULA = (
    "value ~ C(variant, Treatment('base'))"
    " + C(prompt_mode, Treatment('raw'))"
    " + log_params"
    " + C(family)"
    " + C(generation)"
)


def _coef_dict(fit) -> dict[str, dict[str, float]]:
    """Extract a clean (term → β/CI/p) mapping from any statsmodels fit."""
    ci = fit.conf_int()
    return {
        "params":   fit.params.to_dict(),
        "ci_lower": ci[0].to_dict(),
        "ci_upper": ci[1].to_dict(),
        "pvalues":  fit.pvalues.to_dict(),
    }


def fit_summary_model(reg_df: pd.DataFrame, benchmark: str) -> dict[str, Any]:
    """OLS with cluster-robust SEs on the per-checkpoint headline bias score.

    Cluster on `model_id` to handle the within-checkpoint correlation between
    the two prompt_mode observations. If fewer than 2 distinct families /
    generations are present (smoke-test scenarios) the corresponding categorical
    is dropped from the formula.
    """
    if reg_df.empty or "benchmark" not in reg_df.columns:
        return {"benchmark": benchmark, "n": 0, "model_type": "none", "note": "no data"}
    sub = reg_df[reg_df["benchmark"] == benchmark].copy()
    if sub.empty:
        return {"benchmark": benchmark, "n": 0, "model_type": "none", "note": "no data"}

    # Drop empty categoricals to avoid statsmodels rank-deficiency errors.
    formula = _SUMMARY_FORMULA
    if sub["family"].nunique() < 2:
        formula = formula.replace(" + C(family)", "")
    if sub["generation"].nunique() < 2:
        formula = formula.replace(" + C(generation)", "")
    if sub["prompt_mode"].nunique() < 2:
        formula = formula.replace(" + C(prompt_mode, Treatment('raw'))", "")
    if sub["variant"].nunique() < 2:
        formula = formula.replace(" + C(variant, Treatment('base'))", "")

    n_clusters = sub["model_id"].nunique()
    use_cluster = n_clusters >= 4 and len(sub) > n_clusters

    try:
        if use_cluster:
            fit = smf.ols(formula, data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["model_id"].values}
            )
            cov = "cluster_robust"
        else:
            fit = smf.ols(formula, data=sub).fit(cov_type="HC3")
            cov = "HC3"
    except Exception as exc:
        logger.exception("OLS fit failed for %s", benchmark)
        return {"benchmark": benchmark, "n": len(sub), "model_type": "ols", "error": str(exc)}

    out = {
        "benchmark": benchmark,
        "n": len(sub),
        "n_clusters": int(n_clusters),
        "model_type": "ols",
        "cov_type": cov,
        "formula": formula,
        "rsquared": float(getattr(fit, "rsquared", float("nan"))),
        **_coef_dict(fit),
    }
    return out


def fit_crows_glmm(crows_df: pd.DataFrame) -> dict[str, Any]:
    """Logistic GEE on per-pair `stereo_won` outcome, clustered on model_id.

    GEE is preferred over a fully-specified GLMM here because (a) we care about
    a population-averaged effect of `variant`, (b) BinomialBayesMixedGLM is
    notoriously slow/unstable, and (c) GEE's sandwich SEs are robust to
    misspecification of the within-cluster correlation.
    """
    if crows_df.empty:
        return {"n": 0, "model_type": "none", "note": "no data"}

    df = crows_df.copy()
    df["log_params"] = np.log10(df["num_params"].astype(float))
    formula = (
        "stereo_won ~ C(variant, Treatment('base'))"
        " + C(prompt_mode, Treatment('raw'))"
        " + log_params + C(family) + C(generation)"
    )
    if df["family"].nunique() < 2:
        formula = formula.replace(" + C(family)", "")
    if df["generation"].nunique() < 2:
        formula = formula.replace(" + C(generation)", "")
    if df["prompt_mode"].nunique() < 2:
        formula = formula.replace(" + C(prompt_mode, Treatment('raw'))", "")
    if df["variant"].nunique() < 2:
        formula = formula.replace(" + C(variant, Treatment('base'))", "")

    try:
        fit = smf.gee(
            formula, groups="model_id", data=df,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit()
    except Exception as exc:
        logger.exception("GEE fit failed for crows_pairs")
        return {"n": len(df), "model_type": "gee", "error": str(exc)}

    return {
        "n": len(df),
        "n_clusters": int(df["model_id"].nunique()),
        "model_type": "logistic_gee",
        "cov_struct": "independence",
        "formula": formula,
        **_coef_dict(fit),
    }


# ---------------------------------------------------------------------------
# Multiple-comparison correction on the variant coefficient
# ---------------------------------------------------------------------------


def _variant_term_name(params: dict[str, float]) -> str | None:
    """Find the patsy-generated term name for the variant=instruct contrast."""
    for term in params:
        if "variant" in term and "instruct" in term:
            return term
    return None


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni step-down on a {label: p} dict. Returns {label: {p, p_adj, reject}}."""
    items = [(k, v) for k, v in p_values.items() if v is not None and not np.isnan(v)]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float]] = {}
    running_max = 0.0
    for i, (label, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)  # enforce monotonicity
        out[label] = {"p": float(p), "p_adj": float(running_max), "reject": running_max < alpha}
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def coef_table_to_markdown(
    fit: dict[str, Any], *, key_only: bool = True
) -> str:
    """Render a coefficient/CI/p-value table as markdown."""
    if "params" not in fit:
        return f"_{fit.get('error', fit.get('note', 'no fit'))}_"
    rows = ["| Term | β | 95% CI | p |", "|---|---:|:---:|---:|"]
    for term, beta in fit["params"].items():
        if key_only and not any(k in term for k in KEY_TERMS):
            continue
        lo = fit["ci_lower"].get(term, float("nan"))
        hi = fit["ci_upper"].get(term, float("nan"))
        p = fit["pvalues"].get(term, float("nan"))
        rows.append(f"| `{term}` | {beta:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3g} |")
    return "\n".join(rows)


def write_regression_report(
    results_root: Path,
    out_dir: Path,
) -> Path:
    """End-to-end: aggregate, fit per-benchmark + crows GEE, write markdown + JSON."""
    from biaseval.analysis.aggregate_results import aggregate_logit_results

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logit_df = aggregate_logit_results(Path(results_root))
    reg_df = build_regression_frame(logit_df)

    md: list[str] = [
        "# Confound-controlled regression analysis\n",
        ("Each fit estimates the effect of `variant` (base → instruct) on the "
         "headline bias metric **after partialling out** model size "
         "(log₁₀ params), family, generation, and prompt_mode.\n"),
        ("Standard errors are clustered on `model_id` (each checkpoint contributes "
         "two observations: raw and instruct prompt modes).\n"),
    ]

    summary_fits: dict[str, dict[str, Any]] = {}
    variant_pvalues: dict[str, float] = {}
    for bench in HEADLINE_METRIC:
        fit = fit_summary_model(reg_df, bench)
        summary_fits[bench] = fit
        md.append(
            f"## {bench} ({fit.get('model_type', '?')}, "
            f"n={fit.get('n', 0)}, clusters={fit.get('n_clusters', 0)}, "
            f"R²={fit.get('rsquared', float('nan')):.3f})\n"
        )
        md.append(coef_table_to_markdown(fit))
        md.append("")
        if "pvalues" in fit:
            term = _variant_term_name(fit["pvalues"])
            if term is not None:
                variant_pvalues[bench] = float(fit["pvalues"][term])

    # Holm-Bonferroni across the 4 variant tests.
    if variant_pvalues:
        adj = holm_bonferroni(variant_pvalues)
        md.append("## Multiple-comparison correction\n")
        md.append("Holm-Bonferroni step-down on the variant=instruct coefficient "
                   "across the 4 benchmark fits.\n")
        md.append("| Benchmark | p (raw) | p (Holm) | reject @ α=0.05 |")
        md.append("|---|---:|---:|:---:|")
        for bench, info in adj.items():
            mark = "✓" if info["reject"] else "—"
            md.append(f"| {bench} | {info['p']:.3g} | {info['p_adj']:.3g} | {mark} |")
        md.append("")

    # Per-example CrowS GEE.
    md.append("## CrowS-Pairs per-example logistic GEE\n")
    md.append("Outcome: per-pair `stereo_won` (binary). Coefficients are "
               "log-odds; positive = more stereotypical. Standard errors via "
               "GEE sandwich, clustered on `model_id`.\n")
    crows_df = load_crows_per_example(results_root)
    crows_fit = fit_crows_glmm(crows_df)
    summary_fits["crows_pairs_glmm"] = crows_fit
    md.append(
        f"_n={crows_fit.get('n', 0)}, clusters={crows_fit.get('n_clusters', 0)}_\n"
    )
    md.append(coef_table_to_markdown(crows_fit))
    md.append("")

    md_path = out_dir / "regression_report.md"
    md_path.write_text("\n".join(md))

    # Drop raw fits (sans stringy `summary` blobs) as JSON for downstream tooling.
    json_path = out_dir / "regression_summaries.json"
    json_path.write_text(json.dumps(summary_fits, indent=2, default=str))

    logger.info("Wrote regression report to %s", md_path)
    return md_path
