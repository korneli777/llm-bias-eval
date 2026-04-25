"""Generate the 8 thesis figures from aggregated DataFrames.

Functions take a `figures_dir` and return the list of paths written.
Each figure is plain matplotlib + seaborn (no plotly) for clean PDF/PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

CROWS_CATEGORIES = [
    "race-color", "gender", "religion", "age", "nationality",
    "disability", "physical-appearance", "socioeconomic", "sexual-orientation",
]
FAMILY_ORDER = ["llama", "qwen", "gemma", "mistral", "olmo"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def fig_crows_heatmap(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    df = logit_df[(logit_df["benchmark"] == "crows_pairs")
                  & (logit_df["metric"].isin(CROWS_CATEGORIES))]
    if df.empty:
        return Path()
    df = df.assign(label=lambda d: d["family"] + "/" + d["size"] + "/" + d["variant"])
    pivot = df.pivot_table(index="label", columns="metric", values="value", aggfunc="first")
    pivot = pivot.reindex(columns=[c for c in CROWS_CATEGORIES if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(pivot))))
    sns.heatmap(
        pivot, cmap="RdBu_r", center=50, vmin=30, vmax=70,
        annot=True, fmt=".0f", linewidths=0.4, cbar_kws={"label": "Stereotype score (%)"},
        ax=ax,
    )
    ax.set_title("CrowS-Pairs stereotype scores per model × bias category")
    ax.set_xlabel(""); ax.set_ylabel("")
    return _save(fig, figures_dir / "fig1_crows_heatmap.png")


def fig_generation_lines(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    df = logit_df[(logit_df["benchmark"] == "crows_pairs") & (logit_df["metric"] == "overall")]
    if df.empty:
        return Path()
    g = sns.relplot(
        data=df, x="generation", y="value", hue="variant", style="variant",
        col="family", kind="line", marker="o", col_order=FAMILY_ORDER,
        col_wrap=3, height=3.5, facet_kws={"sharex": False},
    )
    g.set_axis_labels("Generation", "CrowS-Pairs (%)")
    g.set_titles("{col_name}")
    g.refline(y=50, linestyle="--", color="grey")
    g.fig.suptitle("Stereotype score across generations (base vs instruct)", y=1.02)
    return _save(g.fig, figures_dir / "fig2_generation_lines.png")


def fig_alignment_delta(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    df = logit_df[(logit_df["benchmark"] == "crows_pairs") & (logit_df["metric"] == "overall")]
    if df.empty:
        return Path()
    pivot = df.pivot_table(index=["family", "generation", "size"], columns="variant", values="value", aggfunc="first")
    pivot = pivot.dropna(subset=["base", "instruct"]).copy()
    pivot["delta"] = pivot["instruct"] - pivot["base"]
    pivot = pivot.sort_values("delta", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(pivot))))
    labels = [f"{f}/{g}/{s}" for (f, g, s) in pivot.index]
    y = np.arange(len(pivot))
    ax.barh(y - 0.2, pivot["base"], height=0.4, label="base", color="#4477AA")
    ax.barh(y + 0.2, pivot["instruct"], height=0.4, label="instruct", color="#EE6677")
    ax.axvline(50, linestyle="--", color="grey")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("CrowS-Pairs (%)")
    ax.set_title("Alignment effect on stereotype score (sorted by |Δ|)")
    ax.legend(); ax.invert_yaxis()
    return _save(fig, figures_dir / "fig3_alignment_delta.png")


def fig_scaling(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    df = logit_df[(logit_df["benchmark"] == "crows_pairs") & (logit_df["metric"] == "overall")]
    if df.empty:
        return Path()
    g = sns.relplot(
        data=df, x="num_params", y="value", hue="variant", style="variant",
        col="family", kind="line", marker="o", col_order=FAMILY_ORDER,
        col_wrap=3, height=3.5,
    )
    for ax in g.axes.flatten():
        ax.set_xscale("log")
        ax.axhline(50, linestyle="--", color="grey")
    g.set_axis_labels("# parameters (log scale)", "CrowS-Pairs (%)")
    g.set_titles("{col_name}")
    g.fig.suptitle("Scaling effect on stereotype score", y=1.02)
    return _save(g.fig, figures_dir / "fig4_scaling.png")


def fig_probe_accuracy(probe_df: pd.DataFrame, figures_dir: Path) -> Path:
    if probe_df.empty:
        return Path()
    probe_df = probe_df.assign(label=lambda d: d["family"] + "/" + d["size"])
    g = sns.relplot(
        data=probe_df, x="layer_normalized", y="mean_accuracy",
        hue="variant", style="attribute",
        col="label", kind="line", col_wrap=3, height=3.0,
    )
    for ax in g.axes.flatten():
        ax.axhline(0.5, linestyle="--", color="grey")
        ax.set_ylim(0.4, 1.05)
    g.set_axis_labels("Layer (normalized)", "Probe accuracy")
    g.fig.suptitle("Layer-wise probe accuracy (base vs instruct)", y=1.02)
    return _save(g.fig, figures_dir / "fig5_probe_accuracy.png")


def fig_expressed_vs_encoded(logit_df: pd.DataFrame, probe_df: pd.DataFrame, figures_dir: Path) -> Path:
    if logit_df.empty or probe_df.empty:
        return Path()
    crows = (logit_df[(logit_df["benchmark"] == "crows_pairs") & (logit_df["metric"] == "overall")]
             .set_index("model_id")["value"].rename("crows_overall"))
    peak = probe_df.groupby("model_id")["mean_accuracy"].max().rename("probe_peak")
    df = pd.concat([crows, peak], axis=1).dropna()
    meta = (probe_df[["model_id", "family", "variant"]].drop_duplicates().set_index("model_id"))
    df = df.join(meta, how="left")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df, x="crows_overall", y="probe_peak", hue="variant", style="family", s=120, ax=ax)
    ax.axvline(50, linestyle="--", color="grey"); ax.axhline(0.5, linestyle="--", color="grey")
    ax.set_xlabel("Expressed bias (CrowS-Pairs %)")
    ax.set_ylabel("Encoded bias (peak probe accuracy)")
    ax.set_title("Expressed vs encoded bias")
    return _save(fig, figures_dir / "fig6_expressed_vs_encoded.png")


def fig_benchmark_correlation(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    rows = []
    metrics = {
        "crows_pairs": "overall",
        "stereoset": "overall_SS",
        "bbq": "overall_bias_ambig",
        "iat": "overall_abs_d",
    }
    for bench, metric in metrics.items():
        sel = logit_df[(logit_df["benchmark"] == bench) & (logit_df["metric"] == metric)]
        if sel.empty:
            continue
        rows.append(sel.set_index("model_id")["value"].rename(bench))
    if len(rows) < 2:
        return Path()
    df = pd.concat(rows, axis=1).dropna()
    corr = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Cross-benchmark correlation (Pearson, n={len(df)})")
    return _save(fig, figures_dir / "fig7_benchmark_correlation.png")


def fig_iat_by_category(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    df = logit_df[(logit_df["benchmark"] == "iat") & (logit_df["metric"].str.endswith("_d"))
                  & (logit_df["metric"] != "overall_abs_d")].copy()
    if df.empty:
        return Path()
    df["category"] = df["metric"].str.removesuffix("_d")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="category", y="value", hue="variant", errorbar="sd", ax=ax)
    ax.axhline(0, color="grey")
    ax.set_ylabel("IAT effect size (d)")
    ax.set_title("IAT effect size by social category, base vs instruct")
    return _save(fig, figures_dir / "fig8_iat_by_category.png")


def generate_all(
    logit_df: pd.DataFrame, probe_df: pd.DataFrame, figures_dir: Path,
) -> list[Path]:
    """Run every figure function and return the paths actually written."""
    figures_dir = Path(figures_dir)
    paths: list[Path] = []
    for fn in (
        fig_crows_heatmap,
        fig_generation_lines,
        fig_alignment_delta,
        fig_scaling,
        fig_benchmark_correlation,
        fig_iat_by_category,
    ):
        try:
            p = fn(logit_df, figures_dir)
        except Exception as exc:  # pragma: no cover
            logger.error("%s failed: %s", fn.__name__, exc)
            continue
        if p and p.exists():
            paths.append(p)
    if not probe_df.empty:
        for fn2 in (fig_probe_accuracy,):
            try:
                p = fn2(probe_df, figures_dir)
                if p and p.exists():
                    paths.append(p)
            except Exception as exc:  # pragma: no cover
                logger.error("%s failed: %s", fn2.__name__, exc)
        try:
            p = fig_expressed_vs_encoded(logit_df, probe_df, figures_dir)
            if p and p.exists():
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            logger.error("fig_expressed_vs_encoded failed: %s", exc)
    return paths
