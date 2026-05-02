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


def fig_alignment_by_size(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Headline figure (Anders-sketched): bias on Y, log(size) on X, base vs instruct.

    Faceted by family. CrowS-Pairs raw mode, English. The single most
    important figure in the thesis — directly answers Anders's framing:
    *"how does bias evolve as size and instruction tuning vary, controlling
    for family?"*
    """
    df = logit_df[
        (logit_df["benchmark"] == "crows_pairs") & (logit_df["metric"] == "overall")
        & (logit_df["prompt_mode"] == "raw")
    ].copy()
    if df.empty:
        return Path()
    df["log_params"] = np.log10(df["num_params"].astype(float))

    g = sns.relplot(
        data=df, x="log_params", y="value",
        hue="variant", style="variant",
        col="family", col_order=[f for f in FAMILY_ORDER if f in df["family"].unique()],
        kind="line", marker="o", height=3.6, aspect=1.05,
        err_style="bars", errorbar="sd", facet_kws={"sharey": True},
    )
    for ax in g.axes.flatten():
        ax.axhline(50, ls="--", color="grey", lw=0.8)
        ax.set_ylim(45, 80)
    g.set_axis_labels("log₁₀(parameters)", "CrowS-Pairs stereotype-pick rate (%)")
    g.fig.suptitle(
        "Alignment effect on CrowS-Pairs scales with model size, varies by vendor",
        y=1.04,
    )
    return _save(g.fig, figures_dir / "fig0_alignment_by_size.png")


def fig_multilingual_heatmap(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Cross-language base→instruct delta on CrowS-Pairs.

    Reads `crows_pairs_<lang>` benchmarks from the aggregate; computes per-
    (model, language) pair-wise delta and plots a heatmap of mean delta per
    (family, language). Negative cells = alignment is reducing bias in that
    language.
    """
    if logit_df.empty:
        return Path()
    df = logit_df[
        logit_df["benchmark"].str.startswith("crows_pairs")
        & (logit_df["metric"] == "overall")
        & (logit_df["prompt_mode"] == "raw")
    ].copy()
    if df.empty:
        return Path()
    df["language"] = df["benchmark"].apply(
        lambda b: b.split("crows_pairs", 1)[1].lstrip("_") or "en"
    )
    if df["language"].nunique() < 2:
        return Path()

    # Pair base ↔ instruct on (family, generation, size, language) so we can
    # subtract within a pair.
    pivot = df.pivot_table(
        index=["family", "generation", "size", "language"],
        columns="variant", values="value", aggfunc="first",
    ).dropna(subset=["base", "instruct"])
    pivot["delta"] = pivot["instruct"] - pivot["base"]

    delta_grid = (
        pivot.reset_index()
        .groupby(["family", "language"])["delta"]
        .mean()
        .unstack("language")
    )
    if delta_grid.empty:
        return Path()
    lang_order = ["en", "fr", "es", "de", "pt", "it"]
    cols = [c for c in lang_order if c in delta_grid.columns]
    if cols:
        delta_grid = delta_grid[cols]

    fig, ax = plt.subplots(figsize=(1.5 * len(delta_grid.columns) + 2, 0.5 * len(delta_grid) + 2))
    sns.heatmap(
        delta_grid, annot=True, fmt="+.2f", center=0, cmap="vlag",
        vmin=-8, vmax=8, ax=ax, cbar_kws={"label": "Δ stereotype-pick rate (inst − base)"},
    )
    ax.set_xlabel("Language")
    ax.set_ylabel("Family")
    ax.set_title("Alignment effect on CrowS-Pairs across languages\n(negative = alignment reducing bias)")
    return _save(fig, figures_dir / "fig18_multilingual_alignment.png")


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
    return _save(g.fig, figures_dir / "fig4_generation_lines.png")


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
    return _save(fig, figures_dir / "fig2b_alignment_delta_simple.png")


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
    return _save(g.fig, figures_dir / "fig3_scaling.png")


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
    return _save(g.fig, figures_dir / "fig8_probe_accuracy.png")


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
    return _save(fig, figures_dir / "fig9_expressed_vs_encoded.png")


def fig_benchmark_correlation(
    logit_df: pd.DataFrame, figures_dir: Path,
    *, results_dir: Path | None = None,
    registry_pairs: list[tuple[str, str, str, str, str]] | None = None,
) -> Path:
    """Cross-benchmark Spearman correlation on the *paired Δ* (instruct − base).

    Each cell answers: "for pairs where alignment helped on benchmark A, did
    it also tend to help on benchmark B?" Low off-diagonal correlations are
    themselves a finding (cf. Cabello, Jørgensen & Søgaard, FAccT 2023).

    If `results_dir` + `registry_pairs` are passed (preferred), uses the new
    `cross_benchmark_consistency()` Δ matrix. Falls back to the legacy
    raw-score Pearson correlation otherwise so callers without the registry
    still get something.
    """
    if results_dir is not None and registry_pairs is not None:
        from biaseval.analysis.statistics import cross_benchmark_consistency
        _, corr = cross_benchmark_consistency(Path(results_dir), registry_pairs)
        if corr.empty:
            return Path()
        title = f"Cross-benchmark Spearman of paired Δ (n_pairs={len(registry_pairs)})"
        fname = "fig11_benchmark_delta_correlation.png"
    else:
        rows = []
        metrics = {
            "crows_pairs": "overall", "stereoset": "overall_SS",
            "bbq": "overall_bias_ambig", "iat": "overall_abs_d",
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
        title = f"Cross-benchmark correlation, raw scores (Pearson, n={len(df)})"
        fname = "fig11b_benchmark_correlation_raw.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    return _save(fig, figures_dir / fname)


def fig_iat_by_category(logit_df: pd.DataFrame, figures_dir: Path) -> Path:
    """IAT effect size per social category, with both d variants side by side.

    `_d` is the log-prob d (our default — Cohen's d on per-token log-prob
    associations). `_d_weat_embed` is the classical Caliskan/Bai WEAT d on
    input-embedding cosine similarities (Step 4). Showing both lets reviewers
    cross-check that the alignment effect doesn't depend on the scoring choice.
    """
    is_iat = logit_df["benchmark"] == "iat"
    excluded = {"overall_abs_d", "overall_abs_d_weat_embed"}
    df_lp = logit_df[is_iat & logit_df["metric"].str.endswith("_d")
                     & ~logit_df["metric"].isin(excluded)].copy()
    df_we = logit_df[is_iat & logit_df["metric"].str.endswith("_d_weat_embed")
                     & ~logit_df["metric"].isin(excluded)].copy()
    if df_lp.empty:
        return Path()
    df_lp["category"] = df_lp["metric"].str.removesuffix("_d")
    df_lp["scoring"] = "log-prob d"
    if not df_we.empty:
        df_we["category"] = df_we["metric"].str.removesuffix("_d_weat_embed")
        df_we["scoring"] = "embedding WEAT d"
        plot_df = pd.concat([df_lp, df_we], ignore_index=True)
    else:
        plot_df = df_lp

    g = sns.catplot(
        data=plot_df, x="category", y="value", hue="variant",
        col="scoring", kind="bar", errorbar="sd", height=4, aspect=1.2,
    )
    for ax in g.axes.flatten():
        ax.axhline(0, color="grey")
        ax.tick_params(axis="x", rotation=45)
    g.set_axis_labels("Social category", "IAT effect size (d)")
    g.fig.suptitle("IAT by social category, base vs instruct (both scorings)", y=1.02)
    return _save(g.fig, figures_dir / "fig12_iat_by_category.png")


def fig_alignment_delta_forest(
    pair_sig_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Forest plot of paired Δ per (base, instruct) pair, effect-size first.

    Inputs: the DataFrame returned by `pair_significance_table()`. Columns
    expected: base_id, instruct_id, family, generation, size, delta,
    delta_ci_lo, delta_ci_hi, cohens_d_paired, cohens_d_label, reject_holm,
    reject_strict.

    Visual:
        - One row per pair, sorted by |cohens_d_paired| (already done by
          pair_significance_table — we just keep its order).
        - Point = Δ on CrowS-Pairs (instruct − base, percentage points).
        - Horizontal bars = bootstrap 95% CI on Δ.
        - Right-side annotation: "d = X.XX [label]   ★ / ★★".
          ★  = survives Holm at α=0.05.   ★★ = also survives α=0.0025.
        - Vertical dashed line at Δ=0 (no effect).
    """
    if pair_sig_df.empty:
        return Path()
    df = pair_sig_df.copy()  # caller already sorted by |d| descending
    n = len(df)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * n)))
    y = np.arange(n)
    err_lo = df["delta"] - df["delta_ci_lo"]
    err_hi = df["delta_ci_hi"] - df["delta"]
    ax.errorbar(df["delta"], y, xerr=[err_lo, err_hi],
                fmt="o", color="#222222", ecolor="#888888",
                capsize=2, markersize=5)
    ax.axvline(0, linestyle="--", color="grey")

    labels = [f"{f}/{g}/{s}" for f, g, s in
              zip(df["family"], df["generation"], df["size"], strict=True)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Δ CrowS-Pairs (pp), instruct − base")
    ax.set_title("Alignment effect per pair, sorted by |Cohen's d|")

    # Annotate each row with d + significance stars.
    xmax = float(df["delta_ci_hi"].max()) + 1.5
    for yi, (_, row) in enumerate(df.iterrows()):
        stars = ("★★" if row.get("reject_strict")
                 else ("★" if row.get("reject_holm") else ""))
        ax.text(xmax, yi,
                f"d = {row['cohens_d_paired']:+.2f} [{row['cohens_d_label']}]  {stars}",
                va="center", fontsize=8)
    ax.set_xlim(right=xmax + 6)
    return _save(fig, figures_dir / "fig2_alignment_delta_forest.png")


def fig_bbq_deferral_decomposition(
    logit_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """BBQ alignment story decomposed into deferral vs conditional bias.

    Per pair, three grouped bars × {base, instruct}:
        - deferral_rate     (% of ambiguous Q's where the model picks "unknown")
        - conditional_bias  (% of committed picks that are stereotype-aligned,
                             0.5 = unbiased)
        - overall_bias_ambig (= (2*cond − 1) * (1 − deferral))

    The visual story is:
        - If instruct's deferral bar shoots up but conditional_bias stays put,
          alignment trained avoidance, not debiasing.
    """
    bbq = logit_df[logit_df["benchmark"] == "bbq"]
    metrics = ["overall_deferral_rate", "overall_conditional_bias",
               "overall_bias_ambig"]
    df = bbq[bbq["metric"].isin(metrics)].copy()
    if df.empty:
        return Path()
    df["pair_label"] = (df["family"].astype(str) + "/"
                        + df["generation"].astype(str) + "/"
                        + df["size"].astype(str))
    df["metric_short"] = df["metric"].str.replace("overall_", "", regex=False)

    g = sns.catplot(
        data=df, x="pair_label", y="value", hue="variant",
        col="metric_short", col_order=[m.replace("overall_", "") for m in metrics],
        kind="bar", errorbar=None, height=4, aspect=1.4, sharey=False,
    )
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=60)
    g.set_axis_labels("Model pair", "Value")
    g.fig.suptitle("BBQ decomposition: deferral vs conditional bias vs overall",
                   y=1.02)
    return _save(g.fig, figures_dir / "fig5_bbq_deferral_decomposition.png")


def fig_implicit_explicit_gap(
    logit_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Per-pair implicit-vs-explicit bias gap (Sun et al. 2x2).

    Two clusters (implicit, explicit) per pair, with base vs instruct bars
    inside each. Faceted by attribute (race, gender). The visual story:
    instruct bar drops to ~0 on explicit but stays high on implicit.
    """
    benches = ["implicit_explicit_race", "implicit_explicit_gender"]
    keep = logit_df[
        logit_df["benchmark"].isin(benches)
        & logit_df["metric"].isin(["implicit_bias_rate", "explicit_bias_rate"])
    ].copy()
    if keep.empty:
        return Path()
    keep["attribute"] = keep["benchmark"].str.replace("implicit_explicit_", "",
                                                       regex=False)
    keep["mode"] = keep["metric"].str.replace("_bias_rate", "", regex=False)
    keep["pair_label"] = (keep["family"].astype(str) + "/"
                          + keep["size"].astype(str))

    g = sns.catplot(
        data=keep, x="pair_label", y="value", hue="variant",
        col="mode", row="attribute", kind="bar",
        errorbar=None, height=3.5, aspect=1.6, sharey=True,
    )
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=60)
        ax.axhline(50, linestyle="--", color="grey", lw=0.5)
    g.set_axis_labels("Model pair", "Bias rate (%)")
    g.fig.suptitle("Implicit vs explicit bias rate, base vs instruct", y=1.02)
    return _save(g.fig, figures_dir / "fig6_implicit_explicit_gap.png")


def fig_jailbreak_reactivation(
    logit_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Per instruct model: CrowS-Pairs score under raw / instruct / jailbreak.

    The headline is whether `jailbreak` score rebounds toward the model's own
    `raw` score (= behavioral suppression) or stays near `instruct` (= genuine
    debiasing). Includes a horizontal reference at 50% (unbiased).
    """
    crows = logit_df[(logit_df["benchmark"] == "crows_pairs")
                     & (logit_df["metric"] == "overall")
                     & (logit_df["variant"] == "instruct")
                     & (logit_df["prompt_mode"].isin(["raw", "instruct", "jailbreak"]))
                     ].copy()
    if crows.empty or "jailbreak" not in set(crows["prompt_mode"]):
        return Path()
    crows["pair_label"] = (crows["family"].astype(str) + "/"
                           + crows["size"].astype(str))

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * crows["pair_label"].nunique() * 3),
                                    4.5))
    sns.barplot(
        data=crows, x="pair_label", y="value", hue="prompt_mode",
        hue_order=["raw", "instruct", "jailbreak"],
        ax=ax, errorbar=None,
    )
    ax.axhline(50, linestyle="--", color="grey", lw=0.7)
    ax.set_ylabel("CrowS-Pairs stereotype score (%)")
    ax.set_xlabel("Instruct model")
    ax.set_title("Jailbreak reactivation: does suppressed bias come back under "
                 "persona injection?")
    ax.tick_params(axis="x", rotation=60)
    return _save(fig, figures_dir / "fig7_jailbreak_reactivation.png")


def fig_cross_benchmark_agreement(
    consistency_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Heatmap: pair × benchmark, colored by the sign of Δ (instruct − base).

    Inputs: the first DataFrame returned by `cross_benchmark_consistency()`.

    Cell colour: green = instruct less biased on that benchmark (Δ < 0),
    red = instruct more biased (Δ > 0), neutral = NaN / not run.
    Right-side annotation: `n_benchmarks_agreeing` out of N benchmarks.
    """
    if consistency_df.empty:
        return Path()
    bench_cols = [c for c in consistency_df.columns if c.startswith("delta_")]
    if not bench_cols:
        return Path()
    df = consistency_df.set_index("instruct_id")[bench_cols].copy()
    df.columns = [c.replace("delta_", "") for c in df.columns]

    # Sign matrix: -1 (instruct less biased), 0 (NaN), +1 (instruct more biased)
    sign = np.where(df.isna(), 0, np.sign(df.values))
    annot = df.map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")

    fig, ax = plt.subplots(figsize=(1.3 * len(df.columns) + 4,
                                    max(3, 0.4 * len(df))))
    sns.heatmap(
        pd.DataFrame(sign, index=df.index, columns=df.columns),
        annot=annot, fmt="", cmap="RdYlGn_r", center=0, vmin=-1, vmax=1,
        cbar_kws={"label": "sign(Δ): green = instruct less biased"},
        ax=ax,
    )
    ax.set_title("Cross-benchmark direction of alignment effect per pair")
    ax.set_ylabel("Instruct model")
    return _save(fig, figures_dir / "fig10_cross_benchmark_agreement.png")


def fig_intervention_by_layer(
    logit_df: pd.DataFrame, intv_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Bias-score delta (baseline − intervened) vs intervention depth.

    For each (model, attribute, method) line, plots how much the headline
    bias score drops when we ablate the latent direction at each depth.
    The depth at which the curve bottoms-out is the **functional locus**
    of demographic representation — distinct from the layer where the
    probe peaks (which is biased toward early layers under keyword labels).
    """
    if intv_df.empty or logit_df.empty:
        return Path()
    headline = {
        "crows_pairs": "overall",
        "stereoset": "overall_SS",
    }
    # Baseline = same model, same benchmark, same prompt_mode, NO intervention.
    base = (
        logit_df[logit_df.apply(lambda r: r["metric"] == headline.get(r["benchmark"]), axis=1)]
        .rename(columns={"value": "bias_baseline"})
        [["model_id", "benchmark", "prompt_mode", "bias_baseline"]]
    )
    intv = intv_df[intv_df.apply(lambda r: r["metric"] == headline.get(r["benchmark"]), axis=1)].copy()
    if intv.empty:
        return Path()
    intv = intv.rename(columns={"value": "bias_intervened"})
    df = intv.merge(base, on=["model_id", "benchmark", "prompt_mode"], how="left")
    df["bias_delta"] = df["bias_baseline"] - df["bias_intervened"]
    df["family_size"] = df["family"] + " " + df["size"]

    g = sns.relplot(
        data=df, x="depth_frac", y="bias_delta",
        hue="variant", style="method",
        col="attribute", row="benchmark",
        kind="line", marker="o", height=3.2, aspect=1.4,
        facet_kws={"sharey": False},
    )
    for ax in g.axes.flatten():
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlim(-0.05, 1.05)
    g.set_axis_labels("Intervention depth (fraction of layers)",
                      "Δ bias  (baseline − intervened)")
    g.fig.suptitle("Causal effect of latent-direction ablation by intervention depth", y=1.02)
    return _save(g.fig, figures_dir / "fig16_intervention_by_layer.png")


def fig_probe_vs_intervention_loci(
    probe_df: pd.DataFrame, intv_df: pd.DataFrame, logit_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Probe-accuracy curve vs intervention-effect curve, on shared depth axis.

    The HEADLINE FIGURE for the probe-locus dissociation argument:
    if the probe peaks at depth ~0.05 (surface keyword detection) but the
    intervention curve dips at depth ~0.5 (where the model actually uses
    the demographic info), the two loci are **dissociated** and probe-peak
    layer selection is methodologically wrong.
    """
    if probe_df.empty or intv_df.empty:
        return Path()

    # Probe curve: mean accuracy vs normalized layer, per attribute, per variant.
    probe_curve = (
        probe_df.assign(metric="probe_accuracy")
        .rename(columns={"layer_normalized": "depth_frac",
                         "mean_accuracy": "value"})
        [["depth_frac", "value", "attribute", "variant", "metric"]]
    )

    # Intervention curve: mean Δ bias vs depth (CrowS-Pairs raw, INLP only — for clarity).
    base = (
        logit_df[(logit_df["benchmark"] == "crows_pairs") & (logit_df["metric"] == "overall")
                 & (logit_df["prompt_mode"] == "raw")]
        .rename(columns={"value": "bias_baseline"})
        [["model_id", "bias_baseline"]]
    )
    intv = intv_df[
        (intv_df["benchmark"] == "crows_pairs") & (intv_df["metric"] == "overall")
        & (intv_df["prompt_mode"] == "raw") & (intv_df["method"] == "inlp")
    ].copy()
    if intv.empty or base.empty:
        return Path()
    intv = intv.rename(columns={"value": "bias_intervened"}).merge(base, on="model_id", how="left")
    intv["delta"] = intv["bias_baseline"] - intv["bias_intervened"]
    intv_curve = (
        intv.assign(metric="intervention_effect")
        .rename(columns={"delta": "value"})
        [["depth_frac", "value", "attribute", "variant", "metric"]]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: probe accuracy.
    sns.lineplot(data=probe_curve, x="depth_frac", y="value",
                 hue="variant", style="attribute", marker="o", ax=axes[0],
                 errorbar="sd")
    axes[0].axhline(0.5, ls="--", color="grey")
    axes[0].set_xlabel("Layer (depth fraction)")
    axes[0].set_ylabel("Probe accuracy")
    axes[0].set_title("Where the demographic info EXISTS\n(probe accuracy by layer)")

    # Right: intervention effect.
    sns.lineplot(data=intv_curve, x="depth_frac", y="value",
                 hue="variant", style="attribute", marker="o", ax=axes[1],
                 errorbar="sd")
    axes[1].axhline(0, ls="--", color="grey")
    axes[1].set_xlabel("Intervention depth (fraction)")
    axes[1].set_ylabel("Δ CrowS bias  (baseline − intervened)")
    axes[1].set_title("Where the model USES the info\n(intervention effect on output bias)")

    fig.suptitle("Dissociation: probe-accuracy locus ≠ causal-effect locus", y=1.02)
    fig.tight_layout()
    return _save(fig, figures_dir / "fig17_probe_vs_intervention_loci.png")


# ---------------------------------------------------------------------------
# Three novel figures specific to our 4-family / 5-prompt-mode design.
# ---------------------------------------------------------------------------


def fig_mistral_prompt_conditional(
    logit_df: pd.DataFrame, figures_dir: Path,
    *, contrast_family: str = "llama",
) -> Path:
    """Mistral observation: alignment effect appears only with chat template.

    Two side-by-side panels for one Mistral pair vs one `contrast_family` pair:
    each panel shows CrowS-Pairs `overall` for the four cells of
    {variant: base, instruct} × {prompt_mode: raw, instruct}. If the alignment
    effect is "real" you see a 2x2 staircase in both; if it's prompt-conditional
    you see a flat raw-mode pair + a separated instruct-mode pair only in
    Mistral.

    No GPU re-run needed — this is just a re-cut of existing CrowS data.
    """
    crows = logit_df[(logit_df["benchmark"] == "crows_pairs")
                     & (logit_df["metric"] == "overall")
                     & (logit_df["prompt_mode"].isin(["raw", "instruct"]))].copy()
    if crows.empty:
        return Path()

    def _largest_pair(family_name: str) -> tuple[str, str] | None:
        fam = crows[crows["family"] == family_name]
        if fam.empty:
            return None
        # Pick the largest size that has both variants in both prompt modes.
        for size in sorted(fam["size"].unique(),
                           key=lambda s: -fam[fam["size"] == s]["num_params"].max()):
            sub = fam[fam["size"] == size]
            cells = sub.groupby(["variant", "prompt_mode"]).size()
            if (cells >= 1).all() and len(cells) == 4:
                return family_name, size
        return None

    a = _largest_pair("mistral")
    b = _largest_pair(contrast_family)
    if a is None or b is None:
        return Path()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (fam, size) in zip(axes, [a, b], strict=True):
        sub = crows[(crows["family"] == fam) & (crows["size"] == size)]
        sns.barplot(data=sub, x="prompt_mode", y="value",
                    hue="variant", order=["raw", "instruct"], ax=ax)
        ax.axhline(50, linestyle="--", color="grey", lw=0.7)
        ax.set_title(f"{fam} {size}")
        ax.set_xlabel("Prompt mode")
        ax.set_ylabel("CrowS-Pairs (%)")
    fig.suptitle("Prompt-conditional alignment: does the effect "
                 "depend on the chat template?", y=1.02)
    fig.tight_layout()
    return _save(fig, figures_dir / "fig13_mistral_prompt_conditional.png")


def fig_per_category_effect_size(
    logit_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Per-pair × per-category Cohen's d on CrowS-Pairs (exploratory companion).

    Heatmap of paired Cohen's d for each of the 9 CrowS-Pairs bias categories,
    one row per (base, instruct) pair, ordered by overall |d|. Sequential
    diverging colormap centered at 0 — negative cells (blue) = instruct less
    biased on that category.

    Important: per-category effects are *exploratory* — we do NOT apply Holm
    across the 9 categories (would obliterate already-marginal effects). This
    figure is for spotting where alignment focuses, not for confirmatory
    significance claims.
    """
    df = logit_df[(logit_df["benchmark"] == "crows_pairs")
                  & (logit_df["metric"].isin(CROWS_CATEGORIES))].copy()
    if df.empty:
        return Path()

    # We approximate per-category paired d as (instruct − base) divided by an
    # implied SD of 0.5 (max for a Bernoulli at p=0.5) — a simple,
    # methodology-honest stand-in when we don't have per-example outcomes here.
    # The actual per-item paired d on CrowS comes from `pair_significance_table`
    # (overall-only). This figure is for shape, not for the headline number.
    pivot = df.pivot_table(
        index=["family", "generation", "size"], columns=["metric", "variant"],
        values="value", aggfunc="first",
    )
    rows = []
    for (family, gen, size), row in pivot.iterrows():
        for cat in CROWS_CATEGORIES:
            try:
                base = float(row[(cat, "base")]) / 100
                inst = float(row[(cat, "instruct")]) / 100
            except (KeyError, ValueError):
                continue
            d_approx = (inst - base) / 0.5
            rows.append({"family": family, "generation": gen, "size": size,
                         "category": cat, "d_approx": d_approx})
    if not rows:
        return Path()
    long = pd.DataFrame(rows)
    long["pair_label"] = long["family"] + "/" + long["generation"] + "/" + long["size"]

    overall_order = (long.groupby("pair_label")["d_approx"]
                     .apply(lambda s: s.abs().mean())
                     .sort_values(ascending=False).index)
    mat = long.pivot(index="pair_label", columns="category", values="d_approx")
    mat = mat.reindex(overall_order)
    mat = mat[[c for c in CROWS_CATEGORIES if c in mat.columns]]

    fig, ax = plt.subplots(figsize=(1.0 * mat.shape[1] + 4,
                                    max(3, 0.4 * len(mat))))
    sns.heatmap(mat, cmap="vlag", center=0,
                vmin=-1.5, vmax=1.5, annot=True, fmt=".2f",
                cbar_kws={"label": "approx. Cohen's d (instruct − base)"},
                ax=ax)
    ax.set_title("Per-category alignment effect (CrowS-Pairs) — exploratory")
    ax.set_ylabel("Pair (sorted by mean |d|)")
    ax.set_xlabel("Bias category")
    return _save(fig, figures_dir / "fig14_per_category_effect_size.png")


def fig_probe_direction_rotation(
    direction_cosines_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """How much did alignment rotate the demographic direction at each layer?

    Inputs: the DataFrame returned by `cross_pair_direction_cosines()`.
    Plots cos(direction_base[L], direction_instruct[L]) vs depth_frac, one
    line per (pair, attribute), faceted by attribute.

    Reading guide:
        cosine ≈ 1 at all layers → alignment didn't rotate the direction →
                                   bias representation is preserved.
        cosine drops in mid/late layers → alignment rotated the direction
                                          where the model would have used it.
    Ties directly to the thesis hypothesis at the geometry level.
    """
    if direction_cosines_df.empty:
        return Path()
    df = direction_cosines_df.copy()
    df["pair_label"] = df["family"] + "/" + df["size"]

    g = sns.relplot(
        data=df, x="depth_frac", y="cosine",
        hue="pair_label", col="attribute", kind="line",
        marker="o", height=4, aspect=1.4,
    )
    for ax in g.axes.flatten():
        ax.axhline(1.0, linestyle="--", color="grey", lw=0.5)
        ax.axhline(0.0, linestyle=":", color="grey", lw=0.5)
        ax.set_ylim(-0.2, 1.1)
    g.set_axis_labels("Layer depth (fraction)",
                      "cos(direction_base, direction_instruct)")
    g.fig.suptitle("Alignment-induced rotation of the demographic direction",
                   y=1.02)
    return _save(g.fig, figures_dir / "fig15_probe_direction_rotation.png")


# ---------------------------------------------------------------------------
# Multilingual statistical figures (Bite 4 of the multilingual extension).
# ---------------------------------------------------------------------------


def fig_multilingual_significance_heatmap(
    pair_sig_per_lang_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Per-pair × per-language Cohen's d, with ★ / ★★ markers from Holm.

    Headline statistical multilingual figure. Inputs: long-format DataFrame
    from `pair_significance_per_language()` with one row per (pair, language).
    Cell value = `cohens_d_paired`. Annotation = numeric d plus ★ / ★★ if
    the row passes Holm at α=0.05 / α=0.0025 within its language family.
    """
    if pair_sig_per_lang_df is None or pair_sig_per_lang_df.empty:
        return Path()
    df = pair_sig_per_lang_df.copy()
    df["pair_label"] = (df["family"].astype(str) + "/"
                        + df["generation"].astype(str) + "/"
                        + df["size"].astype(str))

    pivot_d = df.pivot(index="pair_label", columns="language",
                       values="cohens_d_paired")
    # Order languages canonically and pairs by overall mean |d| desc.
    lang_order = [c for c in ("en", "fr", "es", "de", "pt", "it")
                  if c in pivot_d.columns]
    pivot_d = pivot_d[lang_order] if lang_order else pivot_d
    overall_order = (pivot_d.abs().mean(axis=1).sort_values(ascending=False).index)
    pivot_d = pivot_d.loc[overall_order]

    # Annotation: "+0.32 ★★" etc.
    def _star(row, col):
        sub = df[(df["pair_label"] == row) & (df["language"] == col)]
        if sub.empty:
            return ""
        d = sub["cohens_d_paired"].iloc[0]
        if pd.isna(d):
            return ""
        s = ""
        if bool(sub["reject_strict"].iloc[0]):
            s = " ★★"
        elif bool(sub["reject_holm"].iloc[0]):
            s = " ★"
        return f"{d:+.2f}{s}"

    annot = pd.DataFrame(
        [[_star(r, c) for c in pivot_d.columns] for r in pivot_d.index],
        index=pivot_d.index, columns=pivot_d.columns,
    )

    fig, ax = plt.subplots(
        figsize=(1.4 * len(pivot_d.columns) + 4, max(3, 0.4 * len(pivot_d))),
    )
    sns.heatmap(
        pivot_d, annot=annot, fmt="", cmap="vlag", center=0,
        vmin=-1.0, vmax=1.0, ax=ax,
        cbar_kws={"label": "Cohen's d (paired) — instruct vs base"},
    )
    ax.set_title(
        "Multilingual alignment effect (paired d on CrowS-Pairs)\n"
        "★ = survives Holm at α=0.05; ★★ = also survives α=0.0025"
    )
    ax.set_xlabel("Language")
    ax.set_ylabel("Pair (sorted by mean |d|)")
    return _save(fig, figures_dir / "fig19_multilingual_significance_heatmap.png")


def fig_multilingual_consistency_matrix(
    lang_corr_df: pd.DataFrame, figures_dir: Path,
) -> Path:
    """Spearman correlation of paired Δ across languages.

    Inputs: the second DataFrame returned by `cross_language_consistency`.
    Off-diagonal cells near +1 → alignment effect transfers across languages;
    near 0 → effect is language-specific. The methodological complement to
    fig19's per-pair view.
    """
    if lang_corr_df is None or lang_corr_df.empty:
        return Path()
    fig, ax = plt.subplots(figsize=(1.0 * len(lang_corr_df.columns) + 3,
                                    1.0 * len(lang_corr_df) + 2))
    sns.heatmap(
        lang_corr_df, annot=True, fmt=".2f", cmap="vlag", center=0,
        vmin=-1, vmax=1, ax=ax,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_title("Cross-language consistency of the alignment effect\n"
                 "(Spearman of paired Δ across pairs, per language)")
    return _save(fig, figures_dir / "fig20_multilingual_consistency.png")


def generate_all(
    logit_df: pd.DataFrame, probe_df: pd.DataFrame, figures_dir: Path,
    intv_df: pd.DataFrame | None = None,
    *,
    pair_sig_df: pd.DataFrame | None = None,
    consistency_df: pd.DataFrame | None = None,
    direction_cosines_df: pd.DataFrame | None = None,
    pair_sig_per_lang_df: pd.DataFrame | None = None,
    lang_consistency_df: pd.DataFrame | None = None,
    lang_corr_df: pd.DataFrame | None = None,
    results_dir: Path | None = None,
    registry_pairs: list[tuple[str, str, str, str, str]] | None = None,
) -> list[Path]:
    """Run every figure function and return the paths actually written.

    Optional inputs (all skip gracefully if missing):
        pair_sig_df          — for `fig_alignment_delta_forest`
        consistency_df       — for `fig_cross_benchmark_agreement`
        direction_cosines_df — for `fig_probe_direction_rotation`
        pair_sig_per_lang_df — for `fig_multilingual_significance_heatmap`
        lang_corr_df         — for `fig_multilingual_consistency_matrix`
        results_dir + registry_pairs — for the Δ-based correlation in
                          `fig_benchmark_correlation`. Falls back to raw-score
                          Pearson if either is None.
    """
    figures_dir = Path(figures_dir)
    paths: list[Path] = []

    # Functions whose only input is logit_df.
    logit_only = (
        fig_alignment_by_size,    # headline — Anders-sketched
        fig_multilingual_heatmap, # multilingual extension
        fig_crows_heatmap,
        fig_generation_lines,
        fig_alignment_delta,            # legacy simple bar
        fig_scaling,
        fig_iat_by_category,
        fig_bbq_deferral_decomposition,
        fig_implicit_explicit_gap,
        fig_jailbreak_reactivation,
        fig_mistral_prompt_conditional,
        fig_per_category_effect_size,
    )
    for fn in logit_only:
        try:
            p = fn(logit_df, figures_dir)
        except Exception as exc:  # pragma: no cover
            logger.error("%s failed: %s", fn.__name__, exc)
            continue
        if p and p.exists():
            paths.append(p)

    # Cross-benchmark correlation: Δ-based when registry pairs are provided.
    try:
        p = fig_benchmark_correlation(logit_df, figures_dir,
                                      results_dir=results_dir,
                                      registry_pairs=registry_pairs)
        if p and p.exists():
            paths.append(p)
    except Exception as exc:  # pragma: no cover
        logger.error("fig_benchmark_correlation failed: %s", exc)

    # Pair-stats forest plot (Søgaard-aligned headline statistical figure).
    if pair_sig_df is not None and not pair_sig_df.empty:
        try:
            p = fig_alignment_delta_forest(pair_sig_df, figures_dir)
            if p and p.exists():
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            logger.error("fig_alignment_delta_forest failed: %s", exc)

    # Cross-benchmark agreement heatmap (uses cross_benchmark_consistency output).
    if consistency_df is not None and not consistency_df.empty:
        try:
            p = fig_cross_benchmark_agreement(consistency_df, figures_dir)
            if p and p.exists():
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            logger.error("fig_cross_benchmark_agreement failed: %s", exc)

    # Probe-direction rotation (uses cross_pair_direction_cosines output).
    if direction_cosines_df is not None and not direction_cosines_df.empty:
        try:
            p = fig_probe_direction_rotation(direction_cosines_df, figures_dir)
            if p and p.exists():
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            logger.error("fig_probe_direction_rotation failed: %s", exc)

    # Multilingual statistical figures (need pair_significance_per_language +
    # cross_language_consistency to have been built).
    if pair_sig_per_lang_df is not None and not pair_sig_per_lang_df.empty:
        try:
            p = fig_multilingual_significance_heatmap(
                pair_sig_per_lang_df, figures_dir,
            )
            if p and p.exists():
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            logger.error("fig_multilingual_significance_heatmap failed: %s", exc)
    if lang_corr_df is not None and not lang_corr_df.empty:
        try:
            p = fig_multilingual_consistency_matrix(lang_corr_df, figures_dir)
            if p and p.exists():
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            logger.error("fig_multilingual_consistency_matrix failed: %s", exc)

    # Probing figures.
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

    # Intervention figures.
    if intv_df is not None and not intv_df.empty:
        try:
            p = fig_intervention_by_layer(logit_df, intv_df, figures_dir)
            if p and p.exists():
                paths.append(p)
        except Exception as exc:  # pragma: no cover
            logger.error("fig_intervention_by_layer failed: %s", exc)
        if not probe_df.empty:
            try:
                p = fig_probe_vs_intervention_loci(probe_df, intv_df, logit_df,
                                                    figures_dir)
                if p and p.exists():
                    paths.append(p)
            except Exception as exc:  # pragma: no cover
                logger.error("fig_probe_vs_intervention_loci failed: %s", exc)
    return paths
