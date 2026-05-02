"""Bootstrap CIs, Cohen's d, BH-FDR + paired-pair significance tooling.

The paired-pair tooling lives here so that for every base↔instruct
checkpoint pair we can produce a thesis-ready row of:

    base_score | instruct_score | Δ | 95% CI (Δ) | Cohen's d (paired) | p (paired permutation)

with Holm-Bonferroni correction across all pairs. This is the single
strongest piece of evidence for/against the alignment-reduces-bias claim,
because the paired permutation test treats each CrowS-Pairs *item* as the
unit of analysis (1508 of them per model), not each model — so we get
both per-model and pooled significance at the right grain.

Methodological choices follow our supervisor's published guidance:

* **Effect size first** (Søgaard, NAACL 2013, "Estimating effect size
  across datasets"). Tables are sorted by |Cohen's d|, not by p-value,
  and a `cohens_d_label` column applies Cohen's conventional bands:
      |d| < 0.2  → "negligible"
      0.2 ≤ |d| < 0.5 → "small"
      0.5 ≤ |d| < 0.8 → "medium"
      |d| ≥ 0.8       → "large"
* **Permutation tests, not t-tests** (Søgaard, Johannsen, Plank, Hovy &
  Alonso, CoNLL 2014). Binary CrowS-Pairs outcomes violate t-test
  normality assumptions; we use a paired permutation test with n_perm
  ≥ 10,000 for production runs.
* **Two significance thresholds**: standard α = 0.05 (`reject_holm`) and
  the stricter α = 0.0025 (`reject_strict`) recommended by Søgaard et al.
  (2014) to control NLP false-positive rates.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def bootstrap_ci(
    values: np.ndarray, *, statistic=np.mean, n_iter: int = 1000, alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (point, lo, hi)."""
    rng = rng or RNG
    values = np.asarray(values)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(values))
    boots = np.empty(n_iter, dtype=np.float64)
    n = len(values)
    for i in range(n_iter):
        boots[i] = statistic(values[rng.integers(0, n, n)])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(((a.var(ddof=1) * (a.size - 1)) + (b.var(ddof=1) * (b.size - 1))) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return boolean mask of hypotheses rejected at FDR ≤ alpha."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    cutoff = np.where(passed)[0].max()
    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[: cutoff + 1] = True
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected


def cohens_d_label(d: float) -> str:
    """Cohen's conventional effect-size bands. NaN-safe."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "n/a"
    a = abs(float(d))
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def cohens_d_paired(base: np.ndarray, instruct: np.ndarray) -> float:
    """Cohen's d for *paired* samples (per-item differences).

    For binary CrowS-Pairs outcomes per item i, ``diff_i = instruct_i − base_i``
    has values in {−1, 0, +1}. We report ``mean(diff) / sd(diff)``. This is
    the appropriate paired-comparison effect size — different from the
    independent-samples ``cohens_d`` above which uses pooled SD.
    """
    base = np.asarray(base, dtype=float)
    instruct = np.asarray(instruct, dtype=float)
    if base.size != instruct.size or base.size < 2:
        return float("nan")
    diff = instruct - base
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd)


def paired_permutation_test(
    base: np.ndarray, instruct: np.ndarray, *,
    n_perm: int = 10000, seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired permutation test on the per-item difference in means.

    For each item, randomly swap the (base, instruct) labels with probability
    0.5; recompute Δ = mean(instruct) − mean(base). The two-sided p-value is
    the fraction of permuted |Δ| ≥ observed |Δ|, with a +1 small-sample
    correction in numerator and denominator (standard practice).

    Returns ``{observed_delta, p_value, n_perm}``.
    """
    base = np.asarray(base, dtype=float)
    instruct = np.asarray(instruct, dtype=float)
    if base.size != instruct.size or base.size == 0:
        return {"observed_delta": float("nan"), "p_value": float("nan"), "n_perm": 0}
    obs_delta = float(instruct.mean() - base.mean())
    abs_obs = abs(obs_delta)

    rng = np.random.default_rng(seed)
    n = base.size
    # Vectorised: build n_perm × n random masks, swap, compute deltas.
    flips = rng.random((n_perm, n)) < 0.5
    perm_inst = np.where(flips, base[None, :], instruct[None, :])
    perm_base = np.where(flips, instruct[None, :], base[None, :])
    perm_deltas = perm_inst.mean(axis=1) - perm_base.mean(axis=1)
    n_extreme = int((np.abs(perm_deltas) >= abs_obs).sum())
    return {
        "observed_delta": obs_delta,
        "p_value": (n_extreme + 1) / (n_perm + 1),
        "n_perm": n_perm,
    }


def bootstrap_paired_delta_ci(
    base: np.ndarray, instruct: np.ndarray, *,
    n_iter: int = 10000, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the paired Δ = mean(instruct) − mean(base).

    Resamples *items* (with replacement, preserving the pairing). Returns
    ``(point, lo, hi)``.
    """
    base = np.asarray(base, dtype=float)
    instruct = np.asarray(instruct, dtype=float)
    n = base.size
    if n != instruct.size or n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        deltas[i] = instruct[idx].mean() - base[idx].mean()
    point = float(instruct.mean() - base.mean())
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return point, lo, hi


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni step-down on a {label: p} dict.

    Returns ``{label: {p, p_adj, reject}}``. More conservative than BH-FDR;
    appropriate when we need strong control of family-wise error rate, e.g.
    when claiming significance per individual model pair.
    """
    items = [(k, v) for k, v in p_values.items() if v is not None and not np.isnan(v)]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float]] = {}
    running = 0.0
    for i, (label, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)  # enforce monotonicity along sorted p
        out[label] = {"p": float(p), "p_adj": float(running), "reject": running < alpha}
    return out


def _crows_folder(language: str) -> str:
    """Result-folder name: `crows_pairs` (en) or `crows_pairs_<lang>` (else).

    Mirrors the producer-side convention in `scripts/run_logit_benchmarks.py`.
    """
    return "crows_pairs" if language == "en" else f"crows_pairs_{language}"


def _load_crows_per_example_outcomes(
    results_dir: Path, model_id: str, prompt_mode: str = "raw",
    *, scoring: str = "norm", language: str = "en",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (pair_ids, outcomes) for CrowS-Pairs raw mode for one model.

    scoring='norm' (default) reads `stereo_won` (length-normalised), 'raw'
    reads `stereo_won_raw` (lm-eval-harness convention). Returns None if
    the file doesn't exist or the requested scoring isn't available.
    language defaults to 'en'; pass 'fr', 'es', 'de', 'pt', 'it' for the
    BigScienceBiasEval multilingual mirrors.
    """
    short = model_id.replace("/", "__")
    fp = (results_dir / "logit_scores" / _crows_folder(language)
          / f"{short}__{prompt_mode}.json")
    if not fp.exists():
        return None
    with open(fp) as f:
        per_ex = json.load(f)["result"].get("per_example", [])
    if not per_ex:
        return None
    key = "stereo_won" if scoring == "norm" else "stereo_won_raw"
    if key not in per_ex[0]:
        return None
    pair_ids = np.array([r["pair_id"] for r in per_ex])
    outcomes = np.array([1 if r[key] else 0 for r in per_ex], dtype=float)
    return pair_ids, outcomes


def pair_significance_table(
    results_dir: Path, registry_pairs: list[tuple[str, str, str, str, str]],
    *, prompt_mode: str = "raw", scoring: str = "norm",
    language: str = "en",
    n_perm: int = 10000, n_boot: int = 10000, seed: int = 42,
) -> pd.DataFrame:
    """Per-pair significance table: the headline statistical evidence row.

    For each (base_id, instruct_id) pair where both have CrowS-Pairs results,
    computes:
      base_score, instruct_score, delta, ci_lo, ci_hi, p_perm, cohens_d,
      n_pairs (= shared items between base and instruct).

    Then applies Holm-Bonferroni across pairs and adds `p_adj` + `reject_holm`.
    """
    rows: list[dict[str, Any]] = []
    for base_id, instruct_id, family, generation, size in registry_pairs:
        b = _load_crows_per_example_outcomes(
            results_dir, base_id, prompt_mode, scoring=scoring, language=language,
        )
        i = _load_crows_per_example_outcomes(
            results_dir, instruct_id, prompt_mode, scoring=scoring, language=language,
        )
        if b is None or i is None:
            continue
        b_ids, b_out = b
        i_ids, i_out = i
        # Align on shared pair_ids (both should be 0..1507 but be defensive).
        common = sorted(set(b_ids.tolist()) & set(i_ids.tolist()))
        if not common:
            continue
        b_idx = {int(p): k for k, p in enumerate(b_ids)}
        i_idx = {int(p): k for k, p in enumerate(i_ids)}
        base_arr = np.array([b_out[b_idx[p]] for p in common])
        inst_arr = np.array([i_out[i_idx[p]] for p in common])
        perm = paired_permutation_test(base_arr, inst_arr, n_perm=n_perm, seed=seed)
        d, lo, hi = bootstrap_paired_delta_ci(base_arr, inst_arr, n_iter=n_boot, seed=seed)
        rows.append({
            "family": family, "generation": generation, "size": size,
            "base_id": base_id, "instruct_id": instruct_id,
            "n_pairs": len(common),
            "base_score": 100 * float(base_arr.mean()),
            "instruct_score": 100 * float(inst_arr.mean()),
            "delta": 100 * float(d),
            "delta_ci_lo": 100 * float(lo),
            "delta_ci_hi": 100 * float(hi),
            "p_value": perm["p_value"],
            "cohens_d_paired": cohens_d_paired(base_arr, inst_arr),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Use instruct_id as the per-pair unique key for the multiplicity correction
    # (family/gen/size can collide when two pairs share those metadata values).
    pmap = dict(zip(df["instruct_id"], df["p_value"], strict=True))

    # Holm-adjusted p-values are method-of-correction-only — they don't depend
    # on alpha. We compute them once and emit two boolean reject columns:
    #   reject_holm   — α = 0.05  (standard NLP threshold)
    #   reject_strict — α = 0.0025 (Søgaard, Johannsen, Plank, Hovy & Alonso,
    #                   CoNLL 2014, argue this is the threshold needed to keep
    #                   false-positive rate acceptable in NLP comparisons).
    holm_05 = holm_bonferroni(pmap, alpha=0.05)
    holm_strict = holm_bonferroni(pmap, alpha=0.0025)
    df["p_adj_holm"] = df["instruct_id"].map(lambda k: holm_05[k]["p_adj"])
    df["reject_holm"] = df["instruct_id"].map(lambda k: holm_05[k]["reject"])
    df["reject_strict"] = df["instruct_id"].map(lambda k: holm_strict[k]["reject"])

    # Effect-size first (Søgaard 2013): annotate, reorder, and sort by |d|.
    df["cohens_d_label"] = df["cohens_d_paired"].map(cohens_d_label)
    col_order = [
        "family", "generation", "size", "base_id", "instruct_id", "n_pairs",
        "base_score", "instruct_score", "delta", "delta_ci_lo", "delta_ci_hi",
        "cohens_d_paired", "cohens_d_label",
        "p_value", "p_adj_holm", "reject_holm", "reject_strict",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = (df.assign(_abs_d=df["cohens_d_paired"].abs())
            .sort_values("_abs_d", ascending=False, na_position="last")
            .drop(columns="_abs_d")
            .reset_index(drop=True))
    return df


def pair_significance_per_language(
    results_dir: Path, registry_pairs: list[tuple[str, str, str, str, str]],
    languages: tuple[str, ...] = ("en", "fr", "es", "de", "pt", "it"),
    *, prompt_mode: str = "raw", scoring: str = "norm",
    n_perm: int = 10000, n_boot: int = 10000, seed: int = 42,
) -> pd.DataFrame:
    """Run pair_significance_table once per language; return long-format frame.

    Adds a `language` column and stacks the per-language tables. Holm
    correction is applied within each language (so ★ markers compare to the
    same family of pairs in that language). Languages whose results aren't
    on disk are silently skipped — defensive against partial multilingual
    sweeps.
    """
    parts: list[pd.DataFrame] = []
    for lang in languages:
        df = pair_significance_table(
            results_dir, registry_pairs,
            prompt_mode=prompt_mode, scoring=scoring, language=lang,
            n_perm=n_perm, n_boot=n_boot, seed=seed,
        )
        if df.empty:
            continue
        df = df.assign(language=lang)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Move `language` to the front for readability.
    cols = ["language"] + [c for c in out.columns if c != "language"]
    return out[cols]


def _load_summary_metric(
    results_dir: Path, benchmark: str, model_id: str, metric: str,
    *, prompt_mode: str = "raw",
) -> float | None:
    """Read one summary metric from a logit-benchmark JSON. None if missing."""
    short = model_id.replace("/", "__")
    fp = results_dir / "logit_scores" / benchmark / f"{short}__{prompt_mode}.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        summ = json.load(f)["result"].get("summary", {})
    val = summ.get(metric)
    return None if val is None else float(val)


# Per benchmark: (folder, summary metric, "lower-is-less-biased" transform).
# Cross-benchmark consistency uses these so the sign of every Δ is "negative
# means instruct less biased".
_BENCH_HEADLINE: dict[str, tuple[str, Callable[[float], float]]] = {
    # CrowS-Pairs `overall` is in [0, 100], 50 = neutral but typical scores
    # are 60-70 — lower is monotonically less biased, so use raw.
    "crows_pairs": ("overall", lambda v: v),
    # BBQ bias_ambig is in [-1, 1], 0 = neutral. Distance from 0.
    "bbq": ("overall_bias_ambig", lambda v: abs(v)),
    # StereoSet SS is in [0, 100], 50 = neutral. Distance from 50.
    "stereoset": ("overall_SS", lambda v: abs(v - 50)),
    # IAT overall_abs_d is already a magnitude — lower is less biased.
    "iat": ("overall_abs_d", lambda v: v),
}


def cross_benchmark_consistency(
    results_dir: Path,
    registry_pairs: list[tuple[str, str, str, str, str]],
    *,
    benchmarks: tuple[str, ...] = ("crows_pairs", "bbq", "stereoset", "iat"),
    prompt_mode: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-benchmark agreement on the alignment effect (Søgaard 2013 robustness).

    For each (base, instruct) pair, computes the headline metric on each
    benchmark in a "lower = less biased" frame, then Δ = instruct − base.
    Δ < 0 → instruct less biased on that benchmark.

    Returns
    -------
    (consistency_df, corr_df)
        consistency_df: one row per pair with columns
            family, generation, size, base_id, instruct_id,
            delta_<bench> for each benchmark, n_benchmarks_present,
            n_benchmarks_agreeing, all_agree
        corr_df: Spearman correlation matrix of the per-pair Δ across
            benchmarks (each pair = one observation). Off-diagonal cells
            near +1 → benchmarks agree on which pairs improved most;
            near 0 → benchmarks measure independent dimensions of bias
            (cf. Cabello, Jørgensen & Søgaard 2023, "On the Independence
            of Association Bias and Empirical Fairness").
    """
    rows: list[dict[str, Any]] = []
    for base_id, instruct_id, family, generation, size in registry_pairs:
        row: dict[str, Any] = {
            "family": family, "generation": generation, "size": size,
            "base_id": base_id, "instruct_id": instruct_id,
        }
        deltas: list[float] = []
        for bench in benchmarks:
            metric, transform = _BENCH_HEADLINE[bench]
            b = _load_summary_metric(results_dir, bench, base_id, metric,
                                     prompt_mode=prompt_mode)
            i = _load_summary_metric(results_dir, bench, instruct_id, metric,
                                     prompt_mode=prompt_mode)
            if b is None or i is None:
                row[f"delta_{bench}"] = float("nan")
                continue
            d = transform(i) - transform(b)
            row[f"delta_{bench}"] = d
            deltas.append(d)
        present = [d for d in deltas if not np.isnan(d)]
        row["n_benchmarks_present"] = len(present)
        row["n_benchmarks_agreeing"] = sum(1 for d in present if d < 0)
        row["all_agree"] = (len(present) == len(benchmarks)
                            and row["n_benchmarks_agreeing"] == len(benchmarks))
        rows.append(row)

    consistency_df = pd.DataFrame(rows)

    # Spearman across benchmarks: each pair is one observation.
    delta_cols = [f"delta_{b}" for b in benchmarks if f"delta_{b}" in consistency_df.columns]
    if consistency_df.empty or len(delta_cols) < 2:
        corr_df = pd.DataFrame()
    else:
        corr_df = consistency_df[delta_cols].corr(method="spearman")
        corr_df.index = [c.replace("delta_", "") for c in corr_df.index]
        corr_df.columns = [c.replace("delta_", "") for c in corr_df.columns]
    return consistency_df, corr_df


def cross_language_consistency(
    results_dir: Path,
    registry_pairs: list[tuple[str, str, str, str, str]],
    languages: tuple[str, ...] = ("en", "fr", "es", "de", "pt", "it"),
    *, prompt_mode: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-language agreement on the alignment effect (multilingual robustness).

    Mirrors `cross_benchmark_consistency` but iterates languages on a single
    benchmark (CrowS-Pairs). For each (base, instruct) pair, computes
    Δ_lang = instruct_score − base_score on the multilingual CrowS-Pairs
    overall metric (lower = less biased, so Δ < 0 → instruct less biased
    in that language).

    Returns
    -------
    (consistency_df, corr_df)
        consistency_df: one row per pair with columns
            family, generation, size, base_id, instruct_id,
            delta_<lang> for each language, n_languages_present,
            n_languages_agreeing, all_agree
        corr_df: Spearman correlation matrix of the per-pair Δ across
            languages (each pair = one observation). Off-diagonal cells
            near +1 → alignment effect is consistent across languages;
            near 0 → effect is language-specific (worth a methodological
            note in the thesis).
    """
    rows: list[dict[str, Any]] = []
    for base_id, instruct_id, family, generation, size in registry_pairs:
        row: dict[str, Any] = {
            "family": family, "generation": generation, "size": size,
            "base_id": base_id, "instruct_id": instruct_id,
        }
        deltas: list[float] = []
        for lang in languages:
            folder = _crows_folder(lang)
            b = _load_summary_metric(results_dir, folder, base_id, "overall",
                                     prompt_mode=prompt_mode)
            i = _load_summary_metric(results_dir, folder, instruct_id, "overall",
                                     prompt_mode=prompt_mode)
            if b is None or i is None:
                row[f"delta_{lang}"] = float("nan")
                continue
            d = i - b  # CrowS overall: lower is less biased, no transform.
            row[f"delta_{lang}"] = d
            deltas.append(d)
        present = [d for d in deltas if not np.isnan(d)]
        row["n_languages_present"] = len(present)
        row["n_languages_agreeing"] = sum(1 for d in present if d < 0)
        row["all_agree"] = (len(present) == len(languages)
                            and row["n_languages_agreeing"] == len(languages))
        rows.append(row)

    consistency_df = pd.DataFrame(rows)
    delta_cols = [f"delta_{lang}" for lang in languages
                  if f"delta_{lang}" in consistency_df.columns]
    if consistency_df.empty or len(delta_cols) < 2:
        corr_df = pd.DataFrame()
    else:
        corr_df = consistency_df[delta_cols].corr(method="spearman")
        corr_df.index = [c.replace("delta_", "") for c in corr_df.index]
        corr_df.columns = [c.replace("delta_", "") for c in corr_df.columns]
    return consistency_df, corr_df


def per_example_bootstrap(
    results_dir: Path, benchmark: str, *, n_iter: int = 1000,
) -> dict[tuple[str, str], dict[str, float]]:
    """For one benchmark, compute bootstrap CIs over per-example data per model.

    Currently supported: crows_pairs (binary stereo_won), stereoset (SS).
    Returns {(model_id, prompt_mode): {"point": ..., "lo": ..., "hi": ..., "n": ...}}.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for path in (results_dir / "logit_scores" / benchmark).glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        per_ex = data["result"].get("per_example", [])
        if not per_ex:
            continue
        if benchmark == "crows_pairs":
            arr = np.array([1 if r["stereo_won"] else 0 for r in per_ex], dtype=float)
            def stat(v):
                return 100 * v.mean()
        elif benchmark == "stereoset":
            meaningful = [r for r in per_ex if r.get("meaningful")]
            if not meaningful:
                continue
            arr = np.array([1 if r["stereo_over_anti"] else 0 for r in meaningful], dtype=float)
            def stat(v):
                return 100 * v.mean()
        else:
            continue
        point, lo, hi = bootstrap_ci(arr, statistic=stat, n_iter=n_iter)
        prompt_mode = data["result"].get("prompt_mode", "raw")
        out[(data["spec"]["model_id"], prompt_mode)] = {
            "point": point, "lo": lo, "hi": hi, "n": len(arr),
        }
    return out
