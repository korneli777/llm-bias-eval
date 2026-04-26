"""Bootstrap CIs, Cohen's d, BH-FDR — drives Section 9 statistical tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
