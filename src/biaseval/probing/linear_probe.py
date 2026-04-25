"""Train logistic-regression probes per layer."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_layer_probe(
    X: np.ndarray, y: np.ndarray, *, cv_folds: int = 5, seed: int = 42
) -> dict[str, float]:
    """Stratified k-fold CV accuracy on a logistic-regression probe."""
    if len(np.unique(y)) < 2:
        return {"mean_accuracy": float("nan"), "std_accuracy": float("nan")}

    X = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
    }


def train_probes_all_layers(
    activation_dir: str | Path,
    labels: np.ndarray,
    num_layers: int,
    attribute_name: str,
    *,
    cv_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Iterate every layer's .npy file and train a probe."""
    activation_dir = Path(activation_dir)
    results: list[dict] = []
    for layer_idx in range(num_layers):
        X = np.load(activation_dir / f"layer_{layer_idx}.npy")
        scores = train_layer_probe(X, labels, cv_folds=cv_folds, seed=seed)
        results.append(
            {
                "layer": layer_idx,
                "layer_normalized": layer_idx / max(num_layers - 1, 1),
                "attribute": attribute_name,
                **scores,
            }
        )
    return results
