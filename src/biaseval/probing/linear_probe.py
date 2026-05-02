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


def mean_difference_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sun et al. (2025) direction vector: μ(class=1) − μ(class=0).

    Returns a unit-norm vector in activation space pointing from class 0
    centroid to class 1 centroid. This is the direction we'd add/subtract
    along for activation steering, and the projection axis for ablation —
    so we save it alongside probe accuracy at every layer.
    """
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], dtype=np.float32)
    mu1 = X[y == 1].mean(axis=0)
    mu0 = X[y == 0].mean(axis=0)
    diff = (mu1 - mu0).astype(np.float32)
    nrm = float(np.linalg.norm(diff))
    return diff / nrm if nrm > 0 else diff


def train_probes_all_layers(
    activation_dir: str | Path,
    labels: np.ndarray,
    num_layers: int,
    attribute_name: str,
    *,
    cv_folds: int = 5,
    seed: int = 42,
    save_directions: bool = True,
) -> list[dict]:
    """Iterate every layer's .npy file, train a probe, save direction vector."""
    activation_dir = Path(activation_dir)
    results: list[dict] = []
    if save_directions:
        directions = np.zeros((num_layers, np.load(activation_dir / "layer_0.npy").shape[1]),
                              dtype=np.float32)
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
        if save_directions:
            directions[layer_idx] = mean_difference_direction(X, labels)
    if save_directions:
        np.save(activation_dir / f"direction_{attribute_name}.npy", directions)
    return results
