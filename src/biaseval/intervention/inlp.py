"""Linear-concept erasure: INLP (iterative) and LEACE (closed-form).

INLP — Ravfogel et al., ACL 2020, "Null It Out: Guarding Protected Attributes
by Iterative Nullspace Projection"
    Iteratively (i) trains a linear classifier w on activations X,
    (ii) projects X onto the null-space of w via P_i = (I − w wᵀ / ‖w‖²),
    (iii) repeats on the projected X until probe accuracy hits chance.
    The composed projection P = ∏ P_i removes the targeted attribute.

LEACE — Belrose et al., 2023, "LEACE: Perfect linear concept erasure in
closed form"
    Computes the closed-form projection that exactly nullifies any linear
    classifier's ability to recover the attribute, with provably minimal
    perturbation in mean-squared sense. Faster, single-shot, and more
    principled — but newer / less established in the bias-eval literature.

Both return a P that's idempotent and applied identically by the forward
hook. We ship both so the thesis can show the finding survives the choice
of erasure method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# INLP
# ---------------------------------------------------------------------------


@dataclass
class NullspaceResult:
    """Output of `fit_inlp`."""

    projection: np.ndarray  # (H, H), float32 — the composed projection matrix
    n_iterations: int
    accuracy_curve: list[float]  # probe accuracy per iteration (initial → final)
    converged: bool  # True if final accuracy ≤ chance_threshold
    method: str = "inlp"


def _train_probe_get_w(X: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[np.ndarray, float]:
    """Fit logistic regression; return (weight vector, accuracy)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
    acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean())
    clf.fit(X, y)
    w = clf.coef_.reshape(-1).astype(np.float32)
    return w, acc


def _projection_for(w: np.ndarray) -> np.ndarray:
    """P = I − w wᵀ / ‖w‖²  (orthogonal projection onto the null-space of w)."""
    norm_sq = float(w @ w)
    if norm_sq < 1e-12:
        return np.eye(w.shape[0], dtype=np.float32)
    outer = np.outer(w, w).astype(np.float32) / norm_sq
    return np.eye(w.shape[0], dtype=np.float32) - outer


def fit_inlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 10,
    chance_threshold: float = 0.55,
    seed: int = 42,
) -> NullspaceResult:
    """Iterative Null-space Projection.

    Stops when 5-fold CV probe accuracy on the current projected X falls below
    `chance_threshold`, or after `max_iter` iterations — whichever first.

    The returned projection P is in **row-vector convention**: applied as
    ``x' = x @ P`` for row vectors x. INLP's per-iteration projector is
    symmetric so the convention is moot for it; we use row-form throughout
    for consistency with LEACE.

    Each iteration trains a new probe on the *previously projected* X — the
    composed projection therefore removes whatever linear direction the
    iteration's probe found, accumulated over iterations.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    H = X.shape[1]
    P = np.eye(H, dtype=np.float32)
    X_curr = X.copy()
    curve: list[float] = []
    converged = False

    for it in range(max_iter):
        w, acc = _train_probe_get_w(X_curr, y, seed=seed + it)
        curve.append(acc)
        if acc <= chance_threshold:
            converged = True
            logger.info("INLP converged at iter %d (acc=%.3f ≤ %.3f)", it, acc, chance_threshold)
            break
        P_i = _projection_for(w)         # symmetric, P_i.T == P_i
        X_curr = X_curr @ P_i             # x' = x @ P_i  (row convention)
        P = P @ P_i                       # accumulate composed projection

    if not converged:
        logger.warning("INLP did not converge after %d iters; last acc=%.3f", max_iter, curve[-1])

    return NullspaceResult(
        projection=P.astype(np.float32),
        n_iterations=len(curve),
        accuracy_curve=curve,
        converged=converged,
        method="inlp",
    )


# ---------------------------------------------------------------------------
# LEACE
# ---------------------------------------------------------------------------


@dataclass
class LeaceResult:
    """Output of `fit_leace`."""

    projection: np.ndarray  # (H, H)
    bias: np.ndarray        # (H,) — mean-shift to apply alongside projection
    method: str = "leace"


def fit_leace(X: np.ndarray, y: np.ndarray) -> LeaceResult:
    """LEACE — Least-squares Concept Erasure (Belrose et al. 2023).

    Closed-form orthogonal projection that nullifies the cross-covariance
    between activations X and one-hot labels Y, with minimum mean-squared
    perturbation:

        P = I − Σ_X^{1/2} U Uᵀ Σ_X^{−1/2}

    where U spans the row space of Σ_X^{−1/2} Σ_XY, with Σ_X = Cov(X) and
    Σ_XY = Cov(X, Y).

    For a binary attribute with one-hot Y this reduces to a single
    rank-1 erasure direction in the whitened space. We return both the
    projection P and the centering offset b such that the application is

        x' = (x − b) @ P + b
              ↑ centring   ↑ erasure   ↑ uncentring

    so the post-projection mean is unchanged and only the cross-covariance
    is killed.
    """
    X = np.asarray(X, dtype=np.float64)  # double precision for the SVD
    y = np.asarray(y).reshape(-1)
    H = X.shape[1]

    # One-hot encode Y; for binary this is (n, 2) but we only need rank-1.
    classes = np.unique(y)
    Y = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        Y[y == c, i] = 1.0

    # Centre.
    mean_x = X.mean(axis=0)
    Xc = X - mean_x
    Yc = Y - Y.mean(axis=0)

    # Whitening: Σ_X^{−1/2} via eigendecomposition with a tiny ridge for stability.
    cov_x = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov_x + 1e-6 * np.eye(H))
    eigvals = np.clip(eigvals, 1e-10, None)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # Σ_X^{−1/2}
    W_inv = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T    # Σ_X^{1/2}

    # Cross-covariance in whitened space.
    cov_xy = (Xc.T @ Yc) / max(X.shape[0] - 1, 1)
    M = W @ cov_xy  # (H, K)

    # Orthonormal basis for the column span of M.
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((s > 1e-8).sum())
    U = U[:, :rank]  # (H, rank)

    # Erasure projection in whitened space, mapped back to the original
    # space in **row-vector convention**: applied as x_row' = x_row @ P.
    # The column-form P_col (Belrose Thm 3.1) is `W_inv (I - UU^T) W`; we
    # store its transpose so the row-vector application is consistent with
    # how INLP and the forward hook expect the matrix.
    P_whitened = np.eye(H) - U @ U.T
    P = W @ P_whitened @ W_inv  # = (W_inv (I-UU^T) W)^T  since W, W_inv symmetric
    return LeaceResult(
        projection=P.astype(np.float32),
        bias=mean_x.astype(np.float32),
        method="leace",
    )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def standardise_for_probe(X: np.ndarray) -> np.ndarray:
    """Match the StandardScaler used in `linear_probe.train_layer_probe`."""
    return StandardScaler().fit_transform(X)
