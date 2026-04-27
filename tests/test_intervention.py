"""Tests for INLP / LEACE erasure + the projection forward-hook."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from biaseval.intervention.inlp import (
    fit_inlp,
    fit_leace,
)
from biaseval.intervention.sanity import lm_perplexity, perplexity_check, verify_nullification

# ---------------------------------------------------------------------------
# Synthetic data: a 32-dim Gaussian with one coordinate carrying the label
# ---------------------------------------------------------------------------


def _synth(n: int = 400, dim: int = 32, signal_strength: float = 2.5, seed: int = 0):
    """Two Gaussian blobs separated only along axis 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = rng.integers(0, 2, n)
    X[:, 0] += signal_strength * (2 * y - 1)  # ±signal on axis 0
    return X, y


def _probe_acc(X: np.ndarray, y: np.ndarray, seed: int = 1) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())


# ---------------------------------------------------------------------------
# INLP
# ---------------------------------------------------------------------------


def test_inlp_kills_separability():
    X, y = _synth()
    base_acc = _probe_acc(X, y)
    assert base_acc > 0.9, f"setup broken — base separability only {base_acc}"

    res = fit_inlp(X, y, max_iter=10, chance_threshold=0.55)
    X_proj = X @ res.projection
    post_acc = _probe_acc(X_proj, y)
    assert post_acc <= 0.6, f"INLP did not erase: post_acc={post_acc}"
    assert res.converged, f"INLP should converge on this easy dataset; curve={res.accuracy_curve}"
    # Projection must be idempotent: P @ P ≈ P (orthogonal).
    P2 = res.projection @ res.projection
    assert np.allclose(P2, res.projection, atol=1e-3)


def test_inlp_reports_accuracy_curve():
    X, y = _synth()
    res = fit_inlp(X, y, max_iter=5)
    assert len(res.accuracy_curve) == res.n_iterations
    # Monotone non-increasing in expectation (small jitter possible across iters).
    assert res.accuracy_curve[0] >= res.accuracy_curve[-1] - 0.05


# ---------------------------------------------------------------------------
# LEACE
# ---------------------------------------------------------------------------


def test_leace_kills_separability():
    X, y = _synth()
    res = fit_leace(X, y)
    Xc = X - res.bias
    X_proj = Xc @ res.projection + res.bias
    post_acc = _probe_acc(X_proj, y)
    assert post_acc <= 0.6, f"LEACE did not erase: post_acc={post_acc}"


def test_leace_minimal_perturbation_vs_inlp():
    """LEACE should perturb less than INLP on a clean rank-1 problem."""
    X, y = _synth()
    inlp = fit_inlp(X, y, max_iter=10)
    leace = fit_leace(X, y)
    inlp_diff = np.linalg.norm(X - X @ inlp.projection)
    leace_diff = np.linalg.norm(X - ((X - leace.bias) @ leace.projection + leace.bias))
    assert leace_diff <= inlp_diff * 1.1, \
        f"LEACE perturbation {leace_diff:.2f} should not exceed INLP {inlp_diff:.2f}"


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def test_verify_nullification_passes_after_erasure():
    X, y = _synth()
    res = fit_inlp(X, y, max_iter=10)
    out = verify_nullification(X, y, res.projection, chance_threshold=0.6)
    assert out["passed"]
    assert out["post_intervention_probe_accuracy"] <= 0.6


def test_verify_nullification_fails_with_identity():
    X, y = _synth()
    out = verify_nullification(X, y, np.eye(X.shape[1], dtype=np.float32), chance_threshold=0.6)
    assert not out["passed"]


# ---------------------------------------------------------------------------
# Forward-hook integration on a real (tiny) HF model
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_projection_hook_round_trip_with_identity():
    """Identity projection should be a no-op; outputs must match exactly."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from biaseval.intervention.hooks import ProjectionHook

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.eval()

    H = model.config.hidden_size
    P = np.eye(H, dtype=np.float32)
    enc = tok("hello world", return_tensors="pt")
    with torch.no_grad():
        ref_logits = model(**enc).logits
        with ProjectionHook(model, P, layer_idx=0):
            hooked_logits = model(**enc).logits
    assert torch.allclose(ref_logits, hooked_logits, atol=1e-4)


@pytest.mark.integration
def test_projection_hook_changes_output_when_nontrivial():
    """A non-identity projection must produce different logits."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from biaseval.intervention.hooks import ProjectionHook

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.eval()

    H = model.config.hidden_size
    rng = np.random.default_rng(0)
    w = rng.standard_normal(H).astype(np.float32)
    P = np.eye(H, dtype=np.float32) - np.outer(w, w) / float(w @ w)
    enc = tok("hello world", return_tensors="pt")
    with torch.no_grad():
        ref = model(**enc).logits
        with ProjectionHook(model, P, layer_idx=0):
            hooked = model(**enc).logits
    assert not torch.allclose(ref, hooked, atol=1e-3), "non-identity projection should perturb logits"


@pytest.mark.integration
def test_perplexity_check_handles_identity_projection():
    """Identity projection must give ratio ≈ 1.0 (passed=True)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.eval()

    P = np.eye(model.config.hidden_size, dtype=np.float32)
    out = perplexity_check(model, tok, P, layer_idx=0, blowup_factor=1.5,
                            texts=["The cat sat on the mat.",
                                   "The sun rose over the hill."])
    assert out["passed"]
    assert abs(out["ratio"] - 1.0) < 0.01


@pytest.mark.integration
def test_lm_perplexity_returns_finite_positive():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.eval()
    ppl = lm_perplexity(model, tok, ["The capital of France is Paris."])
    assert ppl > 0 and np.isfinite(ppl)
