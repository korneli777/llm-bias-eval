"""Tests for the probe-direction aggregator (Bite 3)."""

from __future__ import annotations

import numpy as np

from biaseval.analysis.aggregate_results import (
    cross_pair_direction_cosines,
    load_probe_directions,
)


def _write_direction(tmp_path, model_id, attr, vectors):
    short = model_id.replace("/", "__")
    out = tmp_path / "activations" / short
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"direction_{attr}.npy", vectors.astype(np.float32))


def test_load_probe_directions_picks_up_npy(tmp_path):
    rng = np.random.default_rng(0)
    _write_direction(tmp_path, "x/y-base", "gender", rng.normal(size=(4, 8)))
    _write_direction(tmp_path, "x/y-instruct", "gender", rng.normal(size=(4, 8)))
    out = load_probe_directions(tmp_path)
    assert ("x__y-base", "gender") in out
    assert ("x__y-instruct", "gender") in out
    assert out[("x__y-base", "gender")].shape == (4, 8)


def test_cross_pair_direction_cosines_unit_self_similarity(tmp_path):
    """Identical base/instruct directions → cosine == 1 at every layer."""
    vecs = np.eye(4, 8)  # 4 unit vectors of dim 8
    _write_direction(tmp_path, "x/y-base", "gender", vecs)
    _write_direction(tmp_path, "x/y-instruct", "gender", vecs)
    pairs = [("x/y-base", "x/y-instruct", "fam", "G", "7B")]
    df = cross_pair_direction_cosines(tmp_path, pairs, attributes=("gender",))
    assert len(df) == 4
    assert np.allclose(df["cosine"], 1.0)
    # depth_frac runs 0 → 1 across the 4 layers.
    assert df["depth_frac"].iloc[0] == 0.0
    assert df["depth_frac"].iloc[-1] == 1.0


def test_cross_pair_direction_cosines_orthogonal_drops_to_zero(tmp_path):
    """Orthogonal directions → cosine == 0."""
    base_vecs = np.array([[1, 0], [1, 0]], dtype=np.float32)
    inst_vecs = np.array([[0, 1], [0, 1]], dtype=np.float32)
    _write_direction(tmp_path, "x/b", "gender", base_vecs)
    _write_direction(tmp_path, "x/i", "gender", inst_vecs)
    df = cross_pair_direction_cosines(
        tmp_path, [("x/b", "x/i", "fam", "G", "1B")], attributes=("gender",),
    )
    assert np.allclose(df["cosine"], 0.0)


def test_cross_pair_direction_cosines_skips_missing_files(tmp_path):
    """A pair with no direction files → empty DataFrame, no crash."""
    df = cross_pair_direction_cosines(
        tmp_path, [("x/b", "x/i", "fam", "G", "1B")],
    )
    assert df.empty
