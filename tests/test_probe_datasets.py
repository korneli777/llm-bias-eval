"""Tests for the neutral-prompt probe datasets (Step 1, Bouchouchi-style)."""

from __future__ import annotations

import numpy as np

from biaseval.probing.datasets import (
    AFRICAN_AMERICAN_NAMES,
    EUROPEAN_NAMES,
    NAME_TEMPLATE,
    OCCUPATION_TEMPLATE,
    WINOBIAS_OCCUPATIONS_FEMALE,
    WINOBIAS_OCCUPATIONS_MALE,
    build_probe_dataset,
)
from biaseval.probing.linear_probe import mean_difference_direction


def test_gender_probe_uses_neutral_template_and_balances_classes():
    ds = build_probe_dataset("gender")
    assert len(ds) == len(WINOBIAS_OCCUPATIONS_MALE) + len(WINOBIAS_OCCUPATIONS_FEMALE)
    assert ds.labels.count(0) == ds.labels.count(1)
    assert all(s.startswith("The ") and "said that" in s for s in ds.sentences)
    # No explicit gender word — that's the whole point.
    forbidden = {"man", "woman", "he", "she", "his", "her", "male", "female"}
    for s in ds.sentences:
        toks = set(s.lower().replace(".", "").split())
        assert not (toks & forbidden), f"leaked gender keyword in: {s}"


def test_race_probe_uses_name_template_and_balances_classes():
    ds = build_probe_dataset("race")
    assert len(ds) == len(EUROPEAN_NAMES) + len(AFRICAN_AMERICAN_NAMES)
    assert ds.labels.count(0) == ds.labels.count(1)
    assert all("walked into the room" in s for s in ds.sentences)
    # The race words themselves should never appear in the prompt.
    forbidden = {"black", "white", "african", "european", "caucasian"}
    for s in ds.sentences:
        toks = set(s.lower().split())
        assert not (toks & forbidden), f"leaked race keyword in: {s}"


def test_max_per_class_caps_balanced():
    ds = build_probe_dataset("gender", max_per_class=5)
    assert ds.labels.count(0) == 5
    assert ds.labels.count(1) == 5


def test_template_constants_are_simple_strings():
    assert OCCUPATION_TEMPLATE == "The {occupation} said that"
    assert NAME_TEMPLATE == "{name} walked into the room"


def test_mean_difference_direction_is_unit_norm():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (50, 16)), rng.normal(2, 1, (50, 16))])
    y = np.array([0] * 50 + [1] * 50)
    d = mean_difference_direction(X, y)
    assert d.shape == (16,)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-5
    # Should point roughly along the +x direction (class-1 mean shifted by +2 in all dims).
    assert (d > 0).all()


def test_mean_difference_direction_handles_single_class():
    X = np.zeros((10, 8))
    y = np.zeros(10, dtype=int)
    d = mean_difference_direction(X, y)
    assert (d == 0).all()
