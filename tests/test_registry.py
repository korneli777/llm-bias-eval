"""Validate the YAML registry parses to the expected structure."""

from __future__ import annotations

from biaseval.registry import filter_specs, get_probing_subset, load_registry

CONFIG = "configs/models.yaml"


def test_registry_size():
    specs = load_registry(CONFIG)
    assert len(specs) == 60, "expected 30 base+instruct pairs (60 specs)"


def test_registry_pairs():
    specs = load_registry(CONFIG)
    base = [s for s in specs if s.variant == "base"]
    inst = [s for s in specs if s.variant == "instruct"]
    assert len(base) == len(inst) == 30


def test_families_present():
    specs = load_registry(CONFIG)
    families = {s.family for s in specs}
    assert families == {"llama", "qwen", "gemma", "mistral"}


def test_probing_subset_resolvable():
    specs = load_registry(CONFIG)
    by_id = {s.model_id for s in specs}
    subset = get_probing_subset(CONFIG)
    assert subset, "probing_subset is empty"
    missing = subset - by_id
    assert not missing, f"probing_subset references unknown ids: {missing}"


def test_filter_by_family():
    specs = load_registry(CONFIG)
    mistral = list(filter_specs(specs, family="mistral"))
    # 5 generations × (base + instruct), one entry each:
    # v0.1 7B, v0.2 7B, v0.3 7B, Nemo 12B, Small 3 24B
    assert len(mistral) == 10
    assert all(s.family == "mistral" for s in mistral)


def test_gemma3_uses_special_class():
    """Gemma 3 needs Gemma3ForCausalLM for text-only loading."""
    specs = load_registry(CONFIG)
    g3 = [s for s in specs if "gemma-3" in s.model_id]
    assert g3 and all(s.model_class == "Gemma3ForCausalLM" for s in g3)
    # Gemma 4 is intentionally excluded — see configs/models.yaml header.
    g4 = [s for s in specs if "gemma-4" in s.model_id]
    assert not g4, "Gemma 4 should be excluded from the registry"


def test_gated_flag_set():
    specs = load_registry(CONFIG)
    for s in specs:
        if s.family in {"llama", "gemma"}:
            assert s.requires_hf_auth, f"{s.model_id} should require HF auth"
        else:
            assert not s.requires_hf_auth, f"{s.model_id} should NOT require HF auth"


def test_iat_stimuli_loader():
    """Bai et al. CSV parses to the test-dict shape iat.run() expects."""
    import pytest

    from biaseval.data import load_iat_stimuli
    try:
        tests = load_iat_stimuli()
    except Exception as e:
        pytest.skip(f"network unavailable: {e}")
    assert len(tests) >= 15, "expected >=15 IAT subtests in the Bai release"
    cats = {t["category"] for t in tests}
    assert cats == {"race", "gender", "religion", "age", "health"}
    for t in tests:
        for key in ("category", "subcategory", "target_a", "target_b", "attr_a", "attr_b"):
            assert key in t, f"missing {key} in {t}"
        assert t["target_a"]["stimuli"] and t["target_b"]["stimuli"]
        assert len(t["attr_a"]["stimuli"]) == len(t["attr_b"]["stimuli"])
        assert len(t["attr_a"]["stimuli"]) >= 2
