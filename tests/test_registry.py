"""Validate the YAML registry parses to the expected structure."""

from __future__ import annotations

from biaseval.registry import filter_specs, get_probing_subset, load_registry

CONFIG = "configs/models.yaml"


def test_registry_size():
    specs = load_registry(CONFIG)
    assert len(specs) == 44, "expected 22 base+instruct pairs (44 specs)"


def test_registry_pairs():
    specs = load_registry(CONFIG)
    base = [s for s in specs if s.variant == "base"]
    inst = [s for s in specs if s.variant == "instruct"]
    assert len(base) == len(inst) == 22


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
    assert len(mistral) == 4  # 2 sizes × (base + instruct)
    assert all(s.family == "mistral" for s in mistral)


def test_gemma3_4_use_special_class():
    specs = load_registry(CONFIG)
    g3 = [s for s in specs if "gemma-3" in s.model_id]
    g4 = [s for s in specs if "gemma-4" in s.model_id]
    assert g3 and all(s.model_class == "Gemma3ForCausalLM" for s in g3)
    assert g4 and all(s.model_class == "Gemma4ForCausalLM" for s in g4)


def test_gated_flag_set():
    specs = load_registry(CONFIG)
    for s in specs:
        if s.family in {"llama", "gemma"}:
            assert s.requires_hf_auth, f"{s.model_id} should require HF auth"
        else:
            assert not s.requires_hf_auth, f"{s.model_id} should NOT require HF auth"
