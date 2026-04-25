"""Parse configs/models.yaml into ModelSpec objects."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

from biaseval.model_loader import ModelSpec


def load_registry(config_path: str | Path) -> list[ModelSpec]:
    """Flatten the YAML registry into one ModelSpec per (checkpoint, variant).

    Each (size, generation, family) entry in the YAML defines a base ↔ instruct
    pair, so it produces two ModelSpecs.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    specs: list[ModelSpec] = []
    for family_name, family in data["families"].items():
        requires_auth = bool(family.get("requires_hf_auth", False))
        for gen in family["generations"]:
            for entry in gen["models"]:
                shared = {
                    "family": family_name,
                    "generation": gen["name"],
                    "size": entry["size"],
                    "num_params": entry["num_params"],
                    "num_layers": entry["num_layers"],
                    "hidden_size": entry["hidden_size"],
                    "dtype": entry.get("dtype", "bfloat16"),
                    "model_class": entry.get("model_class", "AutoModelForCausalLM"),
                    "requires_hf_auth": requires_auth,
                    "notes": entry.get("notes"),
                }
                specs.append(ModelSpec(model_id=entry["base_id"], variant="base", **shared))
                specs.append(ModelSpec(model_id=entry["instruct_id"], variant="instruct", **shared))
    return specs


def get_probing_subset(config_path: str | Path) -> set[str]:
    """Set of HF model IDs to run probing on."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return set(data.get("probing_subset", []))


def filter_specs(
    specs: list[ModelSpec],
    family: str | None = None,
    variant: str | None = None,
    only_ids: set[str] | None = None,
) -> Iterator[ModelSpec]:
    """Iterate specs matching the given filters. Each filter is optional."""
    for spec in specs:
        if family and spec.family != family:
            continue
        if variant and spec.variant != variant:
            continue
        if only_ids and spec.model_id not in only_ids:
            continue
        yield spec
