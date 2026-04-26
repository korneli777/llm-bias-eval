"""Atomic JSON results IO + checkpoint discovery.

Results live in nested directories per benchmark + model:
    results/logit_scores/{benchmark}/{model_short_name}.json
    results/probe_results/{model_short_name}/{attribute}.json

Each file is written atomically (tmp + rename) so partial writes don't
get mistaken for completed runs by the resume logic.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from biaseval.benchmarks.utils import BenchmarkResult
from biaseval.model_loader import ModelSpec

logger = logging.getLogger(__name__)


def _runtime_metadata() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def logit_result_path(
    results_root: Path, benchmark: str, spec: ModelSpec, prompt_mode: str = "raw"
) -> Path:
    return Path(results_root) / "logit_scores" / benchmark / f"{spec.short_name}__{prompt_mode}.json"


def migrate_legacy_result_paths(results_root: Path) -> int:
    """Rename `{model}.json` → `{model}__raw.json` under logit_scores/.

    Idempotent — leaves already-suffixed files alone. Returns the number of
    files renamed. Run-once helper for results produced before prompt_mode existed.
    """
    n = 0
    base = Path(results_root) / "logit_scores"
    if not base.exists():
        return 0
    for path in base.glob("*/*.json"):
        if "__" in path.stem:
            continue
        new_path = path.with_name(f"{path.stem}__raw.json")
        if new_path.exists():
            continue
        path.rename(new_path)
        n += 1
    return n


def probe_result_path(results_root: Path, spec: ModelSpec, attribute: str) -> Path:
    return Path(results_root) / "probe_results" / spec.short_name / f"{attribute}.json"


def activation_dir(results_root: Path, spec: ModelSpec) -> Path:
    return Path(results_root) / "activations" / spec.short_name


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic write — JSON to .tmp, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def write_benchmark_result(
    results_root: Path, result: BenchmarkResult, spec: ModelSpec
) -> Path:
    path = logit_result_path(results_root, result.benchmark, spec, result.prompt_mode)
    payload = {
        "spec": {
            "model_id": spec.model_id,
            "family": spec.family,
            "generation": spec.generation,
            "size": spec.size,
            "variant": spec.variant,
            "num_params": spec.num_params,
            "num_layers": spec.num_layers,
            "hidden_size": spec.hidden_size,
        },
        "result": result.to_dict(),
        "runtime": _runtime_metadata(),
    }
    write_json(path, payload)
    logger.info("Wrote %s", path)
    return path


def write_probe_result(
    results_root: Path, spec: ModelSpec, attribute: str, layer_results: list[dict]
) -> Path:
    path = probe_result_path(results_root, spec, attribute)
    payload = {
        "spec": {
            "model_id": spec.model_id,
            "family": spec.family,
            "generation": spec.generation,
            "size": spec.size,
            "variant": spec.variant,
            "num_layers": spec.num_layers,
            "hidden_size": spec.hidden_size,
        },
        "attribute": attribute,
        "layers": layer_results,
        "runtime": _runtime_metadata(),
    }
    write_json(path, payload)
    logger.info("Wrote %s", path)
    return path


def is_completed(path: Path) -> bool:
    """Result file exists and parses as valid JSON."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, json.JSONDecodeError):
        return False
