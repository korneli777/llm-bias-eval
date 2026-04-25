"""Stage 2: extract residual-stream activations and train layer-wise probes.

For each model in the probing subset:
  1. Build labeled probe datasets (one per attribute: gender, race).
  2. Extract activations (one .npy per layer in results/activations/<model>/).
  3. Train a logistic-regression probe at every layer with 5-fold CV.
  4. Write per-layer accuracies to results/probe_results/<model>/<attr>.json.

Resumes per (model, attribute).
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from transformers import set_seed

from biaseval.io import (
    activation_dir,
    is_completed,
    probe_result_path,
    write_probe_result,
)
from biaseval.model_loader import load_model, unload_model
from biaseval.probing.datasets import build_probe_dataset
from biaseval.probing.extract_activations import extract_activations
from biaseval.probing.linear_probe import train_probes_all_layers
from biaseval.registry import filter_specs, get_probing_subset, load_registry
from biaseval.tracking import init_run

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--bench-config", default="configs/benchmarks.yaml")
    p.add_argument("--results-root", default="results")
    p.add_argument("--family", default=None)
    p.add_argument("--variant", default=None, choices=["base", "instruct"])
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.bench_config) as f:
        bcfg = yaml.safe_load(f)
    probe_cfg = bcfg.get("probing", {})
    attributes: list[str] = probe_cfg.get("attributes", ["gender", "race"])
    pool: str = probe_cfg.get("pool", "mean")
    cv_folds: int = int(probe_cfg.get("cv_folds", 5))
    max_per_class: int | None = probe_cfg.get("max_sentences")

    probing_ids = get_probing_subset(args.config)
    specs = list(filter_specs(load_registry(args.config),
                              family=args.family, variant=args.variant,
                              only_ids=probing_ids))
    logger.info("Will probe %d models × %d attributes", len(specs), len(attributes))

    results_root = Path(args.results_root)
    n_done, n_skipped, n_errors = 0, 0, 0

    # Pre-build the probe datasets (small, deterministic, model-independent).
    probe_datasets = {attr: build_probe_dataset(attr, max_per_class=max_per_class)
                      for attr in attributes}

    for spec in specs:
        pending = [a for a in attributes
                   if not is_completed(probe_result_path(results_root, spec, a))]
        if not pending:
            logger.info("[skip] %s — all attributes done", spec.model_id)
            n_skipped += len(attributes)
            continue

        # Activations are model-specific but attribute-agnostic — extract once.
        # We extract a superset: union of all attribute datasets, then index.
        all_sentences: list[str] = []
        attr_indices: dict[str, list[int]] = {}
        for attr in pending:
            ds = probe_datasets[attr]
            start = len(all_sentences)
            all_sentences.extend(ds.sentences)
            attr_indices[attr] = list(range(start, start + len(ds.sentences)))

        logger.info("[load] %s — extracting %d sentences", spec.model_id, len(all_sentences))
        try:
            model, tokenizer = load_model(spec)
        except Exception:
            logger.error("[error] failed to load %s\n%s", spec.model_id, traceback.format_exc())
            n_errors += len(pending)
            continue

        # Smaller batch for ≥27B models.
        batch_size = 1 if spec.num_params >= 27_000_000_000 else 4
        adir = activation_dir(results_root, spec)
        try:
            num_layers = extract_activations(
                model, tokenizer, all_sentences, adir,
                pool=pool, batch_size=batch_size,
            )
        except Exception:
            logger.error("[error] activation extraction failed for %s\n%s",
                         spec.model_id, traceback.format_exc())
            unload_model(model)
            n_errors += len(pending)
            continue

        # Free model before scikit-learn work.
        unload_model(model)
        del tokenizer

        for attr in pending:
            run_ctx = init_run(
                name=f"probe/{spec.short_name}/{attr}",
                job_type="probe",
                tags=[spec.family, spec.variant, "probe", attr],
                config={
                    "model_id": spec.model_id, "family": spec.family,
                    "variant": spec.variant, "size": spec.size,
                    "attribute": attr, "pool": pool, "cv_folds": cv_folds,
                    "seed": args.seed,
                },
                enabled=not args.no_wandb,
            )
            with run_ctx as tracker:
                try:
                    ds = probe_datasets[attr]
                    labels = np.array(ds.labels)
                    indices = attr_indices[attr]

                    # Slice the per-layer activations down to this attribute's sentences.
                    sliced_dir = adir / f"_{attr}"
                    sliced_dir.mkdir(exist_ok=True)
                    for li in range(num_layers):
                        full = np.load(adir / f"layer_{li}.npy")
                        np.save(sliced_dir / f"layer_{li}.npy", full[indices])
                        del full

                    layer_results = train_probes_all_layers(
                        sliced_dir, labels, num_layers, attr, cv_folds=cv_folds,
                        seed=args.seed,
                    )
                    write_probe_result(results_root, spec, attr, layer_results)
                    peak = max(r["mean_accuracy"] for r in layer_results)
                    tracker.summary_update({"peak_accuracy": peak,
                                            "n_sentences": len(labels)})
                    n_done += 1
                except Exception:
                    logger.error("[error] probe %s on %s\n%s", attr, spec.model_id,
                                 traceback.format_exc())
                    n_errors += 1

    logger.info("Done. completed=%d skipped=%d errors=%d", n_done, n_skipped, n_errors)
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
