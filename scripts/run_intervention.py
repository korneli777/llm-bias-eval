"""Stage 3.5: causal intervention via linear concept erasure.

For each model in the probing subset:
  1. Read the probe-accuracy JSON to find the peak layer per attribute.
  2. Load the cached activations from `results/activations/<model>/`.
  3. Fit INLP and LEACE projection matrices, save to disk.
  4. Run sanity checks (probe nullification + LM perplexity).
  5. For each (benchmark, prompt_mode, method): re-run the benchmark with a
     ProjectionHook attached at the peak layer. Skip if the JSON exists.

Resumable per (model, attribute, benchmark, prompt_mode, method) cell.

Usage:
    uv run python scripts/run_intervention.py
    uv run python scripts/run_intervention.py --family llama
    uv run python scripts/run_intervention.py --models meta-llama/Llama-3.1-8B \\
        --attributes gender --benchmarks crows_pairs --methods inlp --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import set_seed

from biaseval.benchmarks import crows_pairs, stereoset
from biaseval.intervention import (
    ProjectionHook,
    fit_inlp,
    fit_leace,
    verify_nullification,
)
from biaseval.intervention.sanity import perplexity_check
from biaseval.io import (
    activation_dir,
    intervention_result_path,
    is_completed,
    projection_path,
    write_intervention_result,
)
from biaseval.model_loader import load_model, unload_model
from biaseval.probing.datasets import build_probe_dataset
from biaseval.registry import filter_specs, get_probing_subset, load_registry
from biaseval.tracking import init_run

logger = logging.getLogger(__name__)

BENCHMARK_RUNNERS = {
    "crows_pairs": crows_pairs.run,
    "stereoset": stereoset.run,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--bench-config", default="configs/benchmarks.yaml")
    p.add_argument("--results-root", default="results")
    p.add_argument("--family", default=None,
                   help="Limit to one family (llama, qwen, gemma, mistral)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Limit to specific HF model IDs (overrides --family)")
    p.add_argument("--attributes", nargs="+", default=["gender", "race"],
                   choices=["gender", "race"])
    p.add_argument("--benchmarks", nargs="+", default=["crows_pairs", "stereoset"],
                   choices=list(BENCHMARK_RUNNERS))
    p.add_argument("--prompt-modes", nargs="+", default=["raw", "instruct"],
                   choices=["raw", "instruct"])
    p.add_argument("--methods", nargs="+", default=["inlp", "leace"],
                   choices=["inlp", "leace"])
    p.add_argument("--layer-depths", nargs="+", type=float,
                   default=[0.10, 0.30, 0.50, 0.70, 0.90],
                   help="Fractions of total depth to intervene at "
                        "(e.g. 0.5 = middle layer). Default: 0.1, 0.3, 0.5, 0.7, 0.9")
    p.add_argument("--max-iter", type=int, default=10,
                   help="Max INLP iterations")
    p.add_argument("--chance-threshold", type=float, default=0.55,
                   help="INLP stops when probe accuracy ≤ this")
    p.add_argument("--blowup-factor", type=float, default=1.5,
                   help="Perplexity sanity check passes if post/pre ≤ this")
    p.add_argument("--validate-only", action="store_true",
                   help="Fit projections + run sanity checks, skip benchmark re-runs")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _depths_to_layer_indices(depths: list[float], num_layers: int) -> list[int]:
    """Map depth fractions (e.g. 0.5) to integer layer indices for this model."""
    return sorted({max(0, min(num_layers - 1, round(d * (num_layers - 1)))) for d in depths})


def _load_or_fit_projection(
    spec, results_root: Path, attribute: str, method: str, layer_idx: int,
    activations: np.ndarray, labels: np.ndarray,
    *, max_iter: int, chance_threshold: float,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Return (projection, bias, metadata). Cached per (model, attr, method, layer)."""
    proj_path = projection_path(results_root, spec, attribute, method, layer_idx=layer_idx)
    if proj_path.exists():
        with np.load(proj_path) as npz:
            P = npz["projection"].astype(np.float32)
            bias = npz["bias"].astype(np.float32) if "bias" in npz.files and npz["bias"].size else None
            meta = json.loads(str(npz["metadata"])) if "metadata" in npz.files else {}
        return P, bias, meta

    if method == "inlp":
        res = fit_inlp(activations, labels, max_iter=max_iter,
                       chance_threshold=chance_threshold, seed=42)
        P, bias = res.projection, None
        meta = {"method": "inlp", "layer_idx": layer_idx,
                "n_iterations": res.n_iterations,
                "accuracy_curve": res.accuracy_curve, "converged": res.converged}
    elif method == "leace":
        res = fit_leace(activations, labels)
        P, bias = res.projection, res.bias
        meta = {"method": "leace", "layer_idx": layer_idx}
    else:
        raise ValueError(f"Unknown method: {method}")

    proj_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        proj_path,
        projection=P,
        bias=bias if bias is not None else np.array([], dtype=np.float32),
        metadata=np.array(json.dumps(meta), dtype=object),
    )
    return P, bias, meta


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # Filter to probing-subset models, optionally further by family/IDs.
    probing_ids = get_probing_subset(args.config)
    specs = list(filter_specs(
        load_registry(args.config), family=args.family,
        only_ids=set(args.models) if args.models else probing_ids,
    ))
    logger.info("Will intervene on %d models × %d attributes × %d benchmarks × "
                "%d prompt_modes × %d methods",
                len(specs), len(args.attributes), len(args.benchmarks),
                len(args.prompt_modes), len(args.methods))

    results_root = Path(args.results_root)

    # Pre-build the probe datasets once.
    probe_datasets = {a: build_probe_dataset(a) for a in args.attributes}

    n_done, n_skipped, n_errors = 0, 0, 0

    for spec in specs:
        layer_idxs = _depths_to_layer_indices(args.layer_depths, spec.num_layers)
        logger.info("[%s] num_layers=%d, intervention layers=%s (depths=%s)",
                    spec.short_name, spec.num_layers, layer_idxs, args.layer_depths)

        # Pending = (attr, layer, bench, prompt_mode, method) cells whose
        # output JSON doesn't already exist.
        pending: list[tuple[str, int, str, str, str]] = [
            (attr, L, bench, pm, method)
            for attr in args.attributes
            for L in layer_idxs
            for bench in args.benchmarks
            for pm in args.prompt_modes
            for method in args.methods
            if not is_completed(intervention_result_path(
                results_root, bench, spec,
                attribute=attr, prompt_mode=pm, method=method, layer_idx=L,
            ))
        ]

        if not pending and not args.validate_only:
            logger.info("[skip] %s — all intervention cells done", spec.model_id)
            n_skipped += 1
            continue

        adir = activation_dir(results_root, spec)

        # Fit / load every (attr, layer, method) projection.
        projections: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray | None, dict]] = {}
        sanity_cpu: dict[tuple[str, int, str], dict] = {}
        for attr in args.attributes:
            ds = probe_datasets[attr]
            labels = np.array(ds.labels)
            for L in layer_idxs:
                # Activations: prefer per-attribute sliced dir, fall back to full.
                sliced_dir = adir / f"_{attr}"
                layer_path = sliced_dir / f"layer_{L}.npy"
                if not layer_path.exists():
                    layer_path = adir / f"layer_{L}.npy"
                if not layer_path.exists():
                    logger.error("[error] missing activations at %s", layer_path)
                    n_errors += 1
                    continue
                X = np.load(layer_path)
                if X.shape[0] != labels.shape[0]:
                    logger.error("[error] activation/label size mismatch %s vs %s for %s/%s/L%d",
                                 X.shape, labels.shape, spec.model_id, attr, L)
                    n_errors += 1
                    continue
                for method in args.methods:
                    P, bias, meta = _load_or_fit_projection(
                        spec, results_root, attr, method, L, X, labels,
                        max_iter=args.max_iter, chance_threshold=args.chance_threshold,
                    )
                    projections[(attr, L, method)] = (P, bias, meta)
                    null = verify_nullification(
                        X, labels, P, bias=bias,
                        chance_threshold=args.chance_threshold, seed=args.seed,
                    )
                    sanity_cpu[(attr, L, method)] = {
                        "fit": meta, "nullification": null, "layer_idx": L,
                    }

        if not projections:
            n_errors += 1
            continue

        if not pending:
            n_skipped += 1
            continue

        logger.info("[load] %s (%d pending cells)", spec.model_id, len(pending))
        try:
            model, tokenizer = load_model(spec)
        except Exception:
            logger.error("[error] failed to load %s\n%s", spec.model_id, traceback.format_exc())
            n_errors += 1
            continue

        # GPU sanity: perplexity blow-up per (attr, layer, method).
        for (attr, L, method), info in sanity_cpu.items():
            try:
                P, bias, _ = projections[(attr, L, method)]
                ppl = perplexity_check(
                    model, tokenizer, P, L, bias=bias,
                    blowup_factor=args.blowup_factor,
                )
                info["perplexity"] = ppl
                if not ppl["passed"]:
                    logger.warning(
                        "[ppl-blowup] %s/%s/L%d/%s: pre=%.2f → post=%.2f (×%.2f)",
                        spec.model_id, attr, L, method,
                        ppl["perplexity_pre"], ppl["perplexity_post"], ppl["ratio"],
                    )
            except Exception:
                logger.error("[error] perplexity check failed for %s/%s/L%d/%s\n%s",
                             spec.model_id, attr, L, method, traceback.format_exc())

        if args.validate_only:
            for (attr, L, method), info in sanity_cpu.items():
                report_path = (
                    results_root / "intervention" / spec.short_name
                    / f"{attr}__{method}__L{L:02d}__sanity.json"
                )
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(report_path, "w") as f:
                    json.dump(info, f, indent=2, default=str)
            unload_model(model)
            del tokenizer
            n_done += 1
            continue

        # Real benchmark sweep.
        for attr, L, bench, pm, method in pending:
            key = (attr, L, method)
            if key not in projections:
                continue
            P, bias, _ = projections[key]
            depth_frac = L / max(spec.num_layers - 1, 1)
            run_ctx = init_run(
                name=f"intervene/{bench}/{method}/{attr}/L{L:02d}/{pm}/{spec.short_name}",
                job_type=f"intervene_{bench}",
                tags=[spec.family, spec.variant, bench, f"prompt:{pm}",
                      f"attr:{attr}", f"method:{method}", f"depth:{depth_frac:.2f}"],
                config={
                    "model_id": spec.model_id, "family": spec.family,
                    "variant": spec.variant, "size": spec.size,
                    "benchmark": bench, "prompt_mode": pm,
                    "attribute": attr, "method": method,
                    "layer_idx": L, "depth_frac": depth_frac,
                    "num_layers": spec.num_layers, "seed": args.seed,
                },
                enabled=not args.no_wandb,
            )
            with run_ctx as tracker:
                try:
                    runner = BENCHMARK_RUNNERS[bench]
                    with ProjectionHook(model, P, L, bias=bias):
                        result = runner(model, tokenizer, spec, prompt_mode=pm)
                    write_intervention_result(
                        results_root, result, spec,
                        attribute=attr, method=method, layer_idx=L,
                        sanity=sanity_cpu[key],
                    )
                    tracker.summary_update(result.summary)
                    n_done += 1
                except Exception:
                    logger.error("[error] %s/%s/L%d/%s/%s on %s\n%s",
                                 bench, pm, L, attr, method, spec.model_id,
                                 traceback.format_exc())
                    n_errors += 1

        unload_model(model)
        del tokenizer

    logger.info("Done. completed=%d skipped=%d errors=%d", n_done, n_skipped, n_errors)
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
