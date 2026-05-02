"""Stage 1: run logit-based bias benchmarks on every model.

Iterates over all 48 ModelSpecs × 4 benchmarks. Each model is loaded once,
all enabled benchmarks are run on it, then it is unloaded. Results are
checkpointed per (benchmark, model); a crashed run resumes from the last
completed checkpoint.

Usage:
    uv run python scripts/run_logit_benchmarks.py \\
        --config configs/models.yaml \\
        --bench-config configs/benchmarks.yaml

    # Limit to one family:
    uv run python scripts/run_logit_benchmarks.py --family olmo

    # Smoke test (one tiny model, small example limit):
    uv run python scripts/run_logit_benchmarks.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import set_seed

from biaseval.benchmarks import bbq, crows_pairs, iat, implicit_explicit, stereoset
from biaseval.benchmarks.utils import PROMPT_MODES
from biaseval.io import (
    is_completed,
    logit_result_path,
    migrate_legacy_result_paths,
    write_benchmark_result,
)
from biaseval.model_loader import ModelSpec, load_model, unload_model
from biaseval.registry import filter_specs, load_registry
from biaseval.tracking import init_run

logger = logging.getLogger(__name__)

BENCHMARK_RUNNERS = {
    "crows_pairs": crows_pairs.run,
    "stereoset": stereoset.run,
    "bbq": bbq.run,
    "iat": iat.run,
    "implicit_explicit_race": lambda m, t, s, **kw: implicit_explicit.run(
        m, t, s, attribute="race", **kw),
    "implicit_explicit_gender": lambda m, t, s, **kw: implicit_explicit.run(
        m, t, s, attribute="gender", **kw),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--bench-config", default="configs/benchmarks.yaml")
    p.add_argument("--results-root", default="results")
    p.add_argument("--family", default=None, help="Limit to one family (llama, qwen, gemma, mistral, olmo)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Limit to specific HF model IDs (overrides --family).")
    p.add_argument("--variant", default=None, choices=["base", "instruct"])
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Subset of benchmarks (default: all enabled in benchmarks.yaml)")
    p.add_argument("--prompt-modes", nargs="+", default=list(PROMPT_MODES),
                   choices=list(PROMPT_MODES),
                   help="Prompt conditions to evaluate (default: raw + instruct).")
    p.add_argument("--languages", nargs="+", default=["en"],
                   help="Languages to run CrowS-Pairs in (default: en). "
                        "Use 'fr', 'es', 'de', 'pt', 'it' for non-English. "
                        "Other benchmarks are English-only and ignore this flag.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap dataset size per benchmark (debugging)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Run only the smoke_test_model with --limit 8")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_smoke_spec(config_path: str) -> ModelSpec:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    model_id = data.get("smoke_test_model", "sshleifer/tiny-gpt2")
    return ModelSpec(
        model_id=model_id, family="smoke", generation="smoke", size="tiny",
        variant="base", num_params=1, num_layers=2, hidden_size=64,
        dtype="float32", model_class="AutoModelForCausalLM",
    )


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
        bench_cfg = yaml.safe_load(f)
    enabled = [b for b, c in bench_cfg["benchmarks"].items() if c.get("enabled", True)]
    if args.benchmarks:
        enabled = [b for b in enabled if b in args.benchmarks]
    logger.info("Enabled benchmarks: %s", enabled)

    if args.smoke:
        specs = [_load_smoke_spec(args.config)]
        limit = args.limit or 8
    else:
        specs = list(filter_specs(
            load_registry(args.config),
            family=args.family, variant=args.variant,
            only_ids=set(args.models) if args.models else None,
        ))
        limit = args.limit
    logger.info("Will evaluate %d models × %d benchmarks", len(specs), len(enabled))

    results_root = Path(args.results_root)
    migrated = migrate_legacy_result_paths(results_root)
    if migrated:
        logger.info("Migrated %d legacy result file(s) to __raw.json", migrated)

    prompt_modes: list[str] = args.prompt_modes
    languages: list[str] = args.languages
    n_done, n_skipped, n_errors = 0, 0, 0

    def _result_bench_name(runner_key: str, language: str) -> str:
        """Result-folder name; CrowS-Pairs in non-English languages gets a suffix."""
        if runner_key == "crows_pairs" and language != "en":
            return f"crows_pairs_{language}"
        return runner_key

    def _build_cells(prompt_modes, languages):
        """All (runner_key, prompt_mode, language) tuples per the user's flags.

        CrowS-Pairs gets one cell per (mode, lang); other benchmarks are
        English-only and get one cell per mode.
        """
        for runner_key in enabled:
            if runner_key == "crows_pairs":
                for lang in languages:
                    for pm in prompt_modes:
                        yield runner_key, pm, lang
            else:
                for pm in prompt_modes:
                    yield runner_key, pm, "en"

    for spec in specs:
        all_cells = list(_build_cells(prompt_modes, languages))
        # Jailbreak prompts only make sense on instruct variants — there's no
        # safety filter to bypass on a base model, and the chat template may
        # not exist. Skip the cell silently rather than emitting noise files.
        all_cells = [(rk, pm, lang) for rk, pm, lang in all_cells
                     if not (pm == "jailbreak" and spec.variant != "instruct")]
        pending = [
            (rk, pm, lang) for rk, pm, lang in all_cells
            if not is_completed(
                logit_result_path(results_root, _result_bench_name(rk, lang), spec, pm)
            )
        ]
        if not pending:
            logger.info("[skip] %s — all cells done", spec.model_id)
            n_skipped += len(all_cells)
            continue

        logger.info("[load] %s (pending: %d cells)", spec.model_id, len(pending))
        try:
            model, tokenizer = load_model(spec)
        except Exception:
            logger.error("[error] failed to load %s\n%s", spec.model_id, traceback.format_exc())
            n_errors += len(pending)
            continue

        for runner_key, pm, lang in pending:
            bench_name = _result_bench_name(runner_key, lang)
            run_ctx = init_run(
                name=f"{bench_name}/{pm}/{spec.short_name}",
                job_type=bench_name,
                tags=[spec.family, spec.variant, bench_name, f"prompt:{pm}", f"lang:{lang}"],
                config={
                    "model_id": spec.model_id, "family": spec.family,
                    "variant": spec.variant, "size": spec.size, "benchmark": bench_name,
                    "prompt_mode": pm, "language": lang,
                    "seed": args.seed, "limit": limit,
                },
                enabled=not args.no_wandb,
            )
            with run_ctx as tracker:
                try:
                    runner = BENCHMARK_RUNNERS[runner_key]
                    kwargs: dict = {"prompt_mode": pm}
                    if runner_key == "crows_pairs":
                        kwargs["language"] = lang
                    if limit is not None and runner_key != "iat":
                        kwargs["limit"] = limit
                    result = runner(model, tokenizer, spec, **kwargs)
                    write_benchmark_result(results_root, result, spec)
                    tracker.summary_update(result.summary)
                    n_done += 1
                except Exception:
                    logger.error("[error] %s/%s/%s on %s\n%s",
                                 bench_name, pm, lang, spec.model_id, traceback.format_exc())
                    n_errors += 1

        unload_model(model)
        del tokenizer

    logger.info("Done. completed=%d skipped=%d errors=%d", n_done, n_skipped, n_errors)
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
