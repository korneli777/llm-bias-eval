"""End-to-end MVP validation: full pipeline on 3 small pairs before the big sweep.

Goal — catch broken pipelines, bad tokenizer interactions, and missing-field
bugs *before* committing 40+ GPU-hours to the production sweep. Three families
(Llama / Qwen / Mistral) each test a different code path:
    - Llama-3.2-3B          → baseline tokenizer + chat template
    - Qwen/Qwen2.5-3B       → different tokenizer family
    - mistralai/Mistral-7B  → mistral chat template + the prompt-conditional
                              alignment finding we want to re-check

Usage
-----
    uv run python scripts/run_mvp_validation.py
    # → writes results/mvp/*.json (production schema), figures/mvp/*.png,
    #   and results/mvp/report.md with hard-fail / soft-observe split.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import traceback
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import set_seed

from biaseval.analysis.aggregate_results import (
    aggregate_logit_results,
    aggregate_probe_results,
    cross_pair_direction_cosines,
)
from biaseval.analysis.plotting import generate_all
from biaseval.analysis.statistics import (
    cross_benchmark_consistency,
    cross_language_consistency,
    pair_significance_per_language,
    pair_significance_table,
)
from biaseval.benchmarks import bbq, crows_pairs, iat, implicit_explicit, stereoset
from biaseval.io import is_completed, logit_result_path, write_benchmark_result
from biaseval.model_loader import ModelSpec, load_model, unload_model
from biaseval.probing.datasets import build_probe_dataset
from biaseval.probing.extract_activations import extract_activations
from biaseval.probing.linear_probe import train_probes_all_layers
from biaseval.registry import filter_specs, load_registry

logger = logging.getLogger("mvp")


# Three pairs that together exercise the diverse code paths in our pipeline.
MVP_PAIRS: list[tuple[str, str]] = [
    ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"),
    ("mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
]

# Languages the MVP validates end-to-end. English exercises the production
# default; fr + de stress the multilingual code path (different tokenisation,
# different chat-template behaviour). The full sweep can run more languages.
MVP_LANGUAGES: tuple[str, ...] = ("en", "fr", "de")

# (runner_key, function, accepts_limit). We keep IAT and implicit_explicit at
# full size — they are tiny anyway. CrowS / BBQ / StereoSet honour --limit.
BENCHMARK_RUNNERS = {
    "crows_pairs": (crows_pairs.run, True),
    "stereoset": (stereoset.run, True),
    "bbq": (bbq.run, True),
    "iat": (iat.run, False),
    "implicit_explicit_race": (
        lambda m, t, s, **kw: implicit_explicit.run(m, t, s, attribute="race", **kw),
        False,
    ),
    "implicit_explicit_gender": (
        lambda m, t, s, **kw: implicit_explicit.run(m, t, s, attribute="gender", **kw),
        False,
    ),
}


# ---------------------------------------------------------------------------
# Hard-fail helpers
# ---------------------------------------------------------------------------


def _summary_has_nan_inf(summary: dict) -> list[str]:
    """Return list of metric names containing NaN or Inf. Empty if all clean."""
    bad: list[str] = []
    for k, v in summary.items():
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            bad.append(k)
    return bad


def _per_example_fields_ok(benchmark: str, per_example: list[dict]) -> str | None:
    """Return None if per_example fields look right, else a string reason."""
    if not per_example:
        return None  # benchmarks like IAT have empty per_example by design
    sample = per_example[0]
    required = {
        "crows_pairs": {"pair_id", "stereo_won", "stereo_won_raw"},
        "bbq": {"is_unknown_pred", "is_biased_pred"},
        "stereoset": {"stereo_over_anti"},
        "implicit_explicit_race": {"implicit_biased", "explicit_biased"},
        "implicit_explicit_gender": {"implicit_biased", "explicit_biased"},
    }
    needed = required.get(benchmark, set())
    missing = needed - set(sample)
    if missing:
        return f"missing fields {sorted(missing)} in per_example"
    return None


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def stage1_logit_benchmarks(
    pair_specs: list[tuple[ModelSpec, ModelSpec]],
    results_root: Path,
    *,
    benchmark_limit: int = 100,
    languages: tuple[str, ...] = MVP_LANGUAGES,
    seed: int = 42,
) -> dict:
    """Run all benchmarks × all variants × applicable prompt modes.

    `languages` controls which CrowS-Pairs languages get exercised. Other
    benchmarks are English-only and ignore this. Default = ("en", "fr", "de")
    so the multilingual code path (statistics + figures + tokenizer) gets
    smoke-tested on every model before the full sweep.
    """
    cells_run: list[dict] = []
    hard_fails: list[str] = []

    for base_spec, instruct_spec in pair_specs:
        for spec in (base_spec, instruct_spec):
            try:
                logger.info("[load] %s", spec.model_id)
                model, tokenizer = load_model(spec)
            except Exception as exc:
                hard_fails.append(f"load failed for {spec.model_id}: {exc}")
                logger.error("[FAIL] load %s: %s", spec.model_id, traceback.format_exc())
                continue

            try:
                # Print one wrapped prompt per family for the chat-template check.
                from biaseval.benchmarks.utils import (
                    COMPLETION_INSTRUCTION,
                    JAILBREAK_INSTRUCTION,
                    wrap_chat_template,
                )
                if spec.variant == "instruct":
                    sample = wrap_chat_template(tokenizer, COMPLETION_INSTRUCTION)
                    logger.info("[%s instruct chat-template]\n%r",
                                spec.family, sample[:300])
                    sample_jb = wrap_chat_template(tokenizer, JAILBREAK_INSTRUCTION)
                    logger.info("[%s jailbreak prompt]\n%r",
                                spec.family, sample_jb[:300])

                for runner_key, (fn, accepts_limit) in BENCHMARK_RUNNERS.items():
                    # CrowS-Pairs runs in every language; everything else is en-only.
                    cell_langs = languages if runner_key == "crows_pairs" else ("en",)
                    for lang in cell_langs:
                        for prompt_mode in ("raw", "instruct"):
                            cell = _run_one_cell(
                                fn, runner_key, accepts_limit, model, tokenizer,
                                spec, prompt_mode, results_root,
                                benchmark_limit=benchmark_limit, language=lang,
                            )
                            cells_run.append(cell)
                            hard_fails.extend(cell["hard_fails"])

                # Jailbreak — instruct variants only, CrowS English only.
                if spec.variant == "instruct":
                    cell = _run_one_cell(
                        crows_pairs.run, "crows_pairs", True, model, tokenizer,
                        spec, "jailbreak", results_root,
                        benchmark_limit=benchmark_limit,
                    )
                    cells_run.append(cell)
                    hard_fails.extend(cell["hard_fails"])
            finally:
                unload_model(model)
                del tokenizer

    return {"cells": cells_run, "hard_fails": hard_fails}


def _bench_folder(runner_key: str, language: str) -> str:
    """Result-folder name. CrowS-Pairs gets a language suffix for non-English;
    everything else stays the runner key."""
    if runner_key == "crows_pairs" and language != "en":
        return f"crows_pairs_{language}"
    return runner_key


def _run_one_cell(
    runner, runner_key, accepts_limit, model, tokenizer, spec,
    prompt_mode, results_root, *, benchmark_limit, language: str = "en",
) -> dict:
    """Run one (model, benchmark, prompt_mode, language) cell with hard-fail accounting."""
    bench_folder = _bench_folder(runner_key, language)
    cell_label = f"{bench_folder}/{prompt_mode}"
    info: dict = {
        "model_id": spec.model_id, "family": spec.family, "variant": spec.variant,
        "benchmark": bench_folder, "prompt_mode": prompt_mode, "language": language,
        "ok": False, "elapsed_s": None, "hard_fails": [],
    }
    fp = logit_result_path(results_root, bench_folder, spec, prompt_mode)
    if is_completed(fp):
        logger.info("[skip] %s/%s already done", spec.short_name, cell_label)
        info["ok"] = True
        info["elapsed_s"] = 0.0
        return info

    t0 = time.time()
    try:
        kwargs: dict = {"prompt_mode": prompt_mode}
        if accepts_limit:
            kwargs["limit"] = benchmark_limit
        # Only crows_pairs honours `language`; passing it elsewhere would error.
        if runner_key == "crows_pairs" and language != "en":
            kwargs["language"] = language
        result = runner(model, tokenizer, spec, **kwargs)
        write_benchmark_result(results_root, result, spec)
        info["elapsed_s"] = time.time() - t0

        # Hard-fail checks.
        bad = _summary_has_nan_inf(result.summary)
        if bad:
            info["hard_fails"].append(
                f"{spec.short_name}/{cell_label}: NaN/Inf in {bad}"
            )
        problem = _per_example_fields_ok(runner_key, result.per_example)
        if problem:
            info["hard_fails"].append(
                f"{spec.short_name}/{cell_label}: {problem}"
            )
        info["ok"] = not info["hard_fails"]
    except Exception as exc:
        info["hard_fails"].append(
            f"{spec.short_name}/{cell_label}: exception {exc!r}"
        )
        logger.error("[FAIL] %s/%s\n%s", spec.short_name, cell_label,
                     traceback.format_exc())
    return info


# ---------------------------------------------------------------------------
# Stage 2 — paired statistics + cross-benchmark consistency
# ---------------------------------------------------------------------------


def stage2_statistics(
    pair_specs: list[tuple[ModelSpec, ModelSpec]],
    results_root: Path,
    *,
    languages: tuple[str, ...] = MVP_LANGUAGES,
    n_perm: int = 5000, n_boot: int = 5000, seed: int = 42,
) -> dict:
    """Run paired stats + cross-bench + per-language + cross-language helpers.

    n_perm/n_boot lowered from production (10k) to keep validation fast — still
    enough resolution to verify the columns and signs are right.
    """
    registry_pairs = [
        (b.model_id, i.model_id, b.family, b.generation, b.size)
        for b, i in pair_specs
    ]
    out: dict = {"hard_fails": []}

    try:
        out["pair_sig_df"] = pair_significance_table(
            results_root, registry_pairs,
            n_perm=n_perm, n_boot=n_boot, seed=seed,
        )
        logger.info("pair_significance_table: %d / %d pairs",
                    len(out["pair_sig_df"]), len(registry_pairs))
    except Exception as exc:
        out["hard_fails"].append(f"pair_significance_table: {exc!r}")
        logger.error("[FAIL] pair_significance_table\n%s", traceback.format_exc())
        out["pair_sig_df"] = None

    try:
        consistency_df, corr_df = cross_benchmark_consistency(
            results_root, registry_pairs,
        )
        out["consistency_df"] = consistency_df
        out["corr_df"] = corr_df
        logger.info("cross_benchmark_consistency: %d pairs, corr shape %s",
                    len(consistency_df), corr_df.shape)
    except Exception as exc:
        out["hard_fails"].append(f"cross_benchmark_consistency: {exc!r}")
        logger.error("[FAIL] cross_benchmark_consistency\n%s",
                     traceback.format_exc())
        out["consistency_df"] = None
        out["corr_df"] = None

    try:
        out["pair_sig_per_lang_df"] = pair_significance_per_language(
            results_root, registry_pairs, languages=languages,
            n_perm=n_perm, n_boot=n_boot, seed=seed,
        )
        logger.info("pair_significance_per_language: %d (pair × lang) rows "
                    "across %d langs", len(out["pair_sig_per_lang_df"]),
                    len(languages))
    except Exception as exc:
        out["hard_fails"].append(f"pair_significance_per_language: {exc!r}")
        logger.error("[FAIL] pair_significance_per_language\n%s",
                     traceback.format_exc())
        out["pair_sig_per_lang_df"] = None

    try:
        lang_cons_df, lang_corr_df = cross_language_consistency(
            results_root, registry_pairs, languages=languages,
        )
        out["lang_consistency_df"] = lang_cons_df
        out["lang_corr_df"] = lang_corr_df
        logger.info("cross_language_consistency: %d pairs, corr shape %s",
                    len(lang_cons_df), lang_corr_df.shape)
    except Exception as exc:
        out["hard_fails"].append(f"cross_language_consistency: {exc!r}")
        logger.error("[FAIL] cross_language_consistency\n%s",
                     traceback.format_exc())
        out["lang_consistency_df"] = None
        out["lang_corr_df"] = None

    return out


# ---------------------------------------------------------------------------
# Stage 3 — neutral-prompt probing on every model
# ---------------------------------------------------------------------------


def stage3_probing(
    pair_specs: list[tuple[ModelSpec, ModelSpec]],
    results_root: Path,
    *,
    attributes: tuple[str, ...] = ("gender",),
    seed: int = 42,
) -> dict:
    """Extract activations + train probes for every (model, attribute).

    Default attribute is gender only. Race probing in autoregressive LMs has
    a fundamental keyword-leakage problem (any candidate name or dialect word
    is itself a strong demographic token), so we follow Bouchouchi 2026 and
    restrict probing to gender. Pass `attributes=("gender","race")` if you
    explicitly want the appendix race results.

    Hard-fails:
        - any layer with mean accuracy > 0.90 (keyword leakage signal — the
          neutral-prompt fix is broken or the dataset has slipped a giveaway)
        - any layer with NaN accuracy
        - direction_<attr>.npy missing or wrong shape
    """
    import numpy as np

    out: dict = {"per_model": {}, "hard_fails": []}
    probe_datasets = {a: build_probe_dataset(a) for a in attributes}

    # Re-verify the forbidden-word check at runtime (a second belt to the
    # one in tests/test_probe_datasets.py).
    forbidden = {
        "gender": {"man", "woman", "he", "she", "his", "her", "male", "female"},
        "race":   {"black", "white", "african", "european", "caucasian"},
    }
    for attr, ds in probe_datasets.items():
        for s in ds.sentences:
            toks = set(s.lower().replace(".", "").split())
            leak = toks & forbidden.get(attr, set())
            if leak:
                out["hard_fails"].append(
                    f"probe dataset for {attr!r} leaks keyword {leak} in: {s!r}"
                )

    for base_spec, instruct_spec in pair_specs:
        for spec in (base_spec, instruct_spec):
            try:
                model, tokenizer = load_model(spec)
            except Exception as exc:
                out["hard_fails"].append(
                    f"probing load {spec.model_id}: {exc!r}"
                )
                continue

            try:
                short = spec.short_name
                act_dir = results_root / "activations" / short
                act_dir.mkdir(parents=True, exist_ok=True)
                # Extract one shared activation pool spanning both attributes.
                all_sents: list[str] = []
                attr_idx: dict[str, list[int]] = {}
                for attr in attributes:
                    start = len(all_sents)
                    all_sents.extend(probe_datasets[attr].sentences)
                    attr_idx[attr] = list(range(start, len(all_sents)))
                num_layers = extract_activations(
                    model, tokenizer, all_sents, act_dir,
                    pool="last", batch_size=4,
                )

                model_results: dict[str, dict] = {}
                for attr in attributes:
                    ds = probe_datasets[attr]
                    labels = np.array(ds.labels)
                    sliced_dir = act_dir / f"_{attr}"
                    sliced_dir.mkdir(exist_ok=True)
                    for li in range(num_layers):
                        full = np.load(act_dir / f"layer_{li}.npy")
                        np.save(sliced_dir / f"layer_{li}.npy", full[attr_idx[attr]])
                        del full
                    layer_results = train_probes_all_layers(
                        sliced_dir, labels, num_layers, attr,
                        cv_folds=5, seed=seed, save_directions=True,
                        direction_save_dir=act_dir,
                    )
                    model_results[attr] = layer_results

                    # Hard-fail: keyword leakage signal.
                    accs = [r["mean_accuracy"] for r in layer_results]
                    if any(np.isnan(a) for a in accs):
                        out["hard_fails"].append(
                            f"{short}/{attr}: NaN probe accuracy at some layer"
                        )
                    peak = max(accs)
                    if peak > 0.90:
                        out["hard_fails"].append(
                            f"{short}/{attr}: peak probe accuracy {peak:.2%} > 0.90 "
                            "(possible keyword leakage; neutral-prompt fix may be broken)"
                        )

                    # Direction file shape check.
                    dir_fp = act_dir / f"direction_{attr}.npy"
                    if not dir_fp.exists():
                        out["hard_fails"].append(
                            f"{short}/{attr}: direction_{attr}.npy not written"
                        )
                    else:
                        arr = np.load(dir_fp)
                        if arr.shape != (num_layers, model.config.hidden_size):
                            out["hard_fails"].append(
                                f"{short}/{attr}: direction shape {arr.shape} "
                                f"!= ({num_layers}, {model.config.hidden_size})"
                            )

                out["per_model"][spec.model_id] = {
                    "variant": spec.variant, "family": spec.family,
                    "num_layers": num_layers, "attributes": model_results,
                }
            except Exception as exc:
                out["hard_fails"].append(f"probing {spec.model_id}: {exc!r}")
                logger.error("[FAIL] probing %s\n%s", spec.model_id,
                             traceback.format_exc())
            finally:
                unload_model(model)
                del tokenizer

    return out


# ---------------------------------------------------------------------------
# Stage 4 — figure smoke pass
# ---------------------------------------------------------------------------


def stage4_figures(
    pair_specs: list[tuple[ModelSpec, ModelSpec]],
    results_root: Path,
    figures_root: Path,
    *,
    pair_sig_df,
    consistency_df,
    pair_sig_per_lang_df=None,
    lang_consistency_df=None,
    lang_corr_df=None,
) -> dict:
    """Call every plotting function with the MVP data; hard-fail on any crash.

    Sparse data is fine — the goal is "did the figure code execute without
    error", not "do the figures look good". Visual sanity happens in the
    notebook viewer (Bite 5).
    """
    figures_root.mkdir(parents=True, exist_ok=True)
    out: dict = {"hard_fails": [], "paths": []}

    try:
        logit_df = aggregate_logit_results(results_root)
        probe_df = aggregate_probe_results(results_root)
    except Exception as exc:
        out["hard_fails"].append(f"aggregate_*_results: {exc!r}")
        logger.error("[FAIL] aggregator\n%s", traceback.format_exc())
        return out
    logger.info("Aggregator: %d logit rows, %d probe rows",
                len(logit_df), len(probe_df))

    registry_pairs = [
        (b.model_id, i.model_id, b.family, b.generation, b.size)
        for b, i in pair_specs
    ]
    try:
        direction_cosines_df = cross_pair_direction_cosines(
            results_root, registry_pairs,
        )
    except Exception as exc:
        out["hard_fails"].append(f"cross_pair_direction_cosines: {exc!r}")
        direction_cosines_df = None

    try:
        paths = generate_all(
            logit_df, probe_df, figures_dir=figures_root,
            pair_sig_df=pair_sig_df,
            consistency_df=consistency_df,
            direction_cosines_df=direction_cosines_df,
            pair_sig_per_lang_df=pair_sig_per_lang_df,
            lang_consistency_df=lang_consistency_df,
            lang_corr_df=lang_corr_df,
            results_dir=results_root,
            registry_pairs=registry_pairs,
        )
        out["paths"] = paths
        logger.info("Stage 4 wrote %d figure(s) → %s", len(paths), figures_root)
    except Exception as exc:
        out["hard_fails"].append(f"generate_all: {exc!r}")
        logger.error("[FAIL] generate_all\n%s", traceback.format_exc())

    # Verify every saved path is non-empty (caller asks: "did matplotlib write
    # actual content, not a 0-byte stub from a crashed savefig?").
    for p in out["paths"]:
        try:
            if not p.exists() or p.stat().st_size < 200:
                out["hard_fails"].append(
                    f"figure file empty or tiny: {p} ({p.stat().st_size if p.exists() else 0} bytes)"
                )
        except OSError as exc:
            out["hard_fails"].append(f"could not stat figure {p}: {exc!r}")

    return out


# ---------------------------------------------------------------------------
# Stage 5 — cross-family comparison table + report writer
# ---------------------------------------------------------------------------


# Metrics shown in the cross-family comparison table. Each entry is
# (label, benchmark, metric, prompt_mode, fmt). prompt_mode=None → use raw.
CROSS_FAMILY_METRICS: list[tuple[str, str, str, str | None, str]] = [
    ("CrowS-Pairs (%)",       "crows_pairs", "overall",                  "raw",       "{:.1f}"),
    ("CrowS jailbreak (%)",   "crows_pairs", "overall",                  "jailbreak", "{:.1f}"),
    ("BBQ deferral",          "bbq",         "overall_deferral_rate",    "raw",       "{:.3f}"),
    ("BBQ cond. bias",        "bbq",         "overall_conditional_bias", "raw",       "{:.3f}"),
    ("StereoSet ICAT",        "stereoset",   "overall_ICAT",             "raw",       "{:.1f}"),
    ("IAT |d| (logprob)",     "iat",         "overall_abs_d",            "raw",       "{:.3f}"),
    ("IAT |d| (embed)",       "iat",         "overall_abs_d_weat_embed", "raw",       "{:.3f}"),
    ("Imp/Exp gap race",      "implicit_explicit_race",   "implicit_explicit_gap", "raw", "{:.2f}"),
    ("Imp/Exp gap gender",    "implicit_explicit_gender", "implicit_explicit_gap", "raw", "{:.2f}"),
]


def _lookup(logit_df, model_id: str, benchmark: str, metric: str,
            prompt_mode: str) -> float | None:
    """Single-cell lookup; returns None if missing."""
    if logit_df.empty:
        return None
    sel = logit_df[(logit_df["model_id"] == model_id)
                   & (logit_df["benchmark"] == benchmark)
                   & (logit_df["metric"] == metric)
                   & (logit_df["prompt_mode"] == prompt_mode)]
    return float(sel["value"].iloc[0]) if len(sel) else None


def _build_cross_family_table(
    pair_specs: list[tuple[ModelSpec, ModelSpec]],
    logit_df,
) -> str:
    """Produce a markdown table mirroring the spec layout."""
    families = [(b.family, b.model_id, i.model_id) for b, i in pair_specs]
    head_cells = ["Metric"]
    for fam, _, _ in families:
        head_cells.extend([f"{fam} base", f"{fam} inst", f"{fam} Δ"])
    sep = ["---"] * len(head_cells)
    lines = ["| " + " | ".join(head_cells) + " |",
             "| " + " | ".join(sep) + " |"]
    for label, bench, metric, pm, fmt in CROSS_FAMILY_METRICS:
        row = [label]
        for _, base_id, instr_id in families:
            base = _lookup(logit_df, base_id, bench, metric, pm or "raw")
            instr = _lookup(logit_df, instr_id, bench, metric, pm or "raw")
            row.append("—" if base is None else fmt.format(base))
            row.append("—" if instr is None else fmt.format(instr))
            if base is None or instr is None:
                row.append("—")
            else:
                row.append(fmt.format(instr - base))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _soft_observations(
    pair_specs: list[tuple[ModelSpec, ModelSpec]],
    logit_df, stage3,
) -> list[str]:
    """Directional / magnitude flags. NOT hard-fails — these print and pass."""
    notes: list[str] = []

    # 1) CrowS-Pairs direction per family + cross-family agreement.
    crows_signs = []
    for base_spec, instr_spec in pair_specs:
        b = _lookup(logit_df, base_spec.model_id, "crows_pairs", "overall", "raw")
        i = _lookup(logit_df, instr_spec.model_id, "crows_pairs", "overall", "raw")
        if b is None or i is None:
            continue
        if b < 55:
            notes.append(
                f"{base_spec.family}: CrowS base = {b:.1f}% (close to chance — scoring may be noisy)"
            )
        if i > b + 1.0:
            notes.append(
                f"{base_spec.family}: CrowS instruct ({i:.1f}%) > base ({b:.1f}%) — alignment direction flipped"
            )
        crows_signs.append(i - b)
    if crows_signs and not all(d <= 0 for d in crows_signs):
        notes.append(
            "Cross-family: CrowS Δ signs disagree across families — alignment effect "
            "is not consistent (worth investigating before sweep)"
        )

    # 2) BBQ deferral direction (instruct should defer more).
    for base_spec, instr_spec in pair_specs:
        b = _lookup(logit_df, base_spec.model_id, "bbq", "overall_deferral_rate", "raw")
        i = _lookup(logit_df, instr_spec.model_id, "bbq", "overall_deferral_rate", "raw")
        if b is None or i is None:
            continue
        if i < b:
            notes.append(
                f"{base_spec.family}: BBQ deferral instruct ({i:.2f}) < base ({b:.2f}) — unexpected"
            )

    # 3) IAT magnitude in plausible range.
    for spec in [s for pair in pair_specs for s in pair]:
        v = _lookup(logit_df, spec.model_id, "iat", "overall_abs_d", "raw")
        if v is None:
            continue
        if v < 0.1:
            notes.append(
                f"{spec.short_name}: IAT |d|={v:.3f} < 0.1 — suspiciously small (prompt template?)"
            )
        if v > 2.0:
            notes.append(
                f"{spec.short_name}: IAT |d|={v:.3f} > 2.0 — implausibly large"
            )

    # 4) Jailbreak rebound check.
    for _, instr_spec in pair_specs:
        instr = _lookup(logit_df, instr_spec.model_id, "crows_pairs", "overall", "instruct")
        jb = _lookup(logit_df, instr_spec.model_id, "crows_pairs", "overall", "jailbreak")
        if instr is None or jb is None:
            continue
        if jb < instr - 1.0:
            notes.append(
                f"{instr_spec.family}: jailbreak ({jb:.1f}%) < instruct ({instr:.1f}%) — "
                "no rebound observed"
            )

    # 5) Probe peak accuracy band.
    # We DROP the depth-band check: Bouchouchi 2026 reports late-layer peaks
    # for occupation-gender probes because gender pronouns live in the
    # unembedding direction, so peaks at depth ~0.95 are expected, not broken.
    # Only flag accuracy outside [0.55, 0.90]: below 55% = probe isn't
    # learning; above 90% is the hard-fail keyword-leakage gate, so anything
    # under it but high (0.85–0.90) is just "strong but not suspicious".
    for model_id, info in stage3.get("per_model", {}).items():
        for attr, layer_results in info["attributes"].items():
            accs = [r["mean_accuracy"] for r in layer_results]
            if not accs:
                continue
            peak = max(accs)
            if peak < 0.55:
                notes.append(
                    f"{model_id}/{attr}: probe peak {peak:.2%} < 55% — probe failed to learn"
                )

    # 6) Base vs instruct probe gap.
    by_pair: dict = {}
    for info in stage3.get("per_model", {}).values():
        for attr, layer_results in info["attributes"].items():
            peak = max(r["mean_accuracy"] for r in layer_results)
            by_pair.setdefault((info["family"], attr), {})[info["variant"]] = peak
    for (fam, attr), variants in by_pair.items():
        if "base" in variants and "instruct" in variants:
            gap = abs(variants["base"] - variants["instruct"])
            if gap > 0.15:
                notes.append(
                    f"{fam}/{attr}: probe peak gap base vs instruct = {gap:.2%} > 15pp — "
                    "encoded bias unexpectedly different"
                )

    return notes


def _verdict(hard_fails: list[str], soft_notes: list[str]) -> str:
    if hard_fails:
        return ("**NO** — full sweep blocked. Resolve every hard-fail above first; "
                "the pipeline will produce broken outputs at scale.")
    if soft_notes:
        return ("**CONDITIONAL** — pipeline is mechanically sound but the soft "
                "observations below describe directional or magnitude surprises. "
                "Decide whether each is a real finding or an artifact before "
                "committing to the full sweep.")
    return ("**YES** — every hard-fail check passed and no soft observations were "
            "raised. Safe to queue the full sweep.")


def _build_multilingual_table(stage2: dict) -> str:
    """Per-pair × per-language Cohen's d (with ★/★★ Holm markers).

    Reads `pair_sig_per_lang_df` from stage2. Columns = languages; rows = pairs.
    Each cell shows the d value plus star markers from the within-language
    Holm correction.
    """
    df = stage2.get("pair_sig_per_lang_df")
    if df is None or df.empty:
        return "_No multilingual paired stats._"
    df = df.copy()
    df["pair_label"] = (df["family"].astype(str) + "/"
                        + df["generation"].astype(str) + "/"
                        + df["size"].astype(str))

    def _cell(row):
        d = row["cohens_d_paired"]
        star = ("★★" if row.get("reject_strict") else
                ("★" if row.get("reject_holm") else ""))
        if d is None:
            return "—"
        try:
            f = float(d)
        except (TypeError, ValueError):
            return "—"
        if f != f:  # NaN
            return "—"
        return f"{f:+.2f}{(' ' + star) if star else ''}"

    import pandas as pd
    df["cell"] = df.apply(_cell, axis=1)
    pivot = df.pivot(index="pair_label", columns="language", values="cell")
    lang_order = [c for c in ("en", "fr", "es", "de", "pt", "it") if c in pivot.columns]
    pivot = pivot[lang_order] if lang_order else pivot

    head = "| Pair | " + " | ".join(pivot.columns) + " |"
    sep  = "| --- | " + " | ".join("---" for _ in pivot.columns) + " |"
    body = [
        "| " + r + " | " + " | ".join(("—" if pd.isna(v) else str(v)) for v in pivot.loc[r])
        + " |"
        for r in pivot.index
    ]
    return "\n".join([head, sep, *body])


def _build_probe_layer_table(stage3: dict) -> str:
    """Markdown grid: one row per layer (depth bucketed) per (model, attr).

    Bucketed to 5 depth quintiles so the table stays readable across models
    with different layer counts (28 for Llama-3, 32 for Mistral, 36 for Qwen).
    The "peak" column shows the actual max-accuracy layer for that model.
    """
    if not stage3.get("per_model"):
        return "_No probing results._"

    lines = ["| Model | Attr | L0 | 25% | 50% | 75% | L_last | peak (depth) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model_id, info in sorted(stage3["per_model"].items()):
        n = info["num_layers"]
        for attr, layer_results in info["attributes"].items():
            accs = [r["mean_accuracy"] for r in layer_results]
            if not accs:
                continue
            buckets = [
                accs[0],
                accs[int(0.25 * (n - 1))],
                accs[int(0.50 * (n - 1))],
                accs[int(0.75 * (n - 1))],
                accs[-1],
            ]
            peak_idx = max(range(len(accs)), key=lambda k: accs[k])
            peak_depth = peak_idx / max(n - 1, 1)
            lines.append(
                f"| {model_id} | {attr} | "
                + " | ".join(f"{a:.2%}" for a in buckets)
                + f" | {accs[peak_idx]:.2%} ({peak_depth:.2f}) |"
            )
    return "\n".join(lines)


def stage5_report(
    pair_specs: list[tuple[ModelSpec, ModelSpec]],
    results_root: Path,
    figures_root: Path,
    *,
    stage1: dict, stage2: dict, stage3: dict, stage4: dict,
    elapsed_total_s: float,
) -> Path:
    """Write results/mvp/report.md with verdict + tables + diagnostics."""
    import datetime as dt

    logit_df = aggregate_logit_results(results_root)
    table_md = _build_cross_family_table(pair_specs, logit_df)
    multilingual_md = _build_multilingual_table(stage2)
    probe_layer_md = _build_probe_layer_table(stage3)
    soft_notes = _soft_observations(pair_specs, logit_df, stage3)
    hard_fails = (stage1["hard_fails"] + stage2["hard_fails"]
                  + stage3["hard_fails"] + stage4["hard_fails"])
    verdict = _verdict(hard_fails, soft_notes)

    timing_rows = [
        ("Stage 1 — logit benchmarks",
         sum((c.get("elapsed_s") or 0) for c in stage1["cells"])),
        ("Stage 4 — figures wrote",
         f"{len(stage4.get('paths', []))} files"),
    ]
    cells_n = len(stage1["cells"])
    cells_ok = sum(1 for c in stage1["cells"] if c["ok"])
    figures_n = len(stage4.get("paths", []))
    probe_models = len(stage3.get("per_model", {}))

    lines: list[str] = [
        "# MVP Validation Report",
        "",
        f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_  ",
        f"_Total elapsed: {elapsed_total_s:.0f}s_",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Run summary",
        "",
        f"- Stage 1 (logit benchmarks): {cells_ok}/{cells_n} cells clean — "
        f"sum elapsed {timing_rows[0][1]:.0f}s",
        f"- Stage 2 (statistics): pair_significance n_pairs="
        f"{0 if stage2.get('pair_sig_df') is None else len(stage2['pair_sig_df'])}, "
        f"consistency n_pairs="
        f"{0 if stage2.get('consistency_df') is None else len(stage2['consistency_df'])}, "
        f"multilingual rows="
        f"{0 if stage2.get('pair_sig_per_lang_df') is None else len(stage2['pair_sig_per_lang_df'])}",
        f"- Stage 3 (probing): {probe_models} models probed",
        f"- Stage 4 (figures): {figures_n} figures written → `{figures_root}`",
        "",
        "## Cross-family comparison",
        "",
        "(Δ = instruct − base. Headline metrics in their canonical scoring.)",
        "",
        table_md,
        "",
        "## Multilingual paired d (CrowS-Pairs)",
        "",
        "(Pair × language Cohen's d. ★ = survives Holm at α=0.05 within that "
        "language; ★★ = also survives α=0.0025.)",
        "",
        multilingual_md,
        "",
        "## Probe accuracy by layer depth",
        "",
        "(Bucketed to 5 quintiles so models with different layer counts line up. "
        "`peak (depth)` is the actual max layer.)",
        "",
        probe_layer_md,
        "",
        "## Hard fails",
        "",
        ("_None — all hard-fail checks passed._" if not hard_fails
         else "\n".join(f"- {f}" for f in hard_fails)),
        "",
        "## Soft observations",
        "",
        ("_None — no directional or magnitude surprises._" if not soft_notes
         else "\n".join(f"- {n}" for n in soft_notes)),
        "",
        "## Artifacts",
        "",
        f"- JSONs: `{results_root}/logit_scores/`, `{results_root}/probe_results/`",
        f"- Activations + direction vectors: `{results_root}/activations/`",
        f"- Figures: `{figures_root}/`",
        "",
    ]
    report_path = results_root / "report.md"
    report_path.write_text("\n".join(lines))
    logger.info("Wrote MVP report → %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--results-dir", default="results/mvp",
                   help="Output dir (kept separate from production results/).")
    p.add_argument("--figures-dir", default="figures/mvp",
                   help="Where Stage 4 writes the smoke-pass figure files.")
    p.add_argument("--benchmark-limit", type=int, default=100,
                   help="--limit passed to CrowS/BBQ/StereoSet. IAT and "
                        "implicit_explicit always run full (they are tiny).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_pair_specs(config_path: str) -> list[tuple[ModelSpec, ModelSpec]]:
    """Pull the 3 MVP pairs as (base, instruct) ModelSpec tuples."""
    registry = load_registry(config_path)
    pairs: list[tuple[ModelSpec, ModelSpec]] = []
    for base_id, instruct_id in MVP_PAIRS:
        b_list = list(filter_specs(registry, only_ids={base_id}))
        i_list = list(filter_specs(registry, only_ids={instruct_id}))
        if not b_list or not i_list:
            raise SystemExit(
                f"MVP pair not in registry: {base_id} / {instruct_id}. "
                "Add them to configs/models.yaml or update MVP_PAIRS."
            )
        pairs.append((b_list[0], i_list[0]))
    return pairs


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    pair_specs = _resolve_pair_specs(args.config)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    logger.info("MVP results → %s", results_root)
    logger.info("Pairs: %s", [(b.model_id, i.model_id) for b, i in pair_specs])

    t_start = time.time()
    stage1 = stage1_logit_benchmarks(
        pair_specs, results_root, benchmark_limit=args.benchmark_limit, seed=args.seed,
    )
    logger.info("Stage 1 took %.1fs (%d cells, %d hard-fail)",
                time.time() - t_start, len(stage1["cells"]),
                len(stage1["hard_fails"]))

    t_stats = time.time()
    stage2 = stage2_statistics(pair_specs, results_root)
    logger.info("Stage 2 took %.1fs (%d hard-fail)",
                time.time() - t_stats, len(stage2["hard_fails"]))

    t_probe = time.time()
    stage3 = stage3_probing(pair_specs, results_root)
    logger.info("Stage 3 took %.1fs (%d hard-fail)",
                time.time() - t_probe, len(stage3["hard_fails"]))

    t_fig = time.time()
    stage4 = stage4_figures(
        pair_specs, results_root, Path(args.figures_dir),
        pair_sig_df=stage2.get("pair_sig_df"),
        consistency_df=stage2.get("consistency_df"),
        pair_sig_per_lang_df=stage2.get("pair_sig_per_lang_df"),
        lang_consistency_df=stage2.get("lang_consistency_df"),
        lang_corr_df=stage2.get("lang_corr_df"),
    )
    logger.info("Stage 4 took %.1fs (%d hard-fail, %d figures)",
                time.time() - t_fig, len(stage4["hard_fails"]),
                len(stage4.get("paths", [])))

    elapsed_total = time.time() - t_start
    report_path = stage5_report(
        pair_specs, results_root, Path(args.figures_dir),
        stage1=stage1, stage2=stage2, stage3=stage3, stage4=stage4,
        elapsed_total_s=elapsed_total,
    )

    all_hard_fails = (stage1["hard_fails"]
                      + stage2["hard_fails"]
                      + stage3["hard_fails"]
                      + stage4["hard_fails"])
    if all_hard_fails:
        logger.error("Hard-fail summary (%d total):", len(all_hard_fails))
        for f in all_hard_fails:
            logger.error("  - %s", f)
    else:
        logger.info("All stages clean — no hard fails.")
    logger.info("Report → %s", report_path)

    return 0 if not all_hard_fails else 1


if __name__ == "__main__":
    sys.exit(main())
