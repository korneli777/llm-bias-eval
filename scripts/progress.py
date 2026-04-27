"""Quick progress report on the current sweep.

Walks `results/logit_scores/`, `results/probe_results/` and
`results/intervention_results/` and prints a per-(family, benchmark,
prompt_mode) completion table plus the headline bias numbers from any
finished CrowS-Pairs / StereoSet runs.

Usage:
    uv run python scripts/progress.py
    uv run python scripts/progress.py --results-root /path/to/results
    uv run python scripts/progress.py --headline   # also print bias numbers
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from biaseval.registry import load_registry

BENCHMARKS = ("crows_pairs", "bbq", "stereoset", "iat")
PROMPT_MODES = ("raw", "instruct")
HEADLINE = {
    "crows_pairs": "overall",
    "stereoset": "overall_SS",
    "bbq": "overall_bias_ambig",
    "iat": "overall_abs_d",
}


def _short(model_id: str) -> str:
    return model_id.replace("/", "__")


def _audit_logit(results_root: Path, registry: list) -> dict:
    """Per (family × benchmark × prompt_mode) completion counts."""
    expected_per_cell = defaultdict(int)  # (family, bench, mode) → expected
    done_per_cell = defaultdict(int)
    missing: dict = defaultdict(list)
    for spec in registry:
        for bench in BENCHMARKS:
            for mode in PROMPT_MODES:
                key = (spec.family, bench, mode)
                expected_per_cell[key] += 1
                fp = results_root / "logit_scores" / bench / f"{_short(spec.model_id)}__{mode}.json"
                if fp.exists():
                    done_per_cell[key] += 1
                else:
                    missing[(spec.family, bench)].append(f"{spec.model_id}/{mode}")
    return {"expected": dict(expected_per_cell), "done": dict(done_per_cell), "missing": dict(missing)}


def _audit_probe(results_root: Path, registry: list, probing_ids: set[str]) -> dict:
    """How many (model, attribute) probe JSONs exist."""
    attrs = ("gender", "race")
    expected = sum(1 for s in registry if s.model_id in probing_ids) * len(attrs)
    done = 0
    for s in registry:
        if s.model_id not in probing_ids:
            continue
        for a in attrs:
            if (results_root / "probe_results" / _short(s.model_id) / f"{a}.json").exists():
                done += 1
    return {"expected": expected, "done": done}


def _audit_intervention(results_root: Path, registry: list, probing_ids: set[str]) -> dict:
    """Per (family × benchmark × method) intervention completion."""
    expected = defaultdict(int)
    done = defaultdict(int)
    benches = ("crows_pairs", "stereoset")
    methods = ("inlp", "leace")
    attrs = ("gender", "race")
    for s in registry:
        if s.model_id not in probing_ids:
            continue
        for bench in benches:
            for method in methods:
                key = (s.family, bench, method)
                expected[key] += len(attrs) * len(PROMPT_MODES)  # 2 attrs × 2 prompt modes
                base = results_root / "intervention_results" / bench
                for attr in attrs:
                    for mode in PROMPT_MODES:
                        fp = base / f"{_short(s.model_id)}__{attr}__{mode}__{method}.json"
                        if fp.exists():
                            done[key] += 1
    return {"expected": dict(expected), "done": dict(done)}


def _print_logit_table(audit: dict) -> None:
    families = sorted({f for f, _, _ in audit["expected"]})
    print(f"\n{'='*88}\n  STAGE 1 — Logit benchmarks (per family × benchmark × prompt_mode)\n{'='*88}")
    print(f"  {'family':10s}  ", end="")
    for bench in BENCHMARKS:
        print(f"{bench[:6]:>14s}", end="")
    print()
    print(f"  {'':10s}  " + " ".join(f"{m:>6s} {m:>6s}" for _ in BENCHMARKS for m in PROMPT_MODES))
    for fam in families:
        row = f"  {fam:10s}  "
        for bench in BENCHMARKS:
            for mode in PROMPT_MODES:
                key = (fam, bench, mode)
                d, e = audit["done"].get(key, 0), audit["expected"].get(key, 0)
                marker = "✓" if d == e and e > 0 else " "
                row += f" {d:>2d}/{e:<2d}{marker}"
        print(row)
    total_d = sum(audit["done"].values())
    total_e = sum(audit["expected"].values())
    print(f"\n  TOTAL: {total_d}/{total_e} cells complete ({100*total_d/total_e:.1f}%)")


def _print_probe_table(audit: dict) -> None:
    print(f"\n{'='*88}\n  STAGE 2 — Probing\n{'='*88}")
    print(f"  {audit['done']}/{audit['expected']} (model × attribute) cells complete "
          f"({100 * audit['done'] / max(audit['expected'], 1):.1f}%)")


def _print_intervention_table(audit: dict) -> None:
    if not audit["expected"]:
        return
    print(f"\n{'='*88}\n  STAGE 3.5 — Intervention (per family × benchmark × method)\n{'='*88}")
    families = sorted({f for f, _, _ in audit["expected"]})
    benches = sorted({b for _, b, _ in audit["expected"]})
    methods = sorted({m for _, _, m in audit["expected"]})
    header = f"  {'family':10s}  " + "".join(f"{b}/{m:6s}" for b in benches for m in methods)
    print(header)
    for fam in families:
        row = f"  {fam:10s}  "
        for bench in benches:
            for method in methods:
                key = (fam, bench, method)
                d, e = audit["done"].get(key, 0), audit["expected"].get(key, 0)
                row += f"  {d:>3d}/{e:<3d}"
        print(row)
    total_d = sum(audit["done"].values())
    total_e = sum(audit["expected"].values())
    print(f"\n  TOTAL: {total_d}/{total_e} intervention cells "
          f"({100 * total_d / max(total_e, 1):.1f}%)")


def _print_headline(results_root: Path, registry: list) -> None:
    print(f"\n{'='*88}\n  Headline bias scores (raw prompt mode)\n{'='*88}")
    print(f"  {'model_id':50s}  {'crows':>7s}  {'bbq':>7s}  {'sset':>7s}  {'iat':>6s}")
    by_id = {s.model_id: s for s in registry}
    rows = []
    for spec in registry:
        scores = {}
        for bench in BENCHMARKS:
            fp = results_root / "logit_scores" / bench / f"{_short(spec.model_id)}__raw.json"
            if not fp.exists():
                continue
            with open(fp) as f:
                summary = json.load(f)["result"]["summary"]
            scores[bench] = summary.get(HEADLINE[bench])
        if scores:
            rows.append((spec.model_id, scores))
    rows.sort()
    for model_id, scores in rows:
        def fmt(k: str, sc: dict = scores) -> str:
            return f"{sc.get(k, float('nan')):>7.2f}" if k in sc else " " * 7
        spec = by_id[model_id]
        marker = "*" if spec.variant == "instruct" else " "
        print(f"  {model_id[:50]:50s}{marker} {fmt('crows_pairs')} {fmt('bbq')} "
              f"{fmt('stereoset')} {scores.get('iat', float('nan')):>6.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--headline", action="store_true",
                   help="Also print headline bias scores from completed runs")
    args = p.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"results-root not found: {results_root}", file=sys.stderr)
        return 1

    registry = load_registry(args.config)
    from biaseval.registry import get_probing_subset
    probing_ids = get_probing_subset(args.config)

    _print_logit_table(_audit_logit(results_root, registry))
    _print_probe_table(_audit_probe(results_root, registry, probing_ids))
    _print_intervention_table(_audit_intervention(results_root, registry, probing_ids))

    if args.headline:
        _print_headline(results_root, registry)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
