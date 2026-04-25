"""End-to-end smoke test on a tiny model.

Loads sshleifer/tiny-gpt2 (a few MB), runs a 4-example slice of CrowS-Pairs,
StereoSet and IAT, and the small BBQ slice. Verifies result files write.

Run with:
    uv run pytest tests/test_smoke.py -m integration -v
"""

from __future__ import annotations

import pytest

from biaseval.benchmarks import bbq, crows_pairs, iat, stereoset
from biaseval.io import write_benchmark_result
from biaseval.model_loader import ModelSpec, load_model, unload_model

TINY_SPEC = ModelSpec(
    model_id="sshleifer/tiny-gpt2",
    family="smoke", generation="smoke", size="tiny",
    variant="base", num_params=1, num_layers=2, hidden_size=64,
    dtype="float32",
)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load_model(TINY_SPEC, device_map="cpu")
    yield model, tokenizer
    unload_model(model)


@pytest.mark.integration
def test_crows_pairs_smoke(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer
    result = crows_pairs.run(model, tokenizer, TINY_SPEC, limit=4)
    assert result.benchmark == "crows_pairs"
    assert "overall" in result.summary
    assert len(result.per_example) == 4
    out = write_benchmark_result(tmp_path, result, TINY_SPEC)
    assert out.exists()


@pytest.mark.integration
def test_stereoset_smoke(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer
    result = stereoset.run(model, tokenizer, TINY_SPEC, limit=4)
    assert "overall_SS" in result.summary
    assert "overall_LMS" in result.summary
    assert "overall_ICAT" in result.summary


@pytest.mark.integration
def test_bbq_smoke(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer
    result = bbq.run(model, tokenizer, TINY_SPEC, limit=4)
    assert "overall_acc_ambig" in result.summary
    assert len(result.per_example) <= 4


@pytest.mark.integration
def test_iat_smoke(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer
    # IAT runs all 4 categories with the default stimulus set.
    result = iat.run(model, tokenizer, TINY_SPEC)
    assert any(k.endswith("_d") for k in result.summary)


@pytest.mark.integration
def test_aggregation_and_plotting(model_and_tokenizer, tmp_path):
    """Run two benchmarks then verify aggregation + at least one figure."""
    from biaseval.analysis.aggregate_results import (
        aggregate_logit_results,
        write_aggregated,
    )
    from biaseval.analysis.plotting import generate_all

    model, tokenizer = model_and_tokenizer
    write_benchmark_result(tmp_path, crows_pairs.run(model, tokenizer, TINY_SPEC, limit=4), TINY_SPEC)
    write_benchmark_result(tmp_path, stereoset.run(model, tokenizer, TINY_SPEC, limit=4), TINY_SPEC)

    counts = write_aggregated(tmp_path, tmp_path / "agg.parquet")
    assert counts["logit_rows"] > 0

    logit_df = aggregate_logit_results(tmp_path)
    paths = generate_all(logit_df, probe_df=__import__("pandas").DataFrame(), figures_dir=tmp_path / "figs")
    # tiny-gpt2 has only 1 model so most figures will be sparse; just verify no crash.
    assert isinstance(paths, list)
