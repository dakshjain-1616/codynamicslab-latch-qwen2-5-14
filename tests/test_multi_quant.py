"""
Tests for multi_quant_compare module.
Covers: sweep results, recommendation logic, formatting, edge cases.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codynamicslab_latch_.multi_quant_compare import (
    MultiQuantComparer,
    QuantSweepResult,
    SIMULATED_DELTAS,
    _estimate_vram,
)


@pytest.fixture
def comparer():
    return MultiQuantComparer(
        model_name="CoDynamicsLab/LATCH-Qwen2.5-14B",
        vram_budget_gb=8.0,
        quant_types=["Q4_K_S", "Q4_K_M", "Q5_K_M", "Q8_0"],
        delta_threshold=0.05,
    )


# ── Sweep Results ─────────────────────────────────────────────────────────────

class TestSweepResults:
    def test_sweep_returns_all_types(self, comparer):
        results = comparer.run_sweep(fp16_perplexity=100.0)
        types = {r.quant_type for r in results}
        assert {"Q4_K_S", "Q4_K_M", "Q5_K_M", "Q8_0"}.issubset(types)

    def test_sweep_result_count_matches_input(self, comparer):
        results = comparer.run_sweep(100.0)
        assert len(results) == len(comparer.quant_types)

    def test_fp16_ppl_preserved(self, comparer):
        results = comparer.run_sweep(fp16_perplexity=50.0)
        assert all(r.fp16_perplexity == 50.0 for r in results)

    def test_quantized_ppl_greater_than_fp16(self, comparer):
        results = comparer.run_sweep(100.0)
        # All simulated quants have positive degradation
        assert all(r.quantized_perplexity >= r.fp16_perplexity for r in results)

    def test_delta_non_negative(self, comparer):
        for r in comparer.run_sweep(100.0):
            assert r.delta >= 0.0

    def test_vram_gb_positive(self, comparer):
        for r in comparer.run_sweep(100.0):
            assert r.estimated_vram_gb > 0.0

    def test_exactly_one_recommended(self, comparer):
        results = comparer.run_sweep(100.0)
        recommended = [r for r in results if r.recommended]
        assert len(recommended) == 1


# ── Recommendation Logic ──────────────────────────────────────────────────────

class TestRecommendation:
    def test_recommend_returns_string(self, comparer):
        rec = comparer.recommend(100.0)
        assert isinstance(rec, str)

    def test_recommend_passes_quality_gate(self, comparer):
        rec = comparer.recommend(100.0)
        results = comparer.run_sweep(100.0)
        rec_result = next(r for r in results if r.quant_type == rec)
        assert rec_result.passes_quality_gate is True

    def test_recommended_fits_vram_budget(self, comparer):
        rec = comparer.recommend(100.0)
        results = comparer.run_sweep(100.0)
        rec_result = next(r for r in results if r.quant_type == rec)
        assert rec_result.fits_vram_budget is True

    def test_q2k_fails_quality_gate(self):
        c = MultiQuantComparer(
            quant_types=["Q2_K"], vram_budget_gb=8.0, delta_threshold=0.05
        )
        results = c.run_sweep(100.0)
        assert results[0].passes_quality_gate is False

    def test_no_recommendation_when_all_fail(self):
        c = MultiQuantComparer(quant_types=["Q2_K"], delta_threshold=0.001)
        assert c.recommend(100.0) is None


# ── Formatting ────────────────────────────────────────────────────────────────

class TestFormatting:
    def test_table_contains_all_quant_types(self, comparer):
        results = comparer.run_sweep(100.0)
        table = comparer.format_sweep_table(results)
        for qt in comparer.quant_types:
            assert qt in table

    def test_table_contains_pass_fail(self, comparer):
        results = comparer.run_sweep(100.0)
        table = comparer.format_sweep_table(results)
        assert "PASS" in table or "FAIL" in table

    def test_table_is_markdown(self, comparer):
        results = comparer.run_sweep(100.0)
        table = comparer.format_sweep_table(results)
        assert "|" in table and "---" in table

    def test_to_dict_serialisable(self, comparer):
        import json
        results = comparer.run_sweep(100.0)
        json.dumps([r.to_dict() for r in results])  # should not raise


# ── VRAM Helper ───────────────────────────────────────────────────────────────

class TestVRAMEstimate:
    def test_q4_k_m_less_than_f16(self):
        from codynamicslab_latch_.multi_quant_compare import _FALLBACK_SIZE_FACTORS
        q4_vram = _estimate_vram(_FALLBACK_SIZE_FACTORS["Q4_K_M"])
        f16_vram = _estimate_vram(_FALLBACK_SIZE_FACTORS["F16"])
        assert q4_vram < f16_vram

    def test_vram_positive(self):
        assert _estimate_vram(0.245) > 0.0
