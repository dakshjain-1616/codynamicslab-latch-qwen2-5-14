"""
Tests for perplexity_evaluator module.
Covers: computation logic, delta calculation, threshold enforcement,
        mock evaluator, and test sample generation.
"""

import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("NUM_PERPLEXITY_SAMPLES", "10")
os.environ.setdefault("PERPLEXITY_DELTA_THRESHOLD", "0.05")

from codynamicslab_latch_.perplexity_evaluator import PerplexityEvaluator, MockPerplexityEvaluator, WIKITEXT_SAMPLES


# ─── Delta and Threshold Logic ────────────────────────────────────────────────

class TestDeltaCalculation:
    def test_delta_zero_when_equal(self):
        delta = PerplexityEvaluator.compute_delta(10.0, 10.0)
        assert delta == 0.0

    def test_delta_positive_when_quantized_higher(self):
        delta = PerplexityEvaluator.compute_delta(10.0, 10.5)
        assert delta > 0.0
        assert abs(delta - 0.05) < 1e-9

    def test_delta_positive_when_quantized_lower(self):
        # Delta is absolute value
        delta = PerplexityEvaluator.compute_delta(10.0, 9.5)
        assert delta > 0.0
        assert abs(delta - 0.05) < 1e-9

    def test_delta_formula_correctness(self):
        # |q - fp| / fp
        fp = 20.0
        q = 21.0
        expected = 1.0 / 20.0  # 0.05
        assert abs(PerplexityEvaluator.compute_delta(fp, q) - expected) < 1e-10

    def test_delta_infinity_for_zero_fp16(self):
        delta = PerplexityEvaluator.compute_delta(0.0, 5.0)
        assert delta == float("inf")

    def test_passes_threshold_at_exactly_threshold(self):
        # delta == threshold → should pass (≤)
        assert PerplexityEvaluator.passes_threshold(10.0, 10.5, threshold=0.05)

    def test_passes_threshold_below(self):
        assert PerplexityEvaluator.passes_threshold(10.0, 10.1, threshold=0.05)

    def test_fails_threshold_above(self):
        assert not PerplexityEvaluator.passes_threshold(10.0, 10.6, threshold=0.05)

    def test_passes_threshold_uses_env_default(self):
        # With 1.8% delta and 5% threshold — should pass
        assert PerplexityEvaluator.passes_threshold(100.0, 101.8, threshold=0.05)

    def test_fails_threshold_strict(self):
        # With 6% delta and 5% threshold — should fail
        assert not PerplexityEvaluator.passes_threshold(100.0, 106.1, threshold=0.05)


# ─── Test Sample Generation ───────────────────────────────────────────────────

class TestSampleGeneration:
    def test_default_samples_not_empty(self):
        evaluator = PerplexityEvaluator("mock-model", mock_mode=True)
        samples = evaluator.get_test_samples(10)
        assert len(samples) == 10

    def test_samples_are_strings(self):
        evaluator = PerplexityEvaluator("mock-model", mock_mode=True)
        samples = evaluator.get_test_samples(5)
        assert all(isinstance(s, str) for s in samples)

    def test_samples_are_non_empty(self):
        evaluator = PerplexityEvaluator("mock-model", mock_mode=True)
        samples = evaluator.get_test_samples(5)
        assert all(len(s) > 0 for s in samples)

    def test_can_generate_100_samples(self):
        evaluator = PerplexityEvaluator("mock-model", mock_mode=True)
        samples = evaluator.get_test_samples(100)
        assert len(samples) == 100

    def test_wikitext_samples_covers_quantization_topics(self):
        # At least some samples should mention quantization-related terms
        text = " ".join(WIKITEXT_SAMPLES).lower()
        assert "quantization" in text
        assert "gguf" in text or "q4" in text.lower()
        assert "perplexity" in text


# ─── Mock Evaluator ───────────────────────────────────────────────────────────

class TestMockPerplexityEvaluator:
    """
    Tests for MockPerplexityEvaluator.
    Uses the real gpt2 model via HuggingFace — these tests exercise real
    transformer inference on CPU with a tiny model.
    """

    @pytest.fixture(scope="class")
    def mock_evaluator(self):
        """Shared evaluator with gpt2 for the class."""
        return MockPerplexityEvaluator("CoDynamicsLab/LATCH-Qwen2.5-14B")

    def test_target_model_stored(self, mock_evaluator):
        assert mock_evaluator.target_model == "CoDynamicsLab/LATCH-Qwen2.5-14B"

    def test_proxy_model_set(self, mock_evaluator):
        assert mock_evaluator.model_name_or_path == "openai-community/gpt2" or \
               mock_evaluator.model_name_or_path == mock_evaluator.PROXY_MODEL

    def test_compute_quantized_perplexity_higher_than_fp16(self, mock_evaluator):
        # Q4_K_M should always be >= FP16
        fp16_ppl = 50.0
        q4_ppl = mock_evaluator.compute_quantized_perplexity(fp16_ppl)
        assert q4_ppl > fp16_ppl

    def test_compute_quantized_delta_within_range(self, mock_evaluator):
        fp16_ppl = 50.0
        q4_ppl = mock_evaluator.compute_quantized_perplexity(fp16_ppl)
        delta = PerplexityEvaluator.compute_delta(fp16_ppl, q4_ppl)
        # Simulated delta should be small (< 10%)
        assert delta < 0.10

    def test_full_evaluation_returns_expected_keys(self, mock_evaluator):
        results = mock_evaluator.run_full_evaluation(num_samples=5)
        required_keys = [
            "model", "fp16_perplexity", "quantized_perplexity",
            "delta", "delta_percent", "passes", "num_samples", "mock_mode",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_full_evaluation_passes_quality_gate(self, mock_evaluator):
        results = mock_evaluator.run_full_evaluation(num_samples=5)
        # With the default simulated delta (~1.8%), should pass the 5% threshold
        assert results["passes"] is True
        assert results["delta"] < 0.05

    def test_full_evaluation_fp16_positive(self, mock_evaluator):
        results = mock_evaluator.run_full_evaluation(num_samples=5)
        assert results["fp16_perplexity"] > 0

    def test_full_evaluation_quantized_positive(self, mock_evaluator):
        results = mock_evaluator.run_full_evaluation(num_samples=5)
        assert results["quantized_perplexity"] > 0

    def test_full_evaluation_num_samples_recorded(self, mock_evaluator):
        results = mock_evaluator.run_full_evaluation(num_samples=5)
        assert results["num_samples"] == 5

    def test_full_evaluation_mock_flag_set(self, mock_evaluator):
        results = mock_evaluator.run_full_evaluation(num_samples=5)
        assert results["mock_mode"] is True
