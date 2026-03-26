"""
Tests for quantization_pipeline module.
Covers: initialization, validation, info, full mock run, quality gate enforcement.
"""

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["MOCK_MODE"] = "true"
os.environ.setdefault("NUM_PERPLEXITY_SAMPLES", "5")
os.environ.setdefault("PERPLEXITY_DELTA_THRESHOLD", "0.05")

from codynamicslab_latch_.quantization_pipeline import QuantizationPipeline, QuantizationPipelineError
from codynamicslab_latch_.model_converter import GGUF_MAGIC


@pytest.fixture
def pipeline(tmp_path):
    return QuantizationPipeline(
        model_name="CoDynamicsLab/LATCH-Qwen2.5-14B",
        quant_type="Q4_K_M",
        output_dir=str(tmp_path),
        mock_mode=True,
        num_samples=5,
        fail_on_quality_gate=True,
    )


@pytest.fixture
def pipeline_no_fail(tmp_path):
    return QuantizationPipeline(
        model_name="CoDynamicsLab/LATCH-Qwen2.5-14B",
        quant_type="Q4_K_M",
        output_dir=str(tmp_path),
        mock_mode=True,
        num_samples=5,
        fail_on_quality_gate=False,
    )


# ─── Initialization ───────────────────────────────────────────────────────────

class TestPipelineInit:
    def test_model_name_stored(self, pipeline):
        assert pipeline.model_name == "CoDynamicsLab/LATCH-Qwen2.5-14B"

    def test_quant_type_stored(self, pipeline):
        assert pipeline.quant_type == "Q4_K_M"

    def test_mock_mode_enabled(self, pipeline):
        assert pipeline.mock_mode is True

    def test_output_dir_created(self, pipeline):
        assert pipeline.output_dir.exists()

    def test_converter_initialized(self, pipeline):
        assert pipeline.converter is not None

    def test_reporter_initialized(self, pipeline):
        assert pipeline.reporter is not None


# ─── Validation ───────────────────────────────────────────────────────────────

class TestPipelineValidation:
    def test_valid_pipeline_passes(self, pipeline):
        result = pipeline.validate_inputs()
        assert result["valid"] is True

    def test_valid_pipeline_no_errors(self, pipeline):
        result = pipeline.validate_inputs()
        assert result["errors"] == []

    def test_invalid_quant_type_fails(self, tmp_path):
        p = QuantizationPipeline(
            model_name="test/model",
            quant_type="INVALID",
            output_dir=str(tmp_path),
            mock_mode=True,
        )
        result = p.validate_inputs()
        assert result["valid"] is False
        assert any("Invalid quant_type" in e for e in result["errors"])

    def test_low_sample_count_warns(self, tmp_path):
        p = QuantizationPipeline(
            model_name="test/model",
            quant_type="Q4_K_M",
            output_dir=str(tmp_path),
            mock_mode=True,
            num_samples=5,
        )
        result = p.validate_inputs()
        # Warnings allowed, but should still be valid
        assert result["valid"] is True


# ─── Pipeline Info ────────────────────────────────────────────────────────────

class TestPipelineInfo:
    def test_info_has_model(self, pipeline):
        info = pipeline.get_pipeline_info()
        assert info["model"] == "CoDynamicsLab/LATCH-Qwen2.5-14B"

    def test_info_has_quant_type(self, pipeline):
        info = pipeline.get_pipeline_info()
        assert info["quant_type"] == "Q4_K_M"

    def test_info_has_delta_threshold(self, pipeline):
        info = pipeline.get_pipeline_info()
        assert "delta_threshold" in info

    def test_info_has_vram_estimate(self, pipeline):
        info = pipeline.get_pipeline_info()
        assert "estimated_vram_gb" in info

    def test_info_fits_8gb_vram_for_q4(self, pipeline):
        info = pipeline.get_pipeline_info()
        assert info["fits_8gb_vram"] is True


# ─── Full Mock Run ────────────────────────────────────────────────────────────

class TestPipelineRun:
    def test_run_returns_success(self, pipeline):
        results = pipeline.run()
        assert results["success"] is True

    def test_run_produces_report_file(self, pipeline):
        results = pipeline.run()
        assert Path(results["report_path"]).exists()

    def test_run_produces_results_json(self, pipeline):
        results = pipeline.run()
        assert Path(results["results_path"]).exists()

    def test_run_produces_gguf_file(self, pipeline):
        results = pipeline.run()
        assert Path(results["gguf_path"]).exists()

    def test_run_gguf_has_valid_magic(self, pipeline):
        results = pipeline.run()
        gguf_path = Path(results["gguf_path"])
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
        assert magic == GGUF_MAGIC

    def test_run_passes_quality_gate(self, pipeline):
        results = pipeline.run()
        assert results["passes_quality_gate"] is True

    def test_run_results_json_is_valid(self, pipeline):
        results = pipeline.run()
        data = json.loads(Path(results["results_path"]).read_text())
        assert "quality_gate" in data
        assert data["quality_gate"]["passes"] is True

    def test_quality_gate_fail_raises_when_configured(self, tmp_path):
        """Pipeline must raise QuantizationPipelineError if delta > threshold."""
        pipeline = QuantizationPipeline(
            model_name="CoDynamicsLab/LATCH-Qwen2.5-14B",
            quant_type="Q4_K_M",
            output_dir=str(tmp_path),
            mock_mode=True,
            num_samples=5,
            fail_on_quality_gate=True,
        )
        # Patch the evaluator to return failing results
        failing_results = {
            "model": "CoDynamicsLab/LATCH-Qwen2.5-14B",
            "quantization_type": "Q4_K_M",
            "fp16_perplexity": 10.0,
            "quantized_perplexity": 11.5,
            "delta": 0.15,
            "delta_percent": 15.0,
            "threshold": 0.05,
            "passes": False,
            "num_samples": 5,
            "mock_mode": True,
        }
        mock_evaluator = MagicMock()
        mock_evaluator.run_full_evaluation.return_value = failing_results

        with patch.object(pipeline, "_build_evaluator", return_value=mock_evaluator):
            with pytest.raises(QuantizationPipelineError, match="Quality gate FAILED"):
                pipeline.run()

    def test_quality_gate_fail_no_raise_when_disabled(self, pipeline_no_fail):
        """Pipeline should not raise if fail_on_quality_gate=False."""
        failing_results = {
            "model": "CoDynamicsLab/LATCH-Qwen2.5-14B",
            "quantization_type": "Q4_K_M",
            "fp16_perplexity": 10.0,
            "quantized_perplexity": 11.5,
            "delta": 0.15,
            "delta_percent": 15.0,
            "threshold": 0.05,
            "passes": False,
            "num_samples": 5,
            "mock_mode": True,
        }
        mock_evaluator = MagicMock()
        mock_evaluator.run_full_evaluation.return_value = failing_results

        with patch.object(pipeline_no_fail, "_build_evaluator", return_value=mock_evaluator):
            results = pipeline_no_fail.run()  # should not raise
            assert results["passes_quality_gate"] is False
