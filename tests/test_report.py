"""
Tests for report_generator module.
Covers: Markdown report content, JSON output, pass/fail status, all required sections.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codynamicslab_latch_.report_generator import ReportGenerator


SAMPLE_RESULTS_PASS = {
    "model": "CoDynamicsLab/LATCH-Qwen2.5-14B",
    "quantization_type": "Q4_K_M",
    "fp16_perplexity": 15.2341,
    "quantized_perplexity": 15.5123,
    "delta": 0.01828,
    "delta_percent": 1.828,
    "threshold": 0.05,
    "passes": True,
    "num_samples": 100,
    "mock_mode": True,
}

SAMPLE_RESULTS_FAIL = {
    **SAMPLE_RESULTS_PASS,
    "quantized_perplexity": 17.0000,
    "delta": 0.1161,
    "delta_percent": 11.61,
    "passes": False,
}

SAMPLE_GGUF_INFO = {
    "path": "/outputs/test-q4_k_m.gguf",
    "size_bytes": 7_800_000_000,
    "size_gb": 7.8,
    "valid_gguf_magic": True,
    "mock_mode": True,
}

SAMPLE_VRAM = {
    "weights_gb": 6.62,
    "kv_cache_gb": 0.99,
    "activations_gb": 0.5,
    "total_estimated_gb": 8.11,
    "fits_8gb_vram": False,
}

SAMPLE_INFERENCE = {
    "success": True,
    "mock": True,
    "prompt": "The future of AI is",
    "output": "The future of AI is bright and transformative.",
    "gguf_path": "/outputs/test-q4_k_m.gguf",
}


@pytest.fixture
def reporter(tmp_path):
    return ReportGenerator(str(tmp_path))


@pytest.fixture
def report_md(reporter):
    return reporter.generate_markdown_report(
        SAMPLE_RESULTS_PASS, SAMPLE_GGUF_INFO, SAMPLE_INFERENCE, SAMPLE_VRAM
    )


# ─── Markdown Report Content ──────────────────────────────────────────────────

class TestMarkdownReportContent:
    def test_report_starts_with_title(self, report_md):
        assert report_md.startswith("# LATCH")

    def test_report_contains_pass_badge(self, report_md):
        assert "PASS" in report_md

    def test_report_contains_model_name(self, report_md):
        assert "CoDynamicsLab/LATCH-Qwen2.5-14B" in report_md

    def test_report_contains_fp16_perplexity(self, report_md):
        assert "15.2341" in report_md

    def test_report_contains_q4_perplexity(self, report_md):
        assert "15.5123" in report_md

    def test_report_contains_delta(self, report_md):
        assert "1.828" in report_md

    def test_report_contains_threshold(self, report_md):
        assert "5.0%" in report_md

    def test_report_contains_num_samples(self, report_md):
        assert "100" in report_md

    def test_report_contains_gguf_section(self, report_md):
        assert "GGUF File Info" in report_md

    def test_report_contains_vram_section(self, report_md):
        assert "VRAM" in report_md

    def test_report_contains_inference_section(self, report_md):
        assert "Inference Test" in report_md

    def test_report_contains_quality_gate_section(self, report_md):
        assert "Quality Gate" in report_md

    def test_report_contains_reproduction_section(self, report_md):
        assert "Reproduction" in report_md or "run_quant.py" in report_md

    def test_fail_report_shows_fail(self, reporter):
        md = reporter.generate_markdown_report(SAMPLE_RESULTS_FAIL)
        assert "FAIL" in md

    def test_pass_report_shows_pass(self, report_md):
        assert "PASS" in report_md

    def test_report_contains_inference_output(self, report_md):
        assert "The future of AI is" in report_md


# ─── File Output ─────────────────────────────────────────────────────────────

class TestReportFileOutput:
    def test_save_report_creates_markdown_file(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_PASS)
        assert paths["report"].exists()

    def test_save_report_creates_json_file(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_PASS)
        assert paths["results"].exists()

    def test_json_file_is_valid_json(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_PASS)
        content = paths["results"].read_text()
        data = json.loads(content)  # should not raise
        assert isinstance(data, dict)

    def test_json_contains_perplexity(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_PASS)
        data = json.loads(paths["results"].read_text())
        assert "perplexity" in data

    def test_json_contains_quality_gate(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_PASS)
        data = json.loads(paths["results"].read_text())
        assert "quality_gate" in data

    def test_json_quality_gate_passes_true(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_PASS)
        data = json.loads(paths["results"].read_text())
        assert data["quality_gate"]["passes"] is True

    def test_json_quality_gate_passes_false(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_FAIL)
        data = json.loads(paths["results"].read_text())
        assert data["quality_gate"]["passes"] is False

    def test_json_run_metadata_present(self, reporter, tmp_path):
        paths = reporter.save_report(SAMPLE_RESULTS_PASS)
        data = json.loads(paths["results"].read_text())
        assert "run_metadata" in data
        assert "timestamp" in data["run_metadata"]
