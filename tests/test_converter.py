"""
Tests for model_converter module.
Covers: GGUF file creation, verification, VRAM estimation, inference test mock.
"""

import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("MOCK_MODE", "true")

from codynamicslab_latch_.model_converter import (
    ModelConverter,
    GGUF_MAGIC,
    GGUF_VERSION,
    QUANT_TYPES,
    is_llama_cpp_available,
    _write_mock_gguf,
)


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


@pytest.fixture
def converter(tmp_output):
    return ModelConverter(
        model_name="CoDynamicsLab/LATCH-Qwen2.5-14B",
        output_dir=str(tmp_output),
        mock_mode=True,
    )


# ─── GGUF File Basics ─────────────────────────────────────────────────────────

class TestGGUFCreation:
    def test_mock_gguf_has_correct_magic(self, tmp_output):
        path = tmp_output / "test.gguf"
        _write_mock_gguf(path, "test-model", "Q4_K_M")
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == GGUF_MAGIC

    def test_mock_gguf_has_correct_version(self, tmp_output):
        path = tmp_output / "test.gguf"
        _write_mock_gguf(path, "test-model", "Q4_K_M")
        with open(path, "rb") as f:
            f.read(4)  # magic
            version = struct.unpack("<I", f.read(4))[0]
        assert version == GGUF_VERSION

    def test_mock_gguf_file_exists(self, converter, tmp_output):
        f16_path = converter.convert_to_f16_gguf(tmp_output)
        assert f16_path.exists()

    def test_quantize_produces_gguf_file(self, converter, tmp_output):
        f16_path = tmp_output / "dummy-f16.gguf"
        _write_mock_gguf(f16_path, "test-model", "F16")
        quant_path = converter.quantize(f16_path, "Q4_K_M")
        assert quant_path.exists()

    def test_quantize_invalid_type_raises(self, converter, tmp_output):
        f16_path = tmp_output / "dummy.gguf"
        f16_path.touch()
        with pytest.raises(ValueError, match="Unknown quantization type"):
            converter.quantize(f16_path, "INVALID_TYPE")

    def test_verify_valid_gguf(self, converter, tmp_output):
        path = tmp_output / "valid.gguf"
        _write_mock_gguf(path, "test", "Q4_K_M")
        ok, msg = converter.verify_gguf(path)
        assert ok is True
        assert "Valid GGUF" in msg

    def test_verify_missing_file(self, converter, tmp_output):
        path = tmp_output / "nonexistent.gguf"
        ok, msg = converter.verify_gguf(path)
        assert ok is False
        assert "not found" in msg.lower()

    def test_verify_corrupt_file(self, converter, tmp_output):
        path = tmp_output / "corrupt.gguf"
        path.write_bytes(b"INVALID_HEADER_DATA")
        ok, msg = converter.verify_gguf(path)
        assert ok is False


# ─── File Info ────────────────────────────────────────────────────────────────

class TestFileInfo:
    def test_file_info_has_size(self, converter, tmp_output):
        path = tmp_output / "test.gguf"
        _write_mock_gguf(path, "test", "Q4_K_M")
        info = converter.get_file_info(path)
        assert "size_bytes" in info
        assert info["size_bytes"] > 0

    def test_file_info_valid_gguf_magic(self, converter, tmp_output):
        path = tmp_output / "test.gguf"
        _write_mock_gguf(path, "test", "Q4_K_M")
        info = converter.get_file_info(path)
        assert info["valid_gguf_magic"] is True

    def test_file_info_missing_returns_error(self, converter, tmp_output):
        info = converter.get_file_info(tmp_output / "missing.gguf")
        assert "error" in info


# ─── VRAM Estimation ──────────────────────────────────────────────────────────

class TestVRAMEstimation:
    def test_q4_k_m_fits_8gb(self, converter):
        vram = converter.estimate_vram_requirement("Q4_K_M")
        # Q4_K_M for 14B model should fit in 8GB
        assert vram["fits_8gb_vram"] is True

    def test_f16_does_not_fit_8gb(self, converter):
        vram = converter.estimate_vram_requirement("F16")
        # FP16 for 14B model (~28GB) cannot fit in 8GB
        assert vram["fits_8gb_vram"] is False

    def test_vram_has_required_keys(self, converter):
        vram = converter.estimate_vram_requirement("Q4_K_M")
        for key in ["weights_gb", "kv_cache_gb", "activations_gb", "total_estimated_gb"]:
            assert key in vram

    def test_q4_size_less_than_f16(self, converter):
        q4_vram = converter.estimate_vram_requirement("Q4_K_M")
        f16_vram = converter.estimate_vram_requirement("F16")
        assert q4_vram["weights_gb"] < f16_vram["weights_gb"]

    def test_total_is_sum_of_components(self, converter):
        vram = converter.estimate_vram_requirement("Q4_K_M")
        expected = vram["weights_gb"] + vram["kv_cache_gb"] + vram["activations_gb"]
        assert abs(vram["total_estimated_gb"] - expected) < 0.01


# ─── Inference Test ───────────────────────────────────────────────────────────

class TestInferenceTest:
    def test_mock_inference_succeeds(self, converter, tmp_output):
        path = tmp_output / "test.gguf"
        _write_mock_gguf(path, "test", "Q4_K_M")
        result = converter.run_inference_test(path)
        assert result["success"] is True

    def test_mock_inference_has_output(self, converter, tmp_output):
        path = tmp_output / "test.gguf"
        _write_mock_gguf(path, "test", "Q4_K_M")
        result = converter.run_inference_test(path)
        assert "output" in result
        assert len(result["output"]) > 0

    def test_mock_inference_is_flagged(self, converter, tmp_output):
        path = tmp_output / "test.gguf"
        _write_mock_gguf(path, "test", "Q4_K_M")
        result = converter.run_inference_test(path)
        assert result.get("mock") is True


# ─── Quant Types ─────────────────────────────────────────────────────────────

class TestQuantTypes:
    def test_q4_k_m_in_registry(self):
        assert "Q4_K_M" in QUANT_TYPES

    def test_f16_in_registry(self):
        assert "F16" in QUANT_TYPES

    def test_q4_size_factor_correct(self):
        # Q4_K_M should use a fraction of F16 size (calibrated to ~7.2 GB for 14B)
        assert QUANT_TYPES["Q4_K_M"]["size_factor"] < QUANT_TYPES["F16"]["size_factor"]

    def test_llama_cpp_available_returns_bool(self):
        result = is_llama_cpp_available()
        assert isinstance(result, bool)
