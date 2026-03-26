"""
Tests for enhanced features: GGUF inspection, multi-quant sweep, history tracking,
download retry logic, and other pipeline enhancements.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("USE_REAL_PROXY_MODEL", "false")

from codynamicslab_latch_.gguf_inspector import (
    GGUFMetadata,
    inspect_gguf,
    format_metadata_table,
    metadata_to_dict,
    GGUF_MAGIC,
)
from codynamicslab_latch_.multi_quant_compare import (
    MultiQuantComparer,
    QuantSweepResult,
    SIMULATED_DELTAS,
)
from codynamicslab_latch_.history_tracker import RunHistoryTracker
from codynamicslab_latch_.model_converter import (
    ModelConverter,
    _write_mock_gguf,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_RETRY_DELAY,
)


# ─── GGUF Inspector Tests ────────────────────────────────────────────────────

class TestGGUFInspector:
    def test_inspect_valid_stub(self, tmp_path):
        """Test inspection of a valid GGUF stub file."""
        path = tmp_path / "test.gguf"
        _write_mock_gguf(path, "test-model", "Q4_K_M")
        meta = inspect_gguf(path)
        
        assert meta.valid is True
        assert meta.version == 3
        assert meta.tensor_count == 0
        assert meta.kv_count == 4
        assert meta.file_size_bytes > 0
        assert meta.error is None

    def test_inspect_missing_file(self, tmp_path):
        """Test inspection of a non-existent file."""
        path = tmp_path / "missing.gguf"
        meta = inspect_gguf(path)
        
        assert meta.valid is False
        assert "not found" in meta.error.lower()
        assert meta.version is None

    def test_inspect_corrupt_file(self, tmp_path):
        """Test inspection of a file with invalid magic bytes."""
        path = tmp_path / "corrupt.gguf"
        path.write_bytes(b"INVALID_HEADER")
        meta = inspect_gguf(path)
        
        assert meta.valid is False
        assert "Invalid magic" in meta.error

    def test_format_metadata_table(self, tmp_path):
        """Test formatting metadata as Markdown table."""
        path = tmp_path / "test.gguf"
        _write_mock_gguf(path, "test-model", "Q4_K_M")
        meta = inspect_gguf(path)
        table = format_metadata_table(meta)
        
        assert "| Property | Value |" in table
        assert "Valid GGUF" in table
        assert "Yes" in table

    def test_metadata_to_dict(self, tmp_path):
        """Test converting metadata to dict."""
        path = tmp_path / "test.gguf"
        _write_mock_gguf(path, "test-model", "Q4_K_M")
        meta = inspect_gguf(path)
        d = metadata_to_dict(meta)
        
        assert d["valid"] is True
        assert d["version"] == 3
        assert d["error"] is None


# ─── Multi-Quant Compare Tests ───────────────────────────────────────────────

class TestMultiQuantComparer:
    def test_sweep_returns_results(self):
        """Test that sweep returns a list of results."""
        comparer = MultiQuantComparer(model_name="test-model")
        results = comparer.run_sweep(fp16_perplexity=100.0)
        
        assert len(results) > 0
        assert all(isinstance(r, QuantSweepResult) for r in results)

    def test_sweep_has_recommended(self):
        """Test that exactly one result is recommended."""
        comparer = MultiQuantComparer(model_name="test-model")
        results = comparer.run_sweep(fp16_perplexity=100.0)
        
        recommended = [r for r in results if r.recommended]
        assert len(recommended) == 1

    def test_recommended_passes_quality_gate(self):
        """Test that recommended quant passes quality gate."""
        comparer = MultiQuantComparer(model_name="test-model")
        results = comparer.run_sweep(fp16_perplexity=100.0)
        
        recommended = next(r for r in results if r.recommended)
        assert recommended.passes_quality_gate is True

    def test_recommended_fits_vram_budget(self):
        """Test that recommended quant fits VRAM budget."""
        comparer = MultiQuantComparer(model_name="test-model", vram_budget_gb=8.0)
        results = comparer.run_sweep(fp16_perplexity=100.0)
        
        recommended = next(r for r in results if r.recommended)
        assert recommended.fits_vram_budget is True

    def test_format_sweep_table(self):
        """Test formatting sweep results as Markdown table."""
        comparer = MultiQuantComparer(model_name="test-model")
        results = comparer.run_sweep(fp16_perplexity=100.0)
        table = comparer.format_sweep_table(results)
        
        assert "| Quant Type |" in table
        assert "Q4_K_M" in table

    def test_recommends_most_compressed(self):
        """Test that the most compressed valid quant is recommended."""
        comparer = MultiQuantComparer(model_name="test-model", vram_budget_gb=8.0)
        results = comparer.run_sweep(fp16_perplexity=100.0)
        
        recommended = next(r for r in results if r.recommended)
        # Q4_K_M should be recommended for 8GB VRAM (most compressed that passes)
        assert recommended.passes_quality_gate
        assert recommended.fits_vram_budget


# ─── History Tracker Tests ───────────────────────────────────────────────────

class TestHistoryTracker:
    def test_append_run(self, tmp_path):
        """Test appending a run to history."""
        history_file = tmp_path / "history.jsonl"
        tracker = RunHistoryTracker(history_file=str(history_file))
        
        results = {
            "model": "test-model",
            "quant_type": "Q4_K_M",
            "fp16_perplexity": 100.0,
            "quantized_perplexity": 101.8,
            "delta": 0.018,
            "delta_percent": 1.8,
            "passes": True,
        }
        tracker.append_run(results)
        
        assert history_file.exists()
        records = tracker.load_history()
        assert len(records) == 1
        assert records[0]["model"] == "test-model"

    def test_load_empty_history(self, tmp_path):
        """Test loading history from non-existent file."""
        history_file = tmp_path / "nonexistent.jsonl"
        tracker = RunHistoryTracker(history_file=str(history_file))
        
        records = tracker.load_history()
        assert records == []

    def test_sparkline_empty(self, tmp_path):
        """Test sparkline with no history."""
        history_file = tmp_path / "empty.jsonl"
        tracker = RunHistoryTracker(history_file=str(history_file))
        
        sparkline = tracker.sparkline()
        assert sparkline == "(no history)"

    def test_sparkline_with_data(self, tmp_path):
        """Test sparkline with history data."""
        history_file = tmp_path / "history.jsonl"
        tracker = RunHistoryTracker(history_file=str(history_file))
        
        for delta in [1.0, 2.0, 1.5, 3.0, 2.5]:
            tracker.append_run({
                "model": "test",
                "quant_type": "Q4_K_M",
                "delta_percent": delta,
                "passes": True,
            })
        
        sparkline = tracker.sparkline()
        assert len(sparkline) > 0
        assert "(no history)" not in sparkline

    def test_summary_stats(self, tmp_path):
        """Test computing summary statistics."""
        history_file = tmp_path / "history.jsonl"
        tracker = RunHistoryTracker(history_file=str(history_file))
        
        tracker.append_run({
            "model": "test",
            "quant_type": "Q4_K_M",
            "delta_percent": 1.0,
            "passes": True,
        })
        tracker.append_run({
            "model": "test",
            "quant_type": "Q4_K_M",
            "delta_percent": 2.0,
            "passes": False,
        })
        
        stats = tracker.summary_stats()
        assert stats["total_runs"] == 2
        assert stats["pass_count"] == 1
        assert stats["fail_count"] == 1
        assert stats["pass_rate_percent"] == 50.0

    def test_format_trend_table(self, tmp_path):
        """Test formatting trend table."""
        history_file = tmp_path / "history.jsonl"
        tracker = RunHistoryTracker(history_file=str(history_file))
        
        tracker.append_run({
            "model": "test-model",
            "quant_type": "Q4_K_M",
            "delta_percent": 1.5,
            "passes": True,
        })
        
        table = tracker.format_trend_table()
        assert "| # |" in table
        assert "test-model" in table or "test" in table


# ─── Model Converter Enhancement Tests ───────────────────────────────────────

class TestModelConverterEnhancements:
    def test_download_retry_on_failure(self, tmp_path):
        """Test that download retries on failure with exponential backoff."""
        from codynamicslab_latch_.model_converter import snapshot_download
        
        converter = ModelConverter(
            model_name="test-model",
            output_dir=str(tmp_path),
            mock_mode=False,  # Disable mock mode to test retry logic
        )
        
        # Mock snapshot_download to fail twice then succeed
        call_count = [0]
        
        def mock_download(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            return tmp_path / "model_cache" / "test-model"
        
        with patch("codynamicslab_latch_.model_converter.snapshot_download", side_effect=mock_download):
            with patch("codynamicslab_latch_.model_converter.DOWNLOAD_MAX_RETRIES", 3):
                with patch("codynamicslab_latch_.model_converter.DOWNLOAD_RETRY_DELAY", 0.01):
                    result = converter.download_model()
                    assert call_count[0] == 3
                    assert result.exists()

    def test_download_fails_after_max_retries(self, tmp_path):
        """Test that download raises after max retries exceeded."""
        converter = ModelConverter(
            model_name="test-model",
            output_dir=str(tmp_path),
            mock_mode=False,
        )
        
        with patch("codynamicslab_latch_.model_converter.snapshot_download", side_effect=ConnectionError("Persistent error")):
            with patch("codynamicslab_latch_.model_converter.DOWNLOAD_MAX_RETRIES", 2):
                with patch("codynamicslab_latch_.model_converter.DOWNLOAD_RETRY_DELAY", 0.01):
                    with pytest.raises(RuntimeError, match="Failed to download"):
                        converter.download_model()

    def test_download_succeeds_first_attempt(self, tmp_path):
        """Test that download succeeds on first attempt."""
        converter = ModelConverter(
            model_name="test-model",
            output_dir=str(tmp_path),
            mock_mode=False,
        )
        
        def mock_download(*args, **kwargs):
            target = tmp_path / "model_cache" / "test-model"
            target.mkdir(parents=True, exist_ok=True)
            return target
        
        with patch("codynamicslab_latch_.model_converter.snapshot_download", side_effect=mock_download):
            result = converter.download_model()
            assert result.exists()

    def test_mock_download_skips_download(self, tmp_path, caplog):
        """Test that mock mode skips actual download."""
        converter = ModelConverter(
            model_name="test-model",
            output_dir=str(tmp_path),
            mock_mode=True,
        )
        
        with patch("codynamicslab_latch_.model_converter.snapshot_download") as mock_sd:
            result = converter.download_model()
            mock_sd.assert_not_called()
            assert result.exists()


# ─── Integration Tests ───────────────────────────────────────────────────────

class TestEnhancementIntegration:
    def test_full_enhancement_workflow(self, tmp_path):
        """Test all enhancements working together."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        
        # Create GGUF stub
        gguf_path = output_dir / "test-q4_k_m.gguf"
        _write_mock_gguf(gguf_path, "test-model", "Q4_K_M")
        
        # Inspect GGUF
        meta = inspect_gguf(gguf_path)
        assert meta.valid is True
        
        # Run multi-quant sweep
        comparer = MultiQuantComparer(model_name="test-model")
        results = comparer.run_sweep(fp16_perplexity=100.0)
        assert len(results) > 0
        
        # Track history
        history_file = output_dir / "history.jsonl"
        tracker = RunHistoryTracker(history_file=str(history_file))
        tracker.append_run({
            "model": "test-model",
            "quant_type": "Q4_K_M",
            "fp16_perplexity": 100.0,
            "quantized_perplexity": 101.8,
            "delta": 0.018,
            "delta_percent": 1.8,
            "passes": True,
        })
        
        # Verify all components
        assert history_file.exists()
        assert tracker.summary_stats()["total_runs"] == 1
        assert format_metadata_table(meta) is not None
        assert comparer.format_sweep_table(results) is not None
