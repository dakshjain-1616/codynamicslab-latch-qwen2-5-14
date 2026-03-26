"""
Tests for history_tracker module.
Covers: append, load, sparkline, summary stats, Markdown formatting.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codynamicslab_latch_.history_tracker import RunHistoryTracker


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RUN_PASS = {
    "model": "CoDynamicsLab/LATCH-Qwen2.5-14B",
    "quantization_type": "Q4_K_M",
    "fp16_perplexity": 100.0,
    "quantized_perplexity": 101.8,
    "delta": 0.018,
    "delta_percent": 1.8,
    "passes": True,
    "mock_mode": True,
}

SAMPLE_RUN_FAIL = {
    **SAMPLE_RUN_PASS,
    "quantized_perplexity": 115.0,
    "delta": 0.15,
    "delta_percent": 15.0,
    "passes": False,
}


@pytest.fixture
def tracker(tmp_path):
    return RunHistoryTracker(history_file=str(tmp_path / "run_history.jsonl"))


@pytest.fixture
def tracker_with_runs(tracker):
    """Tracker pre-populated with three runs."""
    tracker.append_run(SAMPLE_RUN_PASS)
    tracker.append_run(SAMPLE_RUN_FAIL)
    tracker.append_run(SAMPLE_RUN_PASS)
    return tracker


# ── Append & Load ─────────────────────────────────────────────────────────────

class TestAppendLoad:
    def test_empty_tracker_returns_no_history(self, tracker):
        assert tracker.load_history() == []

    def test_append_creates_file(self, tracker):
        tracker.append_run(SAMPLE_RUN_PASS)
        assert tracker.history_file.exists()

    def test_append_one_run_loads_one_record(self, tracker):
        tracker.append_run(SAMPLE_RUN_PASS)
        records = tracker.load_history()
        assert len(records) == 1

    def test_append_three_runs_loads_three_records(self, tracker_with_runs):
        records = tracker_with_runs.load_history()
        assert len(records) == 3

    def test_record_has_timestamp(self, tracker):
        tracker.append_run(SAMPLE_RUN_PASS)
        record = tracker.load_history()[0]
        assert "timestamp" in record
        assert len(record["timestamp"]) > 0

    def test_record_has_passes_flag(self, tracker):
        tracker.append_run(SAMPLE_RUN_PASS)
        record = tracker.load_history()[0]
        assert record["passes"] is True

    def test_record_has_delta_percent(self, tracker):
        tracker.append_run(SAMPLE_RUN_PASS)
        record = tracker.load_history()[0]
        assert record["delta_percent"] == pytest.approx(1.8)

    def test_jsonl_file_each_line_valid_json(self, tracker_with_runs):
        lines = tracker_with_runs.history_file.read_text().strip().splitlines()
        for line in lines:
            json.loads(line)  # should not raise


# ── Sparkline ─────────────────────────────────────────────────────────────────

class TestSparkline:
    def test_empty_returns_no_history_string(self, tracker):
        assert tracker.sparkline() == "(no history)"

    def test_sparkline_non_empty_with_runs(self, tracker_with_runs):
        s = tracker_with_runs.sparkline()
        assert s != "(no history)"
        assert len(s) > 0

    def test_sparkline_width_respected(self, tracker_with_runs):
        s = tracker_with_runs.sparkline(width=2)
        assert len(s) <= 2

    def test_custom_values_sparkline(self, tracker):
        s = tracker.sparkline(values=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(s) == 5


# ── Summary Stats ──────────────────────────────────────────────────────────────

class TestSummaryStats:
    def test_empty_tracker_has_zero_runs(self, tracker):
        stats = tracker.summary_stats()
        assert stats["total_runs"] == 0

    def test_total_runs_correct(self, tracker_with_runs):
        stats = tracker_with_runs.summary_stats()
        assert stats["total_runs"] == 3

    def test_pass_count_correct(self, tracker_with_runs):
        stats = tracker_with_runs.summary_stats()
        assert stats["pass_count"] == 2

    def test_fail_count_correct(self, tracker_with_runs):
        stats = tracker_with_runs.summary_stats()
        assert stats["fail_count"] == 1

    def test_pass_rate_correct(self, tracker_with_runs):
        stats = tracker_with_runs.summary_stats()
        assert stats["pass_rate_percent"] == pytest.approx(66.7, abs=0.1)

    def test_avg_delta_present(self, tracker_with_runs):
        stats = tracker_with_runs.summary_stats()
        assert stats["avg_delta_percent"] is not None

    def test_latest_passes_reflects_last_run(self, tracker_with_runs):
        # Last appended run was SAMPLE_RUN_PASS → passes=True
        stats = tracker_with_runs.summary_stats()
        assert stats["latest_passes"] is True


# ── Formatting ────────────────────────────────────────────────────────────────

class TestFormatting:
    def test_empty_trend_table_returns_no_history(self, tracker):
        assert "No run history" in tracker.format_trend_table()

    def test_trend_table_has_headers(self, tracker_with_runs):
        table = tracker_with_runs.format_trend_table()
        assert "Timestamp" in table
        assert "Delta" in table

    def test_trend_table_is_markdown(self, tracker_with_runs):
        table = tracker_with_runs.format_trend_table()
        assert "|" in table and "---" in table

    def test_stats_block_contains_total_runs(self, tracker_with_runs):
        block = tracker_with_runs.format_stats_block()
        assert "3" in block

    def test_stats_block_has_code_fence(self, tracker_with_runs):
        block = tracker_with_runs.format_stats_block()
        assert "```" in block
