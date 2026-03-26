"""
Run history tracker.

Appends each quantization pipeline run to a JSONL log file, generates
ASCII sparklines showing the delta % trend over time, and formats a
Markdown trend table for inclusion in benchmark reports.
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HISTORY_FILE = os.getenv("HISTORY_FILE", "outputs/run_history.jsonl")
MAX_SPARKLINE_WIDTH = int(os.getenv("SPARKLINE_WIDTH", "20"))


class RunHistoryTracker:
    """Records and analyses quantization pipeline run history from a JSONL file."""

    # Unicode block elements for sparkline rendering (space → full block)
    _BLOCKS = " ▁▂▃▄▅▆▇█"

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def append_run(self, results: Dict[str, Any]) -> None:
        """Append a pipeline result dict to the JSONL history file."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": results.get("model", "unknown"),
            "quant_type": results.get("quantization_type", results.get("quant_type", "unknown")),
            "fp16_perplexity": results.get("fp16_perplexity"),
            "quantized_perplexity": results.get("quantized_perplexity"),
            "delta": results.get("delta"),
            "delta_percent": results.get("delta_percent"),
            "passes": results.get("passes"),
            "mock_mode": results.get("mock_mode", True),
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Run appended to history: {self.history_file}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_history(self) -> List[Dict[str, Any]]:
        """Return all records from the history file as a list of dicts."""
        if not self.history_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping malformed line {lineno} in history: {exc}")
        return records

    def get_delta_series(self) -> List[float]:
        """Return time-ordered list of delta_percent values from all runs."""
        return [
            r["delta_percent"]
            for r in self.load_history()
            if r.get("delta_percent") is not None
        ]

    # ── Analysis ──────────────────────────────────────────────────────────────

    def sparkline(
        self,
        values: Optional[List[float]] = None,
        width: int = MAX_SPARKLINE_WIDTH,
    ) -> str:
        """
        Generate a Unicode sparkline for the delta_percent series.

        Uses the last ``width`` values.  Blocks range from space (lowest)
        to full block (highest).  Returns ``"(no history)"`` when empty.
        """
        if values is None:
            values = self.get_delta_series()
        if not values:
            return "(no history)"

        display = values[-width:]
        min_v = min(display)
        max_v = max(display)
        span = max_v - min_v if max_v != min_v else 1.0
        n_blocks = len(self._BLOCKS) - 1

        chars = [
            self._BLOCKS[round((v - min_v) / span * n_blocks)]
            for v in display
        ]
        return "".join(chars)

    def summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics over all recorded runs."""
        records = self.load_history()
        if not records:
            return {"total_runs": 0}

        deltas = [r["delta_percent"] for r in records if r.get("delta_percent") is not None]
        pass_count = sum(1 for r in records if r.get("passes") is True)

        stats: Dict[str, Any] = {
            "total_runs": len(records),
            "pass_count": pass_count,
            "fail_count": len(records) - pass_count,
            "pass_rate_percent": round(pass_count / len(records) * 100, 1),
        }
        if deltas:
            stats["avg_delta_percent"] = round(sum(deltas) / len(deltas), 4)
            stats["min_delta_percent"] = round(min(deltas), 4)
            stats["max_delta_percent"] = round(max(deltas), 4)
        else:
            stats["avg_delta_percent"] = None
            stats["min_delta_percent"] = None
            stats["max_delta_percent"] = None

        last = records[-1]
        stats["latest_passes"] = last.get("passes")
        stats["latest_delta_percent"] = last.get("delta_percent")
        stats["latest_timestamp"] = last.get("timestamp", "")
        return stats

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_trend_table(self, max_rows: int = 10) -> str:
        """Format the most recent runs as a Markdown table."""
        records = self.load_history()
        if not records:
            return "_No run history found._"

        recent = records[-max_rows:]
        header = "| # | Timestamp (UTC) | Model | Quant | Delta % | Passes |"
        sep    = "|---|-----------------|-------|-------|---------|--------|"
        rows = [header, sep]
        for i, r in enumerate(recent, start=len(records) - len(recent) + 1):
            ts = (r.get("timestamp") or "")[:19].replace("T", " ")
            model = (r.get("model") or "?")
            if "/" in model:
                model = model.split("/")[-1]
            qt = r.get("quant_type", "?")
            dp = f"{r['delta_percent']:.3f}%" if r.get("delta_percent") is not None else "N/A"
            p = "✅" if r.get("passes") else "❌"
            rows.append(f"| {i} | {ts} | {model} | {qt} | {dp} | {p} |")
        return "\n".join(rows)

    def format_stats_block(self) -> str:
        """Format summary stats and sparkline as a Markdown code block."""
        stats = self.summary_stats()
        if stats["total_runs"] == 0:
            return "_No run history._"

        sparkline = self.sparkline()
        lines = [
            "```",
            f"Total runs     : {stats['total_runs']}",
            f"Pass / Fail    : {stats['pass_count']} / {stats['fail_count']}",
            f"Pass rate      : {stats['pass_rate_percent']}%",
        ]
        if stats.get("avg_delta_percent") is not None:
            lines += [
                f"Avg delta      : {stats['avg_delta_percent']:.4f}%",
                f"Min / Max delta: {stats['min_delta_percent']:.4f}% / {stats['max_delta_percent']:.4f}%",
            ]
        lines += [
            f"Delta trend    : {sparkline}",
            "```",
        ]
        return "\n".join(lines)
