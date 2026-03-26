"""
Quantization benchmark report generator.
Produces a Markdown report comparing FP16 vs Q4_K_M with pass/fail status.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
PERPLEXITY_DELTA_THRESHOLD = float(os.getenv("PERPLEXITY_DELTA_THRESHOLD", "0.05"))
REPORT_FILENAME = os.getenv("REPORT_FILENAME", "quantization_report.md")
RESULTS_FILENAME = os.getenv("RESULTS_FILENAME", "benchmark_results.json")


class ReportGenerator:
    """Generates benchmark reports comparing FP16 and quantized model quality."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _status_badge(self, passes: bool) -> str:
        if passes:
            return "![PASS](https://img.shields.io/badge/quality--gate-PASS-brightgreen)"
        return "![FAIL](https://img.shields.io/badge/quality--gate-FAIL-red)"

    def _delta_bar(self, delta: float, threshold: float = PERPLEXITY_DELTA_THRESHOLD) -> str:
        """ASCII progress bar for delta vs threshold."""
        ratio = min(delta / threshold, 1.0) if threshold > 0 else 1.0
        filled = int(ratio * 20)
        bar = "█" * filled + "░" * (20 - filled)
        return f"`[{bar}]` {delta*100:.3f}% / {threshold*100:.1f}% threshold"

    def _format_size(self, size_bytes: int) -> str:
        if size_bytes >= 1e9:
            return f"{size_bytes / 1e9:.2f} GB"
        elif size_bytes >= 1e6:
            return f"{size_bytes / 1e6:.2f} MB"
        return f"{size_bytes:,} bytes"

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        gguf_info: Optional[Dict[str, Any]] = None,
        inference_result: Optional[Dict[str, Any]] = None,
        vram_estimate: Optional[Dict[str, Any]] = None,
        gguf_metadata: Optional[Dict[str, Any]] = None,
        multi_quant_sweep: Optional[str] = None,
        step_timings: Optional[Dict[str, float]] = None,
        history_table: Optional[str] = None,
        history_stats: Optional[str] = None,
    ) -> str:
        """Generate the full Markdown benchmark report."""

        model = results.get("model", "Unknown")
        quant_type = results.get("quantization_type", "Q4_K_M")
        fp16_ppl = results.get("fp16_perplexity")
        q4_ppl = results.get("quantized_perplexity")
        delta = results.get("delta")
        delta_pct = results.get("delta_percent", delta * 100 if delta else None)
        passes = results.get("passes", False)
        threshold = results.get("threshold", PERPLEXITY_DELTA_THRESHOLD)
        num_samples = results.get("num_samples", 100)
        mock_mode = results.get("mock_mode", False)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        status_badge = self._status_badge(passes)
        pass_fail = "✅ PASS" if passes else "❌ FAIL"
        mock_note = (
            "\n> **Note:** Results generated in mock/dry-run mode using a proxy model. "
            "Run with a real llama.cpp installation and `MOCK_MODE=false` for production results.\n"
            if mock_mode
            else ""
        )

        # Confidence interval fields (optional)
        fp16_ci_lo = results.get("fp16_ci_lower")
        fp16_ci_hi = results.get("fp16_ci_upper")
        q4_ci_lo = results.get("quantized_ci_lower")
        q4_ci_hi = results.get("quantized_ci_upper")
        fp16_std = results.get("fp16_std_dev")
        q4_std = results.get("quantized_std_dev")

        lines: List[str] = [
            f"# LATCH-Qwen2.5-14B Quantization Benchmark Report",
            f"",
            f"{status_badge}",
            f"",
            mock_note,
            f"**Generated:** {timestamp}  ",
            f"**Model:** `{model}`  ",
            f"**Quantization:** `{quant_type}`  ",
            f"**Quality Gate:** Delta must be ≤ {threshold*100:.1f}% — {pass_fail}  ",
            f"",
            f"---",
            f"",
            f"## Perplexity Comparison",
            f"",
            f"| Metric | FP16 | {quant_type} |",
            f"|--------|------|---------|",
            f"| Perplexity | `{fp16_ppl:.4f}` | `{q4_ppl:.4f}` |" if (fp16_ppl and q4_ppl) else "| Perplexity | N/A | N/A |",
        ]
        if fp16_std is not None and q4_std is not None:
            lines.append(f"| Std Dev | `±{fp16_std:.4f}` | `±{q4_std:.4f}` |")
        if all(v is not None for v in [fp16_ci_lo, fp16_ci_hi, q4_ci_lo, q4_ci_hi]):
            lines.append(f"| 95% CI | `[{fp16_ci_lo:.4f}, {fp16_ci_hi:.4f}]` | `[{q4_ci_lo:.4f}, {q4_ci_hi:.4f}]` |")
        lines += [
            f"| Absolute Delta | — | `{abs(q4_ppl - fp16_ppl):.4f}` |" if (fp16_ppl and q4_ppl) else "| Absolute Delta | — | N/A |",
            f"| Relative Delta | — | `{delta_pct:.3f}%` |" if delta_pct is not None else "| Relative Delta | — | N/A |",
            f"| Threshold | — | `{threshold*100:.1f}%` |",
            f"| Test Samples | `{num_samples}` | `{num_samples}` |",
            f"| Quality Gate | — | **{pass_fail}** |",
            f"",
        ]

        if delta is not None:
            lines += [
                f"### Delta Progress",
                f"",
                f"{self._delta_bar(delta, threshold)}",
                f"",
            ]

        # GGUF file info
        lines += [
            f"---",
            f"",
            f"## GGUF File Info",
            f"",
        ]
        if gguf_info:
            lines += [
                f"| Property | Value |",
                f"|----------|-------|",
                f"| File | `{Path(gguf_info.get('path', 'N/A')).name}` |",
                f"| Size | `{self._format_size(gguf_info.get('size_bytes', 0))}` |",
                f"| Valid GGUF | `{'Yes' if gguf_info.get('valid_gguf_magic') else 'No'}` |",
                f"| Mock Stub | `{'Yes' if gguf_info.get('mock_mode') else 'No'}` |",
                f"",
            ]
        else:
            lines += [f"_GGUF file info not available._", f""]

        # VRAM estimate
        if vram_estimate:
            fits = vram_estimate.get("fits_8gb_vram", False)
            lines += [
                f"---",
                f"",
                f"## VRAM Requirements ({quant_type})",
                f"",
                f"| Component | Estimated |",
                f"|-----------|-----------|",
                f"| Model Weights | `{vram_estimate.get('weights_gb', 0):.2f} GB` |",
                f"| KV Cache | `{vram_estimate.get('kv_cache_gb', 0):.2f} GB` |",
                f"| Activations | `{vram_estimate.get('activations_gb', 0):.2f} GB` |",
                f"| **Total** | **`{vram_estimate.get('total_estimated_gb', 0):.2f} GB`** |",
                f"| Fits 8GB VRAM | `{'✅ Yes' if fits else '❌ No (need 12+ GB)'}` |",
                f"",
            ]

        # Inference test result
        if inference_result:
            success = inference_result.get("success", False)
            is_mock = inference_result.get("mock", True)
            lines += [
                f"---",
                f"",
                f"## Inference Test",
                f"",
                f"| Property | Value |",
                f"|----------|-------|",
                f"| Status | `{'✅ Success' if success else '❌ Failed'}` |",
                f"| Mock Run | `{'Yes' if is_mock else 'No'}` |",
                f"| Prompt | `{inference_result.get('prompt', '')}` |",
                f"",
            ]
            if success and inference_result.get("output"):
                lines += [
                    f"**Generated output:**",
                    f"",
                    f"> {inference_result['output'][:300]}",
                    f"",
                ]
            elif not success:
                lines += [
                    f"**Error:** {inference_result.get('error', 'Unknown error')}",
                    f"",
                ]

        # Optional: GGUF deep metadata (from GGUFInspector)
        if gguf_metadata and gguf_metadata.get("valid"):
            lines += [
                f"---",
                f"",
                f"## GGUF Header Metadata",
                f"",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| GGUF Version | `{gguf_metadata.get('version', 'N/A')}` |",
                f"| Tensor Count | `{gguf_metadata.get('tensor_count', 'N/A')}` |",
                f"| KV Pair Count | `{gguf_metadata.get('kv_count', 'N/A')}` |",
                f"| File Size | `{gguf_metadata.get('file_size_gb', 0):.3f} GB` |",
                f"",
            ]

        # Optional: Multi-quant sweep table
        if multi_quant_sweep:
            lines += [
                f"---",
                f"",
                f"## Quantization Sweep (All Types)",
                f"",
                multi_quant_sweep,
                f"",
            ]

        # Optional: Step timings
        if step_timings:
            lines += [
                f"---",
                f"",
                f"## Step Timings",
                f"",
                f"| Step | Duration (s) |",
                f"|------|-------------|",
            ]
            for step, secs in step_timings.items():
                lines.append(f"| {step} | `{secs:.2f}` |")
            total = sum(step_timings.values())
            lines += [f"| **Total** | **`{total:.2f}`** |", f""]

        # Optional: Run history
        if history_stats or history_table:
            lines += [f"---", f"", f"## Run History", f""]
            if history_stats:
                lines += [history_stats, f""]
            if history_table:
                lines += [history_table, f""]

        # Quality gate summary
        lines += [
            f"---",
            f"",
            f"## Quality Gate Summary",
            f"",
            f"```",
            f"Quality Gate: Perplexity Delta ≤ {threshold*100:.1f}%",
            f"",
            f"  FP16 Perplexity   : {fp16_ppl:.4f}" if fp16_ppl else "  FP16 Perplexity   : N/A",
            f"  Q4_K_M Perplexity : {q4_ppl:.4f}" if q4_ppl else f"  {quant_type} Perplexity : N/A",
            f"  Delta             : {delta_pct:.4f}%" if delta_pct is not None else "  Delta             : N/A",
            f"  Threshold         : {threshold*100:.1f}%",
            f"  Result            : {'PASS ✓' if passes else 'FAIL ✗ — Build rejected'}",
            f"```",
            f"",
            f"---",
            f"",
            f"## Reproduction",
            f"",
            f"```bash",
            f"# Install dependencies",
            f"pip install -r requirements.txt",
            f"",
            f"# Run full quantization pipeline",
            f"python run_quant.py --model {model} --quant-type {quant_type}",
            f"",
            f"# Run demo (mock mode, no GPU required)",
            f"python demo.py",
            f"",
            f"# Run tests",
            f"python -m pytest tests/ -v",
            f"```",
            f"",
            f"---",
            f"",
            f"*Report generated by [LATCH-Qwen2.5-14B-GGUF](https://github.com/dakshjain-1616/LATCH-Qwen2.5-14B-GGUF) "
            f"quantization pipeline.*",
            f"",
        ]

        return "\n".join(lines)

    def save_report(
        self,
        results: Dict[str, Any],
        gguf_info: Optional[Dict[str, Any]] = None,
        inference_result: Optional[Dict[str, Any]] = None,
        vram_estimate: Optional[Dict[str, Any]] = None,
        report_filename: Optional[str] = None,
        results_filename: Optional[str] = None,
        gguf_metadata: Optional[Dict[str, Any]] = None,
        multi_quant_sweep: Optional[str] = None,
        step_timings: Optional[Dict[str, float]] = None,
        history_table: Optional[str] = None,
        history_stats: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Save the markdown report and JSON results to the output directory."""
        report_path = self.output_dir / (report_filename or REPORT_FILENAME)
        results_path = self.output_dir / (results_filename or RESULTS_FILENAME)

        # Generate markdown
        markdown = self.generate_markdown_report(
            results, gguf_info, inference_result, vram_estimate,
            gguf_metadata=gguf_metadata,
            multi_quant_sweep=multi_quant_sweep,
            step_timings=step_timings,
            history_table=history_table,
            history_stats=history_stats,
        )
        report_path.write_text(markdown, encoding="utf-8")
        logger.info(f"Report written: {report_path}")

        # Compile full results JSON
        full_results = {
            "run_metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pipeline_version": "1.1.0",
                "mock_mode": results.get("mock_mode", False),
                "step_timings": step_timings,
            },
            "model": results.get("model"),
            "quantization": {
                "type": results.get("quantization_type", "Q4_K_M"),
                "gguf_info": gguf_info,
            },
            "perplexity": {
                "fp16": results.get("fp16_perplexity"),
                "fp16_std_dev": results.get("fp16_std_dev"),
                "fp16_ci_lower": results.get("fp16_ci_lower"),
                "fp16_ci_upper": results.get("fp16_ci_upper"),
                "quantized": results.get("quantized_perplexity"),
                "quantized_std_dev": results.get("quantized_std_dev"),
                "quantized_ci_lower": results.get("quantized_ci_lower"),
                "quantized_ci_upper": results.get("quantized_ci_upper"),
                "delta": results.get("delta"),
                "delta_percent": results.get("delta_percent"),
                "threshold": results.get("threshold", PERPLEXITY_DELTA_THRESHOLD),
                "num_samples": results.get("num_samples"),
            },
            "gguf_metadata": gguf_metadata,
            "quality_gate": {
                "passes": results.get("passes"),
                "threshold_percent": PERPLEXITY_DELTA_THRESHOLD * 100,
            },
            "vram_estimate": vram_estimate,
            "inference_test": inference_result,
        }
        results_path.write_text(
            json.dumps(full_results, indent=2, default=str), encoding="utf-8"
        )
        logger.info(f"Results JSON written: {results_path}")

        return {"report": report_path, "results": results_path}

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a concise summary to stdout."""
        passes = results.get("passes")
        delta_pct = results.get("delta_percent")
        fp16 = results.get("fp16_perplexity")
        q4 = results.get("quantized_perplexity")
        threshold = results.get("threshold", PERPLEXITY_DELTA_THRESHOLD)

        print("\n" + "=" * 60)
        print("  LATCH QUANTIZATION QUALITY GATE")
        print("=" * 60)
        print(f"  Model         : {results.get('model', 'N/A')}")
        print(f"  Quant Type    : {results.get('quantization_type', 'Q4_K_M')}")
        print(f"  FP16 PPL      : {fp16:.4f}" if fp16 else "  FP16 PPL      : N/A")
        print(f"  Q4_K_M PPL    : {q4:.4f}" if q4 else "  Q4_K_M PPL    : N/A")
        if delta_pct is not None:
            print(f"  Delta         : {delta_pct:.3f}%  (threshold: {threshold*100:.1f}%)")
        if passes is True:
            print("  Quality Gate  : ✅ PASS")
        elif passes is False:
            print("  Quality Gate  : ❌ FAIL — perplexity delta exceeds threshold")
        print("=" * 60 + "\n")
