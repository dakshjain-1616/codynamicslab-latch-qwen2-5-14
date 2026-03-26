"""
04_full_pipeline.py — End-to-end workflow showing every project capability.

Steps:
  1. Run the full quantization pipeline (mock mode)
  2. Inspect the GGUF binary header
  3. Run a multi-quant sweep and pick the best format
  4. Track the run in JSONL history with sparkline trend
  5. Print the final report path and quality-gate verdict
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from pathlib import Path

# Ensure mock mode for offline / no-GPU use
os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("NUM_PERPLEXITY_SAMPLES", "20")

from codynamicslab_latch_ import (
    QuantizationPipeline,
    QuantizationPipelineError,
    MultiQuantComparer,
    RunHistoryTracker,
    inspect_gguf,
    format_metadata_table,
    metadata_to_dict,
)

MODEL      = "CoDynamicsLab/LATCH-Qwen2.5-14B"
QUANT_TYPE = "Q4_K_M"
OUTPUT_DIR = "outputs"

print("=" * 60)
print("  LATCH Full Pipeline Demo")
print("=" * 60)

# ── Step 1: Run the pipeline ──────────────────────────────────────────────────
print("\n[1/4] Running quantization pipeline…")
pipeline = QuantizationPipeline(
    model_name=MODEL,
    quant_type=QUANT_TYPE,
    output_dir=OUTPUT_DIR,
    mock_mode=True,
    num_samples=20,
    fail_on_quality_gate=False,
)

results = pipeline.run()
eval_r   = results["eval_results"]
fp16_ppl = eval_r["fp16_perplexity"]

print(f"  FP16 perplexity   : {fp16_ppl:.4f}")
print(f"  {QUANT_TYPE} perplexity : {eval_r['quantized_perplexity']:.4f}")
print(f"  Delta             : {eval_r['delta_percent']:.3f}%")
print(f"  Quality gate      : {'PASS ✓' if results['passes_quality_gate'] else 'FAIL ✗'}")

# ── Step 2: Inspect GGUF header ───────────────────────────────────────────────
print("\n[2/4] Inspecting GGUF binary header…")
gguf_path = Path(results["gguf_path"])
meta = inspect_gguf(gguf_path)
meta_dict = metadata_to_dict(meta)
print(f"  Valid GGUF  : {meta_dict['valid']}")
print(f"  Version     : {meta_dict['version']}")
print(f"  Tensor count: {meta_dict['tensor_count']}")
print(f"  KV pairs    : {meta_dict['kv_count']}")
print(f"  File size   : {meta_dict['file_size_gb']:.3f} GB")

# ── Step 3: Multi-quant sweep ─────────────────────────────────────────────────
print("\n[3/4] Running multi-quant sweep…")
comparer = MultiQuantComparer(model_name=MODEL, vram_budget_gb=8.0)
sweep    = comparer.run_sweep(fp16_perplexity=fp16_ppl)

print(f"  {'Type':<10} {'Delta':>8} {'VRAM':>8}  Gate")
for r in sweep:
    tag  = " ★" if r.recommended else ""
    gate = "PASS" if r.passes_quality_gate else "FAIL"
    print(f"  {r.quant_type:<10} {r.delta_percent:>7.2f}% {r.estimated_vram_gb:>7.2f} GB  {gate}{tag}")

recommended = next((r.quant_type for r in sweep if r.recommended), None)
print(f"\n  Recommended: {recommended}")

# ── Step 4: History tracking ──────────────────────────────────────────────────
print("\n[4/4] Tracking run in history log…")
tracker = RunHistoryTracker(history_file=f"{OUTPUT_DIR}/run_history.jsonl")
tracker.append_run(eval_r)
stats = tracker.summary_stats()
print(f"  Total runs : {stats['total_runs']}")
print(f"  Pass rate  : {stats.get('pass_rate_percent', 0)}%")
print(f"  Trend      : {tracker.sparkline()}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Output files:")
print(f"    Report  : {results['report_path']}")
print(f"    Results : {results['results_path']}")
print(f"    GGUF    : {results['gguf_path']}")
print("=" * 60)
