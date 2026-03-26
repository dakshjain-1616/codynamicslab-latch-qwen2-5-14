"""
02_advanced_usage.py — Advanced features: GGUF inspection, multi-quant sweep,
                        run-history tracking.

Demonstrates using each enhancement module independently.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from codynamicslab_latch_ import (
    ModelConverter,
    MockPerplexityEvaluator,
    MultiQuantComparer,
    RunHistoryTracker,
    inspect_gguf,
    metadata_to_dict,
)

MODEL = "CoDynamicsLab/LATCH-Qwen2.5-14B"
OUTPUT = "outputs"

# ── 1. Create a GGUF stub and inspect its header ──────────────────────────────
converter = ModelConverter(MODEL, OUTPUT, mock_mode=True)
model_dir = converter.download_model()
f16_path  = converter.convert_to_f16_gguf(model_dir)
quant_path = converter.quantize(f16_path, "Q4_K_M")

meta = inspect_gguf(quant_path)
print("── GGUF Header ──────────────────────────")
print(f"  Valid   : {meta.valid}")
print(f"  Version : {meta.version}")
print(f"  Tensors : {meta.tensor_count}")
print(f"  KV pairs: {meta.kv_count}")
print(f"  Size    : {meta.file_size_gb:.3f} GB")

# ── 2. Multi-quant sweep — find the best quantization for 8 GB VRAM ──────────
evaluator = MockPerplexityEvaluator(MODEL)
eval_results = evaluator.run_full_evaluation(num_samples=10)
fp16_ppl = eval_results["fp16_perplexity"]

comparer = MultiQuantComparer(model_name=MODEL, vram_budget_gb=8.0)
sweep = comparer.run_sweep(fp16_perplexity=fp16_ppl)

print("\n── Quantization Sweep ───────────────────")
for r in sweep:
    tag = " ← recommended" if r.recommended else ""
    print(
        f"  {r.quant_type:<10} delta={r.delta_percent:.2f}%  "
        f"vram={r.estimated_vram_gb:.2f} GB  "
        f"gate={'PASS' if r.passes_quality_gate else 'FAIL'}{tag}"
    )

# ── 3. Append run to history and print stats ──────────────────────────────────
tracker = RunHistoryTracker(history_file=f"{OUTPUT}/run_history.jsonl")
tracker.append_run(eval_results)
stats = tracker.summary_stats()
print("\n── Run History ──────────────────────────")
print(f"  Total runs : {stats['total_runs']}")
print(f"  Pass rate  : {stats.get('pass_rate_percent', 0)}%")
print(f"  Sparkline  : {tracker.sparkline()}")
