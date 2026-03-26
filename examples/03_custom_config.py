"""
03_custom_config.py — Customising behaviour via environment variables and
                       direct constructor arguments.

Shows how to change thresholds, quant type, sample count, and VRAM budget.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Option A: set via environment variables before importing ──────────────────
os.environ["PERPLEXITY_DELTA_THRESHOLD"] = "0.03"   # stricter 3% gate
os.environ["NUM_PERPLEXITY_SAMPLES"]     = "10"
os.environ["VRAM_BUDGET_GB"]             = "12.0"   # 12 GB GPU
os.environ["MOCK_MODE"]                  = "true"

from codynamicslab_latch_ import (
    QuantizationPipeline,
    MultiQuantComparer,
    PERPLEXITY_DELTA_THRESHOLD,
)

print(f"Active threshold: {PERPLEXITY_DELTA_THRESHOLD * 100:.1f}%")

# ── Option B: pass directly to constructors ───────────────────────────────────
pipeline = QuantizationPipeline(
    model_name="CoDynamicsLab/LATCH-Qwen2.5-14B",
    quant_type="Q5_K_M",          # try Q5_K_M instead of the default Q4_K_M
    output_dir="outputs",
    mock_mode=True,
    num_samples=10,
    fail_on_quality_gate=False,   # don't raise — just report
)

info = pipeline.get_pipeline_info()
print("\n── Pipeline config ──────────────────────")
for k, v in info.items():
    print(f"  {k:<25}: {v}")

results = pipeline.run()
print(f"\nQ5_K_M delta  : {results['eval_results']['delta_percent']:.3f}%")
print(f"Quality gate  : {'PASS' if results['passes_quality_gate'] else 'FAIL'}")
print(f"VRAM estimate : {results['vram_estimate']['total_estimated_gb']:.2f} GB")

# ── Multi-quant sweep with custom VRAM budget ─────────────────────────────────
comparer = MultiQuantComparer(vram_budget_gb=12.0)
recommended = comparer.recommend(fp16_perplexity=100.0)
print(f"\nBest quant for 12 GB VRAM: {recommended}")
