"""
01_quick_start.py — Minimal working example.

Runs the full pipeline in mock mode (no GPU / llama.cpp required).
Prints whether the Q4_K_M quality gate passes.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from codynamicslab_latch_ import QuantizationPipeline

pipeline = QuantizationPipeline(
    model_name="CoDynamicsLab/LATCH-Qwen2.5-14B",
    quant_type="Q4_K_M",
    output_dir="outputs",
    mock_mode=True,
    num_samples=5,
    fail_on_quality_gate=False,
)

results = pipeline.run()

print(f"Quality gate passed : {results['passes_quality_gate']}")
print(f"FP16 perplexity     : {results['eval_results']['fp16_perplexity']}")
print(f"Q4_K_M perplexity   : {results['eval_results']['quantized_perplexity']}")
print(f"Delta               : {results['eval_results']['delta_percent']:.3f}%")
print(f"Report written to   : {results['report_path']}")
