"""
Multi-quantization sweep comparator.

Simulates or runs Q2_K / Q4_K_S / Q4_K_M / Q5_K_M / Q8_0 and recommends
the most compressed quantization type that:
  1. Passes the perplexity quality gate (delta ≤ threshold)
  2. Fits within the configured VRAM budget

Works fully in mock mode — no GPU or llama.cpp required.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
VRAM_BUDGET_GB = float(os.getenv("VRAM_BUDGET_GB", "8.0"))
PERPLEXITY_DELTA_THRESHOLD = float(os.getenv("PERPLEXITY_DELTA_THRESHOLD", "0.05"))
SWEEP_QUANT_TYPES = os.getenv(
    "SWEEP_QUANT_TYPES", "Q2_K,Q4_K_S,Q4_K_M,Q5_K_M,Q8_0"
).split(",")

# Simulated perplexity delta per quant type (relative to FP16 baseline).
# Calibrated from published Qwen2.5-14B GGUF benchmarks.
SIMULATED_DELTAS: dict = {
    "Q2_K":  0.085,
    "Q3_K_M": 0.042,
    "Q4_K_S": 0.022,
    "Q4_K_M": 0.018,
    "Q5_K_M": 0.009,
    "Q6_K":  0.003,
    "Q8_0":  0.002,
    "F16":   0.000,
}

# Size factors (fraction of FP16 weight storage) per quant type.
# Used for VRAM estimation when model_converter QUANT_TYPES doesn't include a type.
_FALLBACK_SIZE_FACTORS: dict = {
    "Q2_K":  0.134,
    "Q3_K_M": 0.180,
    "Q4_K_S": 0.234,
    "Q4_K_M": 0.245,
    "Q5_K_M": 0.296,
    "Q6_K":  0.348,
    "Q8_0":  0.503,
    "F16":   1.000,
}

# Qwen2.5-14B architecture constants for KV cache estimation
_PARAM_COUNT_B = float(os.getenv("MODEL_PARAM_COUNT_B", "14.7"))
_KV_LAYERS = int(os.getenv("KV_LAYERS", "48"))
_KV_HEADS = int(os.getenv("KV_HEADS", "8"))
_HEAD_DIM = int(os.getenv("HEAD_DIM", "128"))
_SEQ_LEN = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
_ACTIVATION_GB = float(os.getenv("ACTIVATION_OVERHEAD_GB", "0.3"))


def _estimate_vram(size_factor: float) -> float:
    """Estimate total VRAM requirement (GB) for a given size factor."""
    weights_gb = _PARAM_COUNT_B * 2.0 * size_factor
    kv_cache_gb = 2 * _KV_LAYERS * _KV_HEADS * _HEAD_DIM * _SEQ_LEN * 2 / 1e9
    return round(weights_gb + kv_cache_gb + _ACTIVATION_GB, 2)


@dataclass
class QuantSweepResult:
    """Results for a single quantization type in a sweep."""
    quant_type: str
    fp16_perplexity: float
    quantized_perplexity: float
    delta: float
    delta_percent: float
    passes_quality_gate: bool
    estimated_vram_gb: float
    fits_vram_budget: bool
    size_factor: float
    recommended: bool = field(default=False)

    def to_dict(self) -> dict:
        return {
            "quant_type": self.quant_type,
            "fp16_perplexity": self.fp16_perplexity,
            "quantized_perplexity": self.quantized_perplexity,
            "delta": self.delta,
            "delta_percent": self.delta_percent,
            "passes_quality_gate": self.passes_quality_gate,
            "estimated_vram_gb": self.estimated_vram_gb,
            "fits_vram_budget": self.fits_vram_budget,
            "size_factor": self.size_factor,
            "recommended": self.recommended,
        }


class MultiQuantComparer:
    """
    Sweeps multiple quantization types and recommends the best one.

    Selection criteria (in order):
      1. Must pass the perplexity quality gate (delta ≤ threshold)
      2. Must fit within VRAM budget
      3. Among candidates: lowest size_factor (most compressed)
    """

    def __init__(
        self,
        model_name: str = "CoDynamicsLab/LATCH-Qwen2.5-14B",
        vram_budget_gb: float = VRAM_BUDGET_GB,
        quant_types: Optional[List[str]] = None,
        delta_threshold: float = PERPLEXITY_DELTA_THRESHOLD,
    ):
        self.model_name = model_name
        self.vram_budget_gb = vram_budget_gb
        self.quant_types = [qt.strip() for qt in (quant_types or SWEEP_QUANT_TYPES)]
        self.delta_threshold = delta_threshold

    def _simulate_ppl(self, fp16_ppl: float, quant_type: str) -> float:
        """Apply simulated quantization degradation to FP16 perplexity."""
        delta = SIMULATED_DELTAS.get(quant_type, 0.02)
        return fp16_ppl * (1 + delta)

    def _size_factor(self, quant_type: str) -> float:
        """Return size factor for a quant type, checking model_converter first."""
        try:
            from .model_converter import QUANT_TYPES
            if quant_type in QUANT_TYPES:
                return QUANT_TYPES[quant_type]["size_factor"]
        except ImportError:
            pass
        return _FALLBACK_SIZE_FACTORS.get(quant_type, 0.5)

    def run_sweep(self, fp16_perplexity: float = 100.0) -> List[QuantSweepResult]:
        """
        Simulate sweep over all configured quant types.

        Args:
            fp16_perplexity: Baseline FP16 perplexity to compare against.

        Returns:
            List of QuantSweepResult, one per quant type.  The recommended
            entry has ``recommended=True``.
        """
        results: List[QuantSweepResult] = []

        for qt in self.quant_types:
            qt = qt.strip()
            if not qt:
                continue

            sf = self._size_factor(qt)
            q_ppl = self._simulate_ppl(fp16_perplexity, qt)
            delta = abs(q_ppl - fp16_perplexity) / fp16_perplexity if fp16_perplexity else 0.0
            passes = delta <= self.delta_threshold
            vram_gb = _estimate_vram(sf)
            fits = vram_gb <= self.vram_budget_gb

            results.append(QuantSweepResult(
                quant_type=qt,
                fp16_perplexity=round(fp16_perplexity, 4),
                quantized_perplexity=round(q_ppl, 4),
                delta=round(delta, 6),
                delta_percent=round(delta * 100, 3),
                passes_quality_gate=passes,
                estimated_vram_gb=vram_gb,
                fits_vram_budget=fits,
                size_factor=sf,
            ))

        best = self._pick_best(results)
        if best is not None:
            for r in results:
                if r.quant_type == best.quant_type:
                    r.recommended = True

        logger.info(
            f"Sweep complete ({len(results)} types). "
            f"Recommended: {best.quant_type if best else 'None'}"
        )
        return results

    def _pick_best(self, results: List[QuantSweepResult]) -> Optional[QuantSweepResult]:
        """
        Pick the most compressed quant that passes QA gate and fits VRAM.
        Falls back to QA-passing only if no VRAM-fitting candidate exists.
        """
        candidates = [r for r in results if r.passes_quality_gate and r.fits_vram_budget]
        if not candidates:
            candidates = [r for r in results if r.passes_quality_gate]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.size_factor)

    def recommend(self, fp16_perplexity: float = 100.0) -> Optional[str]:
        """Return the recommended quant type string (or None if none qualify)."""
        for r in self.run_sweep(fp16_perplexity):
            if r.recommended:
                return r.quant_type
        return None

    def format_sweep_table(self, results: List[QuantSweepResult]) -> str:
        """Format sweep results as a Markdown comparison table."""
        header = (
            "| Quant Type | PPL (sim.) | Delta % | VRAM GB | "
            "Fits Budget | Quality Gate | Recommended |"
        )
        sep = (
            "|------------|-----------|---------|---------|"
            "-------------|--------------|-------------|"
        )
        rows = [header, sep]
        for r in results:
            gate = "✅ PASS" if r.passes_quality_gate else "❌ FAIL"
            fits = "✅ Yes" if r.fits_vram_budget else "❌ No"
            rec = "⭐ **Yes**" if r.recommended else ""
            rows.append(
                f"| {r.quant_type} | `{r.quantized_perplexity:.4f}` | "
                f"`{r.delta_percent:.3f}%` | `{r.estimated_vram_gb:.2f}` | "
                f"{fits} | {gate} | {rec} |"
            )
        return "\n".join(rows)
