"""
CoDynamicsLab LATCH – Quality-Gated GGUF Quantization Pipeline.

Public API for CoDynamicsLab/LATCH-Qwen2.5-14B-GGUF (7.2 GB Q4_K_M).
"""

from .quantization_pipeline import QuantizationPipeline, QuantizationPipelineError
from .model_converter import (
    ModelConverter,
    QUANT_TYPES,
    GGUF_MAGIC,
    GGUF_VERSION,
    is_llama_cpp_available,
)
from .perplexity_evaluator import (
    PerplexityEvaluator,
    MockPerplexityEvaluator,
    PERPLEXITY_DELTA_THRESHOLD,
    WIKITEXT_SAMPLES,
)
from .report_generator import ReportGenerator
from .gguf_inspector import (
    GGUFMetadata,
    inspect_gguf,
    format_metadata_table,
    metadata_to_dict,
)
from .multi_quant_compare import MultiQuantComparer, QuantSweepResult
from .history_tracker import RunHistoryTracker

__all__ = [
    # Pipeline
    "QuantizationPipeline",
    "QuantizationPipelineError",
    # Converter
    "ModelConverter",
    "QUANT_TYPES",
    "GGUF_MAGIC",
    "GGUF_VERSION",
    "is_llama_cpp_available",
    # Perplexity
    "PerplexityEvaluator",
    "MockPerplexityEvaluator",
    "PERPLEXITY_DELTA_THRESHOLD",
    "WIKITEXT_SAMPLES",
    # Report
    "ReportGenerator",
    # GGUF Inspector
    "GGUFMetadata",
    "inspect_gguf",
    "format_metadata_table",
    "metadata_to_dict",
    # Multi-quant sweep
    "MultiQuantComparer",
    "QuantSweepResult",
    # History
    "RunHistoryTracker",
]
