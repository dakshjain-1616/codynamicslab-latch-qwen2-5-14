# Examples

All scripts work offline in mock mode — no GPU, no API keys, no llama.cpp required.

Run from the project root or from within this directory:

```bash
python examples/01_quick_start.py
python examples/02_advanced_usage.py
python examples/03_custom_config.py
python examples/04_full_pipeline.py
```

## Script Index

| Script | Lines | What it demonstrates |
|--------|-------|----------------------|
| [`01_quick_start.py`](01_quick_start.py) | ~20 | Minimal end-to-end run: create pipeline, run, print quality-gate verdict |
| [`02_advanced_usage.py`](02_advanced_usage.py) | ~50 | GGUF header inspection, multi-quant sweep, run-history sparkline |
| [`03_custom_config.py`](03_custom_config.py) | ~45 | Customise via env vars and constructor args: threshold, quant type, VRAM budget |
| [`04_full_pipeline.py`](04_full_pipeline.py) | ~80 | Complete workflow: pipeline → GGUF inspect → sweep → history → summary |

## Quick Reference

```python
from codynamicslab_latch_ import (
    QuantizationPipeline,       # orchestrate the full pipeline
    ModelConverter,             # download, convert, quantize
    MockPerplexityEvaluator,    # evaluate quality offline
    MultiQuantComparer,         # sweep all quant types
    RunHistoryTracker,          # JSONL history + sparklines
    inspect_gguf,               # parse GGUF binary header
    ReportGenerator,            # write Markdown + JSON reports
)
```

See [`../.env.example`](../.env.example) for all configuration options.
