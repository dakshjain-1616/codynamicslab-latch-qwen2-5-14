"""
Root pytest configuration for LATCH-Qwen2.5-14B-GGUF test suite.

Sets up sys.path so all project modules are importable from any working
directory and applies environment defaults that keep the test suite fast
and dependency-free.
"""

import os
import sys
from pathlib import Path

# Make project root importable regardless of how pytest is invoked
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Defaults that keep tests fast and offline ─────────────────────────────────
os.environ.setdefault("MOCK_MODE", "true")
# Use pure simulation — no gpt2 download required (set USE_REAL_PROXY_MODEL=true
# to enable real proxy-model evaluation in integration-style runs)
os.environ.setdefault("USE_REAL_PROXY_MODEL", "false")
os.environ.setdefault("NUM_PERPLEXITY_SAMPLES", "5")
os.environ.setdefault("PERPLEXITY_DELTA_THRESHOLD", "0.05")
os.environ.setdefault("OUTPUT_DIR", "outputs")
