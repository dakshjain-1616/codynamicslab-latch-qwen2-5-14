#!/usr/bin/env python3
"""
Entry point for running the LATCH-Qwen2.5-14B-GGUF demo from the scripts/ directory.
Delegates to the root demo.py so it can be invoked as either:
    python scripts/demo.py
    python demo.py
"""

import os
import sys

# Ensure the project root is on sys.path so all modules resolve correctly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Change working directory to project root so relative paths (outputs/) work correctly
os.chdir(_project_root)

# Import and run the root demo
from demo import run_demo  # noqa: E402

if __name__ == "__main__":
    try:
        results = run_demo()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Demo failed: {e}")
        sys.exit(1)
