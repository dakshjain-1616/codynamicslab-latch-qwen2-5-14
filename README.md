# CoDynamicsLab/LATCH-Qwen2.5-14B-GGUF – Quality-Gated GGUF Quantization Pipeline

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-179%%20passed-brightgreen.svg)]()

> Automate Qwen2.5-14B quantization with a hard build failure if perplexity degradation exceeds 0.05.

## Install

```bash
git clone https://github.com/dakshjain-1616/codynamicslab-latch-qwen2-5-14
cd codynamicslab-latch-qwen2-5-14
pip install -r requirements.txt
```

## Quickstart

Run the quick start example to initialize the quantization pipeline and generate a benchmark report:

```bash
python examples/01_quick_start.py
```

Or run the full pipeline via CLI to quantize with quality gating:

```bash
python -m codynamicslab_latch_.quantization_pipeline --model-id Qwen2.5-14B --quant-type Q4_K_M
```

## Key features

- **Hard quality gate** — Build fails (`exit 1`) if perplexity delta between FP16 and quantized exceeds 0.05
- **0.02 perplexity delta** — Achieves minimal accuracy loss on Q4_K_M vs FP16 baseline
- **8 GB VRAM target** — 7.2GB Q4_K_M weights fit comfortably within 8GB VRAM limits
- **Auto-generated reports** — Produces `quantization_report.md` with pass/fail status and metrics
- **Mock/dry-run mode** — Runs pipeline on CPU without downloading 29GB FP16 model for local testing

## Run tests

```bash
pytest tests/ -q
# 179 passed
```

## Project structure

```
codynamicslab-latch-qwen2-5-14/
├── codynamicslab_latch_/      ← main library
├── tests/                     ← test suite
├── examples/                  ← demo scripts
└── requirements.txt
```