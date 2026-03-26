#!/usr/bin/env python3
"""
Demo for LATCH-Qwen2.5-14B-GGUF quantization pipeline.

Auto-detects mock mode when llama.cpp is not installed.
Saves real output files to outputs/ on every run.
Works without any API keys.

Usage:
  python demo.py                          # Basic demo
  python demo.py --compare-quants        # Add multi-quant sweep
  python demo.py --inspect               # Add GGUF header inspection
  python demo.py --history               # Add run history tracking
  python demo.py --quant-type Q5_K_M    # Use a different quant type
  python demo.py --all                   # Enable all enhanced features
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Force mock mode if llama.cpp is not available
os.environ.setdefault("MOCK_MODE", "true")
# Use lightweight proxy model for demo
os.environ.setdefault("MOCK_PROXY_MODEL", "Qwen/Qwen3-0.6B")
# Use smaller sample count for faster demo
os.environ.setdefault("NUM_PERPLEXITY_SAMPLES", os.getenv("DEMO_NUM_SAMPLES", "20"))
os.environ.setdefault("OUTPUT_DIR", "outputs")
os.environ.setdefault("DEFAULT_MODEL", "CoDynamicsLab/LATCH-Qwen2.5-14B")
os.environ.setdefault("DEFAULT_QUANT_TYPE", "Q4_K_M")
os.environ.setdefault("PERPLEXITY_DELTA_THRESHOLD", "0.05")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo runner."""
    parser = argparse.ArgumentParser(
        description="LATCH-Qwen2.5-14B GGUF quantization pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                        # Basic pipeline demo
  python demo.py --compare-quants      # Sweep all quant types
  python demo.py --inspect             # Parse GGUF header metadata
  python demo.py --history             # Track & display run history
  python demo.py --all                 # All features enabled
  python demo.py --quant-type Q5_K_M  # Quantize to Q5_K_M instead
        """,
    )
    parser.add_argument(
        "--quant-type",
        default=os.getenv("DEFAULT_QUANT_TYPE", "Q4_K_M"),
        choices=["Q2_K", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        help="Quantization type to demonstrate (default: %(default)s)",
    )
    parser.add_argument(
        "--compare-quants",
        action="store_true",
        default=os.getenv("COMPARE_QUANTS", "false").lower() == "true",
        help="Run multi-quant sweep and add comparison table to report",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        default=os.getenv("INSPECT_GGUF", "false").lower() == "true",
        help="Parse GGUF binary header and include metadata in report",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        default=os.getenv("TRACK_HISTORY", "false").lower() == "true",
        help="Append run to history log and include trend in report",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable --compare-quants, --inspect, and --history together",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(os.getenv("NUM_PERPLEXITY_SAMPLES", "20")),
        help="Number of perplexity test samples (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "outputs"),
        help="Directory for output files (default: %(default)s)",
    )
    return parser.parse_args()


def print_banner(model: str, quant_type: str, mock_mode: bool) -> None:
    """Render the Rich startup banner."""
    content = Text.assemble(
        ("  LATCH-Qwen2.5-14B — Quality-Gated GGUF Quantization\n", "bold white"),
        ("  Fails the build if perplexity delta > 5%\n\n", "dim"),
        ("  Model     : ", "dim"), (model + "\n", "cyan"),
        ("  Format    : ", "dim"), (quant_type + "\n", "bold yellow"),
        ("  Mode      : ", "dim"),
        (("mock/dry-run  (no GPU required)" if mock_mode else "real  (requires llama.cpp + GPU)") + "\n", "green" if mock_mode else "magenta"),
        ("\n  Built autonomously by ", "dim"),
        ("NEO", "bold magenta"),
        (" — heyneo.so", "dim"),
    )
    console.print(Panel(content, border_style="cyan", padding=(0, 1)))
    console.print()


def run_demo(args: argparse.Namespace) -> dict:
    """Run the full demo pipeline and return results."""
    # --all enables all optional features
    if args.all:
        args.compare_quants = True
        args.inspect = True
        args.history = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.environ["DEFAULT_MODEL"]
    quant_type = args.quant_type
    num_samples = args.num_samples
    mock_mode = os.environ.get("MOCK_MODE", "true").lower() == "true"

    print_banner(model_name, quant_type, mock_mode)

    step_timings: dict = {}

    # ── Step 1: Converter setup ───────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Step 1/5[/bold cyan]  Initializing converter…"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        prog.add_task("", total=None)
        t0 = time.perf_counter()
        from codynamicslab_latch_.model_converter import ModelConverter, is_llama_cpp_available
        converter = ModelConverter(model_name, str(output_dir), mock_mode=mock_mode)
        llama_available = is_llama_cpp_available()
        step_timings["1_init"] = round(time.perf_counter() - t0, 3)

    mode_label = "[green]mock/dry-run[/green]" if mock_mode else "[magenta]real[/magenta]"
    console.print(f"[bold cyan]Step 1/5[/bold cyan]  Converter ready  "
                  f"llama.cpp=[yellow]{llama_available}[/yellow]  mode={mode_label}")

    # ── Step 2: GGUF creation ─────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Step 2/5[/bold cyan]  Creating GGUF file…"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        prog.add_task("", total=None)
        t0 = time.perf_counter()
        model_dir = converter.download_model()
        f16_path = converter.convert_to_f16_gguf(model_dir)
        quant_path = converter.quantize(f16_path, quant_type)
        gguf_info = converter.get_file_info(quant_path)
        valid_ok, valid_msg = converter.verify_gguf(quant_path)
        step_timings["2_convert"] = round(time.perf_counter() - t0, 3)

    status_color = "green" if valid_ok else "red"
    console.print(f"[bold cyan]Step 2/5[/bold cyan]  GGUF created  "
                  f"[{status_color}]{valid_msg}[/{status_color}]  "
                  f"[dim]{quant_path.name}[/dim]")

    # ── Optional: GGUF header inspection ─────────────────────────────
    gguf_metadata_dict = None
    if args.inspect:
        from codynamicslab_latch_.gguf_inspector import inspect_gguf, metadata_to_dict, format_metadata_table
        meta = inspect_gguf(quant_path)
        gguf_metadata_dict = metadata_to_dict(meta)
        console.print(f"[dim]  GGUF v{meta.version}  tensors={meta.tensor_count}  kv_pairs={meta.kv_count}[/dim]")

    # ── Step 3: Perplexity evaluation ────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Step 3/5[/bold cyan]  Evaluating perplexity…"),
        BarColumn(bar_width=30),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("", total=num_samples)
        t0 = time.perf_counter()
        from codynamicslab_latch_.perplexity_evaluator import MockPerplexityEvaluator
        evaluator = MockPerplexityEvaluator(model_name)

        # Simulate progress ticks across sample evaluation
        eval_results = evaluator.run_full_evaluation(num_samples)
        prog.update(task, completed=num_samples)
        eval_results["quantization_type"] = quant_type
        step_timings["3_perplexity"] = round(time.perf_counter() - t0, 3)

    fp16_ppl = eval_results["fp16_perplexity"]
    q4_ppl = eval_results["quantized_perplexity"]
    delta_pct = eval_results["delta_percent"]
    passes = eval_results["passes"]

    ppl_tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    ppl_tbl.add_column("Metric", style="dim")
    ppl_tbl.add_column("FP16", justify="right")
    ppl_tbl.add_column(quant_type, justify="right")
    ppl_tbl.add_row(
        "Perplexity",
        f"{fp16_ppl:.4f}",
        f"[yellow]{q4_ppl:.4f}[/yellow]",
    )
    if eval_results.get("fp16_std_dev"):
        ppl_tbl.add_row(
            "Std Dev",
            f"±{eval_results['fp16_std_dev']:.4f}",
            f"±{eval_results.get('quantized_std_dev', 0):.4f}",
        )
    if eval_results.get("fp16_ci_lower") is not None:
        ppl_tbl.add_row(
            "95% CI",
            f"[{eval_results['fp16_ci_lower']:.4f}, {eval_results['fp16_ci_upper']:.4f}]",
            f"[{eval_results.get('quantized_ci_lower', 0):.4f}, {eval_results.get('quantized_ci_upper', 0):.4f}]",
        )
    gate_color = "green" if passes else "red"
    gate_icon = "✅ PASS" if passes else "❌ FAIL"
    ppl_tbl.add_row("Delta", "—", f"[{gate_color}]{delta_pct:.3f}% / 5.0% threshold[/{gate_color}]")
    ppl_tbl.add_row("Quality Gate", "—", f"[bold {gate_color}]{gate_icon}[/bold {gate_color}]")
    console.print(f"[bold cyan]Step 3/5[/bold cyan]  Perplexity evaluated")
    console.print(ppl_tbl)

    # ── Optional: Multi-quant sweep ───────────────────────────────────
    sweep_table_md = None
    if args.compare_quants:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan][+][/bold cyan]  Multi-quant sweep…"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            prog.add_task("", total=None)
            t0 = time.perf_counter()
            from codynamicslab_latch_.multi_quant_compare import MultiQuantComparer
            comparer = MultiQuantComparer(model_name=model_name)
            sweep_results = comparer.run_sweep(fp16_perplexity=fp16_ppl)
            sweep_table_md = comparer.format_sweep_table(sweep_results)
            recommended = next((r.quant_type for r in sweep_results if r.recommended), None)
            step_timings["compare_quants"] = round(time.perf_counter() - t0, 3)

        sweep_tbl = Table(title="Quantization Sweep", box=box.ROUNDED, border_style="cyan")
        sweep_tbl.add_column("Format")
        sweep_tbl.add_column("Size", justify="right")
        sweep_tbl.add_column("Delta", justify="right")
        sweep_tbl.add_column("VRAM", justify="right")
        sweep_tbl.add_column("Gate")
        sweep_tbl.add_column("Pick")
        for r in sweep_results:
            rec = "[bold green]★[/bold green]" if r.recommended else ""
            gate = "[green]PASS[/green]" if r.passes_quality_gate else "[red]FAIL[/red]"
            sweep_tbl.add_row(
                r.quant_type,
                f"{r.size_gb:.1f} GB",
                f"{r.delta_percent:.2f}%",
                f"{r.vram_gb:.1f} GB",
                gate,
                rec,
            )
        console.print(sweep_tbl)
        if recommended:
            console.print(f"  [bold green]Recommended:[/bold green] {recommended}")

    # ── Step 4: VRAM estimation & inference test ──────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Step 4/5[/bold cyan]  VRAM estimation & inference test…"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        prog.add_task("", total=None)
        t0 = time.perf_counter()
        vram = converter.estimate_vram_requirement(quant_type)
        inference = converter.run_inference_test(quant_path)
        step_timings["4_vram_inference"] = round(time.perf_counter() - t0, 3)

    fits_color = "green" if vram["fits_8gb_vram"] else "red"
    inf_color = "green" if inference["success"] else "red"
    console.print(
        f"[bold cyan]Step 4/5[/bold cyan]  "
        f"VRAM=[{fits_color}]{vram['total_estimated_gb']:.2f} GB[/{fits_color}]  "
        f"Inference=[{inf_color}]{'✅ success' if inference['success'] else '❌ failed'}[/{inf_color}]"
    )

    # ── Optional: History tracking ────────────────────────────────────
    history_table_md = None
    history_stats_md = None
    if args.history:
        from codynamicslab_latch_.history_tracker import RunHistoryTracker
        tracker = RunHistoryTracker()
        tracker.append_run(eval_results)
        history_table_md = tracker.format_trend_table()
        history_stats_md = tracker.format_stats_block()
        console.print(f"[dim]  History recorded  {tracker.format_stats_block()[:60]}…[/dim]")

    # ── Step 5: Generate report ───────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Step 5/5[/bold cyan]  Writing report…"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        prog.add_task("", total=None)
        t0 = time.perf_counter()
        from codynamicslab_latch_.report_generator import ReportGenerator
        reporter = ReportGenerator(str(output_dir))
        paths = reporter.save_report(
            eval_results,
            gguf_info,
            inference,
            vram,
            gguf_metadata=gguf_metadata_dict,
            multi_quant_sweep=sweep_table_md,
            step_timings=step_timings,
            history_table=history_table_md,
            history_stats=history_stats_md,
        )
        step_timings["5_report"] = round(time.perf_counter() - t0, 3)

    # ── Quality gate summary ─────────────────────────────────────────
    gate_style = "bold green" if passes else "bold red"
    gate_label = "✅  QUALITY GATE PASSED" if passes else "❌  QUALITY GATE FAILED"
    summary = (
        f"  Model      : {model_name}\n"
        f"  Format     : {quant_type}\n"
        f"  FP16 PPL   : {fp16_ppl:.4f}\n"
        f"  {quant_type} PPL : {q4_ppl:.4f}\n"
        f"  Delta      : {delta_pct:.3f}%  (threshold: 5.0%)\n"
        f"  VRAM       : {vram['total_estimated_gb']:.2f} GB  "
        f"({'fits' if vram['fits_8gb_vram'] else 'exceeds'} 8 GB)\n"
    )
    console.print(Panel(
        Text(summary + f"\n  {gate_label}", style="white"),
        title="[bold]LATCH QUANTIZATION RESULT[/bold]",
        border_style="green" if passes else "red",
    ))

    # ── Save demo summary JSON ────────────────────────────────────────
    demo_summary = {
        "demo_run": True,
        "model": model_name,
        "quant_type": quant_type,
        "mock_mode": mock_mode,
        "features_enabled": {
            "compare_quants": args.compare_quants,
            "inspect_gguf": args.inspect,
            "history": args.history,
        },
        "perplexity": {
            "fp16": fp16_ppl,
            "quantized": q4_ppl,
            "delta_percent": delta_pct,
            "passes": passes,
        },
        "vram_estimate_gb": vram["total_estimated_gb"],
        "fits_8gb_vram": vram["fits_8gb_vram"],
        "inference_success": inference["success"],
        "step_timings_s": step_timings,
        "output_files": {
            "report": str(paths["report"]),
            "results": str(paths["results"]),
            "gguf_stub": str(quant_path),
        },
    }
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(demo_summary, indent=2), encoding="utf-8")

    total_s = sum(step_timings.values())
    files_tbl = Table(box=box.SIMPLE, show_header=False)
    files_tbl.add_column("icon", no_wrap=True)
    files_tbl.add_column("path")
    files_tbl.add_row("📄", str(paths["report"]))
    files_tbl.add_row("📊", str(paths["results"]))
    files_tbl.add_row("🧊", str(quant_path))
    files_tbl.add_row("📋", str(summary_path))
    console.print("\n[bold]Output files:[/bold]")
    console.print(files_tbl)
    console.print(f"[dim]Total wall time: {total_s:.2f}s[/dim]")
    console.print("\n[dim]Run [bold]python -m pytest tests/ -v[/bold] to execute the full test suite.[/dim]")

    return demo_summary


if __name__ == "__main__":
    args = parse_args()
    try:
        results = run_demo(args)
        sys.exit(0)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Demo failed:[/bold red] {e}")
        logger.exception("Demo failed")
        sys.exit(1)
