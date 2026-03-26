#!/usr/bin/env python3
"""
CLI entry point for the LATCH quantization pipeline.
Usage: python run_quant.py --model CoDynamicsLab/LATCH-Qwen2.5-14B
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

__version__ = "1.2.0"

console = Console()


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging level and format."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def print_banner(model: str, quant_type: str, mock: bool) -> None:
    """Print a Rich startup banner with project info."""
    title = Text("LATCH – Quality-Gated GGUF Quantization", style="bold cyan")
    subtitle = (
        f"[dim]v{__version__}[/dim]  ·  "
        f"[bold]{model}[/bold]  →  [bold yellow]{quant_type}[/bold yellow]"
    )
    if mock:
        subtitle += "  [dim](mock/dry-run)[/dim]"
    content = Text.assemble(
        (f"  Model   : {model}\n", "white"),
        (f"  Format  : {quant_type}\n", "white"),
        (f"  Mode    : {'mock/dry-run' if mock else 'real'}\n", "white"),
        ("\n  Built autonomously by ", "dim"),
        ("NEO", "bold magenta"),
        (" — heyneo.so", "dim"),
    )
    console.print(Panel(content, title=title, subtitle=subtitle, border_style="cyan"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quality-gated GGUF quantization pipeline for LATCH-Qwen2.5-14B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real quantization (requires llama.cpp)
  python run_quant.py --model CoDynamicsLab/LATCH-Qwen2.5-14B

  # Mock/dry-run mode (no GPU or llama.cpp required)
  python run_quant.py --model CoDynamicsLab/LATCH-Qwen2.5-14B --mock

  # Custom quantization type and output directory
  python run_quant.py --model CoDynamicsLab/LATCH-Qwen2.5-14B --quant-type Q5_K_M --output-dir /tmp/gguf

  # Validate inputs without running
  python run_quant.py --model CoDynamicsLab/LATCH-Qwen2.5-14B --validate-only
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"LATCH {__version__}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEFAULT_MODEL", "CoDynamicsLab/LATCH-Qwen2.5-14B"),
        help="HuggingFace model ID or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--quant-type",
        default=os.getenv("DEFAULT_QUANT_TYPE", "Q4_K_M"),
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0", "F16"],
        help="Quantization type (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "outputs"),
        help="Directory for output files (default: %(default)s)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(os.getenv("NUM_PERPLEXITY_SAMPLES", "100")),
        help="Number of perplexity test samples (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("PERPLEXITY_DELTA_THRESHOLD", "0.05")),
        help="Maximum allowed perplexity delta (default: %(default)s)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=os.getenv("MOCK_MODE", "false").lower() == "true",
        help="Run in mock/dry-run mode (no llama.cpp or downloads required)",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit with error code if quality gate fails",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print pipeline config, then exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    logger = logging.getLogger(__name__)

    print_banner(args.model, args.quant_type, args.mock)

    # Set environment variables from CLI args so modules pick them up
    os.environ["DEFAULT_MODEL"] = args.model
    os.environ["DEFAULT_QUANT_TYPE"] = args.quant_type
    os.environ["OUTPUT_DIR"] = args.output_dir
    os.environ["NUM_PERPLEXITY_SAMPLES"] = str(args.num_samples)
    os.environ["PERPLEXITY_DELTA_THRESHOLD"] = str(args.threshold)
    if args.mock:
        os.environ["MOCK_MODE"] = "true"

    from codynamicslab_latch_.quantization_pipeline import QuantizationPipeline, QuantizationPipelineError

    pipeline = QuantizationPipeline(
        model_name=args.model,
        quant_type=args.quant_type,
        output_dir=args.output_dir,
        mock_mode=args.mock or None,
        num_samples=args.num_samples,
        fail_on_quality_gate=not args.no_fail,
    )

    # Validate-only mode
    if args.validate_only:
        validation = pipeline.validate_inputs()
        info = pipeline.get_pipeline_info()

        tbl = Table(title="Pipeline Configuration", box=box.ROUNDED, border_style="cyan")
        tbl.add_column("Setting", style="bold")
        tbl.add_column("Value")
        for k, v in info.items():
            tbl.add_row(str(k), str(v))
        console.print(tbl)

        if validation["warnings"]:
            for w in validation["warnings"]:
                console.print(f"  [yellow]⚠  Warning:[/yellow] {w}")

        if not validation["valid"]:
            console.print("\n[bold red]Validation FAILED:[/bold red]")
            for e in validation["errors"]:
                console.print(f"  [red]✗[/red] {e}")
            return 1

        console.print("[bold green]  ✓ Validation passed[/bold green]")
        return 0

    # Run pipeline
    try:
        results = pipeline.run()
        console.print(f"\n[green]✓ Report:[/green]  {results['report_path']}")
        console.print(f"[green]✓ Results:[/green] {results['results_path']}")
        if results.get("gguf_path"):
            console.print(f"[green]✓ GGUF:[/green]    {results['gguf_path']}")
        return 0

    except QuantizationPipelineError as e:
        console.print(f"\n[bold red]✗ Quality gate failure:[/bold red] {e}")
        return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130

    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline error:[/bold red] {e}")
        logger.exception("Pipeline error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
