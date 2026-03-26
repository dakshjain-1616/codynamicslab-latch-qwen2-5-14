"""
GGUF binary header inspector.
Parses GGUF file metadata (magic, version, tensor count, KV pairs)
without loading model weights.  Works on real and mock stub files.
"""

import os
import struct
import logging
from pathlib import Path
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
INSPECT_GGUF = os.getenv("INSPECT_GGUF", "false").lower() == "true"


class GGUFMetadata(NamedTuple):
    """Parsed header fields from a GGUF binary file."""
    path: str
    valid: bool
    version: Optional[int]
    tensor_count: Optional[int]
    kv_count: Optional[int]
    file_size_bytes: int
    file_size_gb: float
    error: Optional[str]


def inspect_gguf(gguf_path) -> GGUFMetadata:
    """
    Parse the GGUF binary header and return a GGUFMetadata record.

    GGUF header layout (little-endian):
        bytes  0-3   : magic  b"GGUF"
        bytes  4-7   : version  uint32
        bytes  8-15  : tensor_count  uint64
        bytes 16-23  : kv_count  uint64
    """
    gguf_path = Path(gguf_path)

    if not gguf_path.exists():
        return GGUFMetadata(
            path=str(gguf_path),
            valid=False,
            version=None,
            tensor_count=None,
            kv_count=None,
            file_size_bytes=0,
            file_size_gb=0.0,
            error=f"File not found: {gguf_path}",
        )

    file_size = gguf_path.stat().st_size

    try:
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                return GGUFMetadata(
                    path=str(gguf_path),
                    valid=False,
                    version=None,
                    tensor_count=None,
                    kv_count=None,
                    file_size_bytes=file_size,
                    file_size_gb=round(file_size / 1e9, 3),
                    error=f"Invalid magic bytes: {magic!r} (expected {GGUF_MAGIC!r})",
                )

            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                return GGUFMetadata(
                    path=str(gguf_path),
                    valid=False,
                    version=None,
                    tensor_count=None,
                    kv_count=None,
                    file_size_bytes=file_size,
                    file_size_gb=round(file_size / 1e9, 3),
                    error="File too small: missing version field",
                )
            version = struct.unpack("<I", version_bytes)[0]

            tensor_count: Optional[int] = None
            kv_count: Optional[int] = None

            tensor_bytes = f.read(8)
            if len(tensor_bytes) == 8:
                tensor_count = struct.unpack("<Q", tensor_bytes)[0]

            kv_bytes = f.read(8)
            if len(kv_bytes) == 8:
                kv_count = struct.unpack("<Q", kv_bytes)[0]

        file_size_gb = round(file_size / 1e9, 3)
        logger.debug(
            f"Inspected {gguf_path.name}: v{version}, "
            f"{tensor_count} tensors, {kv_count} KV pairs, {file_size_gb:.3f} GB"
        )

        return GGUFMetadata(
            path=str(gguf_path),
            valid=True,
            version=version,
            tensor_count=tensor_count,
            kv_count=kv_count,
            file_size_bytes=file_size,
            file_size_gb=round(file_size / 1e9, 3),
            error=None,
        )

    except (IOError, OSError, struct.error) as e:
        return GGUFMetadata(
            path=str(gguf_path),
            valid=False,
            version=None,
            tensor_count=None,
            kv_count=None,
            file_size_bytes=file_size,
            file_size_gb=round(file_size / 1e9, 3),
            error=str(e),
        )


def format_metadata_table(meta: GGUFMetadata) -> str:
    """Format a GGUFMetadata record as a Markdown table."""
    rows = [
        ("File", Path(meta.path).name),
        ("Valid GGUF", "Yes" if meta.valid else f"No — {meta.error}"),
        ("GGUF Version", str(meta.version) if meta.version is not None else "N/A"),
        ("Tensor Count", f"{meta.tensor_count:,}" if meta.tensor_count is not None else "N/A"),
        ("KV Pair Count", f"{meta.kv_count:,}" if meta.kv_count is not None else "N/A"),
        ("File Size", f"{meta.file_size_gb:.3f} GB ({meta.file_size_bytes:,} bytes)"),
    ]
    lines = ["| Property | Value |", "|----------|-------|"]
    for key, val in rows:
        lines.append(f"| {key} | `{val}` |")
    return "\n".join(lines)


def metadata_to_dict(meta: GGUFMetadata) -> dict:
    """Convert GGUFMetadata to a plain dict (JSON-serialisable)."""
    return {
        "path": meta.path,
        "valid": meta.valid,
        "version": meta.version,
        "tensor_count": meta.tensor_count,
        "kv_count": meta.kv_count,
        "file_size_bytes": meta.file_size_bytes,
        "file_size_gb": meta.file_size_gb,
        "error": meta.error,
    }
