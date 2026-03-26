"""
Tests for gguf_inspector module.
Covers: valid file parsing, invalid files, missing files, formatting helpers.
"""

import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codynamicslab_latch_.gguf_inspector import (
    GGUFMetadata,
    inspect_gguf,
    format_metadata_table,
    metadata_to_dict,
    GGUF_MAGIC,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_gguf(tmp_path) -> Path:
    """Write a minimal valid GGUF stub."""
    path = tmp_path / "valid.gguf"
    with open(path, "wb") as f:
        f.write(GGUF_MAGIC)                          # magic
        f.write(struct.pack("<I", 3))                # version = 3
        f.write(struct.pack("<Q", 42))               # tensor_count = 42
        f.write(struct.pack("<Q", 7))                # kv_count = 7
        f.write(b"\x00" * 64)                        # dummy payload
    return path


@pytest.fixture
def corrupt_gguf(tmp_path) -> Path:
    path = tmp_path / "corrupt.gguf"
    path.write_bytes(b"NOTGGUF_HEADER_DATA_HERE")
    return path


@pytest.fixture
def truncated_gguf(tmp_path) -> Path:
    """Valid magic but missing version field."""
    path = tmp_path / "truncated.gguf"
    path.write_bytes(GGUF_MAGIC)          # only 4 bytes — no version
    return path


@pytest.fixture
def empty_gguf(tmp_path) -> Path:
    path = tmp_path / "empty.gguf"
    path.write_bytes(b"")
    return path


# ── Valid File ────────────────────────────────────────────────────────────────

class TestValidGGUF:
    def test_valid_returns_true(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.valid is True

    def test_version_parsed(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.version == 3

    def test_tensor_count_parsed(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.tensor_count == 42

    def test_kv_count_parsed(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.kv_count == 7

    def test_path_stored(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.path == str(valid_gguf)

    def test_file_size_positive(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.file_size_bytes > 0

    def test_file_size_gb_non_negative(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.file_size_gb >= 0.0

    def test_no_error(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        assert meta.error is None


# ── Invalid / Edge Cases ──────────────────────────────────────────────────────

class TestInvalidGGUF:
    def test_missing_file_returns_invalid(self, tmp_path):
        meta = inspect_gguf(tmp_path / "nonexistent.gguf")
        assert meta.valid is False

    def test_missing_file_has_error(self, tmp_path):
        meta = inspect_gguf(tmp_path / "nonexistent.gguf")
        assert meta.error is not None
        assert "not found" in meta.error.lower() or "nonexistent" in meta.error

    def test_missing_file_zero_size(self, tmp_path):
        meta = inspect_gguf(tmp_path / "nonexistent.gguf")
        assert meta.file_size_bytes == 0

    def test_corrupt_magic_returns_invalid(self, corrupt_gguf):
        meta = inspect_gguf(corrupt_gguf)
        assert meta.valid is False

    def test_corrupt_magic_has_error(self, corrupt_gguf):
        meta = inspect_gguf(corrupt_gguf)
        assert "Invalid magic" in meta.error or "magic" in meta.error.lower()

    def test_truncated_returns_invalid(self, truncated_gguf):
        meta = inspect_gguf(truncated_gguf)
        assert meta.valid is False


# ── Formatting ────────────────────────────────────────────────────────────────

class TestFormatting:
    def test_table_contains_version(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        table = format_metadata_table(meta)
        assert "3" in table

    def test_table_contains_tensor_count(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        table = format_metadata_table(meta)
        assert "42" in table

    def test_table_is_markdown(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        table = format_metadata_table(meta)
        assert "|" in table and "---" in table

    def test_to_dict_is_serialisable(self, valid_gguf):
        import json
        meta = inspect_gguf(valid_gguf)
        d = metadata_to_dict(meta)
        json.dumps(d)  # should not raise

    def test_to_dict_has_required_keys(self, valid_gguf):
        meta = inspect_gguf(valid_gguf)
        d = metadata_to_dict(meta)
        for key in ("path", "valid", "version", "tensor_count", "kv_count",
                    "file_size_bytes", "file_size_gb", "error"):
            assert key in d
