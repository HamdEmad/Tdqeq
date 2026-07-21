"""
test_pipeline.py
End-to-end verification for the refactored Tdqeq pipeline using pytest.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tdqeq.pipeline import Pipeline
from tdqeq.types import RawTable
from tdqeq.config import settings

PDF_PATH = Path("test_input.pdf")


@pytest.fixture(scope="module")
def pipeline():
    if not PDF_PATH.exists():
        pytest.skip(f"Test PDF not found: {PDF_PATH}")

    return Pipeline(
        dpi=150,
        device="cpu",
        batch_size=4,
        mode="auto",
    )


@pytest.fixture(scope="module")
def extracted_tables(pipeline):
    return pipeline.run(PDF_PATH)


def test_pipeline_extraction(extracted_tables):
    """Verify that tables are extracted."""
    assert len(extracted_tables) > 0, "No tables extracted!"
    assert len(extracted_tables) == 13, f"Expected 13 tables, found {len(extracted_tables)}"


def test_json_serialization(extracted_tables, tmp_path):
    """Verify that tables can serialize to JSON."""
    payload_full = [t.to_dict() for t in extracted_tables]
    json_str = json.dumps(payload_full)
    assert len(json_str) > 0


def test_structural_checks(extracted_tables):
    """Verify the dictionary structure of extracted tables."""
    for t in extracted_tables:
        d = t.to_dict()
        assert "cells" in d
        assert "html" in d
        assert "caption" in d
        for c in d["cells"]:
            assert "row" in c and "col" in c and "text" in c


def test_to_pandas(extracted_tables):
    """Verify the RawTable to_pandas conversion."""
    for i, t in enumerate(extracted_tables):
        df = t.to_pandas()
        assert df.shape == (t.row_count, t.col_count), f"Shape mismatch for table {i}"
        if t.cells:
            assert not df.empty or (t.row_count == 0 and t.col_count == 0)
