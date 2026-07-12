# src/types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Coordinate convention (read this before touching any bbox field)
#
# PDF-space:   points (1pt = 1/72 inch), origin top-left, used by fitz
# Pixel-space: pixels at a given DPI, origin top-left, used by YOLO
#
# Conversion:  pixel     = pdf_point * (dpi / 72)
#              pdf_point = pixel     * (72  / dpi)
#
# ALL bboxes are stored in PDF-space EXCEPT:
#   - Detection.bbox              → pixel-space  (direct from YOLO)
#   - Detection.matched_caption_bbox → pixel-space  (direct from YOLO)
#   - ClippedRegion.table_image   → pixel-space  (numpy crop for rapid_table)
#   - ClippedRegion.bbox_pdf      → PDF-space    (converted by TextClipper)
#
# TextClipper is the ONLY box that performs pixel → PDF-space conversion.
# No other module should do this math.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 1 output — PDF Loader
# ---------------------------------------------------------------------------


@dataclass
class Word:
    """Single word extracted by fitz. Coordinates in PDF-space (points)."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    font: str
    size: int
    color_rgb: Tuple[int, int, int]
    bidi: int
    char_flags: int
    alpha: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "text": self.text,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "font": self.font,
            "size": self.size,
            "color_rgb": list(self.color_rgb),
            "bidi": self.bidi,
            "char_flags": self.char_flags,
            "alpha": self.alpha,
        }


@dataclass
class PageBundle:
    """Everything extracted from a single PDF page."""

    page_number: int  # 0-indexed
    image: np.ndarray  # HxWx3 RGB, pixel-space
    image_dpi: int  # DPI used to render image
    page_size: Tuple[float, float]  # (width, height) in PDF points
    words: List[Word]  # all words on page, PDF-space


# ---------------------------------------------------------------------------
# Stage 2 output — Table Detector
# ---------------------------------------------------------------------------


class DetectionLabel(Enum):
    TABLE = "table"
    CAPTION = "caption"


@dataclass
class Detection:
    """Single YOLO detection. All bboxes are in PIXEL-space."""

    page_number: int
    label: DetectionLabel
    bbox: Tuple[float, float, float, float]  # (x0,y0,x1,y1) pixels
    confidence: float
    matched_caption_bbox: Optional[Tuple[float, float, float, float]] = (
        None  # pixels, filled by detector
    )
    _category_id: int = -1  # YOLO category ID, used for tiered caption matching


# ---------------------------------------------------------------------------
# Stage 3 output — Extraction layer
# ---------------------------------------------------------------------------
class TableType(Enum):
    WIRED = "WiredTable"
    WIRELESS = "WirelessTable"


@dataclass
class ClippedRegion:
    """
    Everything needed by the Table Parser (rapid_table) for one table.

    table_words and bbox_pdf are in PDF-space.
    table_image is in pixel-space (numpy crop passed directly to rapid_table).
    caption_text is the resolved string from the nearest caption region.
    """

    detection: Detection
    table_words: List[Word]  # words inside table bbox, PDF-space
    bbox_pdf: Tuple[float, float, float, float]  # table bbox in PDF-space
    table_image: np.ndarray  # cropped table image, pixel-space
    image_dpi: int
    page_size: Tuple[float, float]  # (width, height) PDF
    caption_text: Optional[str] = None  # resolved from matched_caption_bbox
    caption_words: Optional[List[Word]] = None
    ocr_result: List[List] = None


@dataclass
class Cell:
    """Single cell inside a parsed table."""

    row: int
    col: int
    text: str
    bbox: Tuple[float, float, float, float]  # PDF-space
    words: List[Word] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        d: Dict[str, Any] = {
            "row": self.row,
            "col": self.col,
            "text": self.text,
        }
        return d


@dataclass
class RawTable:
    """
    Fully parsed table for a single detected region.
    Coordinates (table_bbox, cell bboxes) are in PDF-space (points).
    """

    page_number: int
    table_bbox: Tuple[float, float, float, float]
    page_size: Tuple[float, float]
    caption: Optional[str]
    cells: List[Cell]
    row_count: int
    col_count: int
    detection_confidence: float
    cls: str
    html: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON export.

        Args:
            exclude_words: if True, omit per-word coordinate data from cells.
                           Produces a much lighter payload when word-level
                           detail is not needed.
        """
        return {
            "page_number": self.page_number,
            "table_bbox": list(self.table_bbox),
            "page_size": list(self.page_size),
            "caption": self.caption,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "detection_confidence": self.detection_confidence,
            "cls": self.cls,
            "html": self.html,
            "cells": [c.to_dict() for c in self.cells],
        }

    def to_pandas(self):
        """Convert this RawTable to a pandas DataFrame representing the table grid.

        The resulting DataFrame will have shape (row_count, col_count),
        with each cell containing the text from the corresponding Cell object.
        Pandas is imported lazily so it is not a hard dependency.

        Returns:
            pandas.DataFrame representing the tabular data

        Raises:
            ImportError: if pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). "
                "Install it with: pip install pandas"
            ) from e

        # Initialize an empty grid
        grid = [["" for _ in range(self.col_count)] for _ in range(self.row_count)]

        # Fill the grid
        for cell in self.cells:
            if 0 <= cell.row < self.row_count and 0 <= cell.col < self.col_count:
                grid[cell.row][cell.col] = cell.text

        return pd.DataFrame(grid)


# ---------------------------------------------------------------------------
# Pipeline-level exceptions
# ---------------------------------------------------------------------------
