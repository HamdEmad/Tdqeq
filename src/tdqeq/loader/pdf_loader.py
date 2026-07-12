# src/loader/pdf_loader.py

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import fitz  # PyMuPDF
import numpy as np

from tdqeq.exceptions import (
    PDFCorruptError,
    PDFLoadError,
    PDFPasswordError,
)
from tdqeq.types import (
    PageBundle,
    Word,
)


class PDFLoader:
    """
    Wraps fitz (PyMuPDF) entirely. No fitz types escape this file.

    All word coordinates in PageBundle are in PDF-space (points).
    All images are RGB numpy arrays rendered at self._dpi.
    image_dpi is stored on every PageBundle so downstream boxes
    can convert pixel-space YOLO bboxes back to PDF-space without
    importing fitz or knowing anything about how rendering worked.
    """

    def __init__(self, dpi: int = 150):
        self._dpi = dpi
        self._scale = dpi / 72.0  # PDF points → pixels multiplier

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def stream(
        self,
        source: Union[str, Path, bytes],
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Iterator[PageBundle]:
        """
        Yield one PageBundle per page.

        Args:
            source:     file path (str or Path) or raw PDF bytes
            page_range: (start, end) 0-indexed inclusive. None = all pages.

        Yields:
            PageBundle for each page in range

        Raises:
            PDFPasswordError: if the PDF is encrypted
            PDFCorruptError:  if the PDF cannot be parsed
            PDFLoadError:     for any other fitz-level failure
        """
        doc = self._open(source)
        try:
            start, end = self._resolve_range(page_range, len(doc))
            for page_num in range(start, end + 1):
                yield self._process_page(doc[page_num], page_num)
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Private: open
    # ------------------------------------------------------------------

    def _open(self, source: Union[str, Path, bytes]) -> fitz.Document:
        try:
            if isinstance(source, bytes):
                doc = fitz.open(stream=source, filetype="pdf")
            else:
                doc = fitz.open(str(source))
        except fitz.FileDataError as e:
            raise PDFCorruptError(f"PDF appears corrupted: {e}") from e
        except Exception as e:
            raise PDFLoadError(f"Failed to open PDF: {e}") from e

        if doc.is_encrypted:
            doc.close()
            raise PDFPasswordError("PDF is password-protected. Skipping.")

        return doc

    # ------------------------------------------------------------------
    # Private: page processing
    # ------------------------------------------------------------------

    def _process_page(self, page: fitz.Page, page_number: int) -> PageBundle:
        """
        Process a single PDF page to extract its image and words.

        Args:
            page (fitz.Page): The PyMuPDF page object.
            page_number (int): The 0-indexed page number.

        Returns:
            PageBundle: A dataclass containing the page image, words, and metadata.

        Raises:
            PDFLoadError: If image rendering or word extraction fails.
        """
        try:
            image = self._render_image(page)
            words = self._extract_words(page)
            page_size = (page.rect.width, page.rect.height)
        except Exception as e:
            raise PDFLoadError(f"Failed to process page {page_number}: {e}") from e

        return PageBundle(
            page_number=page_number,
            image=image,
            image_dpi=self._dpi,
            page_size=page_size,
            words=words,
        )

    def _render_image(self, page: fitz.Page) -> np.ndarray:
        matrix = fitz.Matrix(self._scale, self._scale)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
        return (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, 3)
            .copy()
        )

    def _extract_words(self, page: fitz.Page) -> list[Word]:

        all_words = []

        text_dict = page.get_text("rawdict")

        for block in text_dict["blocks"]:
            if block["type"] != 0:  # Skip non-text blocks
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    # --- Span-level style info ---
                    font = span.get("font", "Unknown")
                    size = span.get("size", 0)
                    color_int = span.get("color", 0)

                    # --- New fields requested ---
                    bidi = span.get("bidi", 0)
                    char_flags = span.get("char_flags", 0)
                    alpha = span.get("alpha", 255)

                    # Convert color integer to RGB
                    if color_int is None:
                        color_rgb = (0, 0, 0)
                    else:
                        r = (color_int >> 16) & 255
                        g = (color_int >> 8) & 255
                        b = color_int & 255
                        color_rgb = (r, g, b)

                    # Process characters to reconstruct words
                    current_word_chars = []
                    current_word_bbox = None

                    for char in span.get("chars", []):
                        c = char.get("c", "")
                        char_bbox = fitz.Rect(char["bbox"])

                        if not c.isspace():
                            current_word_chars.append(c)
                            if current_word_bbox is None:
                                current_word_bbox = char_bbox
                            else:
                                current_word_bbox = current_word_bbox | char_bbox
                        else:
                            if current_word_chars:
                                word_text = "".join(current_word_chars)
                                all_words.append(
                                    {
                                        "x0": round(current_word_bbox.x0, 3),
                                        "y0": round(current_word_bbox.y0, 3),
                                        "x1": round(current_word_bbox.x1, 3),
                                        "y1": round(current_word_bbox.y1, 3),
                                        "text": word_text,
                                        "font": font,
                                        "size": size,
                                        "color_rgb": color_rgb,
                                        "bidi": bidi,
                                        "char_flags": char_flags,
                                        "alpha": alpha,
                                    }
                                )
                                current_word_chars = []
                                current_word_bbox = None

                    # Append last word in span if any
                    if current_word_chars:
                        word_text = "".join(current_word_chars)
                        all_words.append(
                            {
                                "x0": round(current_word_bbox.x0, 3),
                                "y0": round(current_word_bbox.y0, 3),
                                "x1": round(current_word_bbox.x1, 3),
                                "y1": round(current_word_bbox.y1, 3),
                                "text": word_text,
                                "font": font,
                                "size": size,
                                "color_rgb": color_rgb,
                                "bidi": bidi,
                                "char_flags": char_flags,
                                "alpha": alpha,
                            }
                        )

        return [
            Word(
                x0=w["x0"],
                y0=w["y0"],
                x1=w["x1"],
                y1=w["y1"],
                text=w["text"],
                font=w["font"],
                size=w["size"],
                color_rgb=w["color_rgb"],
                bidi=w["bidi"],
                char_flags=w["char_flags"],
                alpha=w["alpha"],
            )
            for w in all_words
            if w["text"].strip()  # drop whitespace-only words
        ]

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _resolve_range(
        self,
        page_range: Optional[Tuple[int, int]],
        total_pages: int,
    ) -> Tuple[int, int]:
        if page_range is None:
            return 0, total_pages - 1
        start, end = page_range
        if start < 0 or end >= total_pages or start > end:
            raise PDFLoadError(
                f"Invalid page_range {page_range} for PDF with {total_pages} pages."
            )
        return start, end
