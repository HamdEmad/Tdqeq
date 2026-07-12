# src/pipeline.py

"""
Pipeline — wires all components together for one PDF at a time.

Flow:
    Phase 1 — Load all pages sequentially (CPU-bound, fast).
    Phase 2 — Batch YOLO detection across all pages at once (GPU-bound).
    Phase 3 — Clip detected regions and batch parse in one call.

Usage:
    pipeline = Pipeline(
        loader=loader,
        detector=detector,
        clipper=clipper,
        parser=parser,
        batch_size=4,
        accelerate=False,
    )

    # Run extraction — returns List[RawTable]
    tables = pipeline.run("path/to/document.pdf")

    # Convert to pandas DataFrame
    df = RawTable.to_pandas(tables)
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from tdqeq.detector.table_detector import TableDetector
from tdqeq.extractor.table_parser import TableParser
from tdqeq.extractor.text_clipper import TextClipper
from tdqeq.loader.pdf_loader import PDFLoader
from tdqeq.types import ClippedRegion, PageBundle, RawTable


class Pipeline:
    """
    End-to-end PDF table extraction pipeline.

    All modules are injected at construction time — easy to test,
    easy to swap individual components.

    Args:
        loader:      PDFLoader instance
        detector:    TableDetector instance
        clipper:     TextClipper instance
        parser:      TableParser instance
        batch_size:  batch size for both YOLO detection and table parsing
        accelerate:  if True, parser always uses the fast SlaNet-Plus model
    """

    def __init__(
        self,
        loader: PDFLoader,
        detector: TableDetector,
        clipper: TextClipper,
        parser: TableParser,
        batch_size: int = 4,
        accelerate: bool = False,
    ):
        self._loader = loader
        self._detector = detector
        self._clipper = clipper
        self._parser = parser
        self._batch_size = batch_size
        self._accelerate = accelerate

        # Apply accelerate flag to parser via its public interface
        self._parser.set_accelerate(accelerate)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        pdf_path: Union[str, Path, bytes],
        page_range: Optional[tuple] = None,
    ) -> List[RawTable]:
        """
        Run the full extraction pipeline on one PDF.

        Phase 1: Load all pages sequentially.
        Phase 2: Batch YOLO detection across all pages.
        Phase 3: Clip regions and batch parse.
        Phase 4: Resolve captions (heuristic fallback + style validation).
        Phase 5: Batch parse all regions.

        Args:
            pdf_path:   path to the PDF file (str or Path) or raw PDF bytes
            page_range: optional (start, end) 0-indexed inclusive.
                        None = all pages.

        Returns:
            List[RawTable] — all extracted tables, sorted by page then
            vertical position.  Empty list if no tables found.
        """
        # ── Phase 1: Load all pages ───────────────────────────────────
        pages = list(self._loader.stream(pdf_path, page_range=page_range))

        if not pages:
            logger.info("  No pages loaded.")
            return []

        logger.info(f"  Loaded {len(pages)} page(s)")

        # ── Phase 2: Batch detect across all pages ────────────────────
        try:
            all_page_detections = self._detector.detect_batch(
                pages, batch_size=self._batch_size
            )
        except Exception as e:
            logger.error(f"  Batch detection failed: {e}")
            return []

        # ── Phase 3: Clip regions ─────────────────────────────────────
        all_regions = []

        for page, detections in zip(pages, all_page_detections):
            if not detections:
                logger.debug(f"  No tables on page {page.page_number}")
                continue

            regions = self._clipper.clip_all(detections, page)
            all_regions.extend(regions)

        if not all_regions:
            logger.info("  No table regions clipped.")
            return []

        # ── Phase 4: Batch parse ──────────────────────────────────────
        logger.info(
            f"  {len(all_regions)} region(s) across {len(pages)} page(s) — "
            "running batch inference..."
        )

        raw_tables = self._parser.parse_all(all_regions)

        logger.info(f"  Extracted {len(raw_tables)} table(s).")
        return raw_tables
