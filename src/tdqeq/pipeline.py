# src/pipeline.py

"""
Pipeline — wires all components together for one PDF at a time.

Flow:
    Phase 1 — Load all pages sequentially (CPU-bound, fast).
    Phase 2 — Batch YOLO detection across all pages at once (GPU-bound).
    Phase 3 — Clip detected regions and batch parse in one call.

Usage:
    pipeline = Pipeline(
        dpi=200,
        device="cpu",
        batch_size=4,
        mode="tdqeq",
    )

    # Run extraction — returns List[RawTable]
    tables = pipeline.run("path/to/document.pdf")

    # Convert to pandas DataFrame
    df = tables[0].to_pandas()
"""

from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

from tdqeq.config import settings
from tdqeq.detector.table_detector import TableDetector
from tdqeq.extractor.table_parser import TableParser
from tdqeq.extractor.text_clipper import TextClipper
from tdqeq.loader.pdf_loader import PDFLoader
from tdqeq.types import ClippedRegion, RawTable


class Pipeline:
    """End-to-end PDF table extraction pipeline.

    All heavy models are instantiated internally based on the parameters
    provided. Weight paths are read from the environment / config file —
    they are never accepted as direct arguments.

    Args:
        dpi:        Resolution (dots-per-inch) used when rasterizing PDF pages.
                    Higher DPI improves OCR quality at the cost of memory.
                    Defaults to ``settings.DEFAULT_DPI``.
        device:     Inference device for YOLO detection and table recognition.
                    ``"cuda"`` for GPU, ``"cpu"`` for CPU.
        batch_size: Batch size for both YOLO detection and rapid_table parsing.
        mode:       Routing mode for model selection:

                    - ``"auto"``:   auto select between faster mode and more
                                    accuracy mode based on the hardness of the table
                    - ``"tdqeq"``:  faster but lower accuracy
                    - ``"tdqeq+"``: high accuracy but slowest
    """

    _VALID_MODES = {"auto", "tdqeq", "tdqeq+"}

    def __init__(
        self,
        dpi: int = settings.DEFAULT_DPI,
        device: str = "cpu",
        batch_size: int = settings.DEFAULT_BATCH_SIZE,
        mode: str = "auto",
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Choose from {self._VALID_MODES}."
            )

        self._mode = mode
        self._batch_size = batch_size

        self._loader = PDFLoader(dpi=dpi)
        self._detector = TableDetector(device=device)
        self._clipper = TextClipper()
        self._parser = TableParser(mode=mode, device=device, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Advanced / testing
    # ------------------------------------------------------------------

    @classmethod
    def _from_components(
        cls,
        loader: PDFLoader,
        detector: TableDetector,
        clipper: TextClipper,
        parser: TableParser,
        batch_size: int = 4,
        mode: str = "auto",
    ) -> "Pipeline":
        """Construct a Pipeline from pre-built component instances.

        Intended for unit testing and advanced scenarios where fine-grained
        control over each component (e.g. mock injection) is required.
        This is a private API — production code should use ``__init__``.

        Args:
            loader:     PDFLoader instance.
            detector:   TableDetector instance.
            clipper:    TextClipper instance.
            parser:     TableParser instance (mode should match ``mode`` param).
            batch_size: Batch size for detection and parsing.
            mode:       Routing mode — must match the parser's mode.

        Returns:
            A fully wired Pipeline instance.
        """
        instance = object.__new__(cls)
        instance._mode = mode
        instance._batch_size = batch_size
        instance._loader = loader
        instance._detector = detector
        instance._clipper = clipper
        instance._parser = parser
        # Ensure parser mode is in sync with pipeline mode
        parser.set_mode(mode)
        return instance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch the table routing mode at runtime.

        Args:
            mode: One of ``"auto"``, ``"tdqeq"``, or ``"tdqeq+"``.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Choose from {self._VALID_MODES}."
            )
        self._mode = mode
        self._parser.set_mode(mode)

    def run(
        self,
        pdf_path: Union[str, Path, bytes],
        page_range: Optional[tuple] = None,
    ) -> List[RawTable]:
        """Run the full extraction pipeline on one PDF.

        Phase 1: Load all pages sequentially.
        Phase 2: Batch YOLO detection across all pages.
        Phase 3: Clip regions and batch parse.

        Args:
            pdf_path:   Path to the PDF file (str or Path) or raw PDF bytes.
            page_range: Optional (start, end) 0-indexed inclusive page range.
                        ``None`` processes all pages.

        Returns:
            List of :class:`~tdqeq.types.RawTable` sorted by page then vertical
            position. Empty list if no tables are found.
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
        all_regions: List[ClippedRegion] = []

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
