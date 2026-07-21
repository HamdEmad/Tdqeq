import numpy as np
from loguru import logger
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from rapid_table import ModelType, RapidTable, RapidTableInput

from tdqeq.exceptions import ExtractionError, ModelNotLoadedError
from tdqeq.types import Cell, ClippedRegion, RawTable, TableType, Word
from tdqeq.config import settings
from tdqeq.extractor.paddle_table_cls import PaddleTableClsModel

# cls_score below this threshold for a WirelessTable → prefer UniTable
WIRELESS_UNITABLE_THRESHOLD = 80


class TableParser:
    """Orchestrates table structure recognition across a batch of clipped regions.

    On construction the correct subset of models is loaded based on the active
    routing mode, so there is no lazy-loading indirection at inference time.

    Args:
        mode:       Routing mode for model selection:
                    - "auto":   auto select between faster mode and more accuracy
                                mode based on the hardness of the table
                    - "tdqeq":  faster but lower accuracy
                    - "tdqeq+": high accuracy but slowest
        device:     Device to run inference on ("cuda" or "cpu")
        batch_size: Batch size for rapid_table and classification inference
    """

    _VALID_MODES = {"auto", "tdqeq", "tdqeq+"}

    def __init__(
        self,
        mode: str = "auto",
        device: str = "cuda",
        batch_size: int = 4,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Choose from {self._VALID_MODES}.")

        self._mode = mode
        self._device = device
        self._batch_size = batch_size

        self._cls: Optional[PaddleTableClsModel] = None
        self._slanet: Optional[RapidTable] = None
        self._unitable: Optional[RapidTable] = None

        self._load_models_for_mode(mode)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch routing mode and reload the required models.

        Args:
            mode: One of "auto", "tdqeq", or "tdqeq+".
        """
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Choose from {self._VALID_MODES}.")
        self._mode = mode
        self._load_models_for_mode(mode)

    def parse(self, region: ClippedRegion) -> RawTable:
        """Parse a single clipped region. Convenience wrapper around parse_all."""
        results = self.parse_all([region])
        if not results:
            raise ExtractionError(
                f"No output from table on page {region.detection.page_number}"
            )
        return results[0]

    def parse_all(self, regions: List[ClippedRegion]) -> List[RawTable]:
        """Parse a batch of clipped regions.

        Flow:
            1. Build OCR map — filter empty regions, convert words to rapid_table format.
            2. Classify all — run batch classification (auto mode only).
            3. Route and group — assign each region to a model.
            4. Run inference — call rapid_table per model group.
            5. Assemble and return sorted RawTable results.

        Args:
            regions: List of clipped table regions to process.

        Returns:
            List of RawTable sorted by (page_number, top y-coordinate).
        """
        if not regions:
            return []

        # ── Stage 1: Filter and build OCR map ─────────────────────────
        valid, ocr_map = self._build_ocr_map(regions)
        if not valid:
            return []

        # ── Stage 2: Classify (auto mode only) ────────────────────────
        cls_info: Dict[int, Tuple[TableType, float]] = self._classify_all(valid)

        # ── Stage 3: Route regions to models ──────────────────────────
        # groups maps model instance → list of (original_idx, region)
        groups: Dict[int, List[Tuple[int, ClippedRegion]]] = {}

        for idx, region in valid:
            table_type, cls_score = cls_info.get(idx, (TableType.WIRELESS, 100.0))
            model = self._route(table_type, cls_score)
            groups.setdefault(id(model), []).append((idx, region))

        # Store the model object keyed by its id for later retrieval
        model_by_id: Dict[int, RapidTable] = {}
        for idx, region in valid:
            table_type, cls_score = cls_info.get(idx, (TableType.WIRELESS, 100.0))
            model = self._route(table_type, cls_score)
            model_by_id[id(model)] = model

        # ── Stage 4: Inference per model group ────────────────────────
        results: Dict[int, RawTable] = {}

        for model_id, group in groups.items():
            model = model_by_id[model_id]
            indices = [idx for idx, _ in group]
            grp_regions = [r for _, r in group]

            try:
                htmls, cells_bboxes = self._run_model(
                    model=model,
                    images=[r.table_image for r in grp_regions],
                    ocr_results=[ocr_map[idx] for idx in indices],
                    tables_header=[r.caption_text for r in grp_regions],
                )
            except ExtractionError:
                continue

            for i, (idx, region) in enumerate(group):
                try:
                    raw_table = self._build_raw_table(
                        region=region,
                        html=htmls[i],
                        cell_bboxes=cells_bboxes[i],
                        model=model,
                        cls_score=cls_info.get(idx, (TableType.WIRELESS, None))[1],
                    )
                    results[idx] = raw_table
                except Exception:
                    continue

        # ── Stage 5: Sort by page then vertical position ──────────────
        return sorted(results.values(), key=lambda t: (t.page_number, t.table_bbox[1]))

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    def words_to_rapid_format(
        self,
        words: List[Word],
        image_dpi: int,
        bbox_pdf: Tuple[float, float, float, float],
    ) -> Tuple:
        """Convert PDF word bounding boxes to the numpy format expected by rapid_table.

        Args:
            words:     List of Word objects with PDF coordinate bboxes.
            image_dpi: DPI used when rasterizing the page image.
            bbox_pdf:  (x0, y0, x1, y1) of the table crop in PDF points.

        Returns:
            Tuple of (bboxes_array, texts_tuple, scores_tuple).
        """
        if not words:
            return (np.zeros((0, 4, 2), dtype=np.float32), (), ())

        scale = image_dpi / 72.0
        origin_x, origin_y = bbox_pdf[0], bbox_pdf[1]

        bboxes, texts = [], []
        for w in words:
            left   = (w.x0 - origin_x) * scale
            top    = (w.y0 - origin_y) * scale
            right  = (w.x1 - origin_x) * scale
            bottom = (w.y1 - origin_y) * scale
            bboxes.append([[left, top], [right, top], [right, bottom], [left, bottom]])
            texts.append(w.text)

        return (
            np.array(bboxes, dtype=np.float32),
            tuple(texts),
            tuple(1.0 for _ in words),
        )

    # ------------------------------------------------------------------
    # Private — parse_all stages
    # ------------------------------------------------------------------

    def _build_ocr_map(
        self,
        regions: List[ClippedRegion],
    ) -> Tuple[List[Tuple[int, ClippedRegion]], Dict[int, Tuple]]:
        """Filter regions without text and convert words to rapid_table OCR format.

        Args:
            regions: All clipped regions from the pipeline.

        Returns:
            valid:   List of (original_index, region) for regions that have words.
            ocr_map: Mapping of original_index → rapid_table OCR tuple.
        """
        valid: List[Tuple[int, ClippedRegion]] = []
        ocr_map: Dict[int, Tuple] = {}

        for idx, region in enumerate(regions):
            if not region.table_words:
                continue
            ocr_map[idx] = self.words_to_rapid_format(
                region.table_words, region.image_dpi, region.bbox_pdf
            )
            valid.append((idx, region))

        return valid, ocr_map

    def _classify_all(
        self,
        valid: List[Tuple[int, ClippedRegion]],
    ) -> Dict[int, Tuple[TableType, float]]:
        """Run batch table classification in auto mode.

        In any other mode classification is unnecessary because routing is fixed.
        If the cls model failed to load (missing weights), falls back to an empty
        dict so callers use the default routing.

        Args:
            valid: List of (original_index, region) pairs.

        Returns:
            Mapping of original_index → (TableType, cls_score).
            Empty dict when classification is skipped or fails.
        """
        if self._mode != "auto" or self._cls is None:
            return {}

        img_info_list = [
            {"wired_table_img": region.table_image, "table_res": {}}
            for _, region in valid
        ]
        try:
            self._cls.batch_predict(img_info_list, batch_size=self._batch_size)
        except Exception as exc:
            logger.warning(
                f"Batch table classification failed: {exc}. "
                "Falling back to default routing (all tables → SlaNet-Plus)."
            )
            return {}

        result: Dict[int, Tuple[TableType, float]] = {}
        for (idx, _), img_info in zip(valid, img_info_list):
            table_res = img_info.get("table_res", {})
            cls_score = float(table_res.get("cls_score", 0.0))
            try:
                table_type = TableType(table_res.get("cls_label", "WirelessTable"))
            except ValueError:
                table_type = TableType.WIRELESS
            result[idx] = (table_type, cls_score)

        return result

    def _route(self, table_type: TableType, cls_score: float) -> RapidTable:
        """Return the appropriate model instance for this table.

        Args:
            table_type: Wired or Wireless classification.
            cls_score:  Confidence score from the classifier (0–100).

        Returns:
            The RapidTable model to use for inference.
        """
        if self._mode == "tdqeq":
            return self._slanet
        if self._mode == "tdqeq+":
            return self._unitable

        # "auto" — dynamic routing based on table type and classifier confidence
        if table_type == TableType.WIRED:
            return self._unitable
        if cls_score < WIRELESS_UNITABLE_THRESHOLD:
            return self._unitable
        return self._slanet

    def _run_model(
        self,
        model: RapidTable,
        images: List[np.ndarray],
        ocr_results: List[Tuple],
        tables_header: List[str],
    ) -> Tuple[List[str], List]:
        """Call rapid_table inference on a batch of table images.

        Args:
            model:         The RapidTable model to use.
            images:        List of table image arrays.
            ocr_results:   List of OCR tuples (bboxes, texts, scores).
            tables_header: Caption text per table (may be empty string).

        Returns:
            (htmls, cells_bboxes) — one entry per image.

        Raises:
            ExtractionError: on inference failure.
        """
        if not ocr_results:
            raise ExtractionError(
                "No words found in table region — table image skipped. "
                "This may be an image-only table or a detection error."
            )

        try:
            result = model(images, ocr_results, batch_size=self._batch_size)
            htmls = result.pred_htmls
            cells_bboxes = result.cell_bboxes
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"rapid_table inference failed: {exc}") from exc

        # Embed caption as a CSS class on the <table> tag when available
        for i, header in enumerate(tables_header):
            if header:
                htmls[i] = htmls[i].replace("<table", f'<table class="{header.strip()}"')

        return htmls, cells_bboxes

    def _build_raw_table(
        self,
        region: ClippedRegion,
        html: str,
        cell_bboxes: List,
        model: RapidTable,
        cls_score: Optional[float],
    ) -> RawTable:
        """Assemble a RawTable from rapid_table outputs and region metadata.

        Args:
            region:      The source clipped region.
            html:        Predicted HTML table string.
            cell_bboxes: Cell bounding boxes in pixel coordinates.
            model:       The model that produced the predictions.
            cls_score:   Classifier confidence score (None if classification skipped).

        Returns:
            A fully populated RawTable.
        """
        cells = self._parse_html(html, cell_bboxes)
        cells = self._reanchor_cells(cells, region.bbox_pdf, region.image_dpi)
        self._assign_words_to_cells(cells, region.table_words)

        model_label = (
            str(model.cfg.model_type.value)
            if hasattr(model.cfg.model_type, "value")
            else str(model.cfg.model_type)
        )

        return RawTable(
            page_number=region.detection.page_number,
            table_bbox=region.bbox_pdf,
            page_size=region.page_size,
            detection_confidence=region.detection.confidence,
            caption=region.caption_text, 
            cls=model_label,
            cls_score=cls_score,
            html=html,
            cells=cells,
            row_count=max((c.row for c in cells), default=0) + 1,
            col_count=max((c.col for c in cells), default=0) + 1
        )

    # ------------------------------------------------------------------
    # Private — geometry helpers
    # ------------------------------------------------------------------

    def _parse_html(self, html: str, cell_bboxes: List) -> List[Cell]:
        """Parse rapid_table HTML output into a list of Cell objects.

        Args:
            html:        HTML string produced by rapid_table.
            cell_bboxes: Parallel list of bounding boxes for each <td>.

        Returns:
            List of Cell objects with row, col, text, and bbox.

        Raises:
            ExtractionError: if parsing yields no cells.
        """
        try:
            rows = BeautifulSoup(html, "html.parser").find_all("tr")
        except Exception as exc:
            raise ExtractionError(f"HTML parsing failed: {exc}") from exc

        cells: List[Cell] = []
        cell_index = 0

        for row_idx, row in enumerate(rows):
            for col_idx, td in enumerate(row.find_all("td")):
                if cell_index < len(cell_bboxes):
                    points = cell_bboxes[cell_index].reshape(-1, 2)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
                else:
                    bbox = (0.0, 0.0, 0.0, 0.0)

                cells.append(
                    Cell(
                        row=row_idx,
                        col=col_idx,
                        text=td.get_text(separator=" ").strip(),
                        bbox=bbox,
                    )
                )
                cell_index += 1

        if not cells:
            raise ExtractionError(
                "rapid_table returned no cells — "
                "image may be blank or model may have failed silently."
            )
        return cells

    def _reanchor_cells(
        self,
        cells: List[Cell],
        bbox_pdf: Tuple[float, float, float, float],
        image_dpi: int,
    ) -> List[Cell]:
        """Convert cell bounding boxes from pixel space back to PDF point space.

        Args:
            cells:     Cells with pixel-coordinate bboxes.
            bbox_pdf:  Table origin in PDF points (x0, y0, x1, y1).
            image_dpi: DPI of the rasterized image.

        Returns:
            New list of Cell objects with PDF-coordinate bboxes.
        """
        scale = 72.0 / image_dpi
        origin_x, origin_y = bbox_pdf[0], bbox_pdf[1]

        return [
            Cell(
                row=cell.row,
                col=cell.col,
                text=cell.text,
                bbox=(
                    round(origin_x + cell.bbox[0] * scale, 2),
                    round(origin_y + cell.bbox[1] * scale, 2),
                    round(origin_x + cell.bbox[2] * scale, 2),
                    round(origin_y + cell.bbox[3] * scale, 2),
                ),
            )
            for cell in cells
        ]

    def _assign_words_to_cells(
        self, cells: List[Cell], table_words: List[Word]
    ) -> None:
        """Map PDF word objects to their containing cells using center-point geometry.

        Each word's center point is compared against midpoint-divided column and
        row boundaries derived from the cell bounding boxes.

        Args:
            cells:       Cells with PDF-coordinate bboxes (mutated in-place).
            table_words: Words extracted from the PDF page within the table region.
        """
        if not cells or not table_words:
            return

        def _midpoint_bounds(ranges: Dict[int, List[float]]) -> Dict[int, Tuple[float, float]]:
            sorted_keys = sorted(ranges)
            bounds = {}
            for i, k in enumerate(sorted_keys):
                lo0, hi0 = ranges[k]
                lo = (ranges[sorted_keys[i - 1]][1] + lo0) / 2.0 if i > 0 else 0.0
                hi = (
                    (hi0 + ranges[sorted_keys[i + 1]][0]) / 2.0
                    if i < len(sorted_keys) - 1
                    else float("inf")
                )
                bounds[k] = (lo, hi)
            return bounds

        col_ranges: Dict[int, List[float]] = {}
        row_ranges: Dict[int, List[float]] = {}

        for cell in cells:
            x0, y0, x1, y1 = cell.bbox
            c, r = cell.col, cell.row
            col_ranges[c] = [min(col_ranges[c][0], x0), max(col_ranges[c][1], x1)] if c in col_ranges else [x0, x1]
            row_ranges[r] = [min(row_ranges[r][0], y0), max(row_ranges[r][1], y1)] if r in row_ranges else [y0, y1]

        col_bounds = _midpoint_bounds(col_ranges)
        row_bounds = _midpoint_bounds(row_ranges)

        slot_words: Dict[Tuple[int, int], List[Word]] = {}
        for word in table_words:
            cx = (word.x0 + word.x1) / 2.0
            cy = (word.y0 + word.y1) / 2.0
            col = next((c for c, (lo, hi) in col_bounds.items() if lo <= cx < hi), None)
            row = next((r for r, (lo, hi) in row_bounds.items() if lo <= cy < hi), None)
            if col is not None and row is not None:
                slot_words.setdefault((row, col), []).append(word)

        for cell in cells:
            words = slot_words.get((cell.row, cell.col), [])
            cell.words = words
            if not cell.text.strip() and words:
                cell.text = " ".join(w.text for w in sorted(words, key=lambda w: w.x0))

    # ------------------------------------------------------------------
    # Private — model lifecycle
    # ------------------------------------------------------------------

    def _load_models_for_mode(self, mode: str) -> None:
        """Load only the models required for the given routing mode.

        Mode → models loaded:
            "auto"   → cls (may be None if weights missing), slanet, unitable
            "tdqeq"  → slanet only
            "tdqeq+" → unitable only

        Args:
            mode: The routing mode to prepare models for.
        """
        self._cls = None
        self._slanet = None
        self._unitable = None

        if mode == "auto":
            self._cls = self._load_cls_model()      # None if weights are missing
            self._slanet = self._load_model(ModelType.SLANETPLUS)
            self._unitable = self._load_model(ModelType.UNITABLE)
        elif mode == "tdqeq":
            self._slanet = self._load_model(ModelType.SLANETPLUS)
        elif mode == "tdqeq+":
            self._unitable = self._load_model(ModelType.UNITABLE)

    def _load_model(self, model_type: ModelType) -> RapidTable:
        """Instantiate a RapidTable model.

        Args:
            model_type: The rapid_table ModelType enum value.

        Returns:
            Loaded RapidTable instance.

        Raises:
            ModelNotLoadedError: if the model cannot be initialized.
        """
        try:
            return RapidTable(RapidTableInput(model_type=model_type))
        except Exception as exc:
            raise ModelNotLoadedError(f"Failed to load {model_type}: {exc}") from exc

    def _load_cls_model(self) -> Optional[PaddleTableClsModel]:
        """Instantiate the PaddleTableClsModel for table type classification.

        Auto-downloads the ONNX weights from HuggingFace Hub if not configured.

        Returns:
            Loaded classifier, or None if loading fails (classification is then skipped).
        """
        try:
            return PaddleTableClsModel(weight=settings.TABLE_CLS_WEIGHTS_PATH)
        except Exception as exc:
            logger.warning(
                f"Failed to load PaddleTableClsModel: {exc}. "
                "Bypassing table classification — all tables will route via SlaNet-Plus."
            )
            return None
