import numpy as np
from loguru import logger
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional
from rapid_table import ModelType, RapidTable, RapidTableInput

from tdqeq.exceptions import (
    ExtractionError,
    ModelNotLoadedError,
)
from tdqeq.types import (
    Cell,
    ClippedRegion,
    RawTable,
    TableType,
    Word,
)
from tdqeq.extractor.paddle_table_cls import PaddleTableClsModel

WIRELESS_UNITABLE_THRESHOLD = 80  # cls_score below this → UniTable

class TableParser:
    def __init__(
        self,
        accelerate: bool = False,
        device: str = "cuda",
        batch_size: int = 4,
    ):
        self._accelerate = accelerate
        self._device = device
        self._batch_size = batch_size

        self._cls = self._load_cls_model()
        self._slanet = self._load_model(model_type=ModelType.SLANETPLUS)
        self._unitable = self._load_model(model_type=ModelType.UNITABLE)

    def set_accelerate(self, flag: bool) -> None:
        """Set the accelerate flag for model routing (public setter)."""
        self._accelerate = flag

    def parse(self, region: ClippedRegion) -> RawTable:
        results = self.parse_all([region])
        if not results:
            raise ExtractionError(
                f"No output from table on page {region.detection.page_number}"
            )
        return results[0]

    def parse_all(self, regions: List[ClippedRegion]) -> List[RawTable]:
        if not regions:
            return []

        valid: List[Tuple[int, ClippedRegion]] = []
        ocr_map: dict = {}

        for idx, region in enumerate(regions):
            if not region.table_words:
                continue
            ocr = self.words_to_rapid_format(
                region.table_words, region.image_dpi, region.bbox_pdf
            )
            ocr_map[idx] = ocr
            valid.append((idx, region))

        if not valid:
            return []

        wired_group: List[Tuple[int, ClippedRegion]] = []
        wireless_group: List[Tuple[int, ClippedRegion]] = []

        for idx, region in valid:
            try:
                table_type, cls_score = self._classify(region.table_image)
            except ExtractionError:
                table_type = TableType.WIRELESS
                cls_score = 0.0

            model = self._route(table_type, cls_score)
            if model is self._unitable:
                wired_group.append((idx, region))
            else:
                wireless_group.append((idx, region))

        results: dict = {}

        for group, model in [
            (wired_group, self._unitable),
            (wireless_group, self._slanet),
        ]:
            if not group:
                continue

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
                    cells = self._parse_html(htmls[i], cells_bboxes[i])
                    cells = self._reanchor_cells(
                        cells, region.bbox_pdf, region.image_dpi
                    )
                    self._assign_words_to_cells(cells, region.table_words)

                    row_count = max((c.row for c in cells), default=0) + 1
                    col_count = max((c.col for c in cells), default=0) + 1

                    results[idx] = RawTable(
                        page_number=region.detection.page_number,
                        table_bbox=region.bbox_pdf,
                        page_size=region.page_size,
                        caption=region.caption_text,
                        cells=cells,
                        row_count=row_count,
                        col_count=col_count,
                        detection_confidence=region.detection.confidence,
                        cls=str(model.cfg.model_type.value)
                        if hasattr(model.cfg.model_type, "value")
                        else str(model.cfg.model_type),
                        html=htmls[i],
                    )
                except Exception:
                    continue

        return sorted(results.values(), key=lambda t: (t.page_number, t.table_bbox[1]))

    def _assign_words_to_cells(
        self, cells: List[Cell], table_words: List[Word]
    ) -> None:
        if not cells or not table_words:
            return

        col_ranges: dict = {}
        for cell in cells:
            c = cell.col
            if c not in col_ranges:
                col_ranges[c] = [cell.bbox[0], cell.bbox[2]]
            else:
                col_ranges[c][0] = min(col_ranges[c][0], cell.bbox[0])
                col_ranges[c][1] = max(col_ranges[c][1], cell.bbox[2])

        sorted_cols = sorted(col_ranges)
        col_bounds = {}
        for i, c in enumerate(sorted_cols):
            x0 = col_ranges[c][0]
            x1 = col_ranges[c][1]
            lo = (col_ranges[sorted_cols[i - 1]][1] + x0) / 2.0 if i > 0 else 0.0
            hi = (
                (x1 + col_ranges[sorted_cols[i + 1]][0]) / 2.0
                if i < len(sorted_cols) - 1
                else float("inf")
            )
            col_bounds[c] = (lo, hi)

        row_ranges: dict = {}
        for cell in cells:
            r = cell.row
            if r not in row_ranges:
                row_ranges[r] = [cell.bbox[1], cell.bbox[3]]
            else:
                row_ranges[r][0] = min(row_ranges[r][0], cell.bbox[1])
                row_ranges[r][1] = max(row_ranges[r][1], cell.bbox[3])

        sorted_rows = sorted(row_ranges)
        row_bounds = {}
        for i, r in enumerate(sorted_rows):
            y0 = row_ranges[r][0]
            y1 = row_ranges[r][1]
            lo = (row_ranges[sorted_rows[i - 1]][1] + y0) / 2.0 if i > 0 else 0.0
            hi = (
                (y1 + row_ranges[sorted_rows[i + 1]][0]) / 2.0
                if i < len(sorted_rows) - 1
                else float("inf")
            )
            row_bounds[r] = (lo, hi)

        slot_words: dict = {}
        for word in table_words:
            cx = (word.x0 + word.x1) / 2.0
            cy = (word.y0 + word.y1) / 2.0

            col = next((c for c, (lo, hi) in col_bounds.items() if lo <= cx < hi), None)
            if col is None:
                continue
            row = next((r for r, (lo, hi) in row_bounds.items() if lo <= cy < hi), None)
            if row is None:
                continue
            slot_words.setdefault((row, col), []).append(word)

        for cell in cells:
            words = slot_words.get((cell.row, cell.col), [])
            cell.words = words
            if not cell.text.strip() and words:
                cell.text = " ".join(w.text for w in sorted(words, key=lambda w: w.x0))

    def _classify(self, image) -> Tuple[TableType, float]:
        if self._cls is None:
            # Fallback if the classification model is not loaded (missing weights)
            return TableType.WIRELESS, 100.0
            
        try:
            table_type, score = self._cls.predict(image)
            return table_type, float(score)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Table classification failed: {e}") from e

    def _route(self, table_type: TableType, cls_score: float = 100.0):
        if self._accelerate:
            return self._slanet

        if table_type == TableType.WIRED:
            return self._unitable

        if cls_score < WIRELESS_UNITABLE_THRESHOLD:
            return self._unitable

        return self._slanet

    def _run_model(
        self,
        model: RapidTable,
        images: List[np.ndarray],
        ocr_results: List[List[Word]],
        tables_header: List[str],
    ) -> Tuple[str, List]:
        if not ocr_results:
            raise ExtractionError(
                "No words found in table region — table image skipped. "
                "This may be an image-only table or a detection error."
            )

        try:
            result = model(images, ocr_results, batch_size=self._batch_size)
            htmls = result.pred_htmls
            cells_bboxes = result.cell_bboxes

            for idx in range(len(htmls)):
                header = tables_header[idx]
                if header:
                    htmls[idx] = htmls[idx].replace(
                        "<table", f'<table class="{header.strip()}"'
                    )
            return htmls, cells_bboxes
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"rapid_table inference failed: {e}") from e

    def words_to_rapid_format(
        self,
        words: List[Word],
        image_dpi: int,
        bbox_pdf: Tuple[float, float, float, float],
    ):
        if not words:
            return (
                np.zeros((0, 4, 2), dtype=np.float32),
                (),
                (),
            )

        scale = image_dpi / 72.0
        origin_x = bbox_pdf[0]
        origin_y = bbox_pdf[1]
        bboxes = []
        texts = []

        for w in words:
            left = (w.x0 - origin_x) * scale
            top = (w.y0 - origin_y) * scale
            right = (w.x1 - origin_x) * scale
            bottom = (w.y1 - origin_y) * scale
            bboxes.append(
                [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ]
            )
            texts.append(w.text)

        return (
            np.array(bboxes, dtype=np.float32),
            tuple(texts),
            tuple(1.0 for _ in words),
        )

    def _parse_html(
        self,
        html: str,
        cell_bboxes: List,
    ) -> List[Cell]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
        except Exception as e:
            raise ExtractionError(f"HTML parsing failed: {e}") from e

        cells = []
        cell_index = 0

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for td in row.find_all("td"):
                text = td.get_text(separator=" ").strip()

                if cell_index < len(cell_bboxes):
                    raw_bbox = cell_bboxes[cell_index]
                    points = raw_bbox.reshape(
                        (-1, 2)
                    )
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
                else:
                    bbox = (0.0, 0.0, 0.0, 0.0)

                cells.append(
                    Cell(
                        row=row_idx,
                        col=col_idx,
                        text=text,
                        bbox=bbox,
                    )
                )

                cell_index += 1
                col_idx += 1

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
        scale = 72.0 / image_dpi
        origin_x = bbox_pdf[0]
        origin_y = bbox_pdf[1]
        reanchored = []

        for cell in cells:
            px0, py0, px1, py1 = cell.bbox
            reanchored.append(
                Cell(
                    row=cell.row,
                    col=cell.col,
                    text=cell.text,
                    bbox=(
                        round(origin_x + px0 * scale, 2),
                        round(origin_y + py0 * scale, 2),
                        round(origin_x + px1 * scale, 2),
                        round(origin_y + py1 * scale, 2),
                    ),
                )
            )

        return reanchored

    def _load_model(
        self,
        model_type: str,
    ) -> RapidTable:
        try:
            return RapidTable(RapidTableInput(model_type=model_type))
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load {model_type} : {e}") from e

    def _load_cls_model(self) -> Optional[PaddleTableClsModel]:
        # User removed the weight parameter, so we bypass it to avoid crashing.
        # It requires an ONNX file which is missing.
        logger.warning("PaddleTableClsModel weights not provided. Bypassing table classification.")
        return None
