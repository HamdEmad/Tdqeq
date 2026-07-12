# src/extractor/text_clipper.py

from typing import List, Optional, Tuple

import numpy as np

from tdqeq.exceptions import ExtractionError
from tdqeq.types import (
    ClippedRegion,
    Detection,
    DetectionLabel,
    PageBundle,
    Word,
)


class TextClipper:
    """
    Converts YOLO pixel-space bboxes to PDF-space and clips words/images.

    This is the ONLY module that performs pixel → PDF-space conversion.
    No fitz required — words are already extracted in PageBundle.words.

    Coordinate formula:
        pdf_coord = pixel_coord * (72 / image_dpi)

    Usage:
        clipper = TextClipper()
        regions = clipper.clip_all(detections, page)
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def clip(self, detection: Detection, page: PageBundle) -> ClippedRegion:
        """
        Clip words and image for a single TABLE detection.

        Args:
            detection: Detection with label=TABLE, bbox in pixel-space
            page:      PageBundle containing words (PDF-space) and image

        Returns:
            ClippedRegion with table_words, table_image, bbox_pdf,
            and caption_text resolved if matched_caption_bbox is set.

        Raises:
            ExtractionError: if detection label is not TABLE
            ExtractionError: if bbox conversion produces invalid coordinates
        """
        if detection.label != DetectionLabel.TABLE:
            raise ExtractionError(
                f"TextClipper.clip() expects TABLE detections, "
                f"got {detection.label.value} on page {detection.page_number}."
            )

        try:
            bbox_pdf = self._to_pdf_bbox(detection.bbox, page.image_dpi)
            table_words = self._clip_words(page.words, bbox_pdf)
            table_image = self._clip_image(page.image, detection.bbox)
            
            caption_words = None
            caption = None
            if detection.matched_caption_bbox is not None:
                caption_pdf_bbox = self._to_pdf_bbox(detection.matched_caption_bbox, page.image_dpi)
                caption_words = self._clip_words(page.words, caption_pdf_bbox)
                if caption_words:
                    caption = self._words_to_text(caption_words)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                f"Clipping failed on page {detection.page_number}: {e}"
            ) from e

        return ClippedRegion(
            detection=detection,
            table_words=table_words,
            bbox_pdf=bbox_pdf,
            table_image=table_image,
            image_dpi=page.image_dpi,
            page_size=page.page_size,
            caption_text=caption,
            caption_words=caption_words,
        )

    def clip_all(
        self,
        detections: List[Detection],
        page: PageBundle,
    ) -> List[ClippedRegion]:
        """
        Clip words and images for all TABLE detections on a page.

        CAPTION detections are silently skipped — their text is resolved
        inside clip() via _resolve_caption when processing the TABLE
        that owns them.

        Args:
            detections: all Detection objects for this page (mixed labels ok)
            page:       PageBundle for this page

        Returns:
            List[ClippedRegion], one per TABLE detection, same order.

        Raises:
            ExtractionError: if any individual clip fails
        """
        return [
            self.clip(det, page)
            for det in detections
            if det.label == DetectionLabel.TABLE
        ]

    # ------------------------------------------------------------------
    # Private: coordinate conversion
    # ------------------------------------------------------------------

    def _to_pdf_bbox(
        self,
        pixel_bbox: Tuple[float, float, float, float],
        image_dpi: int,
    ) -> Tuple[float, float, float, float]:
        """
        Convert a pixel-space bbox to PDF-space (points).

        Formula: pdf_coord = pixel_coord * (72 / image_dpi)

        Args:
            pixel_bbox: (x0, y0, x1, y1) in pixels
            image_dpi:  DPI used when rendering the page image

        Returns:
            (x0, y0, x1, y1) in PDF points

        Raises:
            ExtractionError: if resulting bbox is degenerate (zero area)
        """
        scale = 72.0 / image_dpi
        x0, y0, x1, y1 = pixel_bbox
        pdf_bbox = (
            x0 * scale,
            y0 * scale,
            x1 * scale,
            y1 * scale,
        )

        if pdf_bbox[2] <= pdf_bbox[0] or pdf_bbox[3] <= pdf_bbox[1]:
            raise ExtractionError(
                f"Degenerate bbox after conversion: "
                f"pixel={pixel_bbox} → pdf={pdf_bbox} at dpi={image_dpi}"
            )

        return pdf_bbox

    # ------------------------------------------------------------------
    # Private: word clipping
    # ------------------------------------------------------------------
    def _clip_words(
        self,
        words: List[Word],
        pdf_bbox: Tuple[float, float, float, float],
    ) -> List[Word]:
        """
        Return words that fall inside pdf_bbox or within a maximum margin of 4 units outside it.

        Args:
            words:    all Word objects for the page (PDF-space)
            pdf_bbox: (x0, y0, x1, y1) clipping region in PDF-space

        Returns:
            Filtered list of Word objects, preserving original order.
        """
        if not words:
            return []

        # 1. Unpack base box and apply the maximum allowance margin of 4
        bx0, by0, bx1, by1 = pdf_bbox
        margin = 4.0

        allowed_x0 = bx0 - margin
        allowed_y0 = by0 - margin
        allowed_x1 = bx1 + margin
        allowed_y1 = by1 + margin

        # 2. Extract coordinates into a matrix of shape (N, 4)
        coords = np.array([[w.x0, w.y0, w.x1, w.y1] for w in words])

        # 3. Create boolean masks using the padded boundaries
        inside_mask = (
            (coords[:, 0] >= allowed_x0) &  # w.x0 is within margin left
            (coords[:, 1] >= allowed_y0) &  # w.y0 is within margin top
            (coords[:, 2] <= allowed_x1) &  # w.x1 is within margin right
            (coords[:, 3] <= allowed_y1)    # w.y1 is within margin bottom
        )

        # 4. Use the mask to grab matching words while preserving original order
        return [words[idx] for idx in np.where(inside_mask)[0]]

    # ------------------------------------------------------------------
    # Private: image clipping
    # ------------------------------------------------------------------

    def _clip_image(
        self,
        page_image: np.ndarray,
        pixel_bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Crop the table region from the page image.

        Stays in pixel-space — this crop is passed directly to rapid_table.

        Args:
            page_image:  full page image HxWx3 RGB numpy array
            pixel_bbox:  (x0, y0, x1, y1) in pixels

        Returns:
            Cropped numpy array of the table region (copy, not a view)

        Raises:
            ExtractionError: if crop coordinates fall outside image bounds
        """
        x0, y0, x1, y1 = (int(v) for v in pixel_bbox)
        h, w = page_image.shape[:2]

        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            raise ExtractionError(
                f"Bbox ({x0},{y0},{x1},{y1}) out of image bounds ({w}x{h}). "
                "YOLO bbox may extend beyond page edges."
            )

        if x1 <= x0 or y1 <= y0:
            raise ExtractionError(f"Degenerate crop bbox: ({x0},{y0},{x1},{y1})")

        # .copy() is mandatory — prevents dangling reference to page_image
        return page_image[y0:y1, x0:x1].copy()

    # ------------------------------------------------------------------
    # Private: caption resolution
    # ------------------------------------------------------------------

    def _resolve_caption(
        self,
        caption_bbox: Optional[Tuple[float, float, float, float]],
        page: PageBundle,
    ) -> Optional[str]:
        """
        Resolve a caption bbox to a text string.

        Converts caption_bbox from pixel-space to PDF-space, clips words
        that fall inside it, and joins them in reading order (top-to-bottom,
        left-to-right within each line).

        Args:
            caption_bbox: pixel-space bbox from Detection.matched_caption_bbox,
                          or None if no caption was matched
            page:         PageBundle containing the word list

        Returns:
            Resolved caption string, or None if no caption bbox or no words found.
        """
        if caption_bbox is None:
            return None

        try:
            pdf_bbox = self._to_pdf_bbox(caption_bbox, page.image_dpi)
            caption_words = self._clip_words(page.words, pdf_bbox)
        except ExtractionError:
            return None

        if not caption_words:
            return None

        return self._words_to_text(caption_words)

    def _words_to_text(self, words: List[Word]) -> str:
        """
        Join words in reading order: sort by (y0, x0) then join with spaces.
        Groups words into lines by proximity of y0 coordinates.
        """
        if not words:
            return ""

        # Sort by vertical position first, then horizontal
        sorted_words = sorted(words, key=lambda w: (w.y0, w.x0))

        # Group into lines — words within 5 PDF points vertically are same line
        LINE_TOLERANCE = 5.0
        lines: List[List[Word]] = []
        current_line: List[Word] = [sorted_words[0]]

        for word in sorted_words[1:]:
            if abs(word.y0 - current_line[0].y0) <= LINE_TOLERANCE:
                current_line.append(word)
            else:
                lines.append(sorted(current_line, key=lambda w: w.x0))
                current_line = [word]
        lines.append(sorted(current_line, key=lambda w: w.x0))

        return " ".join(w.text for line in lines for w in line)
