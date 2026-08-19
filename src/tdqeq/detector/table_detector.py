from pathlib import Path
from typing import List, Optional, Union

from doclayout_yolo import YOLOv10
from tqdm import tqdm

from tdqeq.config import settings
from tdqeq.detector import detection_cleaner, image_enhancer
from tdqeq.detector.caption_heuristic import (
    compute_document_baseline,
    find_heuristic_caption_bbox,
)
from tdqeq.exceptions import DetectionError, ModelNotLoadedError
from tdqeq.types import Detection, PageBundle

# ---------------------------------------------------------------------------
# Default HuggingFace model coordinates
# ---------------------------------------------------------------------------
_HF_REPO_ID = "opendatalab/PDF-Extract-Kit-1.0"
_HF_FILENAME = "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
_HF_REPO_TYPE = "model"


class TableDetector:
    """
    Black box: detects tables and captions on PDF pages using YOLOv10.

    Only Detection objects exit this class — no YOLO types, no raw dicts,
    no category_id integers, no poly lists.

    Weights are auto-downloaded from HuggingFace Hub on first use if no
    local path is provided.

    Usage:
        # Auto-download weights (default)
        detector = TableDetector(device="cuda")

        # Custom local weights
        detector = TableDetector(weight="models/yolo/best.pt", device="cuda")

        detections    = detector.detect(page_bundle)
        all_detections = detector.detect_batch(page_bundles, batch_size=4)
    """

    def __init__(
        self,
        weight: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        imgsz: int = 1280,
        conf: float = 0.1,
        iou: float = 0.45,
    ):
        """
        Load YOLOv10 model once at construction time.

        Args:
            weight: path to YOLO weights file (.pt).
                    If None or omitted, the model is automatically downloaded
                    from HuggingFace Hub (opendatalab/PDF-Extract-Kit-1.0).
            device: 'cuda' or 'cpu'
            imgsz:  inference image size
            conf:   confidence threshold
            iou:    IoU threshold for NMS

        Raises:
            ModelNotLoadedError: if weights cannot be loaded or downloaded
        """
        resolved = self._resolve_weight(weight or settings.YOLO_WEIGHTS_PATH)

        try:
            self._model = YOLOv10(str(resolved)).to(device)
        except Exception as e:
            raise ModelNotLoadedError(
                f"Failed to load YOLO weights from {resolved}: {e}"
            ) from e

        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou

    # ------------------------------------------------------------------
    # Weight resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weight(weight: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the YOLO weight path.

        If *weight* is given and points to an existing file, use it.
        Otherwise download the model from HuggingFace Hub and return
        the cached path.

        Raises:
            ModelNotLoadedError: if the given path does not exist, or if
                                 downloading fails.
        """
        if weight is not None:
            path = Path(weight)
            if not path.exists():
                raise ModelNotLoadedError(
                    f"YOLO weights not found at: {path}. "
                    "Either fix the path or omit 'weight' to auto-download."
                )
            return path

        # Auto-download from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ModelNotLoadedError(
                "'huggingface_hub' is required for auto-downloading weights. "
                "Install it with: pip install huggingface_hub"
            ) from e

        try:
            cached = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=_HF_FILENAME,
                repo_type=_HF_REPO_TYPE,
            )
            return Path(cached)
        except Exception as e:
            raise ModelNotLoadedError(
                f"Failed to download YOLO weights from HuggingFace Hub: {e}. "
                "Check your internet connection or supply a local 'weight' path."
            ) from e

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, page: PageBundle) -> List[Detection]:
        """
        Detect tables and captions on a single page.

        Args:
            page: PageBundle from the PDF Loader

        Returns:
            List[Detection] — TABLE detections with matched_caption filled
            where a valid caption was found above the table.

        Raises:
            DetectionError: if YOLO inference fails
        """
        enhanced = self._enhance(page)
        raw = self._predict([enhanced])
        baseline_style = compute_document_baseline(page.words)
        return self._clean(raw[0], page, baseline_style)

    def detect_batch(
        self,
        pages: List[PageBundle],
        batch_size: int = 4,
    ) -> List[List[Detection]]:
        """
        Detect tables and captions across a list of pages using batched
        YOLO inference for GPU efficiency.

        Args:
            pages:      list of PageBundles from the PDF Loader
            batch_size: number of images per YOLO inference call

        Returns:
            List of Detection lists, one per input page (same order).

        Raises:
            DetectionError: if YOLO inference fails on any batch
        """
        if not pages:
            return []

        # Enhance all images up front
        enhanced_images = [self._enhance(p) for p in pages]

        # Compute document baseline from all words across all pages
        all_words = []
        for p in pages:
            all_words.extend(p.words)
        baseline_style = compute_document_baseline(all_words)

        # Run batched YOLO inference
        all_raw: List[List[dict]] = []
        with tqdm(total=len(pages), desc="Detecting tables") as pbar:
            for i in range(0, len(enhanced_images), batch_size):
                batch = enhanced_images[i : i + batch_size]
                batch_raw = self._predict(batch)
                all_raw.extend(batch_raw)
                pbar.update(len(batch))

        # Clean each page's results independently
        return [
            self._clean(all_raw[i], pages[i], baseline_style) for i in range(len(pages))
        ]

    # ------------------------------------------------------------------
    # Private: pipeline stages
    # ------------------------------------------------------------------

    def _enhance(self, page: PageBundle):
        """Delegate to image_enhancer. Returns enhanced np.ndarray."""
        return image_enhancer.enhance(page)

    def _predict(self, images: list) -> List[List[dict]]:
        """
        Run YOLO inference on a list of images.
        Returns raw prediction dicts — never exits this private method.

        Raises:
            DetectionError: wraps any YOLO/PyTorch exception
        """
        try:
            predictions = self._model.predict(
                images,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
            )
        except Exception as e:
            raise DetectionError(f"YOLO inference failed: {e}") from e

        return [self._parse_prediction(pred) for pred in predictions]

    def _parse_prediction(self, prediction) -> List[dict]:
        """
        Convert a single YOLO prediction result to a list of raw dicts.
        poly format: [x0,y0, x1,y0, x1,y1, x0,y1] (clockwise from top-left)
        """
        results = []

        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return results

        for xyxy, conf, cls in zip(
            prediction.boxes.xyxy.cpu(),
            prediction.boxes.conf.cpu(),
            prediction.boxes.cls.cpu(),
        ):
            x0, y0, x1, y1 = xyxy.tolist()  # preserve sub-pixel precision
            results.append(
                {
                    "category_id": int(cls.item()),
                    "poly": [x0, y0, x1, y0, x1, y1, x0, y1],
                    "score": round(float(conf.item()), 3),
                }
            )

        return results

    def _clean(
        self,
        raw: List[dict],
        page: PageBundle,
        baseline_style: tuple,
    ) -> List[Detection]:
        """Delegate to detection_cleaner. Returns typed Detection objects."""
        detections = detection_cleaner.clean(raw, page, baseline_style)

        # Heuristic fallback for un-captioned tables
        all_table_bboxes = [d.bbox for d in detections if d.label.value == "table"]

        for det in detections:
            if det.label.value == "table" and det.matched_caption_bbox is None:
                # Need table bbox in PDF-space. YOLO bbox is in pixel-space.
                # pdf_point = pixel * (72 / dpi)
                scale = 72.0 / page.image_dpi
                tx0, ty0, tx1, ty1 = det.bbox
                table_bbox_pdf = (tx0 * scale, ty0 * scale, tx1 * scale, ty1 * scale)

                all_table_bboxes_pdf = [
                    (b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale)
                    for b in all_table_bboxes
                ]

                heuristic_bbox_pdf = find_heuristic_caption_bbox(
                    page_words=page.words,
                    table_bbox_pdf=table_bbox_pdf,
                    all_table_bboxes=all_table_bboxes_pdf,
                    page_size=page.page_size,
                    baseline_style=baseline_style,
                )

                if heuristic_bbox_pdf:
                    # Convert PDF-space bbox back to pixel-space for matched_caption_bbox
                    inv_scale = page.image_dpi / 72.0
                    hx0, hy0, hx1, hy1 = heuristic_bbox_pdf
                    det.matched_caption_bbox = (
                        hx0 * inv_scale,
                        hy0 * inv_scale,
                        hx1 * inv_scale,
                        hy1 * inv_scale,
                    )

        return detections
