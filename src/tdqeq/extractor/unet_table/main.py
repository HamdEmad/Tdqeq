import html
import logging
import os
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image
from loguru import logger
from bs4 import BeautifulSoup

from tdqeq.config import settings
from tdqeq.exceptions import ModelNotLoadedError

from .span_pre_proc import calculate_contrast
from .table_structure_unet import TSRUnet
from .table_recover import TableRecover
from .utils import InputType, LoadImage
from .utils_table_recover import (
    match_ocr_cell,
    plot_html_table,
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
    gather_ocr_list_by_row,
)

_HF_REPO_ID = "opendatalab/PDF-Extract-Kit-1.0"
_HF_FILENAME = "models/TabRec/UnetStructure/unet.onnx"
_HF_REPO_TYPE = "model"


def resolve_unet_weight(weight: Optional[Union[str, Path]] = None) -> Path:
    if weight is not None:
        path = Path(weight)
        if not path.exists():
            raise ModelNotLoadedError(
                f"U-Net table weights not found at: {path}. "
                "Either fix the path or omit it to auto-download."
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
            f"Failed to download U-Net table weights from HuggingFace Hub: {e}. "
            "Check your internet connection or supply a local weight path."
        ) from e


@dataclass
class WiredTableInput:
    model_path: str
    device: str = "cpu"


@dataclass
class WiredTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None


class WiredTableRecognition:
    def __init__(self, config: WiredTableInput):
        self.table_structure = TSRUnet(asdict(config))
        self.load_img = LoadImage()
        self.table_recover = TableRecover()

    def __call__(
        self,
        img: InputType,
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
        **kwargs,
    ) -> WiredTableOutput:
        s = time.perf_counter()
        need_ocr = True
        col_threshold = 15
        row_threshold = 10
        if kwargs:
            need_ocr = kwargs.get("need_ocr", True)
            col_threshold = kwargs.get("col_threshold", 15)
            row_threshold = kwargs.get("row_threshold", 10)
        img = self.load_img(img)
        polygons, rotated_polygons = self.table_structure(img, **kwargs)
        if polygons is None:
            return WiredTableOutput("", None, None, 0.0)

        try:
            table_res, logi_points = self.table_recover(
                rotated_polygons, row_threshold, col_threshold
            )
            polygons[:, 1, :], polygons[:, 3, :] = (
                polygons[:, 3, :].copy(),
                polygons[:, 1, :].copy(),
            )
            if not need_ocr:
                sorted_polygons, idx_list = sorted_ocr_boxes(
                    [box_4_2_poly_to_box_4_1(box) for box in polygons]
                )
                return WiredTableOutput(
                    "",
                    sorted_polygons,
                    logi_points[idx_list],
                    time.perf_counter() - s,
                )
            cell_box_det_map, not_match_orc_boxes = match_ocr_cell(ocr_result, polygons)
            cell_box_det_map = self.fill_blank_rec(img, polygons, cell_box_det_map)
            t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)

            logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
            cell_box_det_map = {
                i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                for i, t_box_ocr in enumerate(t_rec_ocr_list)
            }
            pred_html = plot_html_table(logi_points, cell_box_det_map)
            polygons = np.array(polygons).reshape(-1, 8)
            logi_points = np.array(logi_points)
            elapse = time.perf_counter() - s

        except Exception:
            logging.warning(traceback.format_exc())
            return WiredTableOutput("", None, None, 0.0)
        return WiredTableOutput(pred_html, polygons, logi_points, elapse)

    def transform_res(
        self,
        cell_box_det_map: Dict[int, List[any]],
        polygons: np.ndarray,
        logi_points: List[np.ndarray],
    ) -> List[Dict[str, any]]:
        res = []
        for i in range(len(polygons)):
            ocr_res_list = cell_box_det_map.get(i)
            if not ocr_res_list:
                continue
            xmin = min([ocr_box[0][0][0] for ocr_box in ocr_res_list])
            ymin = min([ocr_box[0][0][1] for ocr_box in ocr_res_list])
            xmax = max([ocr_box[0][2][0] for ocr_box in ocr_res_list])
            ymax = max([ocr_box[0][2][1] for ocr_box in ocr_res_list])
            dict_res = {
                "t_box": [xmin, ymin, xmax, ymax],
                "t_logic_box": logi_points[i].tolist(),
                "t_ocr_res": [
                    [box_4_2_poly_to_box_4_1(ocr_det[0]), ocr_det[1]]
                    for ocr_det in ocr_res_list
                ],
            }
            res.append(dict_res)
        return res

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threhold=0.3
            )
        return res

    def fill_blank_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[Any]]:
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            box = sorted_polygons[i]
            cell_box_map[i] = [[box, "", 0.1]]
        return cell_box_map


def escape_html(input_string):
    return html.escape(input_string)


def count_table_cells_physical(html_code):
    if not html_code:
        return 0
    html_lower = html_code.lower()
    td_count = html_lower.count('<td')
    th_count = html_lower.count('<th')
    return td_count + th_count


class UnetTableModel:
    def __init__(self):
        model_path = str(resolve_unet_weight(settings.TABLE_UNET_WEIGHTS_PATH))
        wired_input_args = WiredTableInput(model_path=model_path)
        self.wired_table_model = WiredTableRecognition(wired_input_args)

    def predict(self, input_img, ocr_result, wireless_html_code) -> Tuple[str, Optional[np.ndarray]]:
        if isinstance(input_img, Image.Image):
            np_img = np.asarray(input_img)
        elif isinstance(input_img, np.ndarray):
            np_img = input_img
        else:
            raise ValueError("Input must be a pillow object or a numpy array.")
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        if ocr_result is None:
            ocr_result = []

        try:
            wired_table_results = self.wired_table_model(np_img, ocr_result)

            wired_html_code = wired_table_results.pred_html
            wired_len = count_table_cells_physical(wired_html_code)
            wireless_len = count_table_cells_physical(wireless_html_code)
            gap_of_len = wireless_len - wired_len

            wireless_text_count = 0
            wired_text_count = 0
            for ocr_res in ocr_result:
                if ocr_res[1] in wireless_html_code:
                    wireless_text_count += 1
                if ocr_res[1] in wired_html_code:
                    wired_text_count += 1

            wireless_soup = BeautifulSoup(wireless_html_code, 'html.parser') if wireless_html_code else BeautifulSoup("", 'html.parser')
            wired_soup = BeautifulSoup(wired_html_code, 'html.parser') if wired_html_code else BeautifulSoup("", 'html.parser')
            wireless_blank_count = sum(1 for cell in wireless_soup.find_all(['td', 'th']) if not cell.text.strip())
            wired_blank_count = sum(1 for cell in wired_soup.find_all(['td', 'th']) if not cell.text.strip())

            wireless_non_blank_count = wireless_len - wireless_blank_count
            wired_non_blank_count = wired_len - wired_blank_count
            switch_flag = False
            if wireless_non_blank_count > wired_non_blank_count:
                wired_table_scale = round(wired_non_blank_count ** 0.5)
                wired_scale_plus_2_cols = wired_non_blank_count + (wired_table_scale * 2)
                wired_scale_squared_plus_2_rows = wired_table_scale * (wired_table_scale + 2)
                if (wireless_non_blank_count + 3) >= max(wired_scale_plus_2_cols, wired_scale_squared_plus_2_rows):
                    switch_flag = True

            if (
                switch_flag
                or (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))
                or (gap_of_len == 0 and wired_len <= 4)
                or (wired_text_count <= wireless_text_count * 0.6 and  wireless_text_count >=10)
            ):
                # Fall back to wireless HTML and cell bboxes
                return wireless_html_code, None
            else:
                return wired_html_code, wired_table_results.cell_bboxes
        except Exception as e:
            logger.warning(e)
            return wireless_html_code, None
