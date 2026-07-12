"""
Cleans and formats YOLO detections into typed objects.

Copyright (c) 2024 OpenDataLab. All rights reserved.
Portions of this file (layout filtering and geometry helpers) 
are derived from the MinerU project, licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
"""
from typing import Dict, List, Tuple

from tdqeq.detector.boxbase import get_minbox_if_overlap_by_ratio
from tdqeq.types import Detection, DetectionLabel, PageBundle
from tdqeq.detector.caption_heuristic import is_style_different

# ---------------------------------------------------------------------------
# Category ID mapping — specific to doclayout_yolo model
# These are the ONLY place category_id integers appear in the codebase.
# ---------------------------------------------------------------------------
TABLE_ID = 5
CAPTION_ID_PRIMARY = 6  # preferred caption class
CAPTION_ID_FALLBACK = 0  # title class used as fallback caption
CAPTION_CANDIDATE_IDS = [CAPTION_ID_PRIMARY, CAPTION_ID_FALLBACK]

# ---------------------------------------------------------------------------
# Caption matching rules
# Distance is defined in PDF points at 72 DPI.
# At runtime this is scaled to pixels using image_dpi from PageBundle.
# ---------------------------------------------------------------------------
MAX_CAPTION_DISTANCE_PT = 80  # PDF points — relaxed for spacious layouts
CAPTION_BBOX_PADDING_PX = 1   # pixels to expand matched caption bbox on each side

# Priority weight for above vs. below captions (lower = preferred)
# Within each position tier, CAPTION class wins over TITLE, which wins over TEXT.
_ABOVE_PRIORITY = 0

# Category priority — lower is better
_CATEGORY_PRIORITY = {
    CAPTION_ID_PRIMARY: 0,  # CAPTION class — highest confidence
    CAPTION_ID_FALLBACK: 1, # TITLE class — medium confidence
}


def clean(
    raw: List[Dict],
    page: PageBundle,
    baseline_style: Tuple[float, float, Tuple[int, int, int], str],
) -> List[Detection]:
    """
    Clean raw YOLO predictions and return typed Detection objects.

    Pipeline:
        1. run layout filter (merge/deduplicate overlapping boxes)
        2. convert filtered dicts → Detection objects
        3. match nearest caption above each table

    Args:
        raw:            raw YOLO prediction dicts (poly + category_id + score)
        page:           PageBundle containing image_dpi, words, etc.
        baseline_style: document baseline style

    Returns:
        List[Detection] containing TABLE detections with matched captions,
        and unmatched CAPTION detections for reference.
    """
    table_dicts, caption_dicts = _run_layout_filter(raw)

    table_detections = _to_detections(table_dicts, page.page_number, DetectionLabel.TABLE)
    caption_detections = _to_detections(
        caption_dicts, page.page_number, DetectionLabel.CAPTION,
        raw_dicts=caption_dicts,
    )

    matched = _match_captions(table_detections, caption_detections, page, baseline_style)

    return matched


# ---------------------------------------------------------------------------
# Step 1 — layout filter
# ---------------------------------------------------------------------------


def _run_layout_filter(raw: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Run the full layout cleaning pipeline on raw YOLO dicts.
    Separates cleaned results into table dicts and caption dicts.

    Returns:
        (table_dicts, caption_dicts)
    """
    if not raw:
        return [], []

    # Collect caption candidates BEFORE layout filter mutates the list
    caption_dicts = [r for r in raw if int(r["category_id"]) in CAPTION_CANDIDATE_IDS]

    # Layout filter mutates its input list — pass a copy to protect caption_dicts
    raw_copy = list(raw)
    _, filtered_tables, _ = get_res_list_from_layout_res(raw_copy)

    return filtered_tables, caption_dicts


# ---------------------------------------------------------------------------
# Step 2 — dict → Detection
# ---------------------------------------------------------------------------


def _to_detections(
    dicts: List[Dict],
    page_number: int,
    label: DetectionLabel,
    raw_dicts: List[Dict] = None,
) -> List[Detection]:
    """Convert raw YOLO dicts to typed Detection objects.
    
    If raw_dicts is provided, the category_id is stored on the Detection
    so caption matching can use tiered priority.
    """
    detections = []
    for i, d in enumerate(dicts):
        poly = d["poly"]
        bbox = (
            float(poly[0]),  # x0
            float(poly[1]),  # y0
            float(poly[4]),  # x1
            float(poly[5]),  # y1
        )
        det = Detection(
            page_number=page_number,
            label=label,
            bbox=bbox,
            confidence=float(d.get("score", 0.0)),
        )
        # Set category_id for tiered caption matching
        if raw_dicts is not None:
            det._category_id = int(raw_dicts[i].get("category_id", -1))
        detections.append(det)
    return detections


# ---------------------------------------------------------------------------
# Step 3 — caption matching
# ---------------------------------------------------------------------------


def _match_captions(
    tables: List[Detection],
    captions: List[Detection],
    page: PageBundle,
    baseline_style: Tuple[float, float, Tuple[int, int, int], str],
) -> List[Detection]:
    """
    For each table, find the nearest caption bbox that is:
      - directly above the table
      - within MAX_CAPTION_DISTANCE_PT PDF points (scaled to pixels)
      - sharing horizontal x-span with the table
      - has NO intervening text between caption and table (horizontally bound)
      - has a different style than the regular text

    Priority ordering (lower is better):
      Within the ABOVE position tier, the nearest caption wins.

    Caption detections are consumed — a caption matched to one table
    is not matched to another.
    """
    max_dist_px = MAX_CAPTION_DISTANCE_PT * (page.image_dpi / 72.0)
    used_caption_indices = set()

    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox

        best_idx = None
        best_cat_priority = float("inf")
        best_distance = float("inf")

        for i, cap in enumerate(captions):
            if i in used_caption_indices:
                continue

            cx0, cy0, cx1, cy1 = cap.bbox

            # Horizontal overlap check — caption must share x-span with table
            if cx1 < tx0 or cx0 > tx1:
                continue

            # Determine position: MUST be above
            if cy1 <= ty0:
                gap = ty0 - cy1
            else:
                continue

            if gap > max_dist_px:
                continue

            # Intervening tables check
            intervening_table = False
            for other_table in tables:
                if other_table == table:
                    continue
                otx0, oty0, otx1, oty1 = other_table.bbox
                if cy1 <= oty0 and oty1 <= ty0:
                    intervening_table = True
                    break
            if intervening_table:
                continue

            # Convert table and caption bounds to PDF space to check Word intersections
            # pixel = pdf_point * (dpi / 72) => pdf_point = pixel * (72 / dpi)
            scale = 72.0 / page.image_dpi
            cap_y1_pdf = cy1 * scale
            tab_y0_pdf = ty0 * scale
            tab_x1_pdf = tx1 * scale

            # Intervening Word check (horizontally bound: left of page to right edge of table)
            intervening_word = False
            for w in page.words:
                if w.x0 < tab_x1_pdf and cap_y1_pdf < w.y1 and w.y0 < tab_y0_pdf:
                    intervening_word = True
                    break
            if intervening_word:
                continue

            # Extract words within the YOLO caption bbox to check style
            cap_x0_pdf = cx0 * scale
            cap_x1_pdf = cx1 * scale
            cap_y0_pdf = cy0 * scale
            
            caption_words = [
                w for w in page.words
                if cap_y0_pdf <= w.y1 and w.y0 <= cap_y1_pdf and cap_x0_pdf <= w.x1 and w.x0 <= cap_x1_pdf
            ]

            # Enforce style difference
            if not is_style_different(caption_words, baseline_style):
                continue

            # Determine category priority
            cat_priority = _CATEGORY_PRIORITY.get(
                getattr(cap, '_category_id', -1), 1
            )

            # Prefer: category, then nearest gap
            if (cat_priority, gap) < (best_cat_priority, best_distance):
                best_cat_priority = cat_priority
                best_distance = gap
                best_idx = i

        if best_idx is not None:
            expanded_bbox = list(captions[best_idx].bbox)
            expanded_bbox[0] -= CAPTION_BBOX_PADDING_PX
            expanded_bbox[1] -= CAPTION_BBOX_PADDING_PX
            expanded_bbox[2] += CAPTION_BBOX_PADDING_PX
            expanded_bbox[3] += CAPTION_BBOX_PADDING_PX

            table.matched_caption_bbox = expanded_bbox
            used_caption_indices.add(best_idx)

    return tables


# ---------------------------------------------------------------------------
# Imported layout filter — vendored from mineru, internal to this module
# ---------------------------------------------------------------------------


def get_res_list_from_layout_res(
    layout_res,
    iou_threshold=0.7,
    overlap_threshold=0.8,
    area_threshold=0.8,
):
    """
    Extract and clean table regions from raw YOLO layout results.
    Merges high-IoU duplicates, filters nested tables, removes overlaps.

    Returns:
        (ocr_res_list, filtered_table_res_list, single_page_mfdetrec_res)
    """
    ocr_res_list = []
    text_res_list = []
    table_res_list = []
    table_indices = []
    single_page_mfdetrec_res = []

    for i, res in enumerate(layout_res):
        category_id = int(res["category_id"])
        if category_id in [13, 14]:
            single_page_mfdetrec_res.append(
                {
                    "bbox": [
                        int(res["poly"][0]),
                        int(res["poly"][1]),
                        int(res["poly"][4]),
                        int(res["poly"][5]),
                    ],
                }
            )
        elif category_id in [0, 6]:
            ocr_res_list.append(res)
        elif category_id == TABLE_ID:
            table_res_list.append(res)
            table_indices.append(i)
        elif category_id in [1]:
            text_res_list.append(res)

    table_res_list, table_indices = _merge_high_iou_tables(
        table_res_list, layout_res, table_indices, iou_threshold
    )
    filtered_table_res_list = _filter_nested_tables(
        table_res_list, overlap_threshold, area_threshold
    )
    filtered_table_res_list, table_need_remove = _remove_overlaps_min_blocks(
        filtered_table_res_list
    )

    for res in table_need_remove:
        if res in layout_res:
            layout_res.remove(res)

    if len(filtered_table_res_list) < len(table_res_list):
        kept = set(id(t) for t in filtered_table_res_list)
        for table in table_res_list:
            if id(table) not in kept and table in layout_res:
                layout_res.remove(table)

    text_res_list, need_remove = _remove_overlaps_min_blocks(text_res_list)
    ocr_res_list.extend(text_res_list)
    for res in need_remove:
        if res in layout_res:
            layout_res.remove(res)

    combined = ocr_res_list + filtered_table_res_list
    to_remove = _remove_overlaps_low_confidence_blocks(combined, overlap_threshold)
    for block in to_remove:
        if block in ocr_res_list:
            ocr_res_list.remove(block)
        elif block in filtered_table_res_list:
            filtered_table_res_list.remove(block)
        if block in layout_res:
            layout_res.remove(block)

    return ocr_res_list, filtered_table_res_list, single_page_mfdetrec_res


# ---------------------------------------------------------------------------
# Internal geometry helpers (ported from original mineru source)
# ---------------------------------------------------------------------------


def _get_coords_and_area(block):
    xmin = int(block["poly"][0])
    ymin = int(block["poly"][1])
    xmax = int(block["poly"][4])
    ymax = int(block["poly"][5])
    return xmin, ymin, xmax, ymax, (xmax - xmin) * (ymax - ymin)


def _calculate_intersection(b1, b2):
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


def _calculate_iou(b1, b2):
    inter = _calculate_intersection(b1[:4], b2[:4])
    if not inter:
        return 0.0
    iw = inter[2] - inter[0]
    ih = inter[3] - inter[1]
    inter_area = iw * ih
    union = b1[4] + b2[4] - inter_area
    return inter_area / union if union > 0 else 0.0


def _is_inside(small, big, threshold=0.8):
    inter = _calculate_intersection(small[:4], big[:4])
    if not inter:
        return False
    iw = inter[2] - inter[0]
    ih = inter[3] - inter[1]
    return (iw * ih) >= threshold * small[4]


def _do_overlap(b1, b2):
    return _calculate_intersection(b1[:4], b2[:4]) is not None


def _merge_high_iou_tables(
    table_res_list, layout_res, table_indices, iou_threshold=0.7
):
    if len(table_res_list) < 2:
        return table_res_list, table_indices

    info = [_get_coords_and_area(t) for t in table_res_list]
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(table_res_list) - 1:
            j = i + 1
            while j < len(table_res_list):
                if _calculate_iou(info[i], info[j]) > iou_threshold:
                    x1m = min(info[i][0], info[j][0])
                    y1m = min(info[i][1], info[j][1])
                    x2m = max(info[i][2], info[j][2])
                    y2m = max(info[i][3], info[j][3])
                    merged_t = table_res_list[i].copy()
                    merged_t["poly"] = [x1m, y1m, x2m, y1m, x2m, y2m, x1m, y2m]
                    to_remove = sorted(
                        [table_indices[j], table_indices[i]], reverse=True
                    )
                    for idx in to_remove:
                        del layout_res[idx]
                    layout_res.append(merged_t)
                    table_indices = [
                        k
                        if k < min(to_remove)
                        else k - 1
                        if k < max(to_remove)
                        else k - 2
                        if k > max(to_remove)
                        else len(layout_res) - 1
                        for k in table_indices
                        if k not in to_remove
                    ]
                    table_indices.append(len(layout_res) - 1)
                    table_res_list.pop(j)
                    table_res_list.pop(i)
                    table_res_list.append(merged_t)
                    info = [_get_coords_and_area(t) for t in table_res_list]
                    merged = True
                    break
                j += 1
            if merged:
                break
            i += 1
    return table_res_list, table_indices


def _filter_nested_tables(table_res_list, overlap_threshold=0.8, area_threshold=0.8):
    if len(table_res_list) < 3:
        return table_res_list
    info = [_get_coords_and_area(t) for t in table_res_list]
    big_idx = []
    for i in range(len(table_res_list)):
        inside = [
            j
            for j in range(len(table_res_list))
            if i != j and _is_inside(info[j], info[i], overlap_threshold)
        ]
        if len(inside) >= 3:
            no_overlap = not any(
                _do_overlap(info[inside[a]], info[inside[b]])
                for a in range(len(inside))
                for b in range(a + 1, len(inside))
            )
            if no_overlap:
                total = sum(info[j][4] for j in inside)
                if total > area_threshold * info[i][4]:
                    big_idx.append(i)
    return [t for i, t in enumerate(table_res_list) if i not in big_idx]


def _remove_overlaps_min_blocks(res_list):
    for res in res_list:
        res["bbox"] = [
            int(res["poly"][0]),
            int(res["poly"][1]),
            int(res["poly"][4]),
            int(res["poly"][5]),
        ]
    need_remove = []
    for i in range(len(res_list)):
        if res_list[i] in need_remove:
            continue
        for j in range(i + 1, len(res_list)):
            if res_list[j] in need_remove:
                continue
            overlap_box = get_minbox_if_overlap_by_ratio(
                res_list[i]["bbox"], res_list[j]["bbox"], 0.8
            )
            if overlap_box is not None:
                if overlap_box == res_list[i]["bbox"]:
                    small, large = res_list[i], res_list[j]
                elif overlap_box == res_list[j]["bbox"]:
                    small, large = res_list[j], res_list[i]
                else:
                    continue
                if small["score"] <= large["score"]:
                    x1, y1, x2, y2 = large["bbox"]
                    sx1, sy1, sx2, sy2 = small["bbox"]
                    large["bbox"] = [
                        min(x1, sx1),
                        min(y1, sy1),
                        max(x2, sx2),
                        max(y2, sy2),
                    ]
                    if small not in need_remove:
                        need_remove.append(small)
                else:
                    if large not in need_remove:
                        need_remove.append(large)
    for res in need_remove:
        res_list.remove(res)
        del res["bbox"]
    for res in res_list:
        res["poly"] = [
            res["bbox"][0],
            res["bbox"][1],
            res["bbox"][2],
            res["bbox"][1],
            res["bbox"][2],
            res["bbox"][3],
            res["bbox"][0],
            res["bbox"][3],
        ]
        del res["bbox"]
    return res_list, need_remove


def _remove_overlaps_low_confidence_blocks(combined_res_list, overlap_threshold=0.8):
    info = []
    for block in combined_res_list:
        xmin = int(block["poly"][0])
        ymin = int(block["poly"][1])
        xmax = int(block["poly"][4])
        ymax = int(block["poly"][5])
        area = (xmax - xmin) * (ymax - ymin)
        score = block.get("score", 0.5)
        info.append((xmin, ymin, xmax, ymax, area, score, block))

    to_remove = []
    marked = set()

    for i, (xmin, ymin, xmax, ymax, area, score, block) in enumerate(info):
        if i in marked:
            continue
        inside = [
            (j, j_score, j_block)
            for j, (jx0, jy0, jx1, jy1, j_area, j_score, j_block) in enumerate(info)
            if i != j
            and j not in marked
            and _is_inside(info[j], info[i], overlap_threshold)
        ]
        if len(inside) >= 2:
            avg_score = sum(s for _, s, _ in inside) / len(inside)
            if score > avg_score:
                for j, _, j_block in inside:
                    if j_block not in to_remove:
                        to_remove.append(j_block)
                        marked.add(j)
            else:
                if block not in to_remove:
                    to_remove.append(block)
                    marked.add(i)
    return to_remove
