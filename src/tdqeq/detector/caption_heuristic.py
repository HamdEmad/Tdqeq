"""
caption_heuristic.py — Style-matching based fallback caption detection.

Runs when YOLO fails to detect a caption bounding box.
Matches text lines immediately above the table against the document baseline style.
"""

from collections import Counter
from statistics import median
from typing import List, Optional, Tuple

from tdqeq.types import Word

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINE_GROUP_TOLERANCE = 4.5      # vertical tolerance for grouping words on the same line
MIN_WORD_SIZE = 4.0            # minimum font size to consider (filters OCR noise)
HEADER_FOOTER_MARGIN = 30.0     # ignore text in top/bottom margin of page

ALIGN_CENTER_THRESHOLD = 0.10   # max offset ratio for center alignment
ALIGN_LEFT_THRESHOLD = 0.05     # max offset ratio for left alignment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_document_baseline(
    all_words: List[Word],
) -> Tuple[float, float, Tuple[int, int, int], str]:
    """
    Compute document baseline style:
    Returns (median_font_size, bold_ratio, dominant_color, dominant_font).
    """
    if not all_words:
        return 10.0, 0.0, (0, 0, 0), ""

    sizes = [w.size for w in all_words]
    med_size = float(median(sizes))

    body_words = [w for w in all_words if abs(w.size - med_size) <= 2]
    if not body_words:
        body_words = all_words

    bold_count = sum(1 for w in body_words if _is_word_bold(w))
    bold_ratio = bold_count / len(body_words)
    
    colors = [w.color_rgb for w in body_words]
    dom_color = Counter(colors).most_common(1)[0][0] if colors else (0, 0, 0)

    fonts = [w.font for w in body_words if w.font]
    dom_font = Counter(fonts).most_common(1)[0][0] if fonts else ""

    return med_size, bold_ratio, dom_color, dom_font

def is_style_different(
    words: List[Word],
    baseline: Tuple[float, float, Tuple[int, int, int], str],
) -> bool:
    """
    Check if a line's style differs from the baseline style.
    baseline = (med_size, bold_ratio, dom_color, dom_font)
    """
    if not words:
        return False
        
    med_size, bold_ratio, dom_color, dom_font = baseline
    
    line_size = _dominant_size(words)
    line_bold = _is_line_bold(words)
    line_color = _dominant_color(words)
    
    # Difference criteria:
    # 1. Size is significantly larger than baseline
    if line_size >= med_size + 1.0:
        return True
        
    # 2. Boldness differs (line is bold but baseline is NOT)
    # We don't trigger on line_bold == False if baseline is bold, 
    # to avoid false positives on small regular text when baseline is corrupted
    baseline_is_bold = bold_ratio > 0.5
    if line_bold and not baseline_is_bold:
        return True
        
    # 3. Color differs
    if line_color != dom_color:
        return True

    # 4. Font differs with a specific style variant
    fonts = [w.font for w in words if w.font]
    line_font = Counter(fonts).most_common(1)[0][0] if fonts else ""
    if dom_font and line_font and dom_font != line_font:
        lf_lower = line_font.lower()
        df_lower = dom_font.lower()
        if "bold" in lf_lower and "bold" not in df_lower:
            return True
        if "italic" in lf_lower and "italic" not in df_lower:
            return True
        
    return False

def find_heuristic_caption_bbox(
    page_words: List[Word],
    table_bbox_pdf: Tuple[float, float, float, float],
    all_table_bboxes: List[Tuple[float, float, float, float]],
    page_size: Tuple[float, float],
    baseline_style: Tuple[float, float, Tuple[int, int, int]],
    scan_dist: float = 70.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Attempt to find a caption bounding box using heuristics.
    Returns the bounding box of the matching caption words.
    """
    if not page_words:
        return None

    tx0, ty0, tx1, ty1 = table_bbox_pdf

    # 1. Define scan zone above the table
    scan_top = ty0 - scan_dist
    scan_bottom = ty0

    # Boundary guard: don't scan into another table above us
    for other_bbox in all_table_bboxes:
        if other_bbox == table_bbox_pdf:
            continue
        other_y1 = other_bbox[3]
        if scan_top < other_y1 < scan_bottom:
            scan_top = other_y1 + 2.0  # small padding

    # Header exclusion: don't scan into the top margin
    if scan_top < HEADER_FOOTER_MARGIN:
        scan_top = HEADER_FOOTER_MARGIN

    if scan_top >= scan_bottom:
        return None

    # 2. Collect words in the scan zone, restricted horizontally
    zone_words = [
        w for w in page_words
        if w.y1 >= scan_top and w.y1 <= scan_bottom
        and w.y0 < scan_bottom
        and w.x0 < tx1  # Horizontal restriction: left of page to right edge of table
        and w.size >= MIN_WORD_SIZE
    ]

    if not zone_words:
        return None

    # 3. Group into lines
    lines = _group_into_lines(zone_words)
    if not lines:
        return None

    # Sort bottom-to-top (closest to table first)
    lines.sort(key=lambda line: -line[0].y0)

    # 4. Grab at most 5 lines closest to the table
    candidate_lines = lines[:5]

    # 5. Accumulate matching lines bottom-up. Stop immediately on mismatch or empty line
    matched_words = []

    for line in candidate_lines:
        if is_style_different(line, baseline_style):
            matched_words.extend(line)
            
            # Stop if we hit the beginning of a caption (e.g., "Table 1" or "Figure 2")
            # to prevent swallowing another caption situated just above.
            line_sorted = sorted(line, key=lambda w: w.x0)
            if line_sorted:
                first_word = line_sorted[0].text.lower()
                if first_word.startswith(("table", "figure", "tab", "fig")):
                    break
        else:
            # If the closest line has the same style as regular text, 
            # it might be regular text between the caption and the table.
            # Thus, we break and return nothing, enforcing "no text between them".
            break

    if not matched_words:
        return None

    # 6. Calculate bounding box of the matched words
    cx0 = min(w.x0 for w in matched_words)
    cy0 = min(w.y0 for w in matched_words)
    cx1 = max(w.x1 for w in matched_words)
    cy1 = max(w.y1 for w in matched_words)

    return (cx0, cy0, cx1, cy1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_into_lines(words: List[Word]) -> List[List[Word]]:
    """Group words into lines by y0 proximity."""
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w.y0, w.x0))
    lines: List[List[Word]] = []
    current_line: List[Word] = [sorted_words[0]]

    for word in sorted_words[1:]:
        if abs(word.y0 - current_line[0].y0) <= LINE_GROUP_TOLERANCE:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
    lines.append(current_line)

    return lines

def _dominant_size(line: List[Word]) -> float:
    """Return the most common font size in a line."""
    if not line:
        return 0.0
    return float(Counter(w.size for w in line).most_common(1)[0][0])

def _is_line_bold(line: List[Word]) -> bool:
    """Return True if the majority of words in a line are bold."""
    if not line:
        return False
    bold_count = sum(1 for w in line if _is_word_bold(w))
    return bold_count > len(line) / 2

def _dominant_color(line: List[Word]) -> Tuple[int, int, int]:
    """Return the most common color in a line."""
    if not line:
        return (0, 0, 0)
    return Counter(w.color_rgb for w in line).most_common(1)[0][0]

def _is_word_bold(word: Word) -> bool:
    """Check if a word is bold using char_flags or font name."""
    if getattr(word, 'char_flags', 0) & 16:
        return True
    if "bold" in getattr(word, 'font', '').lower():
        return True
    return False
