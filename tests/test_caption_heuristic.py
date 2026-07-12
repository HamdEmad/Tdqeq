import pytest
from typing import List
from tdqeq.types import Word
import tdqeq.detector.caption_heuristic as ch

# Helper to build word lists
def make_word(x0: float, y0: float, x1: float, y1: float, text: str, font: str = "Helvetica", size: float = 10.0, color_rgb=(0,0,0), char_flags=0) -> Word:
    return Word(
        x0=x0, y0=y0, x1=x1, y1=y1,
        text=text, font=font, size=int(size),
        color_rgb=color_rgb, bidi=0, char_flags=char_flags, alpha=255
    )

def test_compute_document_baseline():
    words = [
        make_word(50, 50, 100, 60, "Header", size=14),
        make_word(50, 100, 100, 110, "Body", size=10),
        make_word(50, 120, 100, 130, "Body2", size=10),
        make_word(50, 140, 100, 150, "Body3", size=10),
    ]
    med_size, bold_ratio, dom_color, dom_font = ch.compute_document_baseline(words)
    assert med_size == 10.0
    assert bold_ratio == 0.0
    assert dom_color == (0, 0, 0)

def test_is_style_different():
    baseline = (10.0, 0.0, (0, 0, 0), 'Helvetica') # Size 10, not bold, black
    
    # Same style -> False
    words_same = [make_word(50, 50, 100, 60, "Same", size=10)]
    assert not ch.is_style_different(words_same, baseline)
    
    # Larger size -> True
    words_large = [make_word(50, 50, 100, 60, "Large", size=12)]
    assert ch.is_style_different(words_large, baseline)
    
    # Bold -> True
    words_bold = [make_word(50, 50, 100, 60, "Bold", size=10, char_flags=16)]
    assert ch.is_style_different(words_bold, baseline)
    
    # Different color -> True
    words_color = [make_word(50, 50, 100, 60, "Color", size=10, color_rgb=(255, 0, 0))]
    assert ch.is_style_different(words_color, baseline)

def test_find_heuristic_caption_bbox():
    baseline = (10.0, 0.0, (0, 0, 0), 'Helvetica') # Size 10, not bold, black
    
    words = [
        # Caption line: bold, size 10 (Different style)
        make_word(50, 340, 120, 350, "Table 1.", font="Helvetica-Bold", size=10, char_flags=16),
        # Noise line above caption
        make_word(50, 320, 100, 330, "Unrelated paragraph", size=10)
    ]
    
    # Table bounds: x=50-200, y=360-500
    res = ch.find_heuristic_caption_bbox(
        page_words=words,
        table_bbox_pdf=(50, 360, 200, 500),
        all_table_bboxes=[(50, 360, 200, 500)],
        page_size=(612, 792),
        baseline_style=baseline,
        scan_dist=50.0
    )
    
    assert res is not None
    assert res == (50, 340, 120, 350)

def test_horizontal_restriction_no_intervening_text():
    baseline = (10.0, 0.0, (0, 0, 0), 'Helvetica') # Size 10, not bold, black
    
    words = [
        # Caption line directly above table
        make_word(50, 340, 120, 350, "Table 1.", font="Helvetica-Bold", size=10, char_flags=16),
        # Regular text visually "between" caption y-coord and table y-coord, 
        # BUT it's in a right-hand column (x=300 to 400), table ends at x=200
        make_word(300, 345, 400, 355, "Right column text", size=10),
    ]
    
    # Table bounds: x=50-200, y=360-500
    res = ch.find_heuristic_caption_bbox(
        page_words=words,
        table_bbox_pdf=(50, 360, 200, 500),
        all_table_bboxes=[(50, 360, 200, 500)],
        page_size=(612, 792),
        baseline_style=baseline,
        scan_dist=50.0
    )
    
    # It should still find the caption, ignoring the right column text
    assert res is not None
    assert res == (50, 340, 120, 350)

def test_stop_on_intervening_regular_text():
    baseline = (10.0, 0.0, (0, 0, 0), 'Helvetica') # Size 10, not bold, black
    
    words = [
        # Caption line
        make_word(50, 320, 120, 330, "Table 1.", font="Helvetica-Bold", size=10, char_flags=16),
        # Intervening regular text (same style as baseline) directly above table
        make_word(50, 340, 120, 350, "Some regular text", size=10),
    ]
    
    # Table bounds: x=50-200, y=360-500
    res = ch.find_heuristic_caption_bbox(
        page_words=words,
        table_bbox_pdf=(50, 360, 200, 500),
        all_table_bboxes=[(50, 360, 200, 500)],
        page_size=(612, 792),
        baseline_style=baseline,
        scan_dist=50.0
    )
    
    # Should return None because it hit regular text first, breaking the heuristic accumulation
    assert res is None
