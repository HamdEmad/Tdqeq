import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from rapid_table import ModelType
from tdqeq.extractor.table_parser import TableParser
from tdqeq.types import ClippedRegion, Detection, DetectionLabel, Word, TableType, RawTable


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_dummy_word(text: str) -> Word:
    return Word(
        x0=0.0, y0=0.0, x1=10.0, y1=10.0,
        text=text, font="Helvetica", size=10,
        color_rgb=(0, 0, 0), bidi=0, char_flags=0, alpha=255,
    )


def make_dummy_region(page_num: int, img_shape=(100, 100, 3)) -> ClippedRegion:
    det = Detection(
        page_number=page_num,
        label=DetectionLabel.TABLE,
        bbox=(0.0, 0.0, 100.0, 100.0),
        confidence=0.95,
    )
    return ClippedRegion(
        detection=det,
        table_words=[make_dummy_word("cell1"), make_dummy_word("cell2")],
        bbox_pdf=(0.0, 0.0, 200.0, 200.0),
        table_image=np.zeros(img_shape, dtype=np.uint8),
        image_dpi=72,
        page_size=(612.0, 792.0),
        caption_text="Table Caption",
    )


class DummyResult:
    """Shared mock return value for RapidTable calls."""
    pred_htmls = ["<table><tr><td>cell1</td></tr></table>"]
    cell_bboxes = [np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32)]


def make_load_model_side_effect(mock_slanet, mock_unitable):
    """Factory for a load_model mock that dispatches on ModelType."""
    def side_effect(model_type, *args, **kwargs):
        if model_type == ModelType.SLANETPLUS:
            return mock_slanet
        if model_type == ModelType.UNITABLE:
            return mock_unitable
        return MagicMock()
    return side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("tdqeq.extractor.table_parser.TableParser._load_cls_model")
@patch("tdqeq.extractor.table_parser.TableParser._load_model")
def test_table_parser_batch_classification(mock_load_model, mock_load_cls_model):
    """Auto mode: wired table routes to UniTable, wireless+high-confidence to SlaNetPlus."""
    mock_unitable = MagicMock()
    mock_unitable.cfg.model_type.value = "UniTable"
    mock_unitable.return_value = DummyResult()

    mock_slanet = MagicMock()
    mock_slanet.cfg.model_type.value = "SlaNetPlus"
    mock_slanet.return_value = DummyResult()

    mock_load_model.side_effect = make_load_model_side_effect(mock_slanet, mock_unitable)

    mock_cls = MagicMock()
    def batch_predict_side_effect(img_info_list, batch_size=16):
        img_info_list[0]["table_res"] = {"cls_label": "WiredTable",    "cls_score": 98.0}
        img_info_list[1]["table_res"] = {"cls_label": "WirelessTable", "cls_score": 85.0}
    mock_cls.batch_predict.side_effect = batch_predict_side_effect
    mock_load_cls_model.return_value = mock_cls

    # All three models are preloaded on init for "auto" mode
    parser = TableParser(device="cpu")
    assert parser._cls is mock_cls
    assert parser._slanet is mock_slanet
    assert parser._unitable is mock_unitable

    results = parser.parse_all([make_dummy_region(0), make_dummy_region(1)])

    assert len(results) == 2
    # page 0: WiredTable → UniTable
    assert results[0].cls_score == 98.0
    assert results[0].cls == "UniTable"
    # page 1: WirelessTable, score 85 >= 80 → SlaNetPlus
    assert results[1].cls_score == 85.0
    assert results[1].cls == "SlaNetPlus"

    mock_cls.batch_predict.assert_called_once()
    args, _ = mock_cls.batch_predict.call_args
    assert len(args[0]) == 2
    assert args[0][0]["wired_table_img"] is not None


@patch("tdqeq.extractor.table_parser.TableParser._load_cls_model")
@patch("tdqeq.extractor.table_parser.TableParser._load_model")
def test_table_parser_batch_classification_fallback(mock_load_model, mock_load_cls_model):
    """Auto mode: cls batch_predict failure falls back to SlaNetPlus (wireless default)."""
    mock_unitable = MagicMock()
    mock_unitable.cfg.model_type.value = "UniTable"
    mock_unitable.return_value = DummyResult()

    mock_slanet = MagicMock()
    mock_slanet.cfg.model_type.value = "SlaNetPlus"
    mock_slanet.return_value = DummyResult()

    mock_load_model.side_effect = make_load_model_side_effect(mock_slanet, mock_unitable)

    mock_cls = MagicMock()
    mock_cls.batch_predict.side_effect = ValueError("Input image smaller than target size")
    mock_load_cls_model.return_value = mock_cls

    parser = TableParser(device="cpu")
    results = parser.parse_all([make_dummy_region(0)])

    # Fallback: _classify_all returns {}, so (WIRELESS, 100.0) is used.
    # 100.0 >= 80 threshold → SlaNetPlus
    assert len(results) == 1
    assert results[0].cls_score is None   # not set when classification is skipped
    assert results[0].cls == "SlaNetPlus"


@patch("tdqeq.extractor.table_parser.TableParser._load_cls_model")
@patch("tdqeq.extractor.table_parser.TableParser._load_model")
def test_table_parser_mode_tdqeq(mock_load_model, mock_load_cls_model):
    """tdqeq mode: only SlaNetPlus is loaded; cls and unitable are None."""
    mock_unitable = MagicMock()
    mock_unitable.cfg.model_type.value = "UniTable"
    mock_unitable.return_value = DummyResult()

    mock_slanet = MagicMock()
    mock_slanet.cfg.model_type.value = "SlaNetPlus"
    mock_slanet.return_value = DummyResult()

    mock_load_model.side_effect = make_load_model_side_effect(mock_slanet, mock_unitable)

    parser = TableParser(mode="tdqeq", device="cpu")
    assert parser._cls is None
    assert parser._unitable is None
    assert parser._slanet is mock_slanet

    results = parser.parse_all([make_dummy_region(0)])
    assert len(results) == 1
    assert results[0].cls == "SlaNetPlus"


@patch("tdqeq.extractor.table_parser.TableParser._load_cls_model")
@patch("tdqeq.extractor.table_parser.TableParser._load_model")
def test_table_parser_mode_tdqeq_plus(mock_load_model, mock_load_cls_model):
    """tdqeq+ mode: only UniTable is loaded; cls and slanet are None."""
    mock_unitable = MagicMock()
    mock_unitable.cfg.model_type.value = "UniTable"
    mock_unitable.return_value = DummyResult()

    mock_slanet = MagicMock()
    mock_slanet.cfg.model_type.value = "SlaNetPlus"
    mock_slanet.return_value = DummyResult()

    mock_load_model.side_effect = make_load_model_side_effect(mock_slanet, mock_unitable)

    parser = TableParser(mode="tdqeq+", device="cpu")
    assert parser._cls is None
    assert parser._slanet is None
    assert parser._unitable is mock_unitable

    results = parser.parse_all([make_dummy_region(0)])
    assert len(results) == 1
    assert results[0].cls == "UniTable"
