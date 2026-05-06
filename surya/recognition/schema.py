from typing import Any, List, Optional

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class BlockOCRResult(PolygonBox):
    label: str  # canonicalized layout label (Picture, Text, ...)
    raw_label: str = ""  # original model label
    reading_order: int  # 0-indexed position in layout output
    html: str = ""  # block HTML (BLOCK_PROMPT output, "" if skipped)
    skipped: bool = False  # True if label was in SKIP_OCR_LABELS
    error: bool = False
    char_confidences: Optional[List[float]] = None  # phase 2
    raw_logprobs: Optional[List[Any]] = None  # phase 2 debugging


class PageOCRResult(BaseModel):
    blocks: List[BlockOCRResult]
    image_bbox: List[float]


# ---- Back-compat shims for code paths that still expect text_lines ----
# These are intentionally minimal; downstream consumers should migrate to
# BlockOCRResult / PageOCRResult.


class TextChar(BaseModel):
    text: str
    confidence: float = 0.0


class TextLine(PolygonBox):
    text: str = ""
    chars: List[TextChar] = []


class OCRResult(BaseModel):
    text_lines: List[TextLine]
    image_bbox: List[float]
