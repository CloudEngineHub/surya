"""Parsers for the three task outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Tuple


from surya.logging import get_logger

logger = get_logger()


# ---- Layout (LAYOUT_PROMPT) -------------------------------------------------


@dataclass
class ParsedLayoutBlock:
    label: str
    bbox: Tuple[float, float, float, float]  # 0-1000 normalized
    count: int  # multiple of 50, model's token estimate


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```\s*$", "", cleaned)
    return cleaned


def _coerce_bbox(bbox) -> Tuple[float, float, float, float]:
    if isinstance(bbox, str):
        parts = [float(x) for x in bbox.replace(",", " ").split()]
    else:
        parts = [float(x) for x in bbox]
    if len(parts) != 4:
        raise ValueError(f"Bad bbox: {bbox!r}")
    return (parts[0], parts[1], parts[2], parts[3])


def _coerce_count(value) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def parse_layout(text: str) -> List[ParsedLayoutBlock]:
    """Pull the JSON array out of LAYOUT_PROMPT output and convert to typed blocks.

    Tolerates code fences, missing fields, and stringified bboxes.
    """
    cleaned = _strip_fences(text)
    m = _JSON_ARRAY_RE.search(cleaned)
    if not m:
        raise ValueError(f"No JSON array found in layout output: {text[:500]!r}")
    raw = json.loads(m.group(0))
    out: List[ParsedLayoutBlock] = []
    for item in raw:
        try:
            bbox = _coerce_bbox(item["bbox"])
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping layout block with bad bbox: {e}")
            continue
        label = str(item.get("label", "block"))
        count = _coerce_count(item.get("count"))
        out.append(ParsedLayoutBlock(label=label, bbox=bbox, count=count))
    return out


# ---- Table rec (TABLE_REC_PROMPT) ------------------------------------------


@dataclass
class ParsedTableElement:
    label: str  # "Row" or "Col"
    bbox: Tuple[float, float, float, float]


def parse_table_rec(text: str) -> List[ParsedTableElement]:
    """Parse JSON array of {label: "Row"|"Col", bbox: "x0 y0 x1 y1"} from
    TABLE_REC_PROMPT output. Returns a flat list of Row + Col elements;
    cell derivation is the caller's job."""
    cleaned = _strip_fences(text)
    m = _JSON_ARRAY_RE.search(cleaned)
    if not m:
        raise ValueError(f"No JSON array found in table_rec output: {text[:500]!r}")
    raw = json.loads(m.group(0))
    out: List[ParsedTableElement] = []
    for item in raw:
        label = str(item.get("label", "")).strip()
        if label not in ("Row", "Col"):
            continue
        try:
            bbox = _coerce_bbox(item["bbox"])
        except (KeyError, ValueError):
            continue
        out.append(ParsedTableElement(label=label, bbox=bbox))
    return out


# ---- Block HTML (BLOCK_PROMPT for full table path / general block path) ---


def clean_block_html(html: str) -> str:
    """Light cleanup of model-emitted HTML for a single block.

    Strips code fences, leading/trailing whitespace. Does NOT validate against
    ALLOWED_TAGS — the model is expected to comply, and downstream consumers
    can sanitize further if needed.
    """
    cleaned = _strip_fences(html).strip()
    return cleaned


# ---- Full-page fallback (HIGH_ACCURACY_BBOX_PROMPT) -----------------------


@dataclass
class ParsedFullPageBlock:
    label: str
    bbox: Tuple[float, float, float, float]  # 0-1000 normalized
    html: str  # inner HTML of the wrapping div


def parse_full_page_html(text: str) -> List[ParsedFullPageBlock]:
    """Parse output of HIGH_ACCURACY_BBOX_PROMPT — top-level <div data-bbox=...
    data-label=...>inner HTML</div> blocks. Returns one entry per top-level div."""
    from bs4 import BeautifulSoup

    cleaned = _strip_fences(text).strip()
    if not cleaned:
        return []
    # The model outputs a sequence of top-level divs (no surrounding root).
    # BeautifulSoup parses fine without one.
    soup = BeautifulSoup(cleaned, "html.parser")
    divs = soup.find_all("div", recursive=False)
    out: List[ParsedFullPageBlock] = []
    for div in divs:
        label = div.get("data-label")
        bbox_str = div.get("data-bbox")
        if not label or not bbox_str:
            continue
        try:
            parts = [float(x) for x in bbox_str.split()]
        except ValueError:
            continue
        if len(parts) != 4:
            continue
        # Strip nested data-bbox attrs from the inner HTML so downstream
        # consumers don't see model debug info on every child element.
        for tag in div.find_all(attrs={"data-bbox": True}):
            del tag["data-bbox"]
        for tag in div.find_all(attrs={"data-label": True}):
            del tag["data-label"]
        inner = "".join(str(c) for c in div.contents).strip()
        out.append(
            ParsedFullPageBlock(
                label=str(label),
                bbox=(parts[0], parts[1], parts[2], parts[3]),
                html=inner,
            )
        )
    return out


def denorm_bbox(bbox, img_w: int, img_h: int, scale: int = 1000):
    x0, y0, x1, y1 = bbox
    return (
        x0 / scale * img_w,
        y0 / scale * img_h,
        x1 / scale * img_w,
        y1 / scale * img_h,
    )
