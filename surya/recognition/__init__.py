"""RecognitionPredictor: per-block OCR via BLOCK_PROMPT.

Given page images and corresponding LayoutResult (or any list of LayoutBox),
crops each block, runs BLOCK_PROMPT, returns PageOCRResult per page.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.inference import SuryaInferenceManager, get_default_manager
from surya.inference.parsers import clean_block_html, parse_full_page_html
from surya.inference.prompts import (
    PROMPT_TYPE_BLOCK,
    PROMPT_TYPE_HIGH_ACCURACY_BBOX,
    SKIP_OCR_LABELS,
)
from surya.inference.schema import BatchInputItem
from surya.inference.util import image_token_budget
from surya.layout.label import LAYOUT_PRED_RELABEL
from surya.layout.schema import LayoutResult
from surya.logging import get_logger
from surya.recognition.schema import (
    BlockOCRResult,
    OCRResult,
    PageOCRResult,
    TextLine,
)
from surya.settings import settings

logger = get_logger()


# Surya's canonical labels we shouldn't OCR (mirrors model-emitted SKIP_OCR_LABELS
# after canonicalization).
SKIP_CANON_LABELS = {LAYOUT_PRED_RELABEL.get(lbl, lbl) for lbl in SKIP_OCR_LABELS}


def _crop_block(image: Image.Image, polygon, pad: int = 4) -> Image.Image:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(image.size[0], int(max(xs)) + pad)
    y1 = min(image.size[1], int(max(ys)) + pad)
    if x1 <= x0 or y1 <= y0:
        return image.crop((0, 0, 1, 1))
    return image.crop((x0, y0, x1, y1))


class RecognitionPredictor:
    """Per-block OCR. Construct with a SuryaInferenceManager (or rely on default)."""

    def __init__(self, manager: Optional[SuryaInferenceManager] = None):
        self.manager = manager
        self._disable_tqdm = settings.DISABLE_TQDM

    @property
    def disable_tqdm(self) -> bool:
        return self._disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool) -> None:
        self._disable_tqdm = bool(value)

    def to(self, *args, **kwargs):
        return

    def __call__(
        self,
        images: List[Image.Image],
        layout_results: Optional[List[LayoutResult]] = None,
        *,
        full_page: Optional[bool] = None,
    ) -> List[PageOCRResult]:
        """Run OCR on each page.

        Mode resolution:
          - ``full_page=None`` (default): block mode if ``layout_results`` is
            given, else full-page mode. This is the most-do-what-I-mean form.
          - ``full_page=True``: full-page OCR (single HIGH_ACCURACY_BBOX_PROMPT
            request per page). ``layout_results`` is ignored — a warning is
            logged if it was supplied.
          - ``full_page=False``: block mode (per-layout-block OCR request).
            ``layout_results`` is required.

        Full-page is the more accurate path; block mode is for callers that
        specifically need per-block crops (e.g. for downstream merging with
        text-line detection).
        """
        if not images:
            return []
        if full_page is None:
            full_page = layout_results is None
        if full_page:
            if layout_results is not None:
                logger.warning(
                    "RecognitionPredictor called with full_page=True and "
                    "layout_results; layout_results will be ignored."
                )
            return self._full_page_ocr(images)
        if layout_results is None:
            raise ValueError("layout_results required when full_page=False")
        if len(images) != len(layout_results):
            raise ValueError(
                f"images and layout_results must be same length "
                f"({len(images)} vs {len(layout_results)})"
            )
        manager = self.manager or get_default_manager()

        # Build a flat batch across all pages for max concurrency
        batch: List[BatchInputItem] = []
        block_index_map: List[tuple[int, int]] = []  # (page_idx, block_idx)
        skipped_flags: List[bool] = []

        for page_idx, (img, layout) in enumerate(zip(images, layout_results)):
            for block_idx, box in enumerate(layout.bboxes):
                skip = box.label in SKIP_CANON_LABELS
                skipped_flags.append(skip)
                if skip:
                    continue
                crop = _crop_block(img, box.polygon)
                max_tokens = image_token_budget(
                    box.count, ceiling=settings.SURYA_MAX_TOKENS_BLOCK_CEILING
                )
                batch.append(
                    BatchInputItem(
                        image=crop,
                        prompt_type=PROMPT_TYPE_BLOCK,
                        max_tokens=max_tokens,
                        metadata={"page_idx": page_idx, "block_idx": block_idx},
                    )
                )
                block_index_map.append((page_idx, block_idx))

        outputs = manager.generate(batch) if batch else []

        # Index outputs by (page_idx, block_idx)
        out_by_key = {}
        for out in outputs:
            key = (out.metadata["page_idx"], out.metadata["block_idx"])
            out_by_key[key] = out

        # Assemble PageOCRResult per page
        results: List[PageOCRResult] = []
        for page_idx, (img, layout) in enumerate(zip(images, layout_results)):
            w, h = img.size
            blocks: List[BlockOCRResult] = []
            for block_idx, box in enumerate(layout.bboxes):
                skip = box.label in SKIP_CANON_LABELS
                if skip:
                    blocks.append(
                        BlockOCRResult(
                            polygon=box.polygon,
                            label=box.label,
                            raw_label=box.raw_label,
                            reading_order=box.position,
                            html="",
                            skipped=True,
                            confidence=1.0,
                        )
                    )
                    continue
                out = out_by_key.get((page_idx, block_idx))
                if out is None or out.error:
                    blocks.append(
                        BlockOCRResult(
                            polygon=box.polygon,
                            label=box.label,
                            raw_label=box.raw_label,
                            reading_order=box.position,
                            html="",
                            skipped=False,
                            error=True,
                            confidence=0.0,
                        )
                    )
                    continue
                html = clean_block_html(out.raw)
                conf = out.mean_token_prob if out.mean_token_prob is not None else 1.0
                blocks.append(
                    BlockOCRResult(
                        polygon=box.polygon,
                        label=box.label,
                        raw_label=box.raw_label,
                        reading_order=box.position,
                        html=html,
                        skipped=False,
                        error=False,
                        confidence=conf,
                        raw_logprobs=out.logprobs,
                    )
                )
            results.append(
                PageOCRResult(blocks=blocks, image_bbox=[0, 0, float(w), float(h)])
            )
        return results

    def _full_page_ocr(self, images: List[Image.Image]) -> List[PageOCRResult]:
        """One HIGH_ACCURACY_BBOX_PROMPT request per page; parses divs into blocks."""
        manager = self.manager or get_default_manager()
        batch = [
            BatchInputItem(
                image=img,
                prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
                max_tokens=settings.SURYA_MAX_TOKENS_FULL_PAGE,
                metadata={"page_idx": i},
            )
            for i, img in enumerate(images)
        ]
        outputs = manager.generate(batch)
        out_by_page = {o.metadata["page_idx"]: o for o in outputs}

        results: List[PageOCRResult] = []
        for page_idx, img in enumerate(images):
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            out = out_by_page.get(page_idx)
            if out is None or out.error or not out.raw:
                results.append(PageOCRResult(blocks=[], image_bbox=page_bbox))
                continue
            try:
                parsed = parse_full_page_html(out.raw)
            except Exception as e:
                logger.warning(f"Full-page parse failed for page {page_idx}: {e}")
                results.append(PageOCRResult(blocks=[], image_bbox=page_bbox))
                continue
            confidence = out.mean_token_prob if out.mean_token_prob is not None else 1.0
            blocks: List[BlockOCRResult] = []
            for idx, item in enumerate(parsed):
                x0 = item.bbox[0] / settings.BBOX_SCALE * w
                y0 = item.bbox[1] / settings.BBOX_SCALE * h
                x1 = item.bbox[2] / settings.BBOX_SCALE * w
                y1 = item.bbox[3] / settings.BBOX_SCALE * h
                polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                canon = LAYOUT_PRED_RELABEL.get(item.label, item.label)
                skipped = canon in SKIP_CANON_LABELS
                blocks.append(
                    BlockOCRResult(
                        polygon=polygon,
                        label=canon,
                        raw_label=item.label,
                        reading_order=idx,
                        html="" if skipped else item.html,
                        skipped=skipped,
                        error=False,
                        confidence=confidence,
                    )
                )
            results.append(PageOCRResult(blocks=blocks, image_bbox=page_bbox))
        return results

    def to_legacy_ocr_results(
        self, page_results: List[PageOCRResult]
    ) -> List[OCRResult]:
        """Compatibility shim: map BlockOCRResult → OCRResult.text_lines for old
        downstream code that hasn't migrated yet. One TextLine per block, no chars."""
        out: List[OCRResult] = []
        for page in page_results:
            lines: List[TextLine] = []
            for blk in page.blocks:
                lines.append(
                    TextLine(
                        polygon=blk.polygon,
                        text=blk.html,
                        chars=[],
                        confidence=blk.confidence,
                    )
                )
            out.append(OCRResult(text_lines=lines, image_bbox=page.image_bbox))
        return out
