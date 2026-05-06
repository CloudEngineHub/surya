"""Pipelined layout + block-OCR orchestrator.

Standard flow runs all layouts (batch) then all blocks (batch). The latter
can't start until every page has finished layout, leaving server slots
idle while layouts trickle in.

This orchestrator overlaps the two phases:
  * Submit every layout request into one ThreadPoolExecutor.
  * As each layout future completes, immediately submit that page's block
    OCR requests, **reverse-sorted by count** (LPT scheduling) so big
    blocks start their decode earliest.
  * If a layout fails to parse, fall back to HIGH_ACCURACY_BBOX_PROMPT on
    the full OCR-DPI page — its HTML output yields layout + content in one
    shot (no per-block phase needed for fallback pages).
  * Drain remaining block futures + fallback futures.

Returns the same shape as the serial pair: (List[LayoutResult], List[PageOCRResult]).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from PIL import Image

from surya.inference import SuryaInferenceManager
from surya.inference.backends.openai_client import _generate_one
from surya.inference.parsers import (
    clean_block_html,
    denorm_bbox,
    parse_full_page_html,
    parse_layout,
)
from surya.inference.prompts import LAYOUT_JSON_SCHEMA
from surya.inference.schema import (
    BatchInputItem,
    PROMPT_TYPE_BLOCK,
    PROMPT_TYPE_HIGH_ACCURACY_BBOX,
    PROMPT_TYPE_LAYOUT,
)
from surya.inference.util import image_token_budget
from surya.layout.label import LAYOUT_PRED_RELABEL
from surya.layout.schema import LayoutBox, LayoutResult
from surya.logging import get_logger
from surya.recognition import SKIP_CANON_LABELS, _crop_block
from surya.recognition.schema import BlockOCRResult, PageOCRResult
from surya.settings import settings

logger = get_logger()


def _make_layout_result(
    raw: str,
    error: bool,
    mean_token_prob: Optional[float],
    target_size: Tuple[int, int],
) -> LayoutResult:
    w, h = target_size
    page_bbox = [0, 0, float(w), float(h)]
    if error or not raw:
        return LayoutResult(bboxes=[], image_bbox=page_bbox, raw=raw, error=True)
    try:
        parsed = parse_layout(raw)
    except Exception as e:
        logger.warning(f"Layout parse failed: {e}; raw[:200]={raw[:200]!r}")
        return LayoutResult(bboxes=[], image_bbox=page_bbox, raw=raw, error=True)

    confidence = mean_token_prob if mean_token_prob is not None else 1.0
    boxes: List[LayoutBox] = []
    for idx, blk in enumerate(parsed):
        pixel_bbox = denorm_bbox(blk.bbox, w, h, scale=settings.BBOX_SCALE)
        canon = LAYOUT_PRED_RELABEL.get(blk.label, blk.label)
        boxes.append(
            LayoutBox(
                polygon=list(pixel_bbox),
                label=canon,
                raw_label=blk.label,
                position=idx,
                count=blk.count,
                confidence=confidence,
            )
        )
    return LayoutResult(bboxes=boxes, image_bbox=page_bbox, raw=raw, error=False)


def _make_block_result(
    page_box: LayoutBox,
    raw: str,
    error: bool,
    mean_token_prob: Optional[float],
    skipped: bool,
) -> BlockOCRResult:
    if skipped:
        return BlockOCRResult(
            polygon=page_box.polygon,
            label=page_box.label,
            raw_label=page_box.raw_label,
            reading_order=page_box.position,
            html="",
            skipped=True,
            confidence=1.0,
        )
    if error:
        return BlockOCRResult(
            polygon=page_box.polygon,
            label=page_box.label,
            raw_label=page_box.raw_label,
            reading_order=page_box.position,
            html="",
            skipped=False,
            error=True,
            confidence=0.0,
        )
    html = clean_block_html(raw)
    conf = mean_token_prob if mean_token_prob is not None else 1.0
    return BlockOCRResult(
        polygon=page_box.polygon,
        label=page_box.label,
        raw_label=page_box.raw_label,
        reading_order=page_box.position,
        html=html,
        skipped=False,
        error=False,
        confidence=conf,
    )


def _full_page_to_results(
    raw: str,
    mean_token_prob: Optional[float],
    target_size: Tuple[int, int],
) -> Tuple[LayoutResult, PageOCRResult]:
    """Parse HIGH_ACCURACY_BBOX_PROMPT output into a LayoutResult + PageOCRResult
    pair (one block per top-level <div>)."""
    w, h = target_size
    page_bbox = [0, 0, float(w), float(h)]
    parsed = parse_full_page_html(raw)
    confidence = mean_token_prob if mean_token_prob is not None else 1.0

    boxes: List[LayoutBox] = []
    blocks: List[BlockOCRResult] = []
    for idx, item in enumerate(parsed):
        pixel_bbox = denorm_bbox(item.bbox, w, h, scale=settings.BBOX_SCALE)
        canon = LAYOUT_PRED_RELABEL.get(item.label, item.label)
        polygon = [
            [pixel_bbox[0], pixel_bbox[1]],
            [pixel_bbox[2], pixel_bbox[1]],
            [pixel_bbox[2], pixel_bbox[3]],
            [pixel_bbox[0], pixel_bbox[3]],
        ]
        boxes.append(
            LayoutBox(
                polygon=polygon,
                label=canon,
                raw_label=item.label,
                position=idx,
                count=0,
                confidence=confidence,
            )
        )
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
    layout = LayoutResult(bboxes=boxes, image_bbox=page_bbox, raw=raw, error=False)
    page = PageOCRResult(blocks=blocks, image_bbox=page_bbox)
    return layout, page


def layout_then_blocks(
    manager: SuryaInferenceManager,
    ocr_images: List[Image.Image],
    layout_images: Optional[List[Image.Image]] = None,
    max_workers: Optional[int] = None,
) -> Tuple[List[LayoutResult], List[PageOCRResult]]:
    """Run layout for all pages with block OCR pipelined per page.

    layout_images: optional low-DPI renders for layout. If None, ocr_images are
    used for both. Bbox coords are returned in ocr_images coord space either way.

    On layout failure, falls back to HIGH_ACCURACY_BBOX_PROMPT on the full OCR
    image (when settings.SURYA_LAYOUT_FALLBACK_FULL_PAGE is True).
    """
    if not ocr_images:
        return [], []
    if layout_images is None:
        layout_images = ocr_images

    manager.start()
    backend = manager.backend
    client = backend._client
    model_name = backend.handle.model_name

    timeout = settings.SURYA_INFERENCE_TIMEOUT_SECONDS
    request_logprobs = settings.SURYA_INFERENCE_LOGPROBS
    n_workers = max_workers or settings.SURYA_INFERENCE_PARALLEL

    common_kwargs = dict(
        client=client,
        model_name=model_name,
        max_tokens_default=settings.SURYA_MAX_TOKENS_LAYOUT,
        temperature=0.0,
        top_p=0.1,
        timeout=timeout,
        request_logprobs_default=request_logprobs,
    )

    n_pages = len(ocr_images)
    layout_results: List[Optional[LayoutResult]] = [None] * n_pages
    page_results: List[Optional[PageOCRResult]] = [None] * n_pages
    block_results: dict = {}  # (page_idx, block_idx) -> (raw, error, mean_p, skipped)

    fallback_enabled = settings.SURYA_LAYOUT_FALLBACK_FULL_PAGE

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # ---- Phase 1: submit all layout requests ----
        guided = LAYOUT_JSON_SCHEMA if settings.SURYA_GUIDED_LAYOUT else None
        layout_futures = {}
        for page_idx, lo_img in enumerate(layout_images):
            item = BatchInputItem(
                image=lo_img,
                prompt_type=PROMPT_TYPE_LAYOUT,
                max_tokens=settings.SURYA_MAX_TOKENS_LAYOUT,
                guided_json=guided,
                metadata={"page_idx": page_idx},
            )
            fut = executor.submit(_generate_one, item, **common_kwargs)
            layout_futures[fut] = page_idx

        # ---- Phase 2: as layouts land, submit blocks (LPT sort) or fallback ----
        block_futures = {}
        fallback_futures = {}

        def _submit_fallback(page_idx: int):
            """Schedule a HIGH_ACCURACY_BBOX_PROMPT request for this page."""
            item = BatchInputItem(
                image=ocr_images[page_idx],
                prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
                max_tokens=settings.SURYA_MAX_TOKENS_FULL_PAGE,
                metadata={"page_idx": page_idx},
            )
            fb_fut = executor.submit(_generate_one, item, **common_kwargs)
            fallback_futures[fb_fut] = page_idx

        for fut in as_completed(layout_futures):
            page_idx = layout_futures[fut]
            try:
                gen = fut.result()
            except Exception as e:
                logger.warning(f"Layout request failed for page {page_idx}: {e}")
                if fallback_enabled:
                    _submit_fallback(page_idx)
                else:
                    w, h = ocr_images[page_idx].size
                    layout_results[page_idx] = LayoutResult(
                        bboxes=[],
                        image_bbox=[0, 0, float(w), float(h)],
                        raw=None,
                        error=True,
                    )
                continue

            target_size = ocr_images[page_idx].size
            layout_result = _make_layout_result(
                raw=gen.raw,
                error=gen.error,
                mean_token_prob=gen.mean_token_prob,
                target_size=target_size,
            )
            if layout_result.error:
                if fallback_enabled:
                    logger.info(
                        f"Layout failed for page {page_idx}, falling back to full-page"
                    )
                    _submit_fallback(page_idx)
                    continue
                layout_results[page_idx] = layout_result
                continue

            layout_results[page_idx] = layout_result

            # Per-page LPT: reverse-sort blocks by count
            blocks_sorted = sorted(
                enumerate(layout_result.bboxes),
                key=lambda kv: -kv[1].count,
            )
            for block_idx, blk in blocks_sorted:
                if blk.label in SKIP_CANON_LABELS:
                    block_results[(page_idx, block_idx)] = ("", False, None, True)
                    continue
                crop = _crop_block(ocr_images[page_idx], blk.polygon)
                max_tokens = image_token_budget(
                    blk.count, ceiling=settings.SURYA_MAX_TOKENS_BLOCK_CEILING
                )
                block_item = BatchInputItem(
                    image=crop,
                    prompt_type=PROMPT_TYPE_BLOCK,
                    max_tokens=max_tokens,
                    metadata={"page_idx": page_idx, "block_idx": block_idx},
                )
                bfut = executor.submit(_generate_one, block_item, **common_kwargs)
                block_futures[bfut] = (page_idx, block_idx)

        # ---- Phase 3: drain block futures ----
        for fut in as_completed(block_futures):
            key = block_futures[fut]
            try:
                gen = fut.result()
            except Exception as e:
                logger.warning(f"Block request failed for {key}: {e}")
                block_results[key] = ("", True, None, False)
                continue
            block_results[key] = (gen.raw, gen.error, gen.mean_token_prob, False)

        # ---- Phase 4: drain fallback futures ----
        for fut in as_completed(fallback_futures):
            page_idx = fallback_futures[fut]
            target_size = ocr_images[page_idx].size
            page_bbox = [0, 0, float(target_size[0]), float(target_size[1])]
            try:
                gen = fut.result()
            except Exception as e:
                logger.warning(f"Fallback request failed for page {page_idx}: {e}")
                layout_results[page_idx] = LayoutResult(
                    bboxes=[],
                    image_bbox=page_bbox,
                    raw=None,
                    error=True,
                )
                page_results[page_idx] = PageOCRResult(blocks=[], image_bbox=page_bbox)
                continue
            if gen.error or not gen.raw:
                layout_results[page_idx] = LayoutResult(
                    bboxes=[],
                    image_bbox=page_bbox,
                    raw=gen.raw,
                    error=True,
                )
                page_results[page_idx] = PageOCRResult(blocks=[], image_bbox=page_bbox)
                continue
            try:
                layout, page = _full_page_to_results(
                    gen.raw, gen.mean_token_prob, target_size
                )
                layout_results[page_idx] = layout
                page_results[page_idx] = page
            except Exception as e:
                logger.warning(f"Fallback parse failed for page {page_idx}: {e}")
                layout_results[page_idx] = LayoutResult(
                    bboxes=[],
                    image_bbox=page_bbox,
                    raw=gen.raw,
                    error=True,
                )
                page_results[page_idx] = PageOCRResult(blocks=[], image_bbox=page_bbox)

    # ---- Assemble PageOCRResult for non-fallback pages ----
    for page_idx, layout_result in enumerate(layout_results):
        if page_results[page_idx] is not None:
            # Already populated by fallback path
            continue
        if layout_result is None:
            w, h = ocr_images[page_idx].size
            page_results[page_idx] = PageOCRResult(
                blocks=[], image_bbox=[0, 0, float(w), float(h)]
            )
            continue
        blocks: List[BlockOCRResult] = []
        for block_idx, blk in enumerate(layout_result.bboxes):
            entry = block_results.get((page_idx, block_idx))
            if entry is None:
                blocks.append(
                    BlockOCRResult(
                        polygon=blk.polygon,
                        label=blk.label,
                        raw_label=blk.raw_label,
                        reading_order=blk.position,
                        html="",
                        error=True,
                        confidence=0.0,
                    )
                )
                continue
            raw, error, mean_p, skipped = entry
            blocks.append(_make_block_result(blk, raw, error, mean_p, skipped))
        w, h = ocr_images[page_idx].size
        page_results[page_idx] = PageOCRResult(
            blocks=blocks, image_bbox=[0, 0, float(w), float(h)]
        )

    return list(layout_results), list(page_results)  # type: ignore[arg-type]
