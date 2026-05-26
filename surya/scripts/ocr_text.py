import os
import click
import json
import time
from collections import defaultdict

from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.logging import configure_logging, get_logger
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader

configure_logging()
logger = get_logger()


@click.command(help="OCR text — runs layout then per-block OCR.")
@CLILoader.common_options
def ocr_text_cli(input_path: str, **kwargs):
    # Layout runs on the low-DPI render; recognition gets the high-DPI image
    # so small glyphs are resolved. target_image_sizes makes layout return
    # bboxes already in the high-DPI coord space.
    loader = CLILoader(input_path, kwargs, highres=True)

    manager = SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)
    rec_predictor = RecognitionPredictor(manager)

    start = time.time()
    layouts = layout_predictor(
        loader.images,
        target_image_sizes=[img.size for img in loader.highres_images],
    )
    page_results = rec_predictor(loader.highres_images, layouts)

    if loader.debug:
        logger.debug(f"OCR took {time.time() - start:.2f} seconds")

    out_preds = defaultdict(list)
    for name, page in zip(loader.names, page_results):
        out_pred = page.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    with open(
        os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(out_preds, f, ensure_ascii=False)

    logger.info(f"Wrote results to {loader.result_path}")
