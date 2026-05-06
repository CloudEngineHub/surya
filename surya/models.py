from typing import Dict, Optional

import torch

from surya.detection import DetectionPredictor
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.logging import configure_logging
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor

configure_logging()


def load_predictors(
    device: str | torch.device | None = None,
    dtype: torch.dtype | str | None = None,
    manager: Optional[SuryaInferenceManager] = None,
) -> Dict[str, object]:
    """Build the standard surya predictor set.

    The VLM-backed predictors (layout, recognition, table_rec) share a single
    SuryaInferenceManager. Detection and OCR error keep their own torch models.
    """
    if manager is None:
        manager = SuryaInferenceManager(lazy=True)
    return {
        "layout": LayoutPredictor(manager),
        "recognition": RecognitionPredictor(manager),
        "table_rec": TableRecPredictor(manager),
        "detection": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error": OCRErrorPredictor(device=device, dtype=dtype),
        "manager": manager,
    }
