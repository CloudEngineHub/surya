"""Surya inference manager.

One process owns one SuryaInferenceManager. The manager wraps a single backend
(vllm | llamacpp) which speaks OpenAI-compatible chat completions.

Predictors take the manager via explicit injection (see surya/models.py).
"""

from __future__ import annotations

from typing import List, Optional

from surya.inference.backends.base import Backend
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _autodetect_backend() -> str:
    if settings.SURYA_INFERENCE_BACKEND:
        return settings.SURYA_INFERENCE_BACKEND
    # cuda → vllm, mps/cpu → llamacpp
    try:
        import torch

        if torch.cuda.is_available():
            return "vllm"
    except Exception:
        pass
    return "llamacpp"


def _build_backend(method: str) -> Backend:
    method = method.lower()
    if method == "vllm":
        from surya.inference.backends.vllm import VllmBackend

        return VllmBackend()
    if method == "llamacpp":
        from surya.inference.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend()
    raise ValueError(
        f"Unknown inference backend {method!r}. Supported: 'vllm', 'llamacpp'."
    )


class SuryaInferenceManager:
    """Single entry point for VLM inference. Construct once per process."""

    def __init__(self, method: Optional[str] = None, lazy: bool = True):
        self.method = method or _autodetect_backend()
        self.backend: Backend = _build_backend(self.method)
        if not lazy:
            self.backend.start()

    def start(self) -> None:
        self.backend.start()

    def stop(self) -> None:
        self.backend.stop()

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        return self.backend.generate(batch)


# Module-level lazy singleton for callers that don't want explicit construction
# (notebooks, ad-hoc scripts). Surya's own models.py and marker should use
# explicit construction.
_default_manager: Optional[SuryaInferenceManager] = None


def get_default_manager() -> SuryaInferenceManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = SuryaInferenceManager()
    return _default_manager
