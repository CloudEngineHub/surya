import os
from typing import Callable, Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
from pathlib import Path
from platformdirs import user_cache_dir


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 192  # used for layout, recognition, and table rec
    IMAGE_DPI_HIGHRES: int = 192
    IN_STREAMLIT: bool = False
    DISABLE_TQDM: bool = False
    S3_BASE_URL: str = "https://models.datalab.to"
    PARALLEL_DOWNLOAD_WORKERS: int = 10
    MODEL_CACHE_DIR: str = str(Path(user_cache_dir("datalab")) / "models")
    LOGLEVEL: str = "INFO"

    # Paths
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ---- Surya2 inference (VLM-backed: vllm | llamacpp) ---------------------
    SURYA_MODEL_CHECKPOINT: str = "datalab-to/surya-2"
    SURYA_GGUF_REPO: str = "datalab-to/surya-2-gguf"
    SURYA_GGUF_MODEL_FILE: str = "surya-2.gguf"
    SURYA_GGUF_MMPROJ_FILE: str = "surya-2-mmproj.gguf"
    # If set, used directly instead of HF download (handy for local-conversion testing)
    SURYA_GGUF_LOCAL_MODEL_PATH: Optional[str] = None
    SURYA_GGUF_LOCAL_MMPROJ_PATH: Optional[str] = None

    # Backend selection
    SURYA_INFERENCE_BACKEND: Optional[str] = None  # "vllm" | "llamacpp" | None (auto)
    SURYA_INFERENCE_URL: Optional[str] = None  # external server, skip spawn
    SURYA_INFERENCE_AUTOSTART: bool = True
    SURYA_INFERENCE_HOST: str = "127.0.0.1"
    SURYA_INFERENCE_PORT: Optional[int] = None  # None = pick a free port
    SURYA_INFERENCE_PARALLEL: int = 8
    # Per-parallel-slot KV-cache budget for the llama.cpp backend. Worst-case
    # one OCR request: ~2k for image prefill + SURYA_MAX_TOKENS_FULL_PAGE
    # (8192) generation + ~2k prompt/chat-template overhead ≈ 12k. Below this
    # llama-server silently truncates outputs once a slot fills.
    SURYA_INFERENCE_CTX_PER_SLOT: int = 12288
    # Optional override for the *total* ctx passed to llama-server. When None
    # (default), total = max(16384, PARALLEL * CTX_PER_SLOT). Set this only
    # if you've hand-tuned for a specific machine.
    SURYA_INFERENCE_CTX_SIZE: Optional[int] = None
    SURYA_INFERENCE_TIMEOUT_SECONDS: float = 600.0
    SURYA_INFERENCE_STARTUP_TIMEOUT: float = 600.0
    SURYA_INFERENCE_LOGPROBS: bool = True
    # Force layout/table_rec output through a JSON schema via guided decoding.
    # Eliminates malformed-JSON failures at small decode-throughput cost.
    SURYA_GUIDED_LAYOUT: bool = True
    # Disabled: with no minItems in TABLE_REC_JSON_SCHEMA, the constrained
    # decoder closes the array after one element at temperature=0. The model
    # produces well-formed JSON without the schema.
    SURYA_GUIDED_TABLE_REC: bool = False

    # Token budgets
    SURYA_MAX_TOKENS_LAYOUT: int = 3072
    SURYA_MAX_TOKENS_TABLE_REC: int = 3072
    SURYA_MAX_TOKENS_BLOCK_CEILING: int = 8192
    SURYA_MAX_TOKENS_FULL_PAGE: int = (
        # 12288 (vs 8192) buys +1pp overall on olmOCR-bench and +7.24pp on
        # long_tiny_text — dense pages were truncating at 8k. Total budget
        # fits within VLLM_MAX_MODEL_LEN=18000 after image prefill.
        12288
    )

    # When a layout request fails to produce parseable JSON, fall back to
    # HIGH_ACCURACY_BBOX_PROMPT on the full OCR-DPI page. The HTML output
    # carries both layout (data-bbox + data-label) and OCR content (inner
    # HTML), so we skip per-block OCR for fallback pages.
    SURYA_LAYOUT_FALLBACK_FULL_PAGE: bool = True

    BBOX_SCALE: int = 1000

    # vllm
    VLLM_DOCKER_IMAGE: str = "vllm/vllm-openai:v0.20.1"
    VLLM_API_KEY: str = "EMPTY"
    VLLM_GPUS: str = "0"
    VLLM_GPU_TYPE: str = "4090"
    VLLM_MAX_MODEL_LEN: int = 18000
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.85
    VLLM_ENABLE_MTP: bool = True
    VLLM_MTP_TOKENS: int = 2
    VLLM_EXTRA_ARGS: Optional[str] = None
    DOCKER_HF_CACHE_PATH: str = "~/.cache/huggingface"

    # llama.cpp
    LLAMA_CPP_BINARY: str = "llama-server"
    LLAMA_CPP_NGL: int = 99  # all layers on GPU (Metal on macOS, CUDA on Linux GPU); harmless no-op on pure-CPU builds
    LLAMA_CPP_NO_MMPROJ_OFFLOAD: bool = False
    LLAMA_CPP_EXTRA_ARGS: Optional[str] = None

    # ---- Detection (kept) ---------------------------------------------------
    DETECTOR_BATCH_SIZE: Optional[int] = None
    DETECTOR_MODEL_CHECKPOINT: str = "s3://text_detection/2025_05_07"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1400
    DETECTOR_TEXT_THRESHOLD: float = 0.6
    DETECTOR_BLANK_THRESHOLD: float = 0.35
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(8, os.cpu_count())
    DETECTOR_MIN_PARALLEL_THRESH: int = 3
    DETECTOR_BOX_Y_EXPAND_MARGIN: float = 0.05

    # ---- OCR Error (kept) ---------------------------------------------------
    OCR_ERROR_MODEL_CHECKPOINT: str = "s3://ocr_error_detection/2025_02_18"
    OCR_ERROR_BATCH_SIZE: Optional[int] = None

    # ---- Debug / draw fonts (label rendering on annotated images) ----------
    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        "all": os.path.join(FONT_DIR, "GoNotoCurrent-Regular.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
    }
    RECOGNITION_FONT_DL_BASE: str = (
        "https://github.com/satbyy/go-noto-universal/releases/download/v7.0"
    )

    @computed_field
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        return torch.float16

    @computed_field
    def MODEL_DTYPE_BFLOAT(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        return torch.bfloat16

    @computed_field
    def INFERENCE_MODE(self) -> Callable:
        return torch.inference_mode

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
