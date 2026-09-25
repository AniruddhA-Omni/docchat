"""Runtime settings, loaded from environment variables (prefix ``DOCCHAT_``) and ``.env``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PdfParser = Literal["pymupdf", "docling", "marker", "unstructured"]
OcrBackend = Literal["rapidocr", "tesseract", "paddleocr"]
AnswerMode = Literal["fast", "balanced", "deep"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCCHAT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Ollama -------------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:4b"
    # Vision agent model; empty means "reuse llm_model" (qwen3.5 / gemma4 are multimodal).
    vision_model: str = ""
    embed_model: str = "bge-m3"
    # Ollama's default context window is small; always send num_ctx explicitly.
    num_ctx: int = 16384
    temperature: float = 0.1
    # Thinking mode for reasoning models (qwen3.5 / gemma4). Off by default for latency.
    think: bool = False
    keep_alive: str = "30m"
    request_timeout: float = 300.0

    # --- Storage ------------------------------------------------------------
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    # Local caches for FastEmbed / Hugging Face models, so everything works offline once fetched.
    models_dir: Path = Field(default_factory=lambda: Path.cwd() / "models")
    # Empty -> embedded Qdrant under data_dir/qdrant; set e.g. http://localhost:6333 for Docker.
    qdrant_url: str = ""
    qdrant_collection: str = "docchat"

    # --- Ingestion ----------------------------------------------------------
    pdf_parser: PdfParser = "pymupdf"
    ocr_backend: OcrBackend = "rapidocr"
    chunk_target_tokens: int = 400
    # bge-reranker and bge-m3 (via Ollama) both degrade past ~512 tokens per input.
    chunk_max_tokens: int = 512
    # Describe images / figures with the vision model at ingest so their content is searchable.
    caption_images: bool = True
    max_captions_per_doc: int = 12

    # --- Retrieval ----------------------------------------------------------
    # "ollama" (BGE-M3) for real use; "hashing" is a dependency-free embedder for tests.
    embed_backend: Literal["ollama", "hashing"] = "ollama"
    # fastembed (ONNX, CPU) | sentence-transformers (extra rerank-gpu, CUDA) | none
    reranker_backend: Literal["fastembed", "sentence-transformers", "none"] = "fastembed"
    reranker_model: str = "BAAI/bge-reranker-base"
    retrieve_k: int = 30
    rerank_top_n: int = 6
    # Max concurrent Ollama requests from parallel agents (match OLLAMA_NUM_PARALLEL on the server).
    llm_parallel: int = 2

    # --- Answering ----------------------------------------------------------
    # fast: rule-based verification only; balanced: + LLM claim check;
    # deep: + thinking mode for the answer and one LLM revision pass for unverified claims.
    answer_mode: AnswerMode = "balanced"
    # Token budget for the evidence placed in the synthesis prompt (keep well under num_ctx).
    evidence_budget_tokens: int = 6000
    # "I don't know" gate: abstain when the best source is below these relevance levels.
    abstain_rerank_threshold: float = 0.15
    abstain_dense_threshold: float = 0.45

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model or self.llm_model

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def db_path(self) -> Path:
        """SQLite file for document registry + LangGraph conversation checkpoints."""
        return self.data_dir / "docchat.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def apply_runtime_env(settings: Settings) -> None:
    """Process-wide environment for offline, telemetry-free operation. Call once at startup."""
    defaults = {
        "LANGSMITH_TRACING": "false",
        "LANGGRAPH_STRICT_MSGPACK": "true",
        "DO_NOT_TRACK": "true",
        "SCARF_NO_ANALYTICS": "true",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HOME": str(settings.models_dir / "hf"),
        "FASTEMBED_CACHE_PATH": str(settings.models_dir / "fastembed"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
