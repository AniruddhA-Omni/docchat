"""Map file extensions to the format families the ingestion pipeline understands."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class Format(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TEXT = "text"
    MARKDOWN = "markdown"
    TABULAR = "tabular"
    IMAGE = "image"
    CODE = "code"


CODE_LANGUAGES: dict[str, str] = {
    ".py": "python", ".ipynb": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".cs": "csharp",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift", ".scala": "scala",
    ".sql": "sql", ".sh": "bash", ".ps1": "powershell", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".xml": "xml",
}  # fmt: skip

_BY_EXT: dict[str, Format] = {
    ".pdf": Format.PDF,
    ".docx": Format.DOCX,
    ".pptx": Format.PPTX,
    ".txt": Format.TEXT,
    ".log": Format.TEXT,
    ".md": Format.MARKDOWN,
    ".markdown": Format.MARKDOWN,
    ".csv": Format.TABULAR,
    ".tsv": Format.TABULAR,
    ".xlsx": Format.TABULAR,
    ".xlsm": Format.TABULAR,
    ".xls": Format.TABULAR,
    **dict.fromkeys(
        [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"], Format.IMAGE
    ),
    **dict.fromkeys(CODE_LANGUAGES, Format.CODE),
}

SUPPORTED_EXTENSIONS: list[str] = sorted(ext.lstrip(".") for ext in _BY_EXT)


def detect_format(path: str | Path) -> Format | None:
    return _BY_EXT.get(Path(path).suffix.lower())
