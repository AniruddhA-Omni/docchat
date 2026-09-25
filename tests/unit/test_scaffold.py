from pathlib import Path

import pytest

from docchat.config import Settings
from docchat.ingest.detect import Format, detect_format
from docchat.ingest.safety import safe_filename
from docchat.llm import OllamaStatus


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("report.PDF", Format.PDF),
        ("sales.xlsx", Format.TABULAR),
        ("scan.jpeg", Format.IMAGE),
        ("service.py", Format.CODE),
        ("notes.md", Format.MARKDOWN),
        ("archive.zip", None),
    ],
)
def test_detect_format(name, expected):
    assert detect_format(name) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ('Q3: "final"?.pdf', "Q3_ _final__.pdf"),
        ("CON.txt", "_CON.txt"),
        ("   .hidden. ", "hidden"),
        ("", "file"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_settings_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DOCCHAT_LLM_MODEL", "gemma4:e4b")
    monkeypatch.setenv("DOCCHAT_DATA_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.llm_model == "gemma4:e4b"
    assert s.effective_vision_model == "gemma4:e4b"
    assert s.qdrant_path == tmp_path / "qdrant"


def test_ollama_status_matches_latest_tag():
    status = OllamaStatus(reachable=True, models=["bge-m3:latest", "qwen3.5:4b"])
    assert status.has("bge-m3")
    assert status.has("qwen3.5:4b")
    assert not status.has("gemma4:e4b")
