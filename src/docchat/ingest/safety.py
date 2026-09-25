"""Input hygiene for uploaded files (Windows-safe names, archive bomb checks)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}  # fmt: skip

# DOCX / PPTX / XLSX are zip containers; refuse ones that inflate absurdly.
MAX_UNZIPPED_BYTES = 1_000_000_000
MAX_COMPRESSION_RATIO = 150


def safe_filename(name: str, max_len: int = 150) -> str:
    """Strip directories and characters Windows cannot store; keep the extension."""
    name = _FORBIDDEN.sub("_", Path(name).name).strip(" .")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if stem.upper() in _RESERVED:
        stem = f"_{stem}"
    stem = stem[: max_len - len(ext) - 1] or "file"
    return f"{stem}.{ext}" if ext else stem


def check_zip_container(path: Path) -> None:
    """Raise ValueError if an OOXML file looks like a zip bomb or is not a zip at all."""
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name} is not a valid Office file") from exc
    total = sum(i.file_size for i in infos)
    compressed = sum(i.compress_size for i in infos) or 1
    if total > MAX_UNZIPPED_BYTES or total / compressed > MAX_COMPRESSION_RATIO:
        raise ValueError(f"{path.name} inflates to {total:,} bytes; refusing to parse")
