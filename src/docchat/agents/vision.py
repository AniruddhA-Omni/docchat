"""Vision / OCR agent: re-reads the relevant images with the question in hand.

At ingest every image already has OCR text and (if enabled) a description, which makes it
searchable. At question time this agent sends the image itself plus the question to the
multimodal model (qwen3.5 / gemma4) so chart values and screenshot details are read directly.
"""

from __future__ import annotations

import re
from pathlib import Path

from docchat.agents.types import AgentContext, Evidence
from docchat.llm import LLMError
from docchat.llm.prompts import VISION_PROMPT
from docchat.schemas import Chunk

_ANSWER_RE = re.compile(r"ANSWER:\s*(.*)", re.S | re.I)
_OBS_RE = re.compile(r"OBSERVATION:\s*(.*?)(?:\n\s*ANSWER:|$)", re.S | re.I)
VISION_SCORE = 0.75


def run(ctx: AgentContext, question: str, figure_ids: list[str]) -> tuple[list[Evidence], dict]:
    trace: dict = {"agent": "vision", "images": []}
    llm = ctx.services.vision_llm(ctx.settings)
    if not llm.supports_vision:
        trace["error"] = f"{llm.model} does not support images"
        return [], trace
    evidence = []
    for chunk in _load_chunks(ctx, figure_ids):
        if not chunk.image_path or not Path(chunk.image_path).exists():
            continue
        hint = f"Context: this image is from {chunk.file_name} ({chunk.location() or 'image'})."
        ocr = chunk.text.split("Text in", 1)[-1][:800] if "Text in" in chunk.text else ""
        if ocr:
            hint += f" OCR text found in it (may contain errors): {ocr}"
        prompt = VISION_PROMPT.format(question=question, hint=hint)
        try:
            res = llm.chat([{"role": "user", "content": prompt}], images=[chunk.image_path],
                           tag="vision", think=False, temperature=0.0, num_predict=500)  # fmt: skip
        except LLMError as exc:
            trace["images"].append({"file": chunk.file_name, "error": str(exc)[:200]})
            continue
        text = res.content.strip()
        trace["images"].append({"file": chunk.file_name, "loc": chunk.location(), "reply": text[:400],
                                "seconds": round(res.seconds, 1)})  # fmt: skip
        if not text or "NOT_IN_IMAGE" in text.upper():
            continue
        answer = (_ANSWER_RE.search(text) or [None, text])[1].strip()
        obs = (_OBS_RE.search(text) or [None, ""])[1].strip()
        content = (f"Observation of the image in {chunk.file_name} ({chunk.location() or 'image'}) "
                   f"by the vision model: {obs}\nAnswer read from the image: {answer}")  # fmt: skip
        evidence.append(Evidence(
            id=f"vision:{chunk.chunk_id}", agent="vision", kind="image_obs", content=content,
            score=VISION_SCORE, doc_id=chunk.doc_id, file_name=chunk.file_name,
            file_type=chunk.file_type, location=chunk.location(), page=chunk.page_start,
            bboxes=chunk.bboxes, image_path=chunk.image_path, chunk_kind="figure",
        ))  # fmt: skip
    return evidence, trace


def _load_chunks(ctx: AgentContext, chunk_ids: list[str]) -> list[Chunk]:
    out = []
    for cid in chunk_ids:
        doc_id, _, ordinal = cid.rpartition(":")
        out.extend(ctx.services.store.get_chunks(doc_id, [int(ordinal)]))
    return out
