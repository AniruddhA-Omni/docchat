"""Ollama chat client used by every agent.

Always talks to the native ``/api/chat`` endpoint and always sends ``num_ctx`` (Ollama's default
window is only 4K on machines with < 24 GB VRAM, and prompts over it are silently truncated from
the front). ``think`` is sent explicitly for thinking-capable models (qwen3.5 / gemma4); JSON
calls use a JSON schema, validate with pydantic and retry once with the validation error.
"""

from __future__ import annotations

import base64
import contextvars
import io
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError

from docchat.config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.S)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


@dataclass
class CallStat:
    tag: str
    model: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    done_reason: str | None = None


# Per-request call log (the graph sets a fresh list per turn; shown in the trace view).
CALL_LOG: contextvars.ContextVar[list[CallStat] | None] = contextvars.ContextVar(
    "docchat_llm_calls", default=None
)


@dataclass
class LLMResult:
    content: str
    thinking: str = ""
    model: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    done_reason: str | None = None
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


class LLMError(RuntimeError):
    pass


def encode_image(path: str | Path, max_side: int = 1280) -> str:
    """EXIF-correct, downscale and base64-encode an image for a vision request."""
    from PIL import ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences, thinking tags and chatter."""
    text = _THINK_TAG_RE.sub("", text).strip()
    text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


class LLM:
    def __init__(self, client, settings: Settings, model: str | None = None,
                 semaphore: threading.Semaphore | None = None) -> None:  # fmt: skip
        self.client = client
        self.settings = settings
        self.model = model or settings.llm_model
        self._sem = semaphore or threading.Semaphore(settings.llm_parallel)
        self.json_mode = "schema"  # downgraded to "json" if the startup probe fails

    def with_model(self, model: str) -> LLM:
        if model == self.model:
            return self
        other = LLM(self.client, self.settings, model, self._sem)
        other.json_mode = self.json_mode
        return other

    # -- capabilities ----------------------------------------------------------------------------
    @cached_property
    def capabilities(self) -> set[str]:
        try:
            info = self.client.show(self.model)
        except Exception as exc:
            log.warning("could not read capabilities of %s: %s", self.model, exc)
            return {"completion"}
        caps = set(info.capabilities or [])
        # Clamp num_ctx to the model's trained context length.
        for key, value in (info.modelinfo or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                self._max_ctx = value
        return caps

    @property
    def supports_vision(self) -> bool:
        return "vision" in self.capabilities

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in self.capabilities

    @property
    def num_ctx(self) -> int:
        _ = self.capabilities
        return min(self.settings.num_ctx, getattr(self, "_max_ctx", self.settings.num_ctx))

    # -- chat ------------------------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        tag: str = "chat",
        think: bool | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        images: list[str | Path] | None = None,
        format: dict | str | None = None,
        on_token: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> LLMResult:
        msgs = [dict(m) for m in messages]
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        if images:
            last_user = next(m for m in reversed(msgs) if m["role"] == "user")
            last_user["images"] = [encode_image(p) for p in images]
        options = {
            "num_ctx": self.num_ctx,
            "temperature": self.settings.temperature if temperature is None else temperature,
        }
        if num_predict:
            options["num_predict"] = num_predict
        want_think = self.settings.think if think is None else think
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "options": options,
            "keep_alive": self.settings.keep_alive,
        }
        if self.supports_thinking:
            kwargs["think"] = bool(want_think)
        if format is not None:
            kwargs["format"] = format

        start = time.perf_counter()
        with self._sem:
            try:
                if on_token or on_thinking:
                    result = self._stream(kwargs, on_token, on_thinking)
                else:
                    resp = self.client.chat(**kwargs)
                    result = LLMResult(
                        content=resp.message.content or "",
                        thinking=getattr(resp.message, "thinking", None) or "",
                        prompt_tokens=resp.prompt_eval_count or 0,
                        output_tokens=resp.eval_count or 0,
                        done_reason=resp.done_reason,
                    )
            except Exception as exc:
                raise LLMError(f"Ollama call failed ({self.model}): {exc}") from exc
        result.model = self.model
        result.seconds = time.perf_counter() - start
        result.content = _THINK_TAG_RE.sub("", result.content).strip()
        if result.prompt_tokens >= options["num_ctx"] - 8:
            log.warning("%s: prompt filled the %d-token context window; input may have been "
                        "truncated", tag, options["num_ctx"])  # fmt: skip
        calls = CALL_LOG.get()
        if calls is not None:
            calls.append(CallStat(tag, self.model, result.prompt_tokens, result.output_tokens,
                                  round(result.seconds, 2), result.done_reason))  # fmt: skip
        return result

    def _stream(self, kwargs, on_token, on_thinking) -> LLMResult:
        content, thinking = [], []
        last = None
        for part in self.client.chat(**kwargs, stream=True):
            msg = part.message
            if getattr(msg, "thinking", None):
                thinking.append(msg.thinking)
                if on_thinking:
                    on_thinking(msg.thinking)
            if msg.content:
                content.append(msg.content)
                if on_token:
                    on_token(msg.content)
            last = part
        return LLMResult(
            content="".join(content),
            thinking="".join(thinking),
            prompt_tokens=(last.prompt_eval_count or 0) if last else 0,
            output_tokens=(last.eval_count or 0) if last else 0,
            done_reason=last.done_reason if last else None,
        )

    # -- structured output -----------------------------------------------------------------------
    def json(
        self,
        messages: Sequence[dict[str, Any]],
        schema: type[T],
        *,
        system: str | None = None,
        tag: str = "json",
        num_predict: int = 512,
        images: list[str | Path] | None = None,
    ) -> T:
        fmt: dict | str = schema.model_json_schema() if self.json_mode == "schema" else "json"
        msgs = list(messages)
        error = ""
        for attempt in range(2):
            res = self.chat(msgs, system=system, tag=tag, think=False, temperature=0.0,
                            num_predict=num_predict, images=images if attempt == 0 else None,
                            format=fmt)  # fmt: skip
            try:
                return schema.model_validate(extract_json(res.content))
            except (ValueError, ValidationError) as exc:
                error = str(exc)[:500]
                log.info("%s: invalid JSON (attempt %d): %s", tag, attempt + 1, error)
                msgs = [*msgs, {"role": "assistant", "content": res.content},
                        {"role": "user", "content": "That was not valid JSON for the schema "
                         f"({error}). Reply with only the corrected JSON object."}]  # fmt: skip
        raise LLMError(f"{tag}: model did not return valid JSON: {error}")

    def probe_json(self) -> bool:
        """Check that schema-constrained output works with think=False on this Ollama build."""

        class Probe(BaseModel):
            answer: int

        try:
            self.json([{"role": "user", "content": "What is 2 + 3? Reply as JSON."}], Probe,
                      tag="probe", num_predict=64)  # fmt: skip
            return True
        except LLMError:
            self.json_mode = "json"
            log.warning("schema-constrained output failed for %s; falling back to format=json",
                        self.model)  # fmt: skip
            return False
