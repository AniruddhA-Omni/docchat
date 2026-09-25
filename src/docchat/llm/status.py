"""Thin wrapper around the Ollama client: connectivity checks and model discovery."""

from __future__ import annotations

from dataclasses import dataclass, field

import ollama

from docchat.config import Settings


@dataclass
class OllamaStatus:
    reachable: bool
    models: list[str] = field(default_factory=list)
    error: str = ""

    def has(self, model: str) -> bool:
        """True if ``model`` is pulled (an untagged name matches ``:latest``)."""
        name = model if ":" in model else f"{model}:latest"
        return name in self.models


def get_client(settings: Settings) -> ollama.Client:
    return ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout)


def check_ollama(settings: Settings) -> OllamaStatus:
    try:
        listing = get_client(settings).list()
    except Exception as exc:  # connection refused, timeout, bad host...
        return OllamaStatus(reachable=False, error=str(exc))
    return OllamaStatus(reachable=True, models=sorted(m.model for m in listing.models if m.model))
