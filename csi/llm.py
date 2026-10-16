"""Pluggable LLM client.

Provider order with CSI_LLM_PROVIDER=auto: Groq (hosted, fast) -> Ollama (local, private)
-> offline fallback (no generation; callers use extractive/template logic instead).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx

from csi.config import settings

log = logging.getLogger("csi.llm")


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float


class LLMClient:
    def __init__(self, provider: str | None = None):
        """provider=None follows CSI_LLM_PROVIDER; "groq" / "ollama" / "offline" force one (used by benchmarks)."""
        self.provider, self.model = self._resolve(provider or settings.llm_provider)
        log.info("LLM provider: %s (%s)", self.provider, self.model)

    def _ollama_up(self) -> bool:
        try:
            r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=2)
            names = [m["name"] for m in r.json().get("models", [])]
            return r.status_code == 200 and any(n.startswith(settings.ollama_model) for n in names)
        except Exception:
            return False

    def _resolve(self, choice: str) -> tuple[str, str]:
        if choice in ("groq", "auto") and settings.groq_api_key:
            return "groq", settings.groq_model
        if choice in ("ollama", "auto") and self._ollama_up():
            return "ollama", settings.ollama_model
        return "offline", "none"

    @property
    def available(self) -> bool:
        return self.provider != "offline"

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 400) -> LLMResponse:
        start = time.time()
        if self.provider == "groq":
            r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=settings.llm_timeout,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        elif self.provider == "ollama":
            r = httpx.post(
                f"{settings.ollama_url}/api/chat",
                json={"model": self.model, "stream": False,
                      "options": {"temperature": temperature, "num_predict": max_tokens},
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=settings.llm_timeout,
            )
            r.raise_for_status()
            text = r.json()["message"]["content"]
        else:
            raise LLMUnavailable("No LLM provider configured")
        return LLMResponse(text=text.strip(), provider=self.provider, model=self.model,
                           latency_s=round(time.time() - start, 2))


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
