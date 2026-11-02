"""The OpenAI API and what speaks it: OpenAI itself, OpenRouter, Groq, DeepSeek, Together, Ollama, LM Studio, vLLM...

GET /models of OpenAI gives ids only: the kind is guessed from the id (and what OpenAI's chat models see is known);
OpenRouter's also says what a model takes in and what it costs. Other servers' models are chat models that may take
anything (the server says no if not).

Routers with a video API (OpenRouter, ProxyAPI: POST /videos, its job polled, GET /videos/models listing the models)
add their video models to the list: the video node takes them.
"""

from __future__ import annotations

import logging

import httpx

from nodeloom.providers.base import Model, Provider

log = logging.getLogger(__name__)

# Ids of OpenAI's list that are not chat models, or not ones /chat/completions takes.
NOT_CHAT = ("embed", "tts", "whisper", "transcribe", "moderation", "realtime", "dall-e", "davinci", "babbage", "sora",
            "computer-use", "codex", "deep-research", "-pro")
OPENAI_SEEING = ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4", "chatgpt-4o")


class OpenAIProvider(Provider):
    type = "openai"
    families = {"chat": "chat", "image": "openai-image", "video": "video"}
    default_base_url = "https://api.openai.com/v1"

    @property
    def is_openai(self) -> bool:
        return httpx.URL(self.base_url).host == "api.openai.com"

    @property
    def max_tokens_key(self) -> str:
        # OpenAI's reasoning models refuse max_tokens; servers that copy the API know only max_tokens
        return "max_completion_tokens" if self.is_openai else "max_tokens"

    def guess_kind(self, model_id: str) -> str | None:
        low = model_id.lower()
        if low.startswith(("gpt-image", "chatgpt-image")):
            return "image"
        if any(word in low for word in NOT_CHAT):
            return None
        return "chat"

    async def discover(self) -> list[dict]:
        raw = await super().discover()
        # newest first: the first model is what a new node starts with
        raw = sorted(raw, key=lambda x: -(x.get("created") or 0) if isinstance(x, dict) else 0)
        return raw + await self.video_models()

    async def video_models(self) -> list[dict]:
        """A router's video models (GET /videos/models), marked as such; none where there is no video API."""
        if self.is_openai:  # OpenAI's own /videos (Sora) speaks another dialect
            return []
        try:
            r = await self.client().get(self.base_url + "/videos/models", headers=self.headers())
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception as e:  # noqa: BLE001  no video API is no error
            log.info("provider %s: no video models: %s", self.name, type(e).__name__)
            return []
        items = data.get("data") if isinstance(data, dict) else data
        return [{**m, "kind": "video"} for m in items or [] if isinstance(m, dict) and m.get("id")]

    def from_listing(self, raw: dict) -> Model | None:
        mid = str(raw.get("id") or "")
        if not mid:
            return None
        if raw.get("kind") == "video":
            return self.model(mid, "video")
        arch = raw.get("architecture")
        if isinstance(arch, dict):  # OpenRouter: modalities and prices per token
            if "text" not in (arch.get("output_modalities") or ["text"]):
                return None
            return self.model(mid, "chat", arch.get("input_modalities") or [], _openrouter_price(raw.get("pricing")))
        kind = self.guess_kind(mid)
        inputs = None
        if self.is_openai and kind == "chat":
            inputs = (["image"] if mid.startswith(OPENAI_SEEING) else []) + (["audio"] if "audio" in mid else [])
        return self.model(mid, kind, inputs)


def _openrouter_price(p) -> dict | None:
    try:
        prompt, completion = float(p["prompt"]), float(p["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    if prompt < 0 or completion < 0:  # routers whose price depends on the model they pick
        return None
    return {"input": prompt * 1_000_000, "output": completion * 1_000_000}
