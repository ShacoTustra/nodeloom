"""Google's Gemini API (a key from Google AI Studio): Gemini text models, Nano Banana pictures, Gemini Omni video.

- Text goes through Gemini's OpenAI-compatible /openai/chat/completions (pictures and sound in, function calling for
  the Builder): the text model node and the Builder speak to it as to any other provider.
- Pictures (the gemini-image family: Nano Banana) and video (gemini-video: Omni Flash) go through the Interactions
  API, POST /interactions: a picture in one call, a video in the background (`background: true`), its interaction id
  remembered so a run resumed after a restart follows it, polled with GET /interactions/{id} until it is done.

GET /models lists the models; the id tells the family (…image… and nano-banana: pictures, omni: video). Imagen and
Veo's preview models are shut down in the Gemini API, so there are no nodes for them. Google prices everything in
tokens, pictures and video by their output tokens: a price of {input, output} per million tokens gives the cost.
"""

from __future__ import annotations

import re

from nodeloom.providers.base import Model, Provider

# Models of the list that the nodes have no use for (or that /chat/completions does not take).
NOT_CHAT = ("embedding", "embed", "tts", "live", "transcribe", "aqa", "robotics", "lyria", "native-audio",
            "computer-use", "translate", "learnlm", "deep-research", "antigravity")
SHUT = ("imagen", "veo")  # shut down in the Gemini API


def is_video(model_id: str) -> bool:
    return "omni" in model_id.lower()


def is_image(model_id: str) -> bool:
    low = model_id.lower()
    return ("image" in low or "nano-banana" in low) and not low.startswith(SHUT)


class GoogleProvider(Provider):
    type = "google"
    families = {"chat": "chat", "image": "gemini-image", "video": "gemini-video"}
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    default_file_hosts = ("googleapis.com", "googleusercontent.com")  # a result delivered as a link
    chat_path = "/openai/chat/completions"

    def headers(self) -> dict[str, str]:
        key = self.settings.key()
        return {**({"x-goog-api-key": key} if key else {}), **self.settings.headers}

    def headers_for(self, path: str) -> dict[str, str]:
        # the OpenAI-compatible part takes the key as a bearer token; the rest would read a bearer token as OAuth
        if path.startswith("/openai/"):
            key = self.settings.key()
            return {**({"Authorization": f"Bearer {key}"} if key else {}), **self.settings.headers}
        return self.headers()

    def guess_kind(self, model_id: str) -> str | None:
        low = model_id.lower()
        if low.startswith(SHUT):
            return None
        if is_video(low):
            return "video"
        if is_image(low):
            return "image"
        if any(word in low for word in NOT_CHAT):
            return None
        return "chat" if low.startswith(("gemini", "gemma")) else None

    async def discover(self) -> list[dict]:
        out, token = [], None
        for _ in range(20):
            params = {"pageSize": 1000, **({"pageToken": token} if token else {})}
            r = await self.client().get(self.base_url + "/models", params=params, headers=self.headers())
            r.raise_for_status()
            data = r.json()
            out += [m for m in data.get("models") or [] if isinstance(m, dict)]
            token = data.get("nextPageToken")
            if not token:
                break
        return sorted(out, key=_newest)  # the first model is what a new node starts with

    def from_listing(self, raw: dict) -> Model | None:
        mid = str(raw.get("name") or "").removeprefix("models/")
        if not mid:
            return None
        kind = self.guess_kind(mid)
        methods = raw.get("supportedGenerationMethods")
        if kind == "chat" and isinstance(methods, list) and methods and "generateContent" not in methods:
            return None  # embeddings, question answering...
        inputs = ["image", "audio"] if kind == "chat" and mid.startswith("gemini") else None
        return self.model(mid, kind, inputs)


def _newest(raw: dict) -> tuple:
    """Gemini before Gemma, stable before previews, newer versions first."""
    mid = str(raw.get("name") or "").removeprefix("models/").lower()
    version = re.search(r"-(\d+(?:\.\d+)?)", mid)
    return (not mid.startswith("gemini"), bool(re.search(r"preview|exp", mid)),
            -float(version.group(1)) if version else 0.0, mid)


# ---------------------------------------------------------------- the Interactions API


def media_part(data: str, mime: str) -> dict:
    """A picture, a sound or a video (base64) in an interaction's input."""
    head = mime.split("/")[0]
    return {"type": head if head in ("image", "audio", "video") else "document", "data": data, "mime_type": mime}


def output_parts(interaction: dict) -> list[dict]:
    """The model's output: the content of its model_output steps (text, image, video, audio parts)."""
    parts: list[dict] = []
    for step in interaction.get("steps") or []:
        if isinstance(step, dict) and step.get("type") == "model_output":
            parts += [c for c in step.get("content") or [] if isinstance(c, dict)]
    for key in ("outputs", "output"):  # an answer shaped otherwise
        if not parts and isinstance(interaction.get(key), list):
            parts = [c for c in interaction[key] if isinstance(c, dict)]
    return parts


def usage(interaction: dict) -> dict:
    """Tokens in OpenAI's words, thinking counted as output (it is priced so)."""
    u = interaction.get("usage") or {}
    return {"input_tokens": u.get("total_input_tokens") or 0,
            "output_tokens": (u.get("total_output_tokens") or 0) + (u.get("total_thought_tokens") or 0)}


def error_text(interaction: dict) -> str | None:
    err = interaction.get("error")
    if isinstance(err, dict):
        return err.get("message") or err.get("status")
    return str(err) if err else None
