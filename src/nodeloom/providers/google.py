"""Google's Gemini API (a key from Google AI Studio, or a gateway that passes Google's own API on): Gemini text
models, Nano Banana pictures, Gemini Omni video.

- Text and pictures (the gemini-image family: Nano Banana) go through generateContent
  (POST /models/{model}:generateContent), the API every gateway in front of Gemini passes on: the text model node's
  OpenAI-style request is turned into Google's (`chat`), with pictures, sound and video in it. The Builder, which
  needs function calling, uses Gemini's OpenAI-compatible /openai/chat/completions.
- Video (gemini-video: Omni Flash) goes through the Interactions API, POST /interactions, in the background
  (`background: true`): its interaction id is remembered so a run resumed after a restart follows it, and polled with
  GET /interactions/{id} until it is done.

GET /models lists the models; the id tells the family (…image… and nano-banana: pictures, omni: video). Imagen and
Veo's preview models are shut down in the Gemini API, so there are no nodes for them. Google prices everything in
tokens, pictures and video by their output tokens: a price of {input, output} per million tokens gives the cost.
"""

from __future__ import annotations

import re

from nodeloom import i18n
from nodeloom.providers.base import Model, Provider, ProviderError

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
        inputs = ["image", "audio", "video"] if kind == "chat" and mid.startswith("gemini") else None
        return self.model(mid, kind, inputs)

    async def chat(self, body: dict) -> dict:
        """The text model node's OpenAI-style request through generateContent, answered in OpenAI's shape."""
        answer = await self.request("POST", f"/models/{body['model']}:generateContent", json=native_request(body))
        text = "".join(p["text"] for p in native_parts(answer) if isinstance(p.get("text"), str))
        if not text and (why := native_refusal(answer)):
            raise ProviderError(i18n.tr(f"{self.name} не ответил: {why}", f"{self.name} gave no answer: {why}"))
        u = native_usage(answer)
        return {"choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": u["input_tokens"], "completion_tokens": u["output_tokens"]}}


def _newest(raw: dict) -> tuple:
    """Gemini before Gemma, stable before previews, newer versions first."""
    mid = str(raw.get("name") or "").removeprefix("models/").lower()
    version = re.search(r"-(\d+(?:\.\d+)?)", mid)
    return (not mid.startswith("gemini"), bool(re.search(r"preview|exp", mid)),
            -float(version.group(1)) if version else 0.0, mid)


# ---------------------------------------------------------------- generateContent


def inline(data: str, mime: str) -> dict:
    """A file (base64) in a generateContent request."""
    return {"inlineData": {"mimeType": mime, "data": data}}


def _data_uri(url: str) -> dict | None:
    head, _, data = (url or "").partition(",")
    if not head.startswith("data:") or not data:
        return None
    return inline(data, head[5:].split(";")[0] or "application/octet-stream")


def native_request(body: dict) -> dict:
    """An OpenAI-style chat request (text, image_url, input_audio and video_url parts) as a generateContent one."""
    system: list[dict] = []
    contents: list[dict] = []
    for m in body.get("messages") or []:
        content = m.get("content")
        parts: list[dict | None] = []
        for p in [{"type": "text", "text": content}] if isinstance(content, str) else content or []:
            kind = p.get("type")
            if kind == "text":
                parts.append({"text": p.get("text") or ""})
            elif kind == "image_url":
                parts.append(_data_uri((p.get("image_url") or {}).get("url", "")))
            elif kind == "video_url":
                parts.append(_data_uri((p.get("video_url") or {}).get("url", "")))
            elif kind == "input_audio":
                a = p.get("input_audio") or {}
                parts.append(inline(a.get("data") or "", f"audio/{a.get('format') or 'mp3'}"))
        clean = [x for x in parts if x]
        if m.get("role") == "system":
            system += clean
        else:
            contents.append({"role": "model" if m.get("role") == "assistant" else "user", "parts": clean})
    out: dict = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": system}
    config: dict = {}
    if "temperature" in body:
        config["temperature"] = body["temperature"]
    if limit := body.get("max_tokens") or body.get("max_completion_tokens"):
        config["maxOutputTokens"] = limit
    if config:
        out["generationConfig"] = config
    return out


def native_parts(answer: dict) -> list[dict]:
    """The first candidate's parts, without the model's thoughts."""
    candidates = answer.get("candidates") or [{}]
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    return [p for p in parts if isinstance(p, dict) and not p.get("thought")]


def native_usage(answer: dict) -> dict:
    """Tokens in OpenAI's words, thinking counted as output (it is priced so)."""
    u = answer.get("usageMetadata") or {}
    return {"input_tokens": u.get("promptTokenCount") or 0,
            "output_tokens": (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)}


def native_refusal(answer: dict) -> str | None:
    """Why there is no answer: the prompt was blocked, or the model stopped for a reason other than being done."""
    feedback = answer.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        return feedback.get("blockReasonMessage") or feedback["blockReason"]
    candidates = answer.get("candidates") or []
    if not candidates:
        return "no answer"
    c = candidates[0] or {}
    if c.get("finishReason") not in (None, "STOP"):
        return c.get("finishMessage") or c["finishReason"]
    return None


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
