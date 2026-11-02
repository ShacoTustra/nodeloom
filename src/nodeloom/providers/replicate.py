"""Replicate: thousands of models, each with its own inputs, run as predictions.

Every model in the config (`endpoints`: "owner/name", or "owner/name:<version>" to keep one version; with
REPLICATE_API_TOKEN alone, a list of popular official models) becomes a node of its own: its widgets and links are
read off the OpenAPI schema of the model's version (schema_nodes.py), fetched with the model (GET /models/owner/name)
and kept in the data directory for a day (providers/schemas.py). A run is a prediction: POST /predictions (an
official model by its name, any other by the version its schema came from), its id remembered so a run resumed after
a restart follows it, polled until done; the output's files are copied into the user's results.

Replicate's schemas say of a file output only that it is a uri: whether it is a picture, a video or a sound comes
from the config (`kind`), else from the model's settings and description (`guess_media`).

Files go to Replicate as data URIs when small and through its Files API (POST /files) otherwise. Replicate says no
price in its answers: give one per model ({image = 0.003}, {second = 0.0014}, {request = 0.05}).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time

from nodeloom import i18n, schema_nodes
from nodeloom.config import ModelSettings
from nodeloom.nodes import Ctx, Node, NodeError, Suspended
from nodeloom.providers.base import Model, ProviderError
from nodeloom.providers.schemas import SchemaProvider, camel, title_of

log = logging.getLogger(__name__)

POLL_SECONDS = 2.0
TASK_TIMEOUT = 45 * 60
DATA_URI_MAX = 256 * 1024  # larger files go through the Files API (Replicate's advice)
# With REPLICATE_API_TOKEN alone: well-known official models of each kind (the config's `endpoints` replaces the list).
DEFAULT_ENDPOINTS = [
    "black-forest-labs/flux-schnell", "black-forest-labs/flux-2-pro", "black-forest-labs/flux-kontext-pro",
    "google/nano-banana-pro", "bytedance/seedream-4", "ideogram-ai/ideogram-v3-turbo", "recraft-ai/recraft-v3",
    "kwaivgi/kling-v2.5-turbo-pro", "minimax/hailuo-02", "google/veo-3-fast", "wan-video/wan-2.5-i2v",
    "minimax/speech-02-turbo",
]
ENDPOINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(:[0-9a-f]{64})?$")
KINDS = {"image": "IMAGE", "video": "VIDEO", "audio": "AUDIO"}
STATUS = {"starting": ("запуск", "starting"), "processing": ("генерация", "generating"),
          "succeeded": ("готово", "done")}
AUDIO_FORMATS = {"mp3", "wav", "flac", "ogg", "aac", "opus", "pcm", "m4a"}
VIDEO_FORMATS = {"mp4", "webm", "mov", "gif"}
IMAGE_FORMATS = {"png", "jpg", "jpeg", "webp"}
VIDEO_SETTINGS = ("duration", "fps", "num_frames", "frames", "start_image", "end_image", "first_frame", "last_frame")
AUDIO_SETTINGS = ("voice", "speech", "lyrics", "sample_rate", "bitrate", "audio_format", "emotion")


def node_id(endpoint: str) -> str:
    """black-forest-labs/flux-schnell -> LoomReplicateBlackForestLabsFluxSchnell (a pinned version: the same node)."""
    return "LoomReplicate" + camel(endpoint.partition(":")[0])


def title(endpoint: str) -> str:
    """kwaivgi/kling-v2.5-turbo-pro -> Kling v2.5 Turbo Pro."""
    return title_of(endpoint.partition(":")[0].split("/")[-1])


def guess_media(props: dict, description: str = "") -> str:
    """What a model's bare uri output is: the format it is asked for, else its settings, else its description."""
    for name in ("output_format", "audio_format", "format"):
        values = {str(v).lower() for v in (props.get(name) or {}).get("enum") or []}
        if values & AUDIO_FORMATS:
            return "AUDIO"
        if values & VIDEO_FORMATS:
            return "VIDEO"
        if values & IMAGE_FORMATS:
            return "IMAGE"
    names = [n.lower() for n, p in props.items() if (p or {}).get("format") != "uri"]
    if any(w in n for n in names for w in AUDIO_SETTINGS):
        return "AUDIO"
    if any(w in n for n in names for w in VIDEO_SETTINGS):
        return "VIDEO"
    low = description.lower()
    if re.search(r"\bvideos?\b", low):
        return "VIDEO"
    if re.search(r"\b(speech|music|audio|voice|song|sound)s?\b", low):
        return "AUDIO"
    return "IMAGE"


def category(spec: schema_nodes.Spec) -> str:
    """Where the node goes in the library, in fal's words: text-to-image, image-to-video, speech-to-text..."""
    out = spec.out_fields[0].type
    needs = {f.media for f in spec.fields if f.kind != "value" and not f.optional}
    takes = {f.media for f in spec.fields if f.kind != "value"}
    if out == "IMAGE":
        return "image-to-image" if "IMAGE" in needs else "text-to-image"
    if out == "VIDEO":
        return "video-to-video" if "VIDEO" in needs else "image-to-video" if "IMAGE" in needs else "text-to-video"
    if out == "AUDIO":
        if "AUDIO" in needs:
            return "audio-to-audio"
        return "text-to-speech" if any("voice" in f.name for f in spec.fields) else "text-to-audio"
    if "AUDIO" in takes:
        return "speech-to-text"
    return "image-to-text" if "IMAGE" in needs else "llm"


class ReplicateProvider(SchemaProvider):
    type = "replicate"
    default_base_url = "https://api.replicate.com/v1"
    default_file_hosts = ("replicate.delivery", "replicate.com")  # where outputs point
    auth_file_hosts = ("replicate.delivery",)  # "pass the Authorization header to request them"
    default_endpoints = DEFAULT_ENDPOINTS
    endpoint_pattern = ENDPOINT
    example = "black-forest-labs/flux-schnell"

    # ---- schemas and nodes

    async def fetch(self, endpoint: str) -> dict | None:
        """The model (its description, whether it is official) with the version to run: the latest, or the pinned
        one."""
        name, _, version = endpoint.partition(":")
        try:
            model = await self.request("GET", f"/models/{name}")
            if version:
                model["latest_version"] = await self.request("GET", f"/models/{name}/versions/{version}")
        except ProviderError as e:
            log.warning("replicate: no schema of %s: %s", endpoint, e)
            return None
        if not ((model.get("latest_version") or {}).get("openapi_schema") or {}).get("components"):
            log.warning("replicate: %s has no version with a schema", endpoint)
            return None
        return model

    def node(self, endpoint: str, schema: dict, ms: ModelSettings) -> Node:
        name, _, pinned = endpoint.partition(":")
        version = schema["latest_version"]
        api = version["openapi_schema"]
        props = schema_nodes.properties(api, "Input")
        about = re.sub(r"\s+", " ", str(schema.get("description") or "")).strip()
        media = KINDS.get((ms.kind or "").lower()) or guess_media(props, about)
        spec = schema_nodes.read(api, "Input", "Output", root_media=media)
        # an official model runs by its name (always its current version); any other by the version of its schema
        ref = endpoint if pinned else name if schema.get("is_official") else f"{name}:{version['id']}"
        cat = ms.category or category(spec)
        model = Model(self.name, endpoint, "node", "replicate", None, ms.price or None)
        return Node(node_id(endpoint), ms.title or title(endpoint), f"Generate/Replicate/{cat}",
                    (about[:300] + "…" if len(about) > 300 else about) or "A model of Replicate.",
                    spec.inputs, spec.outputs, self.runner(ref, spec, model), output_node=True, cached=True,
                    search=["replicate", name, cat])

    # ---- running

    async def file_value(self, ctx: Ctx, ref: dict, media: str) -> str:
        if ref.get("asset"):
            await ctx.data_uri(ref)  # says what a character is for
        data, ct = await ctx.read(ref), ref.get("ct") or "application/octet-stream"
        if len(data) <= DATA_URI_MAX:
            return f"data:{ct};base64," + base64.b64encode(data).decode()
        answer = await ctx.call(self, "POST", "/files", data={"type": ct, "filename": ref.get("name") or "file"},
                                upload=[("content", (ref.get("name") or "file", data, ct))])
        url = (answer.get("urls") or {}).get("get")
        if not url:
            raise NodeError(i18n.tr("Replicate не принял файл", "Replicate did not take the file"))
        return url

    def runner(self, ref: str, spec: schema_nodes.Spec, model: Model):
        provider = self

        async def run(ctx: Ctx, **values) -> tuple[list, dict | None]:
            prediction_id = ctx.pending_task
            if prediction_id is None:
                body = {"version": ref, "input": await provider.body(ctx, spec, values)}
                answer = await ctx.call(provider, "POST", "/predictions", body)
                prediction_id = answer.get("id")
                if not prediction_id:
                    raise NodeError(i18n.tr(f"{provider.name} не вернул номер задачи",
                                            f"{provider.name} returned no prediction id"))
                if ctx.remember_task:
                    await ctx.remember_task(prediction_id)
            started = time.monotonic()
            try:
                while True:
                    pred = await ctx.call(provider, "GET", f"/predictions/{prediction_id}")
                    status = pred.get("status", "")
                    ru, en = STATUS.get(status, (status, status))
                    seconds = int(time.monotonic() - started)
                    await ctx.progress(i18n.tr(f"Replicate: {ru}, {seconds} с", f"Replicate: {en}, {seconds} s"))
                    if status == "succeeded":
                        break
                    if status in ("failed", "canceled", "aborted"):
                        err = str(pred.get("error") or status).strip()
                        raise NodeError(f"{provider.name}: {err[:500]}")
                    if time.monotonic() - started > TASK_TIMEOUT:
                        raise NodeError(i18n.tr(f"Replicate не ответил за {TASK_TIMEOUT // 60} минут",
                                                f"Replicate gave no result in {TASK_TIMEOUT // 60} minutes"))
                    if ctx.suspending():
                        raise Suspended  # the prediction id is remembered: the resumed run follows it
                    await asyncio.sleep(POLL_SECONDS)
            except asyncio.CancelledError:
                if not ctx.suspending():  # the user cancelled: the prediction stops (and stops costing)
                    asyncio.get_running_loop().create_task(_cancel(ctx, provider, prediction_id))
                raise
            output = pred.get("output")
            ctx.charge(model, {"requests": 1, "seconds": (pred.get("metrics") or {}).get("predict_time"),
                               "generated_images": _images(spec, output)})
            return await provider.outputs(ctx, spec, output)

        return run


def _images(spec: schema_nodes.Spec, output) -> int:
    if not any(of.type == "IMAGE" and of.name == schema_nodes.ROOT for of in spec.out_fields):
        return 0
    return len([u for u in output if u]) if isinstance(output, list) else int(bool(output))


async def _cancel(ctx: Ctx, p: ReplicateProvider, prediction_id: str) -> None:
    try:
        await ctx.call(p, "POST", f"/predictions/{prediction_id}/cancel")
    except NodeError:
        pass
