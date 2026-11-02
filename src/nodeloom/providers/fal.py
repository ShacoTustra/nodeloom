"""fal.ai: hundreds of image, video and sound models behind one queue API, each with its own inputs.

Every endpoint in the config (or, with FAL_KEY alone, a list of popular ones) becomes a node of its own: its widgets
and links are read off the endpoint's OpenAPI schema (schema_nodes.py), fetched from fal and kept in the data
directory for a day. A run goes through fal's queue: POST queue.fal.run/<endpoint> gives a request id (remembered,
so a run resumed after a restart follows it), its status is polled, then the result is fetched and its files are
copied into the user's results. Pictures, videos and sounds go to fal as data URIs.

fal says no price in its answers: give one per endpoint in the config ({request = 0.04}) to see the cost of a run.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time

import httpx

from nodeloom import files, i18n, schema_nodes
from nodeloom.config import ModelSettings, get_settings
from nodeloom.nodes import Ctx, Node, NodeError, Suspended, _flat, show_images, show_video
from nodeloom.providers.base import Model, Provider

log = logging.getLogger(__name__)

SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json"
SCHEMA_TTL = 24 * 3600  # a kept schema older than this is used while a fresh one is fetched
NODES_TTL = 600.0
RETRY_TTL = 60.0
POLL_SECONDS = 3.0
TASK_TIMEOUT = 45 * 60
# With FAL_KEY alone: well-known models of each kind (the config's `endpoints` replaces the list).
DEFAULT_ENDPOINTS = [
    "fal-ai/flux/dev", "fal-ai/flux-pro/kontext", "fal-ai/nano-banana", "fal-ai/nano-banana/edit", "fal-ai/qwen-image",
    "fal-ai/kling-video/v2.1/standard/image-to-video", "fal-ai/veo3/fast", "fal-ai/veo3/image-to-video",
    "fal-ai/minimax/hailuo-02/standard/image-to-video", "fal-ai/wan/v2.2-a14b/image-to-video",
    "fal-ai/elevenlabs/tts/turbo-v2.5",
]
ENDPOINT = re.compile(r"^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)+$")
SMALL = {"to", "of", "and", "a", "in"}
STATUS = {"IN_QUEUE": ("в очереди", "in queue"), "IN_PROGRESS": ("генерация", "generating"),
          "COMPLETED": ("готово", "done")}


def node_id(endpoint: str) -> str:
    """fal-ai/kling-video/v2.1/standard/image-to-video -> LoomFalKlingVideoV21StandardImageToVideo."""
    parts = endpoint.split("/")
    parts = parts[1:] if parts[0] == "fal-ai" else parts
    return "LoomFal" + "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", "-".join(parts)) if w)


def title(endpoint: str) -> str:
    """fal-ai/kling-video/v2.1/standard/image-to-video -> Kling Video v2.1 Standard Image to Video."""
    parts = endpoint.split("/")
    words = " ".join(parts[1:] if parts[0] == "fal-ai" else parts).replace("-", " ").split()
    return " ".join(w if re.match(r"^v?\d", w) or w in SMALL and i else w[:1].upper() + w[1:]
                    for i, w in enumerate(words))


def _about(info: dict) -> str:
    text = str((info.get("x-fal-metadata") or {}).get("about") or "").strip()
    first = re.split(r"\n\s*\n", text)[0].strip()
    return (first[:300] + "…" if len(first) > 300 else first) or "A model of fal.ai."


class FalProvider(Provider):
    type = "fal"
    families: dict[str, str] = {}  # its models are nodes of their own, not lists of the generic nodes
    default_base_url = "https://queue.fal.run"
    default_file_hosts = (".fal.media", ".fal.run", ".fal.ai", "storage.googleapis.com")  # where results point

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nodes: list[Node] | None = None
        self._nodes_at = 0.0
        self._nodes_ttl = NODES_TTL
        self._schema_client: httpx.AsyncClient | None = None
        self._refreshing: set[str] = set()

    def headers(self) -> dict[str, str]:
        key = self.settings.key()
        return {**({"Authorization": f"Key {key}"} if key else {}), **self.settings.headers}

    async def models(self) -> list[Model]:
        return []

    def endpoints(self) -> dict[str, ModelSettings]:
        e = self.settings.endpoints
        if e is None:
            return {x: ModelSettings() for x in DEFAULT_ENDPOINTS}
        return dict(e) if isinstance(e, dict) else {x: ModelSettings() for x in e}

    async def close(self) -> None:
        await super().close()
        if self._schema_client is not None:
            await self._schema_client.aclose()
            self._schema_client = None

    # ---- schemas and nodes

    async def schema(self, endpoint: str) -> dict | None:
        """The endpoint's OpenAPI schema: the copy in the data directory (fetched again in the background once it is
        a day old), or fresh from fal the first time."""
        cache = get_settings().data_dir / "cache" / "fal" / (endpoint.replace("/", "__") + ".json")
        if cache.is_file():
            if time.time() - cache.stat().st_mtime > SCHEMA_TTL and endpoint not in self._refreshing:
                self._refreshing.add(endpoint)
                task = asyncio.get_running_loop().create_task(self._fetch(endpoint, cache))
                task.add_done_callback(lambda _: self._refreshing.discard(endpoint))
            return json.loads(cache.read_text())
        return await self._fetch(endpoint, cache)

    async def _fetch(self, endpoint: str, cache) -> dict | None:
        if self._schema_client is None:
            self._schema_client = httpx.AsyncClient(timeout=httpx.Timeout(15, connect=10))
        try:
            r = await self._schema_client.get(SCHEMA_URL, params={"endpoint_id": endpoint})
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("fal: no schema of %s: %s %s", endpoint, type(e).__name__, e)
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
        return data

    async def nodes(self) -> list[Node]:
        """A node for every endpoint whose schema is known and usable (made again every few minutes)."""
        if self._nodes is not None and time.monotonic() - self._nodes_at < self._nodes_ttl:
            return self._nodes
        wanted = {}
        for endpoint, ms in self.endpoints().items():
            if ENDPOINT.match(endpoint):
                wanted[endpoint] = ms
            else:
                log.warning("fal: %r is not an endpoint id like fal-ai/flux/dev", endpoint)
        schemas = await asyncio.gather(*(self.schema(e) for e in wanted))
        made, missing = [], False
        for (endpoint, ms), schema in zip(wanted.items(), schemas, strict=True):
            if schema is None:
                missing = True
                continue
            try:
                made.append(self.node(endpoint, schema, ms))
            except (schema_nodes.Unsupported, KeyError, TypeError) as e:
                log.warning("fal: %s is left out: %s", endpoint, e)
        self._nodes, self._nodes_at = made, time.monotonic()
        self._nodes_ttl = RETRY_TTL if missing else NODES_TTL
        return made

    def node(self, endpoint: str, schema: dict, ms: ModelSettings) -> Node:
        paths = schema["paths"]

        def ref(op: dict) -> str:
            return op["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
        input_name = ref(paths[f"/{endpoint}"]["post"]["requestBody"])
        output_name = ref(paths[f"/{endpoint}/requests/{{request_id}}"]["get"]["responses"]["200"])
        spec = schema_nodes.read(schema, input_name, output_name)
        info = schema.get("info") or {}
        category = ms.category or (info.get("x-fal-metadata") or {}).get("category") or "other"
        model = Model(self.name, endpoint, "node", "fal", None, ms.price or None)
        return Node(node_id(endpoint), ms.title or title(endpoint), f"Generate/fal/{category}", _about(info),
                    spec.inputs, spec.outputs, self.runner(endpoint, spec, model), output_node=True, cached=True,
                    search=["fal", endpoint, category])

    # ---- running

    def _app(self, endpoint: str) -> str:
        """The queue's requests live under the app (owner/alias), not under the endpoint's own path."""
        return "/".join(endpoint.split("/")[:2])

    def runner(self, endpoint: str, spec: schema_nodes.Spec, model: Model):
        provider = self

        async def run(ctx: Ctx, **values) -> tuple[list, dict | None]:
            app = provider._app(endpoint)
            request_id = ctx.pending_task
            if request_id is None:
                body = await _body(ctx, spec, values)
                answer = await ctx.call(provider, "POST", f"/{endpoint}", body)
                request_id = answer.get("request_id")
                if not request_id:
                    raise NodeError(i18n.tr("fal не вернул номер задачи", "fal returned no request id"))
                if ctx.remember_task:
                    await ctx.remember_task(request_id)
            started = time.monotonic()
            try:
                while True:
                    st = await ctx.call(provider, "GET", f"/{app}/requests/{request_id}/status")
                    status = st.get("status", "")
                    ru, en = STATUS.get(status, (status.lower(), status.lower()))
                    place = f" ({st['queue_position']})" if st.get("queue_position") else ""
                    seconds = int(time.monotonic() - started)
                    await ctx.progress(i18n.tr(f"fal: {ru}{place}, {seconds} с", f"fal: {en}{place}, {seconds} s"))
                    if status == "COMPLETED":
                        break
                    if time.monotonic() - started > TASK_TIMEOUT:
                        raise NodeError(i18n.tr(f"fal не ответил за {TASK_TIMEOUT // 60} минут",
                                                f"fal gave no result in {TASK_TIMEOUT // 60} minutes"))
                    if ctx.suspending():
                        raise Suspended  # the request id is remembered: the resumed run follows it
                    await asyncio.sleep(POLL_SECONDS)
            except asyncio.CancelledError:
                if not ctx.suspending():  # the user cancelled: a queued request is dropped at fal
                    asyncio.get_running_loop().create_task(_cancel(ctx, provider, app, request_id))
                raise
            result = await ctx.call(provider, "GET", f"/{app}/requests/{request_id}")
            ctx.charge(model, {"requests": 1})
            return await _outputs(ctx, provider, spec, result)

        return run


async def _cancel(ctx: Ctx, p: FalProvider, app: str, request_id: str) -> None:
    try:
        await ctx.call(p, "PUT", f"/{app}/requests/{request_id}/cancel")
    except NodeError:
        pass


async def _uri(ctx: Ctx, ref: dict, media: str) -> str:
    if media == "IMAGE":
        return await ctx.data_uri(ref)
    return f"data:{ref.get('ct') or 'application/octet-stream'};base64," + base64.b64encode(await ctx.read(ref)).decode()


async def _body(ctx: Ctx, spec: schema_nodes.Spec, values: dict) -> dict:
    labels = {i.name: i.label for i in spec.inputs}
    body: dict = {}
    for f in spec.fields:
        v = values.get(f.name)
        if f.kind == "value":
            if v is None or (v == "" and f.optional):
                continue
            body[f.name] = v
            continue
        refs = _flat(v) if f.kind == "media_list" or f.media == "IMAGE" else ([v] if v else [])
        if not refs:
            if not f.optional:
                raise NodeError(i18n.tr(f"Подключите «{labels.get(f.name, f.name)}»",
                                        f"Connect “{labels.get(f.name, f.name)}”"))
            continue
        uris = [await _uri(ctx, r, f.media) for r in refs]
        body[f.name] = uris if f.kind == "media_list" else uris[0]
    return body


async def _outputs(ctx: Ctx, p: FalProvider, spec: schema_nodes.Spec, result: dict) -> tuple[list, dict | None]:
    values: list = []
    images, video, audio, text = [], None, None, None
    for of in spec.out_fields:
        v = result.get(of.name)
        if of.type == "STRING":
            values.append("" if v is None else str(v))
            text = text or values[-1]
            continue
        items = v if isinstance(v, list) else [v]
        refs = [await ctx.download(p, it["url"]) for it in items if isinstance(it, dict) and it.get("url")]
        if of.type == "IMAGE":
            values.append(refs)
            images += refs
        else:
            values.append(refs[0] if refs else None)
            if refs and of.type == "VIDEO":
                video = video or refs[0]
            elif refs:
                audio = audio or refs[0]
    if not (images or video or audio or text):
        raise NodeError(i18n.tr("fal не вернул результата", "fal returned no result"))
    if video:
        return values, show_video(video)
    if images:
        return values, show_images(images)
    if audio:
        return values, {"audio": [files.view_item(audio)]}
    return values, {"text": [text]}
