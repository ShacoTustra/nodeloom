"""fal.ai: hundreds of image, video and sound models behind one queue API, each with its own inputs.

Every endpoint in the config (or, with FAL_KEY alone, a list of popular ones) becomes a node of its own: its widgets and
links are read off the endpoint's OpenAPI schema (schema_nodes.py), fetched from fal and kept in the data directory for
a day (providers/schemas.py). A run goes through fal's queue: POST queue.fal.run/<endpoint> gives a request id
(remembered, so a run resumed after a restart follows it), its status is polled, then the result is fetched and its
files are copied into the user's results. Pictures, videos and sounds go to fal as data URIs.

fal says no price in its answers: give one per endpoint in the config ({request = 0.04}) to see the cost of a run.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx

from nodeloom import i18n, schema_nodes
from nodeloom.config import ModelSettings
from nodeloom.nodes import Ctx, Node, NodeError, Suspended
from nodeloom.providers.base import Model
from nodeloom.providers.schemas import SchemaProvider, camel, title_of

log = logging.getLogger(__name__)

SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json"
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
STATUS = {"IN_QUEUE": ("в очереди", "in queue"), "IN_PROGRESS": ("генерация", "generating"),
          "COMPLETED": ("готово", "done")}


def _own(endpoint: str) -> str:
    """The endpoint without fal's own prefix: fal-ai/flux/dev -> flux/dev."""
    return endpoint.removeprefix("fal-ai/")


def node_id(endpoint: str) -> str:
    """fal-ai/kling-video/v2.1/standard/image-to-video -> LoomFalKlingVideoV21StandardImageToVideo."""
    return "LoomFal" + camel(_own(endpoint))


def title(endpoint: str) -> str:
    """fal-ai/kling-video/v2.1/standard/image-to-video -> Kling Video v2.1 Standard Image to Video."""
    return title_of(_own(endpoint))


def _about(info: dict) -> str:
    text = str((info.get("x-fal-metadata") or {}).get("about") or "").strip()
    first = re.split(r"\n\s*\n", text)[0].strip()
    return (first[:300] + "…" if len(first) > 300 else first) or "A model of fal.ai."


class FalProvider(SchemaProvider):
    type = "fal"
    default_base_url = "https://queue.fal.run"
    default_file_hosts = ("fal.media", "fal.run", "fal.ai", "storage.googleapis.com")  # where results point
    default_endpoints = DEFAULT_ENDPOINTS
    endpoint_pattern = ENDPOINT
    example = "fal-ai/flux/dev"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema_client: httpx.AsyncClient | None = None

    def headers(self) -> dict[str, str]:
        key = self.settings.key()
        return {**({"Authorization": f"Key {key}"} if key else {}), **self.settings.headers}

    async def close(self) -> None:
        await super().close()
        if self._schema_client is not None:
            await self._schema_client.aclose()
            self._schema_client = None

    # ---- schemas and nodes

    async def fetch(self, endpoint: str) -> dict | None:
        if self._schema_client is None:  # fal's schemas are public: no key goes with them
            self._schema_client = httpx.AsyncClient(timeout=httpx.Timeout(15, connect=10))
        try:
            r = await self._schema_client.get(SCHEMA_URL, params={"endpoint_id": endpoint})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("fal: no schema of %s: %s %s", endpoint, type(e).__name__, e)
            return None

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
                body = await provider.body(ctx, spec, values)
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
            return await provider.outputs(ctx, spec, result)

        return run


async def _cancel(ctx: Ctx, p: FalProvider, app: str, request_id: str) -> None:
    try:
        await ctx.call(p, "PUT", f"/{app}/requests/{request_id}/cancel")
    except NodeError:
        pass
