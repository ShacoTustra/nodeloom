"""Providers whose every model is a node of its own, made of the model's schema (fal, Replicate).

What they share: the models to make nodes of (the config's `endpoints`, or the type's popular ones), their schemas
kept in the data directory (fetched again in the background once a day old), the nodes made of them every few
minutes, and what such a node does with its inputs and its result: the request's body (files as the provider takes
them) and the node's outputs (the result's files copied into the user's results). The type says how a schema is
fetched, how a node is made of it and how a run goes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import ClassVar

from nodeloom import files, i18n, schema_nodes
from nodeloom.config import ModelSettings, get_settings
from nodeloom.nodes import Ctx, Node, NodeError, _flat, show_images, show_video
from nodeloom.providers.base import Model, Provider

log = logging.getLogger(__name__)

SMALL = {"to", "of", "and", "a", "in"}
SCHEMA_TTL = 24 * 3600  # a kept schema older than this is used while a fresh one is fetched
NODES_TTL = 600.0
RETRY_TTL = 60.0


def camel(text: str) -> str:
    """kling-video/v2.1 -> KlingVideoV21: a part of a node id."""
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", text) if w)


def title_of(text: str) -> str:
    """kling-video v2.1 standard image-to-video -> Kling Video v2.1 Standard Image to Video."""
    words = re.sub(r"[-/_]", " ", text).split()
    return " ".join(w if re.match(r"^v?\d", w) or w in SMALL and i else w[:1].upper() + w[1:]
                    for i, w in enumerate(words))


class SchemaProvider(Provider):
    families: ClassVar[dict[str, str]] = {}  # its models are nodes of their own, not lists of the generic nodes
    default_endpoints: ClassVar[list[str]] = []
    endpoint_pattern: ClassVar[re.Pattern]
    example: ClassVar[str]  # an endpoint id, for the message about one that is not

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nodes: list[Node] | None = None
        self._nodes_at = 0.0
        self._nodes_ttl = NODES_TTL
        self._refreshing: set[str] = set()

    async def models(self) -> list[Model]:
        return []

    def endpoints(self) -> dict[str, ModelSettings]:
        e = self.settings.endpoints
        if e is None:
            return {x: ModelSettings() for x in self.default_endpoints}
        return dict(e) if isinstance(e, dict) else {x: ModelSettings() for x in e}

    # ---- schemas and nodes

    def cache_path(self, endpoint: str) -> Path:
        return get_settings().data_dir / "cache" / self.name / (re.sub(r"[/:]", "__", endpoint) + ".json")

    async def schema(self, endpoint: str) -> dict | None:
        """The endpoint's schema: the copy in the data directory (fetched again in the background once it is a day
        old), or fresh from the provider the first time."""
        cache = self.cache_path(endpoint)
        if cache.is_file():
            if time.time() - cache.stat().st_mtime > SCHEMA_TTL and endpoint not in self._refreshing:
                self._refreshing.add(endpoint)
                task = asyncio.get_running_loop().create_task(self._refresh(endpoint, cache))
                task.add_done_callback(lambda _: self._refreshing.discard(endpoint))
            return json.loads(cache.read_text())
        return await self._refresh(endpoint, cache)

    async def _refresh(self, endpoint: str, cache: Path) -> dict | None:
        data = await self.fetch(endpoint)
        if data is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        return data

    async def fetch(self, endpoint: str) -> dict | None:
        """The endpoint's schema from the provider; None (and a warning in the log) when there is none."""
        raise NotImplementedError

    def node(self, endpoint: str, schema: dict, ms: ModelSettings) -> Node:
        raise NotImplementedError

    async def nodes(self) -> list[Node]:
        """A node for every endpoint whose schema is known and usable (made again every few minutes)."""
        if self._nodes is not None and time.monotonic() - self._nodes_at < self._nodes_ttl:
            return self._nodes
        wanted = {}
        for endpoint, ms in self.endpoints().items():
            if self.endpoint_pattern.match(endpoint):
                wanted[endpoint] = ms
            else:
                log.warning("%s: %r is not an endpoint id like %s", self.name, endpoint, self.example)
        schemas = await asyncio.gather(*(self.schema(e) for e in wanted))
        made, missing = [], False
        for (endpoint, ms), schema in zip(wanted.items(), schemas, strict=True):
            if schema is None:
                missing = True
                continue
            try:
                made.append(self.node(endpoint, schema, ms))
            except (schema_nodes.Unsupported, KeyError, TypeError, AttributeError) as e:
                log.warning("%s: %s is left out: %s", self.name, endpoint, e)
        self._nodes, self._nodes_at = made, time.monotonic()
        self._nodes_ttl = RETRY_TTL if missing else NODES_TTL
        return made

    # ---- running

    async def file_value(self, ctx: Ctx, ref: dict, media: str) -> str:
        """A file of the user as the API takes it in a request: a data URI."""
        return await data_uri(ctx, ref, media)

    async def body(self, ctx: Ctx, spec: schema_nodes.Spec, values: dict) -> dict:
        """The request's fields of the node's inputs; NodeError when a required link is not connected."""
        labels = {i.name: i.label for i in spec.inputs}
        body: dict = {}
        for f in spec.fields:
            v = values.get(f.name)
            if f.kind == "value":
                if v is None or (v == "" and f.optional):
                    continue
                body[f.name] = _cast(v, f.cast)
                continue
            refs = _flat(v) if f.kind == "media_list" or f.media == "IMAGE" else ([v] if v else [])
            if not refs:
                if not f.optional:
                    raise NodeError(i18n.tr(f"Подключите «{labels.get(f.name, f.name)}»",
                                            f"Connect “{labels.get(f.name, f.name)}”"))
                continue
            uris = [await self.file_value(ctx, r, f.media) for r in refs]
            body[f.name] = uris if f.kind == "media_list" else uris[0]
        return body

    async def outputs(self, ctx: Ctx, spec: schema_nodes.Spec, result) -> tuple[list, dict | None]:
        """The node's outputs of the result (its files copied into the user's results) and what the node shows."""
        values: list = []
        shown: list[dict] = []
        text = None
        for of in spec.out_fields:
            v = result if of.name == schema_nodes.ROOT else (result.get(of.name) if isinstance(result, dict) else None)
            if of.type == "STRING":
                values.append("".join(map(str, v)) if isinstance(v, list) else "" if v is None else str(v))
                text = text or values[-1]
                continue
            items = v if isinstance(v, list) else [v]
            urls = [it.get("url") if isinstance(it, dict) else it for it in items]
            refs = [await ctx.download(self, u) for u in urls if isinstance(u, str) and u]
            values.append(refs if of.type == "IMAGE" else refs[0] if refs else None)
            shown += refs
        if not (shown or text):
            raise NodeError(i18n.tr(f"{self.name} не вернул результата", f"{self.name} returned no result"))
        by_media = {m: [r for r in shown if r.get("media") == m] for m in ("video", "image", "audio")}
        if by_media["video"]:
            return values, show_video(by_media["video"][0])
        if by_media["image"]:
            return values, show_images(by_media["image"])
        if by_media["audio"]:
            return values, {"audio": [files.view_item(by_media["audio"][0])]}
        return values, {"text": [text or ""]}


def _cast(v, cast: str):
    try:
        return int(v) if cast == "int" else float(v) if cast == "float" else v
    except (TypeError, ValueError):
        return v


async def data_uri(ctx: Ctx, ref: dict, media: str) -> str:
    if media == "IMAGE":
        return await ctx.data_uri(ref)
    return f"data:{ref.get('ct') or 'application/octet-stream'};base64," + base64.b64encode(await ctx.read(ref)).decode()
