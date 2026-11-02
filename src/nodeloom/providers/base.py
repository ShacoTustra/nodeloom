"""What every provider has: an HTTP client with its key, the model list (discovered, then cached), error messages for
the user and downloads of result files."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import httpx

from nodeloom import i18n
from nodeloom.config import ModelSettings, ProviderSettings

log = logging.getLogger(__name__)

# The node families: which nodes list a model. A chat model of any provider goes to the text model node; pictures
# and video are as many families as there are APIs for them.
FAMILIES = ("chat", "openai-image", "video", "seedream", "seedance", "gemini-image", "gemini-video")
KINDS = ("chat", "image", "video")
SENSES = ("image", "audio", "video")  # what a chat model may take in besides text
REDIRECTS = 5  # a result's link may lead on to the file (Google's downloads do), each hop checked
MODELS_TTL = 600.0  # the provider's model list is asked again after this long
RETRY_TTL = 60.0  # and after a failure, sooner


class ProviderError(Exception):
    """A message for the user: shown in the editor's error dialog as is."""


@dataclass(frozen=True)
class Model:
    provider: str
    id: str
    kind: str  # chat | image | video
    family: str  # one of FAMILIES
    inputs: frozenset[str] | None = None  # a chat model's inputs besides text; None: not known (anything may go)
    price: dict | None = None
    # what a router's video model can do, as its /videos/models says: supported_durations, supported_resolutions,
    # supported_aspect_ratios, supported_frame_images (nodes.video_problems checks a node's settings against it)
    caps: dict | None = field(default=None, compare=False, hash=False)

    @property
    def value(self) -> str:
        return f"{self.provider}/{self.id}"

    def takes(self, sense: str) -> bool:
        return self.inputs is None or sense in self.inputs


class Provider:
    type: ClassVar[str]
    families: ClassVar[dict[str, str]]  # kind -> family
    default_base_url: ClassVar[str | None] = None
    # Hosts result files are downloaded from ("example.com": the host and its subdomains); none: any public host.
    default_file_hosts: ClassVar[tuple[str, ...]] = ()
    # Hosts besides the API's own that want the key with a download (the API's own always gets it).
    auth_file_hosts: ClassVar[tuple[str, ...]] = ()
    chat_path: ClassVar[str] = "/chat/completions"  # where its OpenAI-style chat lives
    max_tokens_key: ClassVar[str] = "max_tokens"  # the name /chat/completions takes the answer's length limit by

    def __init__(self, name: str, settings: ProviderSettings):
        self.name = name
        self.settings = settings
        self.base_url = (settings.base_url or self.default_base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError(f"provider {name}: base_url is required")
        self._client: httpx.AsyncClient | None = None
        self._files: httpx.AsyncClient | None = None
        self._found: list[Model] = []  # what GET /models said last time
        self._models: list[Model] | None = None
        self._models_at = 0.0
        self._models_ttl = MODELS_TTL
        self._lock = asyncio.Lock()

    # ---- http

    def headers(self) -> dict[str, str]:
        key = self.settings.key()
        return {**({"Authorization": f"Bearer {key}"} if key else {}), **self.settings.headers}

    async def chat(self, body: dict) -> dict:
        """An OpenAI-style chat request (the text model node's): the API's /chat/completions, or what the provider
        turns it into."""
        return await self.request("POST", self.chat_path, json=body)

    def headers_for(self, path: str) -> dict[str, str]:
        """The headers of a call of `path` (an API whose parts take the key differently says so here)."""
        return self.headers()

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # trust_env: HTTPS_PROXY and the system's certificates apply, as for any client on this machine
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.settings.timeout_s, connect=15))
        return self._client

    def files_client(self) -> httpx.AsyncClient:
        if self._files is None:
            self._files = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15), follow_redirects=False)
        return self._files

    async def close(self) -> None:
        for c in (self._client, self._files):
            if c is not None:
                await c.aclose()
        self._client = self._files = None

    async def request(self, method: str, path: str, *, json: dict | None = None, files=None,
                      data: dict | None = None) -> dict:
        """A call of the provider's API; ProviderError with the provider's own message when it fails."""
        try:
            r = await self.client().request(method, self.base_url + path, json=json, files=files, data=data,
                                            headers=self.headers_for(path))
        except httpx.HTTPError as e:
            raise ProviderError(i18n.tr(f"{self.name} не ответил: {type(e).__name__}",
                                        f"{self.name} did not answer: {type(e).__name__}")) from None
        if r.status_code >= 400:
            raise ProviderError(self.error_text(r.status_code, r.content))
        try:
            return r.json() if r.content else {}
        except ValueError:
            raise ProviderError(i18n.tr(f"{self.name}: ответ не JSON", f"{self.name}: the answer is not JSON")) from None

    def error_text(self, status: int, body: bytes) -> str:
        msg = None
        try:
            data = json.loads(body)
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code")
            elif isinstance(err, str):
                msg = err
            msg = msg or (data.get("message") or data.get("detail") if isinstance(data, dict) else None)
            if isinstance(msg, list):  # FastAPI's (fal's) validation errors: [{"loc": [..., field], "msg"}]
                msg = "; ".join(f"{(e.get('loc') or ['?'])[-1]}: {e.get('msg')}" if isinstance(e, dict) else str(e)
                                for e in msg)
        except (ValueError, AttributeError):
            text = body.decode(errors="replace").strip()
            msg = text[:300] if text and not text.startswith("<") else None
        if status in (401, 403):
            head = i18n.tr(f"{self.name} не принял ключ", f"{self.name} refused the key")
        elif status == 429:
            head = i18n.tr(f"{self.name}: слишком много запросов или кончились деньги",
                           f"{self.name}: too many requests or out of credit")
        else:
            head = f"{self.name}"
        return f"{head}: {msg}" if msg else i18n.tr(f"{head}: ошибка {status}", f"{head}: error {status}")

    # ---- files

    async def _allowed(self, url: str) -> bool:
        u = httpx.URL(url)
        host = u.host or ""
        if u.scheme != "https" or not host:
            return False
        if host == httpx.URL(self.base_url).host:
            return True
        hosts = self.settings.file_hosts if self.settings.file_hosts is not None else self.default_file_hosts
        if hosts:
            return _within(host, hosts)
        try:  # no list: any public host (never this machine or its network)
            infos = await asyncio.get_running_loop().getaddrinfo(host, u.port or 443)
        except OSError:
            return False
        return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)

    def _file_headers(self, url: str) -> dict[str, str]:
        host = httpx.URL(url).host or ""
        if host == httpx.URL(self.base_url).host or _within(host, self.auth_file_hosts):
            return self.headers()
        return {}

    @asynccontextmanager
    async def download(self, url: str) -> AsyncIterator[tuple[Path, str]]:
        """A generated file from the provider's storage (its links expire) in a temporary file, with its type."""
        tmp = Path(tempfile.mkstemp(prefix="loom-")[1])
        try:
            for _ in range(REDIRECTS + 1):
                if not await self._allowed(url):
                    raise ProviderError(i18n.tr("Недопустимый адрес файла", "Invalid file address"))
                async with self.files_client().stream("GET", url, headers=self._file_headers(url)) as r:
                    if r.is_redirect and r.headers.get("location"):
                        url = str(r.url.join(r.headers["location"]))
                        continue
                    if r.status_code != 200:
                        raise ProviderError(i18n.tr(f"Файл не скачан: HTTP {r.status_code}",
                                                    f"The file was not downloaded: HTTP {r.status_code}"))
                    ct = (r.headers.get("content-type") or "application/octet-stream").split(";")[0].strip()
                    with tmp.open("wb") as f:
                        async for chunk in r.aiter_bytes():
                            f.write(chunk)
                    break
            else:
                raise ProviderError(i18n.tr("Файл не скачан: слишком много переадресаций",
                                            "The file was not downloaded: too many redirects"))
            yield tmp, ct
        except httpx.HTTPError as e:
            raise ProviderError(i18n.tr(f"Файл не скачан: {type(e).__name__}",
                                        f"The file was not downloaded: {type(e).__name__}")) from None
        finally:
            tmp.unlink(missing_ok=True)

    # ---- models

    def family_of(self, model_id: str, kind: str) -> str | None:
        """The node family a model of this kind goes to (a provider with two APIs for one kind tells by the id)."""
        return self.families.get(kind)

    async def models(self) -> list[Model]:
        """The provider's models: the config's list, or what GET /models says (asked again every few minutes)."""
        if self.settings.models is not None:
            return self._from_config()
        if self._models is not None and time.monotonic() - self._models_at < self._models_ttl:
            return self._models
        async with self._lock:
            if self._models is not None and time.monotonic() - self._models_at < self._models_ttl:
                return self._models
            try:
                raw = await self.discover()
                self._found = [m for m in (self.from_listing(x) for x in raw if isinstance(x, dict)) if m is not None]
                self._models_ttl = MODELS_TTL
            except Exception as e:  # noqa: BLE001  the provider is away: the last list (or the extras), asked again soon
                log.warning("provider %s: GET /models failed: %s", self.name, e)
                self._models_ttl = RETRY_TTL
            self._models = self._merge(self._found)
            self._models_at = time.monotonic()
            return self._models

    async def discover(self) -> list[dict]:
        r = await self.client().get(self.base_url + "/models", headers=self.headers())
        r.raise_for_status()
        data = r.json()
        return data.get("data") or [] if isinstance(data, dict) else data

    def from_listing(self, raw: dict) -> Model | None:
        """A model of the provider's GET /models, or None when the nodes have no use for it."""
        raise NotImplementedError

    def guess_kind(self, model_id: str) -> str | None:
        return "chat"

    def model(self, model_id: str, kind: str | None, inputs=None, price: dict | None = None,
              caps: dict | None = None) -> Model | None:
        kind = kind or self.guess_kind(model_id)
        family = self.family_of(model_id, kind or "")
        if family is None:
            return None
        if kind != "chat":
            inputs = None
        return Model(self.name, model_id, kind, family,
                     frozenset(x for x in inputs if x in SENSES) if inputs is not None else None, price or None,
                     caps or None)

    def _configured(self, model_id: str, ms: ModelSettings, base: Model | None = None) -> Model | None:
        kind = ms.kind or (base.kind if base else None)
        inputs = ms.inputs if ms.inputs is not None else (sorted(base.inputs) if base and base.inputs is not None else None)
        return self.model(model_id, kind, inputs, ms.price or (base.price if base else None),
                          base.caps if base and base.kind == kind else None)

    def _from_config(self) -> list[Model]:
        configured = self.settings.models
        items = configured.items() if isinstance(configured, dict) else ((m, ModelSettings()) for m in configured or [])
        out = [m for m in (self._configured(mid, ms) for mid, ms in items) if m is not None]
        return self._merge(out)

    def _merge(self, found: list[Model]) -> list[Model]:
        """The discovered or configured models with `extra_models` over them (settings for some, more models)."""
        by_id = {m.id: m for m in found}
        for mid, ms in self.settings.extra_models.items():
            if (m := self._configured(mid, ms, by_id.get(mid))) is not None:
                by_id[mid] = m
            else:
                by_id.pop(mid, None)
        return list(by_id.values())


def _within(host: str, hosts) -> bool:
    """The host is one of `hosts` or under one of them ("fal.media" lets in v3.fal.media, not evilfal.media)."""
    for h in hosts:
        h = h.lstrip(".").lower()
        if h and (host == h or host.endswith("." + h)):
            return True
    return False
