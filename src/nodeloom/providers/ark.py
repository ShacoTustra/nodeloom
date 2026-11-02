"""BytePlus ModelArk and Volcengine Ark (the same API), and gateways in front of them.

Text goes through /chat/completions (OpenAI's format, with `video_url` parts for models that watch video), pictures
through Seedream's /images/generations, video through Seedance tasks (/contents/generations/tasks). GET /models says
each model's domain and what it takes in; a gateway may say `kind` and `input_modalities` instead.

Ready characters for Seedance (asset://<id>: faces Seedance accepts, while a photo of a real person is refused) come
from a gateway that lists them at GET /assets; ModelArk's own library needs its control plane, not supported yet.
"""

from __future__ import annotations

import httpx

from nodeloom.providers.base import Model, Provider, ProviderError

DOMAINS = {"llm": "chat", "vlm": "chat", "imagegeneration": "image", "videogeneration": "video"}


class ArkProvider(Provider):
    type = "ark"
    families = {"chat": "chat", "image": "seedream", "video": "seedance"}
    default_base_url = "https://ark.ap-southeast.bytepluses.com/api/v3"
    default_file_hosts = ("volces.com", "bytepluses.com", "byteplus.com")  # where result links point

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_assets: bool | None = None

    def guess_kind(self, model_id: str) -> str | None:
        low = model_id.lower()
        if "seedream" in low or "seededit" in low:
            return "image"
        if "seedance" in low:
            return "video"
        if "embedding" in low:
            return None
        return "chat"

    def from_listing(self, raw: dict) -> Model | None:
        mid = str(raw.get("id") or "")
        if not mid or raw.get("status") == "Shutdown" or raw.get("lifecycle") == "Shutdown":
            return None
        kind = raw.get("kind")  # a gateway's
        if kind is None and raw.get("domain"):
            kind = DOMAINS.get(str(raw["domain"]).lower(), "other")
        if kind not in (None, "chat", "image", "video"):
            return None
        modalities = raw.get("modalities") if isinstance(raw.get("modalities"), dict) else {}
        inputs = raw.get("input_modalities") or modalities.get("input_modalities")
        return self.model(mid, kind, inputs if isinstance(inputs, list) else None)

    # ---- ready characters (a gateway's /assets)

    async def has_assets(self) -> bool:
        if self._has_assets is None:
            try:
                await self.assets({"limit": 1})
            except ProviderError:
                self._has_assets = False
        return bool(self._has_assets)

    async def assets(self, params: dict) -> dict:
        """{"total", "items": [{"id", "title", "subtitle", "gender", "age", ...}]} of the gateway's library."""
        try:
            r = await self.client().get(self.base_url + "/assets", params=params, headers=self.headers())
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}") from None
        if r.status_code >= 400:
            raise ProviderError(self.error_text(r.status_code, r.content))
        data = r.json()
        self._has_assets = True
        return {"total": int(data.get("total") or 0), "items": [x for x in data.get("items") or [] if x.get("id")]}

    async def asset_preview(self, asset_id: str) -> httpx.Response:
        return await self.client().get(f"{self.base_url}/assets/{asset_id}/preview", headers=self.headers())
