"""The application: the editor's routes (api.py), the run dispatcher and the hourly cleanup, and the Host check."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.responses import PlainTextResponse

from nodeloom import api, bus, db, providers, runs
from nodeloom.config import APP_NAME, get_settings

log = logging.getLogger(__name__)

CLEANUP_S = 3600


class HostCheck:
    """Only the Host names of ALLOWED_HOSTS reach the app. A page of another site can point its own name at 127.0.0.1
    (DNS rebinding) and then talk to the editor as if it were the same origin; its Host header gives it away."""

    def __init__(self, app, hosts: list[str]):
        self.app = app
        self.any = "*" in hosts
        self.hosts = {h.lower() for h in hosts}

    def _allowed(self, host: str) -> bool:
        host = host.lower()
        if host.startswith("["):  # [::1]:8188
            name = host[1:].partition("]")[0]
        else:
            name = host.rpartition(":")[0] if host.count(":") == 1 else host
        return name in self.hosts

    async def __call__(self, scope, receive, send):
        if self.any or scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        host = next((v.decode("latin-1") for k, v in scope.get("headers") or [] if k == b"host"), "")
        if self._allowed(host):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4403})
            return
        await PlainTextResponse(f"Host {host!r} is not allowed: add it to ALLOWED_HOSTS", status_code=403)(
            scope, receive, send)


async def _cleanup_loop() -> None:
    while True:
        try:
            await runs.cleanup()
        except Exception:  # noqa: BLE001  the database may be away for a moment
            log.exception("cleanup")
        await asyncio.sleep(CLEANUP_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    names = list(providers.registry())
    if names:
        log.info("providers: %s", ", ".join(names))
    else:
        log.warning("no model providers configured: set OPENAI_API_KEY, OPENROUTER_API_KEY or ARK_API_KEY, or "
                    "[providers.*] in config.toml")
    runs.dispatcher.start()
    cleanup = asyncio.create_task(_cleanup_loop())
    # the providers' own nodes (fal's schemas) made before the editor first asks for them
    prefetch = asyncio.create_task(providers.ensure_nodes())
    try:
        yield
    finally:
        for task in (cleanup, prefetch):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await runs.dispatcher.stop()
        await bus.stop()
        await providers.close()
        await db.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(api.router)
    app.add_middleware(HostCheck, hosts=get_settings().allowed_hosts)
    return app
