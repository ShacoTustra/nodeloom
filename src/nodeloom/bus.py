"""Run events to the user's open editors.

A run executes in whichever process claimed it, while the user's browser holds a WebSocket to any process. Events go through Redis pub/sub (one channel per user, one pattern subscription per process); without
Redis (development, tests) they are delivered within the process. An event addressed to a client id (the browser
tab that queued the run) reaches only that tab, as in ComfyUI; queue status goes to all of the user's tabs.

Events are not stored: a tab that reconnects asks /api/jobs and /api/history for the state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from nodeloom import shared

log = logging.getLogger(__name__)

CHANNEL = "nodeloom:u:"

_local: dict[str, set[asyncio.Queue]] = {}
_listener: asyncio.Task | None = None


def _deliver(user_id: str, msg: dict) -> None:
    for q in list(_local.get(user_id, ())):
        with contextlib.suppress(asyncio.QueueFull):
            q.put_nowait(msg)


async def publish(user_id: str, type_: str, data: dict, sid: str | None = None) -> None:
    msg = {"type": type_, "data": data, "sid": sid}
    r = shared.redis()
    if r is not None:
        try:
            await r.publish(CHANNEL + user_id, json.dumps(msg, ensure_ascii=False))
            return
        except shared.ERRORS as e:
            shared.unavailable(e)
    _deliver(user_id, msg)


async def _listen() -> None:
    """Forward the users' channels to this process's sockets; reconnects after a failure."""
    while True:
        client = shared.listener()
        if client is None:
            return
        try:
            pubsub = client.pubsub()
            await pubsub.psubscribe(CHANNEL + "*")
            async for m in pubsub.listen():
                if m.get("type") != "pmessage":
                    continue
                channel = m["channel"].decode() if isinstance(m["channel"], bytes) else m["channel"]
                _deliver(channel[len(CHANNEL):], json.loads(m["data"]))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  Redis gone for a moment: retry
            log.warning("event bus: %s", e)
            await asyncio.sleep(2)


def _ensure_listener() -> None:
    global _listener
    if shared.redis() is not None and (_listener is None or _listener.done()):
        _listener = asyncio.create_task(_listen())


@contextlib.contextmanager
def subscribe(user_id: str, maxsize: int = 1000):
    """A queue of the user's events for one socket."""
    _ensure_listener()
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _local.setdefault(user_id, set()).add(q)
    try:
        yield q
    finally:
        subs = _local.get(user_id)
        if subs is not None:
            subs.discard(q)
            if not subs:
                _local.pop(user_id, None)


async def stop() -> None:
    global _listener
    if _listener is not None:
        _listener.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await _listener
        _listener = None
