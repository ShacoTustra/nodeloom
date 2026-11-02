"""State shared by the processes of a cluster: Redis (REDIS_URL, or Sentinel with REDIS_SENTINELS).

Only run events go there (bus.py); nothing that must not be lost leaves the database. Every user of Redis fails open:
if it is unreachable, events stay within the process for a while.

Without REDIS_URL / REDIS_SENTINELS (one process: the usual local setup, tests) the callers keep their state in
process memory.
"""

from __future__ import annotations

import logging
import time

from redis.asyncio import Redis
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import RedisError

from nodeloom.config import get_settings

log = logging.getLogger(__name__)

_client: Redis | None = None
_configured: bool | None = None
_last_warning = 0.0

# What counts as "Redis is not there right now".
ERRORS = (RedisError, OSError, TimeoutError)


def redis() -> Redis | None:
    """The shared Redis (the current master through Sentinel), or None when it is not configured."""
    global _client, _configured
    if _configured is None:
        s = get_settings()
        if s.redis_sentinels:
            hosts = []
            for item in s.redis_sentinels.split(","):
                host, _, port = item.strip().rpartition(":")
                hosts.append((host, int(port)))
            kw = {"password": s.redis_password} if s.redis_password else {}
            sentinel = Sentinel(hosts, sentinel_kwargs=kw, socket_timeout=0.5, socket_connect_timeout=0.5, **kw)
            _client = sentinel.master_for(s.redis_master, socket_timeout=0.5, socket_connect_timeout=0.5)
        elif s.redis_url:
            _client = Redis.from_url(s.redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
        _configured = _client is not None
    return _client


def listener() -> Redis | None:
    """A separate connection without a read timeout, for pub/sub subscriptions that wait for messages."""
    s = get_settings()
    kw = {"password": s.redis_password} if s.redis_password else {}
    if _configured and _client is not None and not (s.redis_sentinels or s.redis_url):
        return _client  # set by use() in tests
    if s.redis_sentinels:
        hosts = []
        for item in s.redis_sentinels.split(","):
            host, _, port = item.strip().rpartition(":")
            hosts.append((host, int(port)))
        sentinel = Sentinel(hosts, sentinel_kwargs=kw, socket_timeout=0.5, socket_connect_timeout=0.5, **kw)
        return sentinel.master_for(s.redis_master, socket_timeout=None, socket_connect_timeout=2)
    if s.redis_url:
        return Redis.from_url(s.redis_url, socket_timeout=None, socket_connect_timeout=2)
    return None


def use(client: Redis | None) -> None:
    """Set the client explicitly (tests)."""
    global _client, _configured
    _client, _configured = client, client is not None


def reset() -> None:
    global _client, _configured
    _client, _configured = None, None


def unavailable(e: Exception) -> None:
    """Log a Redis failure at most once a minute: the callers carry on without it."""
    global _last_warning
    now = time.monotonic()
    if now - _last_warning > 60:
        _last_warning = now
        log.warning("redis unavailable, events stay in this process: %s", e)
