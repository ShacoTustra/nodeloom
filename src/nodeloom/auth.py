"""Who is using the editor. For now one local user without signing in, as in ComfyUI itself: the server listens on
localhost by default, and requests are checked so that other sites cannot reach it (see api.py: the Host header and
the origin of state-changing requests).

Everything else is already per user (runs, files, results, settings), so several users need only a new
`current_user`: a session, a proxy's header or a token, mapped to a user id.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from nodeloom import i18n
from nodeloom.config import get_settings

LOCAL_USER = "local"


@dataclass(frozen=True)
class User:
    id: str
    lang: str  # en | ru


def language(saved: str | None, accept: str | None) -> str:
    """The editor's language: the user's own choice (the editor's Comfy.Locale), else LANGUAGE, else the browser's."""
    return i18n.norm(saved) or i18n.norm(get_settings().language) or i18n.from_accept(accept) or i18n.DEFAULT


def user_id(request: Request) -> str:
    return LOCAL_USER
