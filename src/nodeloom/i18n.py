"""The language of what the backend says: English (the default) or Russian.

A request takes the user's language (auth.current_user: the editor's own setting, else LANGUAGE, else the browser's);
a run, which executes outside the request, takes the language it was queued in (runs.execute). Messages are written
in both languages where they are made: `tr("Russian", "English")`.
"""

from __future__ import annotations

from contextvars import ContextVar

LANGS = ("en", "ru")
DEFAULT = "en"

_lang: ContextVar[str] = ContextVar("loom_lang", default=DEFAULT)


def norm(value: str | None) -> str | None:
    v = (value or "").strip().lower()[:2]
    return v if v in LANGS else None


def from_accept(header: str | None) -> str | None:
    """The first language of Accept-Language that we speak ("ru-RU,ru;q=0.9,en;q=0.8" -> "ru")."""
    for part in (header or "").split(","):
        if lang := norm(part.split(";")[0]):
            return lang
    return None


def current() -> str:
    return _lang.get()


def use(value: str | None) -> None:
    """Set the language for the rest of this task."""
    _lang.set(norm(value) or DEFAULT)


def tr(ru: str, en: str) -> str:
    return ru if _lang.get() == "ru" else en
