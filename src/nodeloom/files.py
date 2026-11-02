"""The user's files: uploads (`input`), results (`output`) and intermediate files (`temp`).

Files live in the media store (media_store.py) under <user id>/<kind>/<name>; the database row keeps the size for
the optional quota (STORAGE_LIMIT_MB). Inside the engine a file travels between nodes as a small dict ("ref"):
{"kind", "name", "media": "image"|"video"|"audio", "ct", "w"?, "h"?}.
"""

from __future__ import annotations

import io
import mimetypes
import re
import secrets
from pathlib import Path

from fastapi import Request, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nodeloom import i18n, media_store
from nodeloom.config import get_settings
from nodeloom.models import StoredFile

KINDS = ("input", "output", "temp")
MEDIA = {"image": "IMAGE", "video": "VIDEO", "audio": "AUDIO"}
# What may be uploaded: formats the models take. No SVG or anything else a browser would run.
UPLOAD_TYPES = {
    "image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff",
    "video/mp4", "video/quicktime", "video/webm",
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/mp4", "audio/aac", "audio/ogg",
    "audio/flac",
}
PIL_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif", "BMP": "image/bmp",
             "TIFF": "image/tiff", "MPO": "image/jpeg"}
# Files are served from our own origin: never sniffed, never run.
SERVE_HEADERS = {"X-Content-Type-Options": "nosniff",
                 "Content-Security-Policy": "sandbox; default-src 'none'; img-src 'self' data:; media-src 'self'; "
                                            "style-src 'unsafe-inline'"}
MAX_UPLOAD = 200 * 1024 * 1024  # ModelArk takes reference videos up to 200 MB


class FileError(Exception):
    pass


def media_of(content_type: str) -> str | None:
    head = (content_type or "").split("/")[0]
    return head if head in MEDIA else None


def key_of(user_id: str, kind: str, name: str) -> str:
    return f"{user_id}/{kind}/{name}"


def clean_name(name: str) -> str:
    """A safe file name: no folders, no control characters, at most 120 characters (the extension kept)."""
    name = Path((name or "").replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f\"<>|:*?/]", "_", name).strip(" .") or "file"
    stem, ext = Path(name).stem, Path(name).suffix[:10].lower()
    return stem[: 120 - len(ext)] + ext


def check_upload(data: bytes, content_type: str) -> str:
    """The content type to store an upload under, or FileError. Images must open in Pillow (their real format wins
    over what the browser said); video and audio go by the declared type from the allowlist."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct not in UPLOAD_TYPES:
        raise FileError(i18n.tr("Можно загрузить картинку (PNG, JPEG, WebP), видео (MP4, MOV) или аудио (MP3, WAV)",
                                "You can upload an image (PNG, JPEG, WebP), a video (MP4, MOV) or audio (MP3, WAV)"))
    if media_of(ct) == "image":
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
                fmt = im.format
        except Exception:  # noqa: BLE001  anything Pillow can't read is not an image for us
            raise FileError(i18n.tr("Файл не похож на картинку", "The file does not look like an image")) from None
        if fmt not in PIL_TYPES:
            raise FileError(i18n.tr("Этот формат картинки не подходит", "This image format is not supported"))
        ct = PIL_TYPES[fmt]
    return ct


def image_meta(data: bytes) -> dict:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            return {"w": im.width, "h": im.height}
    except Exception:  # noqa: BLE001  not an image Pillow reads: no size
        return {}


def ref(f: StoredFile) -> dict:
    out = {"kind": f.kind, "name": f.name, "media": media_of(f.content_type), "ct": f.content_type}
    out.update({k: v for k, v in (f.meta or {}).items() if k in ("w", "h", "duration")})
    return out


def view_item(r: dict) -> dict:
    """How the editor addresses a file in a node's output: /view?filename=&subfolder=&type=."""
    return {"filename": r["name"], "subfolder": "", "type": r["kind"]}


# ---------------------------------------------------------------- quota


async def used(session: AsyncSession, user_id: str) -> int:
    """Bytes of the user's files."""
    return int((await session.execute(select(func.coalesce(func.sum(StoredFile.size), 0)).where(
        StoredFile.user_id == user_id))).scalar_one())


def limit() -> int | None:
    mb = get_settings().storage_limit_mb
    return mb * 1024 * 1024 if mb > 0 else None


async def check_room(session: AsyncSession, user_id: str, extra: int = 0) -> None:
    room = limit()
    if room is not None and await used(session, user_id) + extra > room:
        raise FileError(i18n.tr("Место для файлов закончилось: удалите старые результаты или загрузки",
                                "No room left for files: delete old results or uploads"))


# ---------------------------------------------------------------- read and write


async def get(session: AsyncSession, user_id: str, kind: str, name: str) -> StoredFile | None:
    return (await session.execute(select(StoredFile).where(
        StoredFile.user_id == user_id, StoredFile.kind == kind, StoredFile.name == name))).scalar_one_or_none()


async def _free_name(session: AsyncSession, user_id: str, kind: str, name: str) -> str:
    stem, ext = Path(name).stem, Path(name).suffix
    candidate, n = name, 1
    while await get(session, user_id, kind, candidate) is not None:
        candidate, n = f"{stem} ({n}){ext}", n + 1
    return candidate


async def save(session: AsyncSession, user_id: str, kind: str, data: bytes | Path, content_type: str, *,
               name: str | None = None, overwrite: bool = False, meta: dict | None = None,
               run_id: str | None = None) -> StoredFile:
    """Store a file and its row (flushes, the caller commits). Without a name one is made up from the type."""
    if kind not in KINDS:
        raise FileError(i18n.tr("Неизвестная папка", "Unknown folder"))
    if name is None:
        ext = mimetypes.guess_extension(content_type) or ".bin"
        name = f"{secrets.token_hex(6)}{'.jpg' if ext in ('.jpe', '.jpeg') else ext}"
    name = clean_name(name)
    size = data.stat().st_size if isinstance(data, Path) else len(data)
    existing = await get(session, user_id, kind, name)
    if existing is not None and not overwrite:
        name, existing = await _free_name(session, user_id, kind, name), None
    if meta is None:
        meta = image_meta(data if isinstance(data, bytes) else data.read_bytes()) if media_of(content_type) == "image" else {}
    key = key_of(user_id, kind, name)
    if isinstance(data, Path):
        await media_store.store().put_file(key, data)
    else:
        await media_store.store().put(key, data)
    if existing is not None:
        existing.content_type, existing.size, existing.meta = content_type, size, meta
        f = existing
    else:
        f = StoredFile(user_id=user_id, kind=kind, name=name, key=key, content_type=content_type, size=size, meta=meta,
                       run_id=run_id)
        session.add(f)
    await session.flush()
    return f


async def delete(session: AsyncSession, rows: list[StoredFile]) -> int:
    """Remove files and their rows (flushes, the caller commits); returns the bytes freed."""
    freed = 0
    for f in rows:
        try:
            await media_store.store().delete(f.key)
        except Exception:  # noqa: BLE001  gone already or the store hiccups: the row goes anyway, no quota held
            pass
        freed += f.size
        await session.delete(f)
    await session.flush()
    return freed


def refs_in(value) -> list[dict]:
    """File refs anywhere in a node's outputs (lists of images, a video, ...)."""
    if isinstance(value, dict):
        return [value] if {"kind", "name", "media"} <= value.keys() else []
    if isinstance(value, list):
        return [r for v in value for r in refs_in(v)]
    return []


async def all_exist(session: AsyncSession, user_id: str, refs: list[dict]) -> bool:
    for r in refs:
        if await get(session, user_id, r["kind"], r["name"]) is None:
            return False
    return True


async def read(user_id: str, r: dict) -> bytes:
    try:
        return await media_store.store().get(key_of(user_id, r["kind"], r["name"]))
    except media_store.MissingFile:
        raise FileError(i18n.tr(f"Файл {r['name']} не найден", f"File {r['name']} not found")) from None


async def response(session: AsyncSession, user_id: str, kind: str, name: str, request: Request) -> Response:
    f = await get(session, user_id, kind, name)
    if f is None:
        return Response(status_code=404)
    try:
        resp = await media_store.store().response(f.key, request, None)
    except media_store.MissingFile:
        return Response(status_code=404)
    resp.headers["Content-Type"] = f.content_type
    resp.headers.update(SERVE_HEADERS)
    return resp


async def names(session: AsyncSession, user_id: str, kind: str, media: str) -> list[str]:
    """The user's files of a media type, newest first (the choices of an upload node)."""
    rows = await session.execute(select(StoredFile.name, StoredFile.content_type).where(
        StoredFile.user_id == user_id, StoredFile.kind == kind).order_by(StoredFile.created_at.desc()))
    return [n for n, ct in rows if media_of(ct) == media]
