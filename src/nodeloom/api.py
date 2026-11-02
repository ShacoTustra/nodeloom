"""ComfyUI's frontend and the part of ComfyUI's API it uses, served at the site's root as ComfyUI serves them.

The user is auth.current_user's (one local user for now). The frontend sends no CSRF header of ours, so requests that
change state must come from our own pages (Sec-Fetch-Site / Origin), and the WebSocket checks its Origin likewise;
app.py lets in only the Host names in ALLOWED_HOSTS.

What ComfyUI has and this backend does not (local models, custom node managers, Comfy's cloud) answers empty, so the
frontend shows nothing for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import urllib.parse
from datetime import UTC
from functools import lru_cache
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nodeloom import agent, auth, bus, engine, files, i18n, nodes, providers, runs
from nodeloom.auth import User
from nodeloom.config import APP_NAME
from nodeloom.db import get_session, sessionmaker, utcnow
from nodeloom.models import Run, StoredFile, Userdata

log = logging.getLogger(__name__)

router = APIRouter(include_in_schema=False)

WEB = Path(__file__).parent / "web"
FRONTEND_VERSION = "1.52.7"
COMFYUI_VERSION = "0.37.2"  # what the frontend expects of its backend
SETTINGS_FILE = "comfy.settings.json"
# The editor talks only to us: no Comfy cloud, registry or CDN (ComfyUI sends the same with --disable-api-nodes).
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self'; connect-src 'self' data:; "
       "frame-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'")
USERDATA_FILE_MAX = 5 * 1024 * 1024
USERDATA_TOTAL_MAX = 100 * 1024 * 1024
EXT = "/extensions/nodeloom"  # our extension of the editor: web/


def frontend_dir() -> Path:
    import comfyui_frontend_package

    return Path(comfyui_frontend_package.__file__).parent / "static"


# ---------------------------------------------------------------- the user


def _err(status: int, message: str, code: str) -> HTTPException:
    return HTTPException(status, {"error": {"code": code, "message": message}})


def _same_origin(request: Request) -> bool:
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site in ("same-origin", "none")
    origin = request.headers.get("origin")
    return origin is None or urllib.parse.urlsplit(origin).netloc == request.headers.get("host")


async def _user(user_id: str, accept: str | None, session: AsyncSession) -> User:
    """The user and their language (the editor's own Comfy.Locale, else LANGUAGE, else the browser's), which the
    rest of the request speaks."""
    saved = (await _settings(session, user_id)).get("Comfy.Locale")
    user = User(user_id, auth.language(saved if isinstance(saved, str) else None, accept))
    i18n.use(user.lang)
    return user


async def current_user(request: Request, session: AsyncSession = Depends(get_session)) -> User:
    if request.method not in ("GET", "HEAD") and not _same_origin(request):
        raise _err(403, i18n.tr("Запрос не со страницы редактора", "The request is not from the editor's page"), "csrf")
    return await _user(auth.user_id(request), request.headers.get("accept-language"), session)


# ---------------------------------------------------------------- the editor's files


def _static(path: Path, immutable: bool) -> Response:
    cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
    if path.suffix == ".js" and (patched := _patched(str(path))) is not None:
        return Response(patched, media_type="text/javascript", headers={"Cache-Control": cache})
    return FileResponse(path, headers={"Cache-Control": cache})


# Strings of the stock frontend that no setting reaches, replaced as its files are served. Check them when the
# frontend package is upgraded: a string that is gone is simply not replaced.
FRONTEND_PATCHES = [
    # the name of a new workflow tab, in the page's language (<html lang>, set from the user's choice)
    (b"`Unsaved Workflow.json`",
     "`${document.documentElement.lang==='en'?'Untitled':'Без названия'}.json`".encode()),
]


@lru_cache(maxsize=64)
def _patched(path: str) -> bytes | None:
    data = Path(path).read_bytes()
    if not any(old in data for old, _ in FRONTEND_PATCHES):
        return None
    for old, new in FRONTEND_PATCHES:
        data = data.replace(old, new)
    return data


@router.get("/")
async def index(user: User = Depends(current_user)):
    return Response(_index_html(user.lang), media_type="text/html",
                    headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP})


@lru_cache
def _index_html(lang: str) -> str:
    """The frontend's page with our loading screen (web/splash, over the editor until it is ready; the editor's own
    splash stays hidden), the language (<html lang>, which nodeloom.js reads), our title and our icon."""
    html = (frontend_dir() / "index.html").read_text()
    html = html.replace('<html lang="en">', f'<html lang="{lang}">', 1)
    html = html.replace("<title>ComfyUI</title>", f'<title>{APP_NAME}</title><meta name="application-name" '
                                                   f'content="{APP_NAME}">', 1)
    html = html.replace('href="./assets/favicon.ico"', f'href=".{EXT}/favicon.ico"', 1)
    # the page's colours before the app starts, whatever the frontend remembered
    html = re.sub(r"bg = isDark \? '[^']*' : '[^']*'", "bg = '#121212'", html, count=1)
    html = html.replace("bg = localStorage.getItem('comfy-splash-bg')", "bg = null", 1)
    html = html.replace("</head>", f'<link rel="stylesheet" href=".{EXT}/splash/splash.css">'
                                   f'<script src=".{EXT}/splash/splash.js"></script></head>', 1)
    splash = (WEB / "splash" / "splash.html").read_text().replace("{name}", APP_NAME)
    html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + splash, html, count=1)
    return html


@router.get("/user.css")
async def user_css():
    return Response("", media_type="text/css", headers={"Cache-Control": "no-cache"})


@router.get(EXT + "/{path:path}")
async def our_extension(path: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    p = (WEB / path).resolve()
    if WEB.resolve() not in p.parents or not p.is_file() or p.name == "settings.json":
        return Response(status_code=404)
    if path == "starter.json":  # the graph of a new tab, like the templates: the user's models and language
        ch = await runs.choices(session, user.id)
        graph = json.loads(_starter(ch).read_text())
        return JSONResponse(_translated(_fit_models(graph, ch)), headers={"Cache-Control": "no-cache"})
    return _static(p, immutable=False)


# the first whose nodes the user has models for
STARTERS = ("starter.json", "starter-openai.json", "starter-google.json")


def _starter(choices: dict[str, list]) -> Path:
    for name in STARTERS:
        graph = json.loads((WEB / name).read_text())
        if all(nodes.available(nodes.BY_ID[n["type"]], choices) for n in graph["nodes"] if n["type"] in nodes.BY_ID):
            return WEB / name
    return WEB / "starter-empty.json"


# ---------------------------------------------------------------- server info


@router.get("/api/features")
async def features(user: User = Depends(current_user)):
    return {"supports_preview_metadata": False, "max_upload_size": files.MAX_UPLOAD, "node_replacements": False,
            "assets": False, "extension": {"manager": {"supports_v4": False}}}


@router.get("/api/system_stats")
async def system_stats(user: User = Depends(current_user)):
    return {"system": {"os": "linux", "ram_total": 0, "ram_free": 0, "comfyui_version": COMFYUI_VERSION,
                       "required_frontend_version": FRONTEND_VERSION, "installed_templates_version": None,
                       "required_templates_version": None, "python_version": "", "pytorch_version": "",
                       "embedded_python": False, "deploy_environment": "local",
                       # as ComfyUI with Comfy's cloud nodes off: the frontend then skips its update check too
                       "argv": ["main.py", "--disable-api-nodes"]},
            "devices": []}


@router.get("/api/users")
async def users_info(user: User = Depends(current_user)):
    return {"storage": "server", "migrated": True}


@router.get("/api/extensions")
async def extensions(user: User = Depends(current_user)):
    core = frontend_dir() / "extensions" / "core"
    stock = sorted(f"/extensions/core/{p.relative_to(core).as_posix()}" for p in core.rglob("*.js")) if core.is_dir() else []
    return [*stock, EXT + "/nodeloom.js"]


# Our wording over the frontend's messages (merged into its locale).
# The results gallery is built from the run history here, so clearing the history deletes its pictures and videos
# (in Comfy's cloud they stay; the stock texts promise that).
I18N = {"ru": {
    "sideToolbar": {"nodeLibraryTab": {"sections": {"comfyNodes": "Ноды"}},
                    "workflows": "Воркфлоу", "labels": {"workflows": "Воркфлоу"}},
    "menuLabels": {"Workflows": "Воркфлоу"},
    "commands": {"Workspace_ToggleSidebarTab_workflows": {"label": "Показать или скрыть воркфлоу",
                                                          "tooltip": "Воркфлоу"}},
    "g": {"searchWorkflows": "Поиск воркфлоу", "noWorkflowsFound": "Воркфлоу не найдены."},
    "queueProgressOverlay": {
        "clearHistoryDialogAssetsNote": "Картинки и видео этих заданий тоже удалятся и освободят место. "
                                        "Загруженные файлы останутся.",
        "clearHistoryMenuAssetsNote": "Картинки и видео удалятся вместе с заданиями."},
}, "en": {
    "sideToolbar": {"nodeLibraryTab": {"sections": {"comfyNodes": "Nodes"}}},
    "queueProgressOverlay": {
        "clearHistoryDialogAssetsNote": "The images and videos of these jobs are deleted too and free up space. "
                                        "Uploaded files stay.",
        "clearHistoryMenuAssetsNote": "The images and videos are deleted together with the jobs."},
}}


@router.get("/api/i18n")
async def i18n_strings(user: User = Depends(current_user)):
    """Our strings over the editor's own, in both languages; the nodes' Russian (nodes.i18n_ru) for Russian users."""
    return {"ru": {**I18N["ru"], **nodes.i18n_ru()}, "en": I18N["en"]}


@router.get("/internal/files/input")
async def input_files(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    """The user's uploads, newest first (the "Импортированные" tab of the results panel)."""
    out: list[str] = []
    for media in files.MEDIA:
        out += await files.names(session, user.id, "input", media)
    return out


@router.get("/api/object_info")
async def object_info(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    return nodes.object_info(await runs.choices(session, user.id))


@router.get("/api/object_info/{node_class}")
async def object_info_one(node_class: str, user: User = Depends(current_user),
                          session: AsyncSession = Depends(get_session)):
    info = nodes.object_info(await runs.choices(session, user.id))
    return {node_class: info[node_class]} if node_class in info else {}


# nothing local: no models, embeddings, blueprints, templates of ComfyUI's, node replacements, logs
for _path, _value in (("/api/embeddings", []), ("/api/models", []), ("/api/experiment/models", []),
                      ("/api/global_subgraphs", {}), ("/api/node_replacements", {}), ("/api/free", {}),
                      ("/api/logs", "")):
    router.add_api_route(_path, (lambda v: (lambda user=Depends(current_user): v))(_value),
                         methods=["POST"] if _path == "/api/free" else ["GET"])


@router.get("/api/workflow_templates")
async def workflow_templates(user: User = Depends(current_user)):
    return {}  # ours are served as the editor's own templates, below


@router.get("/templates/{name}")
async def template(name: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    """Our templates in the place of ComfyUI's own (the comfyui-workflow-templates package): index.json lists
    them by category, <id>.json is the graph, <id>-1.jpg the picture. The index lists only the templates whose nodes
    the user has models for, and a graph comes with the user's models in place of the ones it was made with. Written
    in English; a Russian user gets them through templates/ru.json (titles, descriptions, example prompts, labels;
    never slot or output names)."""
    if name.startswith("index") and name.endswith(".json") and name != "index.json":
        name = "index.json" if name.startswith("index.") else name  # index.ru.json, index.en.json, ...
    folder = (WEB / "templates").resolve()
    p = (folder / name).resolve()
    if p.parent != folder or not p.is_file() or p.name == "ru.json":
        return Response(status_code=404)
    if p.suffix != ".json":
        return _static(p, immutable=False)
    ch = await runs.choices(session, user.id)
    data = json.loads(p.read_text())
    if p.name == "index.json":
        data = [{**group, "templates": [t for t in group["templates"] if _usable(t["name"], ch)]} for group in data]
        data = [group for group in data if group["templates"]]
    else:
        data = _fit_models(data, ch)
    return JSONResponse(_translated(data), headers={"Cache-Control": "no-cache"})


@lru_cache
def _template_nodes(name: str) -> frozenset[str]:
    p = WEB / "templates" / f"{name}.json"
    return frozenset(n.get("type") for n in json.loads(p.read_text()).get("nodes") or []) if p.is_file() else frozenset()


def _usable(name: str, choices: dict[str, list]) -> bool:
    return all(t == "Note" or (t in nodes.BY_ID and nodes.available(nodes.BY_ID[t], choices))
               for t in _template_nodes(name))


def _fit_models(graph: dict, choices: dict[str, list]) -> dict:
    """A template's (or the starter's) models that the user does not have become the user's own: the same model of
    another provider ("ark/seedream-x" as "mygateway/seedream-x"), else the first of the list. The model is every
    generating node's first widget (tests/test_templates.py checks it)."""
    for n in graph.get("nodes") or []:
        spec = nodes.BY_ID[n["type"]].input("model") if n.get("type") in nodes.BY_ID else None
        mine = choices.get(spec.options) if spec is not None else None
        if not mine:
            continue
        values, named = n.get("widgets_values"), n.get("widgets_values_named")
        current = (named or {}).get("model", values[0] if values else None)
        if current not in mine:
            model_id = str(current or "").partition("/")[2]
            fit = next((v for v in mine if v.partition("/")[2] == model_id), mine[0])
            if values:
                values[0] = fit
            if isinstance(named, dict):
                named["model"] = fit
    return graph


@lru_cache
def _template_ru() -> dict[str, dict[str, str]]:
    return json.loads((WEB / "templates" / "ru.json").read_text())


TEXT_KEYS = {"title", "description", "tags", "localized_name", "label"}  # what a template says to people
VALUE_KEYS = {"widgets_values", "widgets_values_named"}  # what its nodes are set to: prompts, list words


def _translated(value, table: str | None = None):
    """A template in the user's language: for a Russian user the strings templates/ru.json knows, under the keys that
    hold text for people ("strings": titles, descriptions, labels) or widget values ("values": example prompts, list
    words). Nothing else: not slot or output names ("name"), which must stay the nodes' own in both languages, nor
    types and media kinds that happen to be English words."""
    if i18n.current() != "ru":
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "name" and table != "values":
                out[k] = v
            elif table == "values" or k in VALUE_KEYS:
                out[k] = _translated(v, "values")
            else:
                out[k] = _translated(v, "strings" if k in TEXT_KEYS else None)
        return out
    if isinstance(value, list):
        return [_translated(v, table) for v in value]
    if isinstance(value, str) and table is not None:
        return _template_ru()[table].get(value, value)
    return value


# ---------------------------------------------------------------- settings and userdata


async def _userdata(session: AsyncSession, user_id: str, path: str) -> Userdata | None:
    return await session.get(Userdata, (user_id, path))


async def _put_userdata(session: AsyncSession, user_id: str, path: str, content: bytes) -> Userdata:
    if len(content) > USERDATA_FILE_MAX:
        raise _err(413, i18n.tr("Файл слишком большой", "The file is too large"), "too_large")
    total = (await session.execute(select(func.coalesce(func.sum(Userdata.size), 0)).where(
        Userdata.user_id == user_id, Userdata.path != path))).scalar_one()
    if total + len(content) > USERDATA_TOTAL_MAX:
        raise _err(413, i18n.tr("Место для воркфлоу закончилось: удалите ненужные",
                                   "No room left for workflows: delete the ones you do not need"), "quota_exceeded")
    row = await _userdata(session, user_id, path)
    if row is None:
        row = Userdata(user_id=user_id, path=path, content=content, size=len(content))
        session.add(row)
    else:
        row.content, row.size, row.modified_at = content, len(content), utcnow()
    await session.flush()
    return row


async def _settings(session: AsyncSession, user_id: str) -> dict:
    row = await _userdata(session, user_id, SETTINGS_FILE)
    if row is None:
        return json.loads((WEB / "settings.json").read_text())
    try:
        return json.loads(row.content)
    except ValueError:
        return {}


@router.get("/api/settings")
async def get_settings_all(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    return {**await _settings(session, user.id), "Comfy.Locale": user.lang}


@router.get("/api/settings/{sid}")
async def get_setting(sid: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    if sid == "Comfy.Locale":
        return user.lang
    return (await _settings(session, user.id)).get(sid)


@router.post("/api/settings")
async def post_settings(request: Request, user: User = Depends(current_user),
                        session: AsyncSession = Depends(get_session)):
    current = await _settings(session, user.id)
    body = await request.json()
    if isinstance(body, dict):
        current.update(body)
    await _put_userdata(session, user.id, SETTINGS_FILE, json.dumps(current, ensure_ascii=False).encode())
    await session.commit()
    return Response(status_code=200)


@router.post("/api/settings/{sid}")
async def post_setting(sid: str, request: Request, user: User = Depends(current_user),
                       session: AsyncSession = Depends(get_session)):
    current = await _settings(session, user.id)
    current[sid] = await request.json()
    await _put_userdata(session, user.id, SETTINGS_FILE, json.dumps(current, ensure_ascii=False).encode())
    await session.commit()
    return Response(status_code=200)


def _clean_path(path: str) -> str:
    path = path.replace("\\", "/").strip("/")
    parts = [p for p in path.split("/") if p]
    if not parts or any(p in (".", "..") for p in parts) or len(path) > 500:
        raise _err(400, i18n.tr("Недопустимый путь", "Invalid path"), "invalid_path")
    return "/".join(parts)


def _info(row: Userdata, prefix: str = "") -> dict:
    return {"path": row.path[len(prefix):], "size": row.size, "modified": row.modified_at.timestamp(),
            "created": row.created_at.timestamp()}


@router.get("/api/userdata")
async def list_userdata(dir: str = Query(...), recurse: bool = False, full_info: bool = False, split: bool = False,
                        user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    prefix = _clean_path(dir) + "/"
    rows = (await session.execute(select(Userdata).where(
        Userdata.user_id == user.id, Userdata.path.startswith(prefix, autoescape=True))
        .order_by(Userdata.path))).scalars()
    out = []
    for row in rows:
        rel = row.path[len(prefix):]
        if not recurse and "/" in rel:
            continue
        if full_info:
            out.append(_info(row, prefix))
        elif split:
            out.append([rel, *rel.split("/")])
        else:
            out.append(rel)
    return out


@router.get("/api/v2/userdata")
async def list_userdata_v2(path: str = "", user: User = Depends(current_user),
                           session: AsyncSession = Depends(get_session)):
    prefix = (_clean_path(path) + "/") if path.strip("/") else ""
    rows = (await session.execute(select(Userdata).where(
        Userdata.user_id == user.id, Userdata.path.startswith(prefix, autoescape=True)))).scalars()
    seen, out = set(), []
    for row in rows:
        rel = row.path[len(prefix):]
        head, slash, _ = rel.partition("/")
        if slash:
            if head not in seen:
                seen.add(head)
                out.append({"name": head, "path": prefix + head, "type": "directory"})
        else:
            out.append({"name": rel, "path": row.path, "type": "file", "size": row.size,
                        "modified": row.modified_at.timestamp()})
    return out


@router.get("/api/userdata/{file:path}")
async def get_userdata(file: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    row = await _userdata(session, user.id, _clean_path(file))
    if row is None:
        return Response(status_code=404)
    ct = "application/json" if row.path.endswith(".json") else "application/octet-stream"
    return Response(row.content, media_type=ct, headers={"Cache-Control": "no-store", **files.SERVE_HEADERS})


@router.post("/api/userdata/{file:path}")
async def post_userdata(file: str, request: Request, overwrite: bool = True, full_info: bool = False,
                        user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    if file.endswith("/publish"):
        return Response(status_code=404)
    if "/move/" in file:
        src, _, dest = file.partition("/move/")
        return await _move(src, dest, overwrite, user, session)
    path = _clean_path(file)
    if not overwrite and await _userdata(session, user.id, path) is not None:
        return Response("File already exists", status_code=409)
    row = await _put_userdata(session, user.id, path, await request.body())
    await session.commit()
    return _info(row) if full_info else row.path


async def _move(src: str, dest: str, overwrite: bool, user: User, session: AsyncSession):
    src, dest = _clean_path(src), _clean_path(dest)
    row = await _userdata(session, user.id, src)
    if row is None:
        return Response(status_code=404)
    existing = await _userdata(session, user.id, dest)
    if existing is not None:
        if not overwrite:
            return Response("File already exists", status_code=409)
        await session.delete(existing)
    new = Userdata(user_id=user.id, path=dest, content=row.content, size=row.size, created_at=row.created_at)
    await session.delete(row)
    await session.flush()
    session.add(new)
    await session.commit()
    return _info(new)


@router.delete("/api/userdata/{file:path}")
async def delete_userdata(file: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    await session.execute(delete(Userdata).where(Userdata.user_id == user.id,
                                                       Userdata.path == _clean_path(file)))
    await session.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------- files


@router.post("/api/upload/image")
async def upload(request: Request, image: UploadFile, user: User = Depends(current_user),
                 session: AsyncSession = Depends(get_session)):
    form = await request.form()
    overwrite = str(form.get("overwrite", "")).lower() == "true"
    data = await image.read(files.MAX_UPLOAD + 1)
    if len(data) > files.MAX_UPLOAD:
        raise _err(413, i18n.tr(f"Файл больше {files.MAX_UPLOAD >> 20} МБ", f"The file is over {files.MAX_UPLOAD >> 20} MB"),
                   "too_large")
    try:
        ct = files.check_upload(data, image.content_type or "")
    except files.FileError as e:
        raise _err(400, str(e), "invalid_file") from None
    try:
        await files.check_room(session, user.id, len(data))
        f = await files.save(session, user.id, "input", data, ct, name=image.filename or None, overwrite=overwrite)
    except files.FileError as e:
        raise _err(413, str(e), "quota_exceeded") from None
    await session.commit()
    return {"name": f.name, "subfolder": "", "type": "input"}


@router.get("/api/view")
async def view(request: Request, filename: str, type: str = "output", user: User = Depends(current_user),
               session: AsyncSession = Depends(get_session)):
    if type == "asset":  # the preview of a ready character (the "Seedance character" node)
        return await asset_preview(filename)
    if type not in files.KINDS:
        return Response(status_code=404)
    name = filename
    if name.endswith("]") and " [" in name:  # ComfyUI's annotated names: "a.png [input]"
        name, _, kind = name[:-1].rpartition(" [")
        type = kind if kind in files.KINDS else type
    return await files.response(session, user.id, type, name, request)


# ---------------------------------------------------------------- runs


def _history_item(run: Run) -> dict:
    status_str = {"success": "success", "error": "error", "cancelled": "error"}.get(run.status)
    return {"prompt": [0, run.id, run.prompt, _extra(run), []],
            "outputs": run.outputs or {},
            "status": {"status_str": status_str, "completed": run.status == "success",
                       "messages": run.messages or []},
            "meta": {nid: {"node_id": nid, "display_node": nid, "parent_node": None, "real_node_id": nid}
                     for nid in (run.outputs or {})}}


def _extra(run: Run) -> dict:
    return {k: v for k, v in (run.extra or {}).items() if not k.startswith("loom_")}


def _ms(dt) -> int | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # SQLite gives back naive datetimes; they are UTC
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _preview(outputs: dict) -> tuple[int, dict | None]:
    count, preview, text = 0, None, None
    for nid, out in (outputs or {}).items():
        for media, items in (out or {}).items():
            if media == "animated" or not isinstance(items, list):
                continue
            for item in items:
                # a character shown by "Show" is not a file: the editor's job list accepts only input/output/temp
                # files and drops the whole list over one other item
                if isinstance(item, dict) and item.get("type", "output") in files.KINDS:
                    count += 1
                    if preview is None:
                        kind = "video" if out.get("animated") else media
                        preview = {**item, "nodeId": nid, "mediaType": kind}
                elif media == "text" and text is None:
                    text = {"content": str(item)[:1024], "nodeId": nid, "mediaType": "text"}
    return count, preview or text


JOB_STATUS = {"pending": "pending", "running": "in_progress", "cancelling": "in_progress", "success": "completed",
              "error": "failed", "cancelled": "cancelled"}


def _new_outputs(run: Run) -> dict:
    """What the run made itself: results reused from earlier runs are left out, so the editor's results gallery
    shows each picture once, not once per run (runs older than this field count everything)."""
    new = (run.extra or {}).get("loom_new")
    outputs = run.outputs or {}
    return outputs if new is None else {k: v for k, v in outputs.items() if k in set(new)}


def _job(run: Run, full: bool = False) -> dict:
    count, preview = _preview(_new_outputs(run))
    workflow = ((run.extra or {}).get("extra_pnginfo") or {}).get("workflow") or {}
    job = {"id": run.id, "status": JOB_STATUS[run.status], "priority": 0, "create_time": _ms(run.created_at),
           "execution_start_time": _ms(run.started_at), "execution_end_time": _ms(run.finished_at),
           "execution_error": run.error if run.status == "error" else None,
           "outputs_count": count, "previewable_outputs_count": count, "preview_output": preview,
           "workflow_id": workflow.get("id") if isinstance(workflow, dict) else None}
    job = {k: v for k, v in job.items() if v is not None}
    if full:
        job["outputs"] = run.outputs or {}
        job["execution_status"] = _history_item(run)["status"]
        job["workflow"] = {"prompt": run.prompt, "extra_data": _extra(run)}
    return job


@router.post("/api/nodeloom/agent")
async def agent_turn(request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    """The Builder: {messages: [{role, text}], graph: the canvas as the API prompt or null} → NDJSON events
    (step labels, then done with the text and the graph to put on the canvas, or error). Nothing is run."""
    body = await request.json()
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    graph = body.get("graph") if isinstance(body.get("graph"), dict) else None
    ch = await runs.choices(session, user.id)
    await session.close()

    async def events():
        i18n.use(user.lang)  # the stream runs after the request's own task
        async for ev in agent.run(ch, messages, graph):
            yield (json.dumps(ev, ensure_ascii=False) + "\n").encode()

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/api/nodeloom/status")
async def status(user: User = Depends(current_user)):
    """The providers and how many models each has, for the editor's notice when nothing is configured."""
    out = []
    for p in providers.registry().values():
        models = await p.models()
        made = await p.nodes() if hasattr(p, "nodes") else []
        out.append({"name": p.name, "type": p.type, "models": len(models), "nodes": len(made)})
    return {"providers": out}


@router.get("/api/nodeloom/uploads")
async def uploads(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    """The user's uploads, newest first, and the room their files take (the "Uploads" tab: the editor's own assets
    panel deletes imported files only in Comfy's cloud)."""
    rows = (await session.execute(select(StoredFile).where(StoredFile.user_id == user.id, StoredFile.kind == "input")
                                  .order_by(StoredFile.created_at.desc()))).scalars()
    items = [{"name": f.name, "media": files.media_of(f.content_type), "size": f.size, "created": _ms(f.created_at)}
             for f in rows]
    return {"items": items, "used": await files.used(session, user.id), "limit": files.limit()}


@router.delete("/api/nodeloom/uploads/{name:path}")
async def delete_upload(name: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    """Delete an upload; the results made from it stay in the results gallery."""
    f = await files.get(session, user.id, "input", name)
    if f is None:
        return Response(status_code=404)
    freed = await files.delete(session, [f])
    await session.commit()
    return {"freed": freed}


@router.get("/api/nodeloom/assets")
async def assets(gender: str | None = None, age_min: int | None = None, age_max: int | None = None,
                 q: str | None = None, offset: int = 0, limit: int = 60, user: User = Depends(current_user)):
    """Ready characters for the "Seedance character" node's picker: the first ark provider that lists them."""
    library = await agent._library()
    if library is None:
        return {"total": 0, "items": [], "available": False}
    params = {k: v for k, v in {"gender": gender, "age_min": age_min, "age_max": age_max, "q": q}.items()
              if v not in (None, "")}
    try:
        found = await library.assets({**params, "offset": max(offset, 0), "limit": max(1, min(limit, 200))})
    except providers.ProviderError as e:
        raise _err(502, str(e), "provider") from None
    return {**found, "available": True}


async def asset_preview(asset_id: str) -> Response:
    library = await agent._library()
    if library is None or not nodes.ASSET_ID.match(asset_id or ""):
        return Response(status_code=404)
    r = await library.asset_preview(asset_id)
    if r.status_code != 200 or not (r.headers.get("content-type") or "").startswith("image/"):
        return Response(status_code=404)
    return Response(r.content, media_type=r.headers["content-type"],
                    headers={"Cache-Control": "private, max-age=86400", **files.SERVE_HEADERS})


def _prompt_error(kind: str, message: str, details: str = "", status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message, "details": details, "extra_info": {}},
                         "node_errors": {}}, status_code=status)


@router.post("/api/prompt")
async def post_prompt(request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    body = await request.json()
    extra = dict(body.get("extra_data") or {})
    extra["create_time"] = int(time.time() * 1000)
    run_id = body.get("prompt_id")
    if run_id is not None:
        try:
            import uuid
            if str(uuid.UUID(run_id)) != run_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            return _prompt_error("invalid_prompt", i18n.tr("Неверный prompt_id", "Invalid prompt_id"))
        if await session.get(Run, run_id) is not None:
            return _prompt_error("invalid_prompt", i18n.tr("Такой запуск уже есть", "This run already exists"))
    try:
        run = await runs.submit(session, user.id, prompt=body.get("prompt") or {}, extra=extra,
                                client_id=body.get("client_id"), targets=body.get("partial_execution_targets"),
                                run_id=run_id)
    except engine.PromptInvalid as e:
        return JSONResponse(e.body, status_code=400)
    except runs.QueueFull:
        return _prompt_error("queue_full", i18n.tr("Слишком много запусков в очереди", "Too many runs in the queue"),
                             i18n.tr("Дождитесь окончания текущих", "Wait for the current ones to finish"), 429)
    n = await runs.queue_remaining(session, user.id)
    return {"prompt_id": run.id, "number": n, "node_errors": {}}


@router.get("/api/prompt")
async def get_prompt(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    return {"exec_info": {"queue_remaining": await runs.queue_remaining(session, user.id)}}


async def _active(session: AsyncSession, user_id: str) -> list[Run]:
    return list((await session.execute(select(Run).where(
        Run.user_id == user_id, Run.status.in_(("pending", "running", "cancelling")))
        .order_by(Run.created_at))).scalars())


@router.get("/api/queue")
async def get_queue(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    active = await _active(session, user.id)

    def item(i: int, r: Run) -> list:
        return [i, r.id, r.prompt, _extra(r), []]

    return {"queue_running": [item(i, r) for i, r in enumerate(active) if r.status != "pending"],
            "queue_pending": [item(i, r) for i, r in enumerate(active) if r.status == "pending"]}


@router.post("/api/queue")
async def post_queue(request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    body = await request.json()
    if body.get("clear"):
        ids = [r.id for r in await _active(session, user.id) if r.status == "pending"]
        await runs.cancel(session, user.id, ids or ["-"])
    if body.get("delete"):
        await runs.cancel(session, user.id, [str(x) for x in body["delete"]])
    return Response(status_code=200)


@router.post("/api/interrupt")
async def interrupt(request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    try:
        body = await request.json()
    except ValueError:
        body = {}
    pid = (body or {}).get("prompt_id")
    await runs.cancel(session, user.id, [pid] if pid else None)
    return Response(status_code=200)


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    run = await session.get(Run, job_id)
    if run is None or run.user_id != user.id:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    touched = await runs.cancel(session, user.id, [job_id])
    return {"id": job_id, "cancelled": bool(touched)}


@router.post("/api/jobs/cancel")
async def cancel_jobs(request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    body = await request.json()
    ids = [str(x) for x in (body.get("job_ids") or body.get("ids") or [])]
    touched = await runs.cancel(session, user.id, ids or ["-"])
    return {"cancelled": touched}


@router.get("/api/jobs")
async def jobs(status: str | None = None, limit: int = 100, offset: int = 0, sort_order: str = "desc",
               workflow_id: str | None = None, user: User = Depends(current_user),
               session: AsyncSession = Depends(get_session)):
    wanted = {s.strip() for s in (status or "").split(",") if s.strip()} or set(JOB_STATUS.values())
    db_status = [k for k, v in JOB_STATUS.items() if v in wanted]
    q = select(Run).where(Run.user_id == user.id, Run.status.in_(db_status))
    total = (await session.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    order = Run.created_at.asc() if sort_order == "asc" else Run.created_at.desc()
    rows = (await session.execute(q.order_by(order).offset(max(offset, 0)).limit(min(max(limit, 1), 500)))).scalars()
    items = [_job(r) for r in rows]
    if workflow_id:
        items = [j for j in items if j.get("workflow_id") == workflow_id]
    return {"jobs": items, "pagination": {"offset": offset, "limit": limit, "total": total,
                                          "has_more": offset + len(items) < total}}


@router.get("/api/jobs/{job_id}")
async def job(job_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    run = await session.get(Run, job_id)
    if run is None or run.user_id != user.id:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return _job(run, full=True)


@router.get("/api/history")
async def history(max_items: int | None = None, offset: int = -1, user: User = Depends(current_user),
                  session: AsyncSession = Depends(get_session)):
    q = (select(Run).where(Run.user_id == user.id, Run.status.in_(runs.FINAL))
         .order_by(Run.created_at.desc()).limit(min(max_items or 200, 500)))
    if offset > 0:
        q = q.offset(offset)
    return {r.id: _history_item(r) for r in (await session.execute(q)).scalars()}


@router.get("/api/history/{prompt_id}")
async def history_one(prompt_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    run = await session.get(Run, prompt_id)
    if run is None or run.user_id != user.id or run.status not in runs.FINAL:
        return {}
    return {run.id: _history_item(run)}


@router.post("/api/history")
async def post_history(request: Request, user: User = Depends(current_user),
                       session: AsyncSession = Depends(get_session)):
    """Delete finished runs and the files they made (the results gallery is built from runs: deleting a picture there
    deletes its run). Uploads stay."""
    body = await request.json()
    q = select(Run).where(Run.user_id == user.id, Run.status.in_(runs.FINAL))
    if body.get("clear"):
        doomed = list((await session.execute(q)).scalars())
    elif body.get("delete"):
        doomed = list((await session.execute(q.where(Run.id.in_([str(x) for x in body["delete"]])))).scalars())
    else:
        doomed = []
    if doomed:
        await files.delete(session, await _files_of(session, user.id, doomed))
        await session.execute(delete(Run).where(Run.id.in_([r.id for r in doomed])))
        await session.commit()
    return Response(status_code=200)


async def _files_of(session: AsyncSession, user_id: str, doomed: list[Run]) -> list[StoredFile]:
    """Results of these runs: recorded with the run, or (runs before that) named in what the run made itself."""
    ids = [r.id for r in doomed]
    rows = {f.id: f for f in (await session.execute(select(StoredFile).where(
        StoredFile.user_id == user_id, StoredFile.run_id.in_(ids)))).scalars()}
    for run in doomed:
        for out in _new_outputs(run).values():
            for items in (out or {}).values():
                for item in items if isinstance(items, list) else []:
                    if isinstance(item, dict) and item.get("type") == "output" and item.get("filename"):
                        f = await files.get(session, user_id, "output", item["filename"])
                        if f is not None and f.run_id in (None, run.id):
                            rows[f.id] = f
    return list(rows.values())


# ---------------------------------------------------------------- the WebSocket


@router.websocket("/ws")
async def ws(socket: WebSocket, clientId: str | None = None):
    origin = socket.headers.get("origin")
    if origin and urllib.parse.urlsplit(origin).netloc != socket.headers.get("host"):
        await socket.close(code=4403)
        return
    user_id = auth.user_id(socket)
    async with sessionmaker()() as s:
        remaining = await runs.queue_remaining(s, user_id)
    import secrets
    sid = clientId or secrets.token_hex(16)
    await socket.accept()
    await socket.send_json({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": remaining}},
                                                       "sid": sid}})
    await socket.send_json({"type": "feature_flags", "data": {"supports_preview_metadata": False,
                                                              "max_upload_size": files.MAX_UPLOAD}})

    async def receive():
        with contextlib.suppress(WebSocketDisconnect):
            while True:
                await socket.receive_text()  # feature flags from the client; nothing to do with them

    reader = asyncio.create_task(receive())
    with bus.subscribe(user_id) as q:
        try:
            while not reader.done():
                get = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({get, reader}, timeout=25, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    if reader.done():
                        break
                    continue
                msg = get.result()
                if msg.get("sid") not in (None, sid):
                    continue
                if msg["type"] == "__text":
                    await socket.send_bytes(runs.encode_text(msg["data"]["node"], msg["data"]["text"]))
                else:
                    await socket.send_json({"type": msg["type"], "data": msg["data"]})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()


# ---------------------------------------------------------------- the frontend's own files (last: catch-all)


@router.get("/{path:path}")
async def frontend_file(path: str):
    root = frontend_dir().resolve()
    p = (root / path).resolve()
    if root not in p.parents or not p.is_file():
        return Response(status_code=404)
    return _static(p, immutable=path.startswith("assets/"))
