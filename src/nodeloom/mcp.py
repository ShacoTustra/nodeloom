"""MCP: the nodes for agents outside the editor (Claude Code, Cursor, Claude Desktop, any MCP client).

The server speaks MCP's Streamable HTTP at /mcp: JSON-RPC over POST, each answer a JSON body (no server-sent stream),
no session; it acts as the one local user. An agent reads the catalog of nodes and models, checks and runs graphs in
the Builder's form (nodes with values and links, agent.py), follows runs, uploads input files, puts a graph into the
open editor (a new tab) and reads the one on its canvas. Runs made here are in the editor's queue and results like
any other, and they reuse earlier results the same way.

The editor tools talk to the browser through the run events (bus.py): the editor answers at
/api/nodeloom/editor/reply/{request} (nodeloom.js). That answer must reach this process, as it does with one process.

`nodeloom mcp` (cli.py) is the same over stdio for clients that only start commands: it forwards to a running server.
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import json
import logging
import mimetypes
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select

from nodeloom import __version__, agent, bus, engine, files, i18n, runs
from nodeloom.api import _same_origin, current_user
from nodeloom.auth import LOCAL_USER, User
from nodeloom.config import APP_NAME
from nodeloom.db import sessionmaker
from nodeloom.models import Run

log = logging.getLogger(__name__)

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")  # newest first
WAIT_DEFAULT, WAIT_MAX = 50, 300  # seconds a call may wait for a run (clients give up on a call after a minute or so)
EDITOR_WAIT = 5.0
PREVIEW_SIDE, MAX_PREVIEWS = 768, 4
STATUS = {"pending": "queued", "running": "running", "cancelling": "running", "success": "completed",
          "error": "failed", "cancelled": "cancelled"}
DONE = ("success", "error", "cancelled")

INSTRUCTIONS = f"""{APP_NAME} runs graphs of nodes that call AI models behind APIs: text, pictures, video, sound.
Call list_nodes first: it lists the nodes, their settings and the models this user has. A graph is a list of nodes
{{"id": "1", "type": "LoomTextModel", "values": {{...}}, "links": {{"prompt": "2.0"}}}}, where a link is
"<node id>.<output index>". check_graph checks one for free; run_graph runs it (paid models cost money) and waits up
to wait_seconds: a video takes minutes, so follow it with get_run. Results are files at URLs on this server; pictures
also come back as previews. A run reuses every result whose node and inputs did not change: running a graph again
costs nothing for what is the same. open_in_editor puts a graph on the user's canvas; get_editor_graph reads it."""

GRAPH_SCHEMA = {"type": "array", "description": "The graph: its nodes", "items": {
    "type": "object", "required": ["id", "type"], "properties": {
        "id": {"type": "string"},
        "type": {"type": "string", "description": "A node type from list_nodes, e.g. LoomTextModel"},
        "title": {"type": "string", "description": "A title on the canvas (open_in_editor)"},
        "values": {"type": "object", "description": "Settings by input name"},
        "links": {"type": "object", "description": "Link inputs: \"<node id>.<output index>\", or a list of them "
                                                   "for an input that takes several"}}}}

router = APIRouter()


class ToolError(Exception):
    """A tool's failure, told to the agent (isError) rather than as a protocol error."""


@dataclass
class Call:
    base_url: str  # this server as the client reached it: links to result files


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    run: Callable[[Call, dict], Awaitable[dict]]
    read_only: bool = False
    open_world: bool = False  # it calls paid models on the internet

    def listed(self) -> dict:
        return {"name": self.name, "description": self.description,
                "inputSchema": {"type": "object", **self.schema},
                "annotations": {"readOnlyHint": self.read_only, "destructiveHint": False,
                                "openWorldHint": self.open_world}}


# ---------------------------------------------------------------- the endpoint


@router.post("/mcp")
async def endpoint(request: Request) -> Response:
    # A page of another site cannot get here: the Host check (app.HostCheck), a browser's cross-site request refused
    # by its Sec-Fetch-Site / Origin, and JSON only (a cross-site form cannot send it, a cross-site fetch with it
    # needs a CORS preflight, which this server never allows).
    if not _same_origin(request):
        return JSONResponse(_error(None, -32600, "Requests from other sites are refused"), status_code=403)
    if not request.headers.get("content-type", "").split(";")[0].strip().lower() == "application/json":
        return JSONResponse(_error(None, -32600, "Send JSON (Content-Type: application/json)"), status_code=415)
    try:
        body = json.loads(await request.body())
    except ValueError:
        return JSONResponse(_error(None, -32700, "Parse error"), status_code=400)
    i18n.use("en")
    call = Call(str(request.base_url).rstrip("/"))
    messages = body if isinstance(body, list) else [body]
    answers = [a for a in [await handle(call, m) for m in messages] if a is not None]
    if not answers:
        return Response(status_code=202)  # notifications and responses only
    return JSONResponse(answers if isinstance(body, list) else answers[0])


@router.get("/mcp")
@router.delete("/mcp")
async def no_stream() -> Response:
    """No server-sent stream and no sessions to end."""
    return Response(status_code=405, headers={"Allow": "POST"})


def _result(mid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


async def handle(call: Call, msg) -> dict | None:
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid request")
    method, mid = msg.get("method"), msg.get("id")
    if method is None or mid is None:
        return None  # a notification (initialized, cancelled) or an answer to us: nothing to say
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        return _result(mid, {
            "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "nodeloom", "title": APP_NAME, "version": __version__},
            "instructions": INSTRUCTIONS})
    if method == "ping":
        return _result(mid, {})
    if method == "tools/list":
        return _result(mid, {"tools": [t.listed() for t in TOOLS.values()]})
    if method == "tools/call":
        tool = TOOLS.get(str(params.get("name")))
        if tool is None:
            return _error(mid, -32602, f"Unknown tool: {params.get('name')}")
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            return _result(mid, await tool.run(call, args))
        except ToolError as e:
            return _result(mid, {"content": [_text(str(e))], "isError": True})
        except Exception:  # noqa: BLE001  a bug: the agent hears of it, the log has the details
            log.exception("mcp: %s failed", tool.name)
            return _result(mid, {"content": [_text(f"{tool.name} failed: an error in {APP_NAME} (see its log)")],
                                 "isError": True})
    return _error(mid, -32601, f"Method not found: {method}")


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _answer(data: dict, *extra: dict) -> dict:
    return {"content": [_text(json.dumps(data, ensure_ascii=False, indent=1)), *extra], "structuredContent": data}


# ---------------------------------------------------------------- graphs


async def _choices() -> dict[str, list]:
    async with sessionmaker()() as s:
        return await runs.choices(s, LOCAL_USER)


async def _graph(args: dict) -> tuple[dict, dict, dict[str, list], dict[str, dict], list[str], list[str]]:
    """The agent's node list → the API prompt, titles, the lists, the models' abilities, errors and notes."""
    nodes = args.get("nodes")
    if not isinstance(nodes, list):
        raise ToolError("Give the graph as `nodes`: a list of {id, type, values, links}")
    ch = await _choices()
    caps = await agent.model_caps()
    prompt, titles, problems = agent.to_prompt(nodes)
    errors, notes = agent.check(prompt, ch, caps) if prompt and not problems else ([], [])
    notes = [n.replace("the user picks the file", "no file chosen: upload_file, then set its name") for n in notes]
    return prompt, titles, ch, caps, problems + errors, notes


async def list_nodes(call: Call, args: dict) -> dict:
    ch = await _choices()
    text = agent.catalog(ch, await agent.model_caps(), uploads=True)
    return {"content": [_text("The nodes (only these exist):\n" + text)]}


async def check_graph(call: Call, args: dict) -> dict:
    _, _, _, _, errors, notes = await _graph(args)
    return _answer({"ok": not errors and not notes, "errors": errors, "notes": notes})


async def run_graph(call: Call, args: dict) -> dict:
    prompt, _, _, _, errors, notes = await _graph(args)
    if errors or notes:
        raise ToolError("The graph cannot run:\n- " + "\n- ".join(errors + notes))
    async with sessionmaker()() as s:
        try:
            run = await runs.submit(s, LOCAL_USER, prompt=prompt, extra={"loom_source": "mcp"}, client_id=None,
                                    targets=None)
        except engine.PromptInvalid as e:
            raise ToolError(f"The graph cannot run: {e.body.get('error', {}).get('message')}: "
                            f"{json.dumps(e.body.get('node_errors') or {}, ensure_ascii=False)}") from None
        except runs.QueueFull:
            raise ToolError("Too many runs are queued: wait for some to finish") from None
    return await _report(call, run.id, _wait(args))


async def get_run(call: Call, args: dict) -> dict:
    return await _report(call, str(args.get("run_id") or ""), _wait(args, 0))


async def cancel_run(call: Call, args: dict) -> dict:
    run_id = str(args.get("run_id") or "")
    async with sessionmaker()() as s:
        touched = await runs.cancel(s, LOCAL_USER, [run_id])
    if not touched:
        raise ToolError(f"No queued or running run {run_id}")
    return _answer({"run_id": run_id, "cancelled": True})


async def list_runs(call: Call, args: dict) -> dict:
    limit = max(1, min(int(args.get("limit") or 10), 50))
    async with sessionmaker()() as s:
        rows = (await s.execute(select(Run).where(Run.user_id == LOCAL_USER).order_by(Run.created_at.desc())
                                .limit(limit))).scalars().all()
    return _answer({"runs": [{"run_id": r.id, "status": STATUS.get(r.status, r.status), "nodes": len(r.prompt or {}),
                              "created": r.created_at.isoformat() if r.created_at else None, **_cost(r)}
                             for r in rows]})


def _wait(args: dict, default: int = WAIT_DEFAULT) -> float:
    try:
        return max(0.0, min(float(args.get("wait_seconds", default)), WAIT_MAX))
    except (TypeError, ValueError):
        return float(default)


def _cost(run: Run) -> dict:
    extra = run.extra or {}
    if "loom_cost_micro" not in extra:
        return {}
    return {"cost_usd": round(extra["loom_cost_micro"] / 1_000_000, 4), "cost_complete": not extra.get("loom_unpriced")}


async def _report(call: Call, run_id: str, wait: float) -> dict:
    """The run's state once it is done or `wait` seconds have passed: outputs (files as links, pictures as
    previews), the error, the cost."""
    deadline = time.monotonic() + wait
    while True:
        async with sessionmaker()() as s:
            run = await s.get(Run, run_id)
            if run is None or run.user_id != LOCAL_USER:
                raise ToolError(f"No run {run_id}")
            if run.status in DONE or time.monotonic() >= deadline:
                return await _describe(call, s, run)
        await asyncio.sleep(0.5)


async def _describe(call: Call, s, run: Run) -> dict:
    data: dict = {"run_id": run.id, "status": STATUS.get(run.status, run.status), **_cost(run)}
    if run.status not in DONE:
        data["note"] = "Not finished yet: call get_run with this run_id (and wait_seconds) to follow it"
    if run.status == "error" and run.error:
        e = run.error
        data["error"] = {"node": e.get("node_id"), "type": e.get("node_type"), "message": e.get("exception_message")}
    outputs, previews = [], []
    for nid, ui in (run.outputs or {}).items():
        item: dict = {"node": nid, "type": (run.prompt.get(nid) or {}).get("class_type")}
        if ui.get("text"):
            item["text"] = [str(x) for x in ui["text"]]
        found = []
        for media in ("images", "audio"):
            for f in ui.get(media) or []:
                if not isinstance(f, dict) or f.get("type") not in files.KINDS:
                    continue
                stored = await files.get(s, LOCAL_USER, f["type"], f["filename"])
                if stored is None:
                    continue
                kind = files.media_of(stored.content_type)
                url = str(httpx.URL(f"{call.base_url}/api/view", params={"filename": f["filename"], "type": f["type"]}))
                meta = {k: v for k, v in (stored.meta or {}).items() if k in ("w", "h")}
                found.append({"name": f["filename"], "media": kind, "url": url, **meta})
                if kind == "image" and len(previews) < MAX_PREVIEWS:
                    previews.append(await _preview(files.ref(stored)))
        if found:
            item["files"] = found
        outputs.append(item)
    data["outputs"] = outputs
    return _answer(data, *[p for p in previews if p])


async def _preview(ref: dict) -> dict | None:
    """A picture small enough for the agent to look at."""
    from PIL import Image

    try:
        raw = await files.read(LOCAL_USER, ref)
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            im.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
        return {"type": "image", "data": base64.b64encode(buf.getvalue()).decode(), "mimeType": "image/jpeg"}
    except Exception:  # noqa: BLE001  no preview: the link is enough
        return None


# ---------------------------------------------------------------- files


async def upload_file(call: Call, args: dict) -> dict:
    name = str(args.get("name") or "").strip()
    if args.get("data_base64"):
        try:
            data = base64.b64decode(str(args["data_base64"]), validate=True)
        except ValueError:
            raise ToolError("data_base64 is not base64") from None
        ct = mimetypes.guess_type(name)[0] or ""
    elif args.get("url"):
        data, ct = await _fetch(str(args["url"]))
        name = name or httpx.URL(str(args["url"])).path.rsplit("/", 1)[-1]
    else:
        raise ToolError("Give the file as data_base64 or as a url")
    if len(data) > files.MAX_UPLOAD:
        raise ToolError(f"The file is over {files.MAX_UPLOAD >> 20} MB")
    try:
        ct = files.check_upload(data, ct)
        async with sessionmaker()() as s:
            await files.check_room(s, LOCAL_USER, len(data))
            meta = files.image_meta(data) if files.media_of(ct) == "image" else None
            f = await files.save(s, LOCAL_USER, "input", data, ct, name=name or None, meta=meta)
            await s.commit()
    except files.FileError as e:
        raise ToolError(str(e)) from None
    return _answer({"name": f.name, "media": files.media_of(ct),
                    "use": "set it as the file of a node: “Image” image, “Video file” file or “Audio file” audio"})


async def _fetch(url: str) -> tuple[bytes, str]:
    """A file from a public address (never this machine or its network), redirects checked hop by hop."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10), follow_redirects=False) as c:
        for _ in range(4):
            u = httpx.URL(url)
            if u.scheme not in ("http", "https") or not u.host or not await _public(u.host, u.port):
                raise ToolError("Only files at public http(s) addresses can be fetched")
            try:
                r = await c.get(url)
            except httpx.HTTPError as e:
                raise ToolError(f"The file was not downloaded: {type(e).__name__}") from None
            if r.is_redirect and r.headers.get("location"):
                url = str(r.url.join(r.headers["location"]))
                continue
            if r.status_code != 200:
                raise ToolError(f"The file was not downloaded: HTTP {r.status_code}")
            return r.content, (r.headers.get("content-type") or "").split(";")[0].strip()
    raise ToolError("The file was not downloaded: too many redirects")


async def _public(host: str, port: int | None) -> bool:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port or 443)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


# ---------------------------------------------------------------- the editor


_waiting: dict[str, asyncio.Future] = {}


async def _ask_editor(event: str, data: dict) -> dict:
    """Send the open editor an event and wait for its answer (nodeloom.js posts it to editor_reply)."""
    request = secrets.token_hex(8)
    fut = asyncio.get_running_loop().create_future()
    _waiting[request] = fut
    try:
        await bus.publish(LOCAL_USER, event, {**data, "request": request})
        return await asyncio.wait_for(fut, EDITOR_WAIT)
    except TimeoutError:
        raise ToolError(f"No editor is open: open {APP_NAME} in a browser and call again") from None
    finally:
        _waiting.pop(request, None)


def editor_reply(request: str, body: dict) -> bool:
    fut = _waiting.get(request)
    if fut is None or fut.done():
        return False
    fut.set_result(body)
    return True


async def open_in_editor(call: Call, args: dict) -> dict:
    prompt, titles, _, _, errors, notes = await _graph(args)
    if errors:
        raise ToolError("Fix the graph first:\n- " + "\n- ".join(errors))
    name = str(args.get("name") or "").strip()[:80]
    answer = await _ask_editor("nodeloom.graph", {"prompt": prompt, "titles": titles, "layout": agent.layout(prompt),
                                                  "name": name})
    if not answer.get("ok"):
        raise ToolError(f"The editor did not take the graph: {answer.get('error') or 'unknown error'}")
    return _answer({"opened": True, "nodes": len(prompt), "tab": answer.get("tab") or name or None,
                    **({"notes": notes} if notes else {})})


async def get_editor_graph(call: Call, args: dict) -> dict:
    answer = await _ask_editor("nodeloom.ask_graph", {})
    graph = answer.get("graph") if isinstance(answer.get("graph"), dict) else {}
    return _answer({"tab": answer.get("tab"), "nodes": agent.compact(graph)})


# ---------------------------------------------------------------- the tools

_NODES = {"properties": {"nodes": GRAPH_SCHEMA}, "required": ["nodes"]}
_RUN = {"run_id": {"type": "string"}}
_WAIT = {"wait_seconds": {"type": "number", "minimum": 0, "maximum": WAIT_MAX,
                          "description": f"How long to wait for the run to finish (default {WAIT_DEFAULT})"}}

TOOLS = {t.name: t for t in [
    Tool("list_nodes", "The nodes that exist, their settings and links, and the models this user has for them. Read "
         "it before making a graph.", {"properties": {}}, list_nodes, read_only=True),
    Tool("check_graph", "Check a graph without running it: the errors a run would stop at.", _NODES, check_graph,
         read_only=True),
    Tool("run_graph", "Run a graph (paid models cost money) and wait for it up to wait_seconds. Returns the outputs: "
         "texts, files as links, pictures as previews, and the cost. Nodes whose inputs did not change since an "
         "earlier run reuse its results for free.",
         {"properties": {"nodes": GRAPH_SCHEMA, **_WAIT}, "required": ["nodes"]}, run_graph, open_world=True),
    Tool("get_run", "A run's state and outputs; with wait_seconds, waits for it to finish first.",
         {"properties": {**_RUN, **_WAIT}, "required": ["run_id"]}, get_run, read_only=True),
    Tool("cancel_run", "Stop a queued or running run.", {"properties": _RUN, "required": ["run_id"]}, cancel_run),
    Tool("list_runs", "The latest runs, newest first.",
         {"properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}}, list_runs, read_only=True),
    Tool("upload_file", "Add a picture, a video or a sound for the file nodes (“Image”, “Video file”, “Audio "
         "file”): as base64 or from a public URL. Returns the name to set.",
         {"properties": {"name": {"type": "string", "description": "A file name with its extension"},
                         "data_base64": {"type": "string"}, "url": {"type": "string"}}}, upload_file),
    Tool("open_in_editor", "Put a graph on the user's canvas, in a new tab of the open editor, for them to see, "
         "change and run.",
         {"properties": {"nodes": GRAPH_SCHEMA, "name": {"type": "string", "description": "The tab's name"}},
          "required": ["nodes"]}, open_in_editor),
    Tool("get_editor_graph", "The graph on the user's canvas now (the open editor's active tab), in the same form "
         "as run_graph takes.", {"properties": {}}, get_editor_graph, read_only=True),
]}


@router.post("/api/nodeloom/editor/reply/{request}")
async def reply(request: str, body: dict, user: User = Depends(current_user)) -> dict:
    """The open editor's answer to an agent's request (nodeloom.js); the editor's own pages only, as every editor
    request."""
    return {"taken": editor_reply(request, body)}
