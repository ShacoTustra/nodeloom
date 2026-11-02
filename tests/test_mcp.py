"""MCP for agents outside the editor: the protocol at /mcp, the tools (catalog, check, run, follow, upload), the editor
tools answered by a stand-in editor, and the stdio bridge."""

import asyncio
import base64
import io
import json

import pytest

from nodeloom import bus, mcp
from nodeloom.auth import LOCAL_USER
from tests.conftest import ARK, PNG, SAME, image_reply, run_all

JSON = {"Content-Type": "application/json"}


async def rpc(env, method, params=None, mid=1, headers=None):
    msg = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {}),
           **({"id": mid} if mid is not None else {})}
    return await env.client.post("/mcp", content=json.dumps(msg), headers={**JSON, **(headers or {})})


async def tool(env, tool_name, **args):
    r = await rpc(env, "tools/call", {"name": tool_name, "arguments": args})
    assert r.status_code == 200, r.text
    return r.json()["result"]


def picture_graph(name):
    return [{"id": "1", "type": "LoomLoadImage", "values": {"image": name}},
            {"id": "2", "type": "LoomSeedreamImage", "values": {"model": "ark/dola-seedream-5-0-pro",
                                                                "prompt": "the cat at night"},
             "links": {"references": ["1.0"]}}]


async def test_the_protocol(env):
    r = await rpc(env, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "1"}})
    init = r.json()["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "nodeloom"
    assert init["capabilities"] == {"tools": {"listChanged": False}} and "list_nodes" in init["instructions"]
    assert (await rpc(env, "initialize", {"protocolVersion": "1999-01-01"})).json()["result"]["protocolVersion"] \
        == mcp.PROTOCOL_VERSIONS[0]
    assert (await rpc(env, "notifications/initialized", mid=None)).status_code == 202
    assert (await rpc(env, "ping")).json() == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert (await rpc(env, "resources/list")).json()["error"]["code"] == -32601
    tools = (await rpc(env, "tools/list")).json()["result"]["tools"]
    assert {t["name"] for t in tools} == {"list_nodes", "check_graph", "run_graph", "get_run", "cancel_run",
                                         "list_runs", "upload_file", "open_in_editor", "get_editor_graph"}
    assert all(t["inputSchema"]["type"] == "object" for t in tools)
    unknown = await rpc(env, "tools/call", {"name": "rm_rf", "arguments": {}})
    assert unknown.json()["error"]["code"] == -32602
    batch = await env.client.post("/mcp", headers=JSON, content=json.dumps(
        [{"jsonrpc": "2.0", "id": 7, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/x"}]))
    assert batch.json() == [{"jsonrpc": "2.0", "id": 7, "result": {}}]


async def test_other_sites_and_other_bodies_are_refused(env):
    assert (await rpc(env, "ping", headers={"Sec-Fetch-Site": "cross-site"})).status_code == 403
    assert (await rpc(env, "ping", headers={"Origin": "https://evil.example"})).status_code == 403
    r = await env.client.post("/mcp", content='{"jsonrpc": "2.0", "id": 1, "method": "ping"}',
                              headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    assert (await env.client.post("/mcp", content="{", headers=JSON)).json()["error"]["code"] == -32700
    assert (await env.client.get("/mcp")).status_code == 405


async def test_catalog_and_check(env):
    text = (await tool(env, "list_nodes"))["content"][0]["text"]
    assert "LoomSeedreamImage" in text and "ark/dola-seedream-5-0-pro" in text
    assert "the name of an uploaded file (upload_file adds one); uploaded now: none" in text
    bad = await tool(env, "check_graph", nodes=[{"id": "1", "type": "LoomSeedreamImage",
                                                 "values": {"model": "ark/nope", "prompt": "x"}}])
    assert bad["structuredContent"]["ok"] is False and "ark/nope" in json.dumps(bad["structuredContent"]["errors"])
    empty = await tool(env, "check_graph", nodes=picture_graph(""))
    assert empty["structuredContent"]["notes"] == ['node 1 “Image”: no file chosen: upload_file, then set its name']
    assert (await tool(env, "check_graph", nodes="nope"))["isError"]


async def test_upload_run_and_follow(env):
    up = await tool(env, "upload_file", name="cat.png", data_base64=base64.b64encode(PNG).decode())
    name = up["structuredContent"]["name"]
    assert name == "cat.png" and up["structuredContent"]["media"] == "image"
    made = env.mock.post(f"{ARK}/images/generations").respond(json=image_reply())
    started = await tool(env, "run_graph", nodes=picture_graph(name), wait_seconds=0)
    run = started["structuredContent"]
    assert run["status"] == "queued" and "get_run" in run["note"]
    await run_all(env)
    done = await tool(env, "get_run", run_id=run["run_id"], wait_seconds=5)
    out = done["structuredContent"]
    assert out["status"] == "completed" and out["cost_usd"] == 0.04 and out["cost_complete"]
    pic = next(o for o in out["outputs"] if o["node"] == "2")
    assert pic["type"] == "LoomSeedreamImage" and pic["files"][0]["media"] == "image"
    assert pic["files"][0]["url"].startswith("http://localhost/api/view?filename=")
    view = await env.client.get(pic["files"][0]["url"])
    assert view.status_code == 200 and view.content == PNG
    assert [c["type"] for c in done["content"]] == ["text", "image"]  # the result, as a preview
    body = json.loads(made.calls[0].request.content)
    assert body["prompt"] == "the cat at night" and body["image"].startswith("data:image/png;base64,")
    # the same graph again: everything reused, nothing paid
    again = (await tool(env, "run_graph", nodes=picture_graph(name), wait_seconds=0))["structuredContent"]
    await run_all(env)
    assert (await tool(env, "get_run", run_id=again["run_id"]))["structuredContent"]["status"] == "completed"
    assert made.call_count == 1
    runs = (await tool(env, "list_runs", limit=5))["structuredContent"]["runs"]
    assert [r["run_id"] for r in runs] == [again["run_id"], run["run_id"]]
    assert (await tool(env, "cancel_run", run_id=run["run_id"]))["isError"]  # finished
    assert (await tool(env, "get_run", run_id="nope"))["isError"]


async def test_runs_refuse_what_cannot_run(env):
    refused = await tool(env, "run_graph", nodes=picture_graph(""))
    assert refused["isError"] and "upload_file" in refused["content"][0]["text"]
    private = await tool(env, "upload_file", url="http://127.0.0.1:8188/api/view?filename=x")
    assert private["isError"] and "public" in private["content"][0]["text"]
    assert (await tool(env, "upload_file", name="x.png"))["isError"]


async def test_the_editor_tools(env, monkeypatch):
    monkeypatch.setattr(mcp, "EDITOR_WAIT", 0.3)
    nobody = await tool(env, "open_in_editor", nodes=picture_graph(""))
    assert nobody["isError"] and "No editor is open" in nobody["content"][0]["text"]
    monkeypatch.setattr(mcp, "EDITOR_WAIT", 5.0)
    seen = []

    async def editor():
        """What nodeloom.js does: take the event, answer at /api/nodeloom/editor/reply/{request}."""
        with bus.subscribe(LOCAL_USER) as q:
            while True:
                msg = await q.get()
                if msg["type"] == "nodeloom.graph":
                    seen.append(msg["data"])
                    answer = {"ok": True, "tab": "cats.json"}
                elif msg["type"] == "nodeloom.ask_graph":
                    answer = {"tab": "cats.json", "graph": {
                        "5": {"class_type": "LoomText", "inputs": {"value": "hi"}},
                        "6": {"class_type": "LoomShow", "inputs": {"value": ["5", 0]}}}}
                else:
                    continue
                r = await env.client.post(f"/api/nodeloom/editor/reply/{msg['data']['request']}", json=answer,
                                          headers=SAME)
                assert r.json() == {"taken": True}

    task = asyncio.create_task(editor())
    await asyncio.sleep(0)
    try:
        opened = await tool(env, "open_in_editor", nodes=picture_graph(""), name="cats")
        assert opened["structuredContent"]["opened"] is True and opened["structuredContent"]["tab"] == "cats.json"
        assert seen[0]["name"] == "cats" and seen[0]["prompt"]["2"]["class_type"] == "LoomSeedreamImage"
        assert seen[0]["layout"]["2"][0] > seen[0]["layout"]["1"][0]  # the picture's source on its left
        got = (await tool(env, "get_editor_graph"))["structuredContent"]
        assert got == {"tab": "cats.json", "nodes": [
            {"id": "5", "type": "LoomText", "values": {"value": "hi"}},
            {"id": "6", "type": "LoomShow", "values": {}, "links": {"value": "5.0"}}]}
    finally:
        task.cancel()
    # an answer for nobody, or from another site, is not taken
    assert (await env.client.post("/api/nodeloom/editor/reply/zzz", json={}, headers=SAME)).json() == {"taken": False}
    assert (await env.client.post("/api/nodeloom/editor/reply/zzz", json={},
                                  headers={"Sec-Fetch-Site": "cross-site"})).status_code == 403


def test_the_stdio_bridge(monkeypatch, capsys):
    import respx

    from nodeloom import cli
    lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}), "not json"]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(lines) + "\n"))
    with respx.mock:
        respx.post("http://127.0.0.1:9999/mcp").mock(side_effect=lambda req: (
            __import__("httpx").Response(202) if b'"id"' not in req.content
            else __import__("httpx").Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})))
        cli.main(["mcp", "--url", "http://127.0.0.1:9999"])
    out = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert {"jsonrpc": "2.0", "id": 1, "result": {}} in out
    assert {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}} in out
    assert len(out) == 2  # the notification has no answer


def test_the_bridge_says_when_the_server_is_down(monkeypatch, capsys):
    from nodeloom import cli
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}) + "\n"))
    cli.main(["mcp", "--url", "http://127.0.0.1:9"])
    answer = json.loads(capsys.readouterr().out)
    assert answer["id"] == 3 and "is not running" in answer["error"]["message"]


@pytest.fixture(autouse=True)
def _english():
    from nodeloom import i18n
    i18n.use("en")
