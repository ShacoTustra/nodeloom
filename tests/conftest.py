"""The app in process with its own SQLite and media directory, two providers from a config file and their APIs mocked
(respx): an ark provider (BytePlus ModelArk) and an openai one. Runs are driven by the tests (`run_all`), not by the
dispatcher's loop."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import httpx
import pytest
import respx

from nodeloom import config, db, media_store, nodes, runs

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
ARK = "https://ark.test/api/v3"
OPENAI = "https://openai.test/v1"
TOS = "https://ark-out.tos-ap-southeast-1.volces.com"
SAME = {"Sec-Fetch-Site": "same-origin"}

CONFIG = f"""
[providers.ark]
type = "ark"
base_url = "{ARK}"
api_key = "ark-key"

[providers.ark.models."seed-2-0-lite"]
kind = "chat"
inputs = ["image", "audio", "video"]
price = {{ input = 0.25, output = 2.0 }}

[providers.ark.models."deepseek-v4-flash"]
kind = "chat"
inputs = []

[providers.ark.models."dola-seedream-5-0-pro"]
kind = "image"
price = {{ image = 0.04 }}

[providers.ark.models."dreamina-seedance-2-0"]
kind = "video"
price = {{ output = 7.0 }}

[providers.ark.models."dreamina-seedance-2-5"]
kind = "video"

[providers.openai]
type = "openai"
base_url = "{OPENAI}"
api_key = "sk-test"

[providers.openai.models."gpt-5-mini"]
inputs = ["image"]

[providers.openai.models."gpt-image-1"]
kind = "image"
price = {{ input = 5.0, image_input = 10.0, output = 40.0 }}
"""


@dataclass
class Env:
    client: httpx.AsyncClient
    mock: respx.MockRouter


def configure(monkeypatch, tmp_path, config_text: str | None = CONFIG, **env) -> None:
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ARK_API_KEY", "LANGUAGE", "AGENT_MODEL", "STORAGE_LIMIT_MB"):
        monkeypatch.delenv(name, raising=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    if config_text is not None:
        (tmp_path / "config.toml").write_text(config_text)
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.toml"))
    else:
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "none.toml"))
    for k, v in env.items():
        monkeypatch.setenv(k.upper(), str(v))
    config.get_settings.cache_clear()
    media_store._store.cache_clear()


async def start(monkeypatch, tmp_path, config_text: str | None = CONFIG, **env):
    configure(monkeypatch, tmp_path, config_text, **env)
    await db.dispose()
    await db.init_db()
    monkeypatch.setattr(nodes, "POLL_SECONDS", 0.01)
    runs.dispatcher.stopping = False
    from nodeloom.app import create_app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://localhost")


@pytest.fixture
async def env(monkeypatch, tmp_path):
    client = await start(monkeypatch, tmp_path)
    with respx.mock(assert_all_called=False) as mock:
        mock.route(host="localhost").pass_through()
        yield Env(client, mock)
    for t in list(runs.dispatcher.tasks.values()):
        t.cancel()
    await asyncio.gather(*runs.dispatcher.tasks.values(), return_exceptions=True)
    runs.dispatcher.tasks.clear()
    await client.aclose()
    await db.dispose()


async def run_all(env: Env) -> None:
    """Claim the queued runs and wait for them, as a process's dispatcher would."""
    for _ in range(5):
        await runs.dispatcher.claim()
        tasks = list(runs.dispatcher.tasks.values())
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


async def upload(env: Env, name="frame.png", data=PNG, ct="image/png") -> str:
    r = await env.client.post("/api/upload/image", headers=SAME, files={"image": (name, data, ct)},
                              data={"type": "input"})
    assert r.status_code == 200, r.text
    return r.json()["name"]


async def queue(env: Env, prompt: dict, **kw) -> httpx.Response:
    return await env.client.post("/api/prompt", headers=SAME, json={"prompt": prompt, "client_id": "tab1", **kw})


async def job(env: Env, run_id: str) -> dict:
    return (await env.client.get(f"/api/jobs/{run_id}")).json()


async def error_of(env: Env, graph: dict) -> str:
    """Queue a graph, run it and say how it ended: its error message, or its status."""
    r = await queue(env, graph)
    assert r.status_code == 200, r.text
    await run_all(env)
    j = await job(env, r.json()["prompt_id"])
    return (j.get("execution_error") or {}).get("exception_message", j["status"])


def image_reply(n=1):
    return {"model": "dola-seedream-5-0-pro-260628", "created": 1,
            "data": [{"b64_json": base64.b64encode(PNG).decode(), "size": "2048x2048"} for _ in range(n)],
            "usage": {"generated_images": n, "output_tokens": 16384 * n, "total_tokens": 16384 * n}}


def chat_reply(content="ok", usage=None):
    return {"id": "c", "object": "chat.completion", "model": "seed-2-0-lite-260428",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}


def video_mocks(env: Env, task_id="cgt-1"):
    create = env.mock.post(f"{ARK}/contents/generations/tasks").respond(json={"id": task_id})
    env.mock.get(f"{ARK}/contents/generations/tasks/{task_id}").mock(side_effect=[
        httpx.Response(200, json={"id": task_id, "model": "dreamina-seedance-2-0-260128", "status": "running"}),
        httpx.Response(200, json={"id": task_id, "model": "dreamina-seedance-2-0-260128", "status": "succeeded",
                                  "resolution": "720p", "content": {"video_url": f"{TOS}/v.mp4",
                                                                    "last_frame_url": f"{TOS}/last.png"},
                                  "usage": {"completion_tokens": 1000, "total_tokens": 1000}}),
    ])
    env.mock.get(f"{TOS}/v.mp4").respond(content=b"\x00\x00\x00\x18ftypmp42", headers={"content-type": "video/mp4"})
    env.mock.get(f"{TOS}/last.png").respond(content=PNG, headers={"content-type": "image/png"})
    return create


IMAGE_GRAPH = {
    "1": {"class_type": "LoomText", "inputs": {"value": "a cat in a hat"}},
    "2": {"class_type": "LoomSeedreamImage", "inputs": {"model": "ark/dola-seedream-5-0-pro", "prompt": ["1", 0],
                                                        "size": "2K", "count": 1, "seed": 0, "watermark": False}},
}
