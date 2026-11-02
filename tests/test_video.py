"""The routers' video API (OpenRouter, ProxyAPI; mocked): video models listed next to the chat ones, a job submitted
with its frames, polled, downloaded with the key, its cost taken from the answer, followed after a restart."""

import json
import subprocess

import httpx
import pytest

from nodeloom import db, providers
from tests.conftest import error_of, job, queue, run_all, start, upload

R = "https://router.test/api/v1"
ROUTER = f"""
[providers.router]
type = "openai"
base_url = "{R}"
api_key = "or-key"
"""


def mp4(tmp_path) -> bytes:
    out = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=green:s=320x240:d=1", "-pix_fmt",
                    "yuv420p", str(out)], check=True)
    return out.read_bytes()


@pytest.fixture
async def venv(monkeypatch, tmp_path):
    import respx

    from nodeloom import nodes
    from tests.conftest import Env
    client = await start(monkeypatch, tmp_path, ROUTER)
    monkeypatch.setattr(nodes, "POLL_SECONDS", 0.01)
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{R}/models").respond(json={"data": [
            {"id": "openai/gpt-5-mini", "architecture": {"input_modalities": ["text", "image"],
                                                         "output_modalities": ["text"]}},
            {"id": "google/veo-3.1", "architecture": {"output_modalities": ["video"]}}]})
        mock.get(f"{R}/videos/models").respond(json={"data": [
            {"id": "google/veo-3.1", "supported_durations": [4, 6, 8], "supported_frame_images": ["first_frame"]},
            {"id": "bytedance/seedance-2.0", "supported_durations": [4, 5, 6]}]})
        yield Env(client, mock)
    await client.aclose()
    await db.dispose()


async def test_video_models_come_from_the_video_api(venv):
    ch = await providers.choices()
    assert ch["models:video"] == ["router/google/veo-3.1", "router/bytedance/seedance-2.0"]
    assert ch["models:chat"] == ["router/openai/gpt-5-mini"]
    info = (await venv.client.get("/api/object_info")).json()
    assert info["LoomVideo"]["output"] == ["VIDEO", "IMAGE"]


async def test_a_video_from_the_first_frame(venv, tmp_path):
    clip = mp4(tmp_path)
    made = venv.mock.post(f"{R}/videos").respond(202, json={"id": "j1", "status": "pending",
                                                            "polling_url": f"{R}/videos/j1"})
    venv.mock.get(f"{R}/videos/j1").mock(side_effect=[
        httpx.Response(200, json={"id": "j1", "status": "in_progress"}),
        httpx.Response(200, json={"id": "j1", "status": "completed", "unsigned_urls": [f"{R}/videos/j1/content?index=0"],
                                  "usage": {"cost": 0.5, "is_byok": False}})])
    content = venv.mock.get(f"{R}/videos/j1/content").respond(content=clip, headers={"content-type": "video/mp4"})
    pic = await upload(venv)
    r = await queue(venv, {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
                           "2": {"class_type": "LoomVideo", "inputs": {
                               "model": "router/google/veo-3.1", "prompt": "the cat jumps", "first_frame": ["1", 0],
                               "duration": 8, "resolution": "720p", "aspect_ratio": "16:9", "sound": False}}})
    assert r.status_code == 200, r.text
    await run_all(venv)
    j = await job(venv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert j["outputs"]["2"]["animated"] == [True]
    body = json.loads(made.calls[0].request.content)
    assert body["model"] == "google/veo-3.1" and body["duration"] == 8 and body["generate_audio"] is False
    assert body["frame_images"][0]["frame_type"] == "first_frame"
    assert body["frame_images"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "input_references" not in body
    assert content.calls[0].request.headers["authorization"] == "Bearer or-key"  # unsigned: the key goes along
    async with db.sessionmaker()() as s:
        from nodeloom.models import Run
        assert (await s.get(Run, r.json()["prompt_id"])).extra["loom_costs"]["2"] == 500_000  # the router's cost


async def test_settings_left_to_the_model_and_failures(venv):
    made = venv.mock.post(f"{R}/videos").respond(202, json={"id": "j2", "status": "pending"})
    venv.mock.get(f"{R}/videos/j2").respond(json={"id": "j2", "status": "failed", "error": "Content policy violation"})
    graph = {"1": {"class_type": "LoomVideo", "inputs": {"model": "router/bytedance/seedance-2.0", "prompt": "go"}}}
    assert await error_of(venv, graph) == "The video failed: Content policy violation"
    assert json.loads(made.calls[0].request.content) == {"model": "bytedance/seedance-2.0", "prompt": "go"}
    pic = await upload(venv)
    graph = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
             "2": {"class_type": "LoomVideo", "inputs": {"model": "router/google/veo-3.1", "prompt": "go",
                                                         "last_frame": ["1", 0]}}}
    assert await error_of(venv, graph) == "A last frame needs a first frame"


async def test_a_resumed_run_follows_its_job(venv, tmp_path):
    from nodeloom import nodes
    from nodeloom.auth import LOCAL_USER
    from nodeloom.nodes import Ctx
    made = venv.mock.post(f"{R}/videos").respond(202, json={"id": "new"})
    venv.mock.get(f"{R}/videos/old").respond(json={"id": "old", "status": "completed"})  # no urls: the content path
    venv.mock.get(f"{R}/videos/old/content").respond(content=mp4(tmp_path), headers={"content-type": "video/mp4"})

    async def progress(_):
        pass
    ctx = Ctx(LOCAL_USER, "1", db.sessionmaker(), progress, pending_task="old")
    values, _ = await nodes.BY_ID["LoomVideo"].run(ctx, model="router/google/veo-3.1", prompt="go")
    assert values[0]["media"] == "video" and not made.called
