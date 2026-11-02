"""Google's Gemini API (mocked): the model list split into the node families, text through the OpenAI-compatible
endpoint, Nano Banana pictures and Omni video through the Interactions API (the video in the background, followed
after a restart)."""

import base64
import json
import subprocess

import httpx
import pytest

from nodeloom import db, providers
from tests.conftest import PNG, error_of, job, queue, run_all, start, upload

G = "https://generativelanguage.googleapis.com/v1beta"
GOOGLE = """
[providers.google]
type = "google"
api_key = "g-key"

[providers.google.extra_models."gemini-3.1-flash-image"]
price = { input = 1.5, output = 30.0 }
"""
LISTED = [  # GET /models, in two pages
    [("gemini-2.5-flash", ["generateContent", "countTokens"]), ("gemini-3.8-flash", ["generateContent"]),
     ("gemini-4-argon", ["generateContent"]), ("gemini-3.1-pro-preview", ["generateContent"]),
     ("gemini-embedding-001", ["embedContent"]), ("gemini-3.8-flash-tts", ["generateContent"]),
     ("gemini-3.8-live", ["bidiGenerateContent"]), ("gemma-4-31b-it", ["generateContent"])],
    [("gemini-3.1-flash-image", ["generateContent"]), ("gemini-nano-banana-2.1", ["generateContent"]),
     ("gemini-omni-1.1-flash", ["generateContent"]), ("imagen-4.0-generate-001", ["predict"]),
     ("veo-3.1-generate-preview", ["predictLongRunning"]), ("lyria-3.5", ["generateContent"]),
     ("gemini-3.5-transcribe", ["generateContent"])],
]


def listing(request):
    page = 1 if request.url.params.get("pageToken") == "p2" else 0
    return httpx.Response(200, json={
        "models": [{"name": f"models/{mid}", "supportedGenerationMethods": m} for mid, m in LISTED[page]],
        **({"nextPageToken": "p2"} if page == 0 else {})})


@pytest.fixture
async def genv(monkeypatch, tmp_path):
    import respx

    from nodeloom import nodes
    from tests.conftest import Env
    client = await start(monkeypatch, tmp_path, GOOGLE)
    monkeypatch.setattr(nodes, "POLL_SECONDS", 0.01)
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{G}/models").mock(side_effect=listing)
        yield Env(client, mock)
    await client.aclose()
    await db.dispose()


async def test_the_model_list_goes_to_the_node_families(genv):
    by = {}
    for m in await providers.registry()["google"].models():
        by.setdefault(m.family, []).append(m.id)
    assert by["chat"] == ["gemini-4-argon", "gemini-3.8-flash", "gemini-2.5-flash", "gemini-3.1-pro-preview",
                          "gemma-4-31b-it"]  # newest first, previews after, no embeddings, voices or music
    assert by["gemini-image"] == ["gemini-3.1-flash-image", "gemini-nano-banana-2.1"]
    assert by["gemini-video"] == ["gemini-omni-1.1-flash"]  # Imagen and Veo are shut down in the Gemini API
    _, flash = await providers.resolve("google/gemini-3.8-flash")
    assert flash.takes("image") and flash.takes("audio") and not flash.takes("video")
    info = (await genv.client.get("/api/object_info")).json()
    assert info["LoomGeminiImage"]["input"]["required"]["model"][1]["options"] == [f"google/{m}" for m in by["gemini-image"]]
    assert info["LoomGeminiVideo"]["category"] == "Generate/Google"


async def test_text_goes_through_the_openai_compatible_endpoint(genv):
    chat = genv.mock.post(f"{G}/openai/chat/completions").respond(json={
        "choices": [{"message": {"role": "assistant", "content": "A red fox."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    pic = await upload(genv)
    r = await queue(genv, {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
                           "2": {"class_type": "LoomTextModel", "inputs": {
                               "model": "google/gemini-3.8-flash", "prompt": "what is it?", "images.image 1": ["1", 0]}}})
    assert r.status_code == 200, r.text
    await run_all(genv)
    j = await job(genv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert j["outputs"]["2"]["text"] == ["A red fox."]
    sent = chat.calls[0].request
    assert sent.headers["authorization"] == "Bearer g-key" and "x-goog-api-key" not in sent.headers
    body = json.loads(sent.content)
    assert body["model"] == "gemini-3.8-flash" and body["messages"][0]["content"][0]["type"] == "image_url"


def picture_answer(**extra):
    return {"id": "v1_img", "status": "completed", "steps": [
        {"type": "thought", "content": [{"type": "thought", "text": "..."}]},
        {"type": "model_output", "content": [
            {"type": "text", "text": "Here it is"},
            {"type": "image", "mime_type": "image/png", "data": base64.b64encode(PNG).decode()}]}],
        "usage": {"total_input_tokens": 300, "total_output_tokens": 1120, "total_thought_tokens": 100}, **extra}


async def test_nano_banana_draws_with_references(genv):
    made = genv.mock.post(f"{G}/interactions").respond(json=picture_answer())
    pic = await upload(genv)
    r = await queue(genv, {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
                           "2": {"class_type": "LoomGeminiImage", "inputs": {
                               "model": "google/gemini-3.1-flash-image", "prompt": "the cat at night",
                               "references.reference 1": ["1", 0], "aspect_ratio": "16:9", "size": "2K"}}})
    assert r.status_code == 200, r.text
    await run_all(genv)
    j = await job(genv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert len(j["outputs"]["2"]["images"]) == 1
    sent = made.calls[0].request
    assert sent.headers["x-goog-api-key"] == "g-key" and "authorization" not in sent.headers
    body = json.loads(sent.content)
    assert body["model"] == "gemini-3.1-flash-image" and body["store"] is False
    assert body["input"][0] == {"type": "image", "mime_type": "image/png", "data": base64.b64encode(PNG).decode()}
    assert body["input"][1] == {"type": "text", "text": "the cat at night"}
    assert body["response_format"] == {"type": "image", "aspect_ratio": "16:9", "image_size": "2K"}
    async with db.sessionmaker()() as s:
        from nodeloom.models import Run
        cost = (await s.get(Run, r.json()["prompt_id"])).extra["loom_costs"]["2"]
    assert cost == round(1.5 * 300 + 30.0 * 1220)  # thinking is output too


async def test_a_refusal_says_why(genv):
    genv.mock.post(f"{G}/interactions").respond(json={"id": "v1_no", "status": "completed", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "I can't draw real people like that."}]}]})
    graph = {"1": {"class_type": "LoomGeminiImage", "inputs": {"model": "google/gemini-3.1-flash-image",
                                                               "prompt": "x"}}}
    assert await error_of(genv, graph) == "No image came back: I can't draw real people like that."


def mp4(tmp_path) -> bytes:
    out = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=blue:s=320x240:d=1", "-pix_fmt",
                    "yuv420p", str(out)], check=True)
    return out.read_bytes()


def video_answer(clip: bytes, iid="v1_vid"):
    return {"id": iid, "status": "completed", "steps": [{"type": "model_output", "content": [
        {"type": "video", "mime_type": "video/mp4", "data": base64.b64encode(clip).decode()}]}],
        "usage": {"total_input_tokens": 50, "total_output_tokens": 46336}}


async def test_omni_makes_a_video_in_the_background(genv, tmp_path):
    clip = mp4(tmp_path)
    made = genv.mock.post(f"{G}/interactions").respond(json={"id": "v1_vid", "status": "in_progress"})
    genv.mock.get(f"{G}/interactions/v1_vid").mock(side_effect=[
        httpx.Response(200, json={"id": "v1_vid", "status": "in_progress"}),
        httpx.Response(200, json=video_answer(clip))])
    pic = await upload(genv)
    r = await queue(genv, {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
                           "2": {"class_type": "LoomGeminiVideo", "inputs": {
                               "model": "google/gemini-omni-1.1-flash", "prompt": "the cat jumps",
                               "first_frame": ["1", 0], "aspect_ratio": "9:16", "resolution": "1080p"}}})
    assert r.status_code == 200, r.text
    await run_all(genv)
    j = await job(genv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert j["outputs"]["2"]["animated"] == [True]
    body = json.loads(made.calls[0].request.content)
    assert body["background"] is True and body["generation_config"] == {"video_config": {"task": "image_to_video"}}
    assert body["response_format"] == {"type": "video", "aspect_ratio": "9:16", "resolution": "1080p"}
    assert [p["type"] for p in body["input"]] == ["image", "text"]


async def test_video_rules_and_failures(genv):
    pic = await upload(genv)
    base = {"model": "google/gemini-omni-1.1-flash", "prompt": "go"}
    graph = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
             "2": {"class_type": "LoomGeminiVideo", "inputs": {**base, "first_frame": ["1", 0],
                                                                "references.reference 1": ["1", 0]}}}
    assert await error_of(genv, graph) == "References do not go with the first and last frames: keep one or the other"
    graph["2"]["inputs"] = {**base, "last_frame": ["1", 0]}
    assert await error_of(genv, graph) == "A last frame needs a first frame"
    genv.mock.post(f"{G}/interactions").respond(json={"id": "v1_bad", "status": "in_progress"})
    genv.mock.get(f"{G}/interactions/v1_bad").respond(json={
        "id": "v1_bad", "status": "failed", "error": {"code": 400, "message": "The prompt was blocked."}})
    graph = {"1": {"class_type": "LoomGeminiVideo", "inputs": {**base, "prompt": "something else"}}}
    assert await error_of(genv, graph) == "The video failed: The prompt was blocked."


async def test_a_resumed_run_follows_its_interaction(genv, tmp_path):
    from nodeloom import nodes
    from nodeloom.auth import LOCAL_USER
    from nodeloom.nodes import Ctx
    made = genv.mock.post(f"{G}/interactions").respond(json={"id": "v1_new", "status": "in_progress"})
    genv.mock.get(f"{G}/interactions/v1_old").respond(json=video_answer(mp4(tmp_path), "v1_old"))

    async def progress(_):
        pass
    ctx = Ctx(LOCAL_USER, "1", db.sessionmaker(), progress, pending_task="v1_old")
    values, ui = await nodes.BY_ID["LoomGeminiVideo"].run(ctx, model="google/gemini-omni-1.1-flash", prompt="go")
    assert values[0]["media"] == "video" and values[1][0]["media"] == "image" and not made.called


def test_the_key_alone_gives_the_gemini_api(monkeypatch, tmp_path):
    from tests.conftest import configure
    configure(monkeypatch, tmp_path, None, gemini_api_key="AIza-x")
    p = providers.registry()["google"]
    assert p.headers() == {"x-goog-api-key": "AIza-x"} and p.base_url == G
    assert p.headers_for("/openai/chat/completions") == {"Authorization": "Bearer AIza-x"}


async def test_a_gemini_user_starts_with_a_gemini_graph(genv):
    starter = (await genv.client.get("/extensions/nodeloom/starter.json")).json()
    assert {n["type"] for n in starter["nodes"]} == {"LoomText", "LoomTextModel", "LoomJoinText", "LoomGeminiImage"}
    models = [n["widgets_values_named"]["model"] for n in starter["nodes"] if "model" in n["widgets_values_named"]]
    assert models == ["google/gemini-3.8-flash", "google/gemini-3.1-flash-image"]
