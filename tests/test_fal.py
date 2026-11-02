"""fal: nodes made of the endpoints' OpenAPI schemas (real schemas in tests/data/fal), runs through fal's queue (mocked),
resumed by request id, costs per request."""

import json
from pathlib import Path

import httpx
import pytest

from nodeloom import db, nodes, providers, schema_nodes
from nodeloom.providers import fal
from tests.conftest import CONFIG, PNG, SAME, error_of, job, queue, run_all, start, upload

DATA = Path(__file__).parent / "data" / "fal"
QUEUE = "https://queue.fal.run"
KLING = "fal-ai/kling-video/v2.1/standard/image-to-video"
FAL = """
[providers.fal]
type = "fal"
api_key = "fk"

[providers.fal.endpoints."fal-ai/flux/dev"]
[providers.fal.endpoints."fal-ai/nano-banana/edit"]
[providers.fal.endpoints."fal-ai/kling-video/v2.1/standard/image-to-video"]
price = { request = 0.28 }
[providers.fal.endpoints."fal-ai/elevenlabs/tts/turbo-v2.5"]
title = "Voice"
[providers.fal.endpoints."fal-ai/minimax/hailuo-02/standard/image-to-video"]
"""


def schema(endpoint: str) -> dict:
    return json.loads((DATA / (endpoint.replace("/", "_") + ".json")).read_text())


def spec(endpoint: str) -> schema_nodes.Spec:
    s = schema(endpoint)
    paths = s["paths"]
    inp = paths[f"/{endpoint}"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
    out = paths[f"/{endpoint}/requests/{{request_id}}"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"]["$ref"].split("/")[-1]
    return schema_nodes.read(s, inp, out)


def test_schemas_become_widgets_links_and_outputs():
    flux = spec("fal-ai/flux/dev")
    by = {i.name: i for i in flux.inputs}
    assert flux.inputs[0].name == "prompt" and by["prompt"].type == "STRING" and by["prompt"].extra["multiline"]
    assert not by["prompt"].optional and not by["prompt"].advanced
    assert by["image_size"].type == "COMBO" and by["image_size"].default == "landscape_4_3"
    assert "square_hd" in by["image_size"].options and not by["image_size"].advanced  # the enum next to {width, height}
    assert by["seed"].type == "INT" and by["seed"].extra["control_after_generate"] == "fixed"
    assert by["guidance_scale"].type == "FLOAT" and by["guidance_scale"].extra["min"] == 1 and by["guidance_scale"].advanced
    assert by["enable_safety_checker"].type == "BOOLEAN" and "sync_mode" not in by
    assert [(o.type, o.label) for o in flux.outputs] == [("IMAGE", "images")]

    edit = spec("fal-ai/nano-banana/edit")
    images = next(i for i in edit.inputs if i.name == "image_urls")
    assert images.type == "IMAGE" and images.grow and images.label == "image"
    assert next(f for f in edit.fields if f.name == "image_urls").optional is False  # required: checked when it runs
    assert [o.type for o in edit.outputs] == ["IMAGE", "STRING"]

    kling = spec(KLING)
    by = {i.name: i for i in kling.inputs}
    assert by["image_url"].type == "IMAGE" and not by["image_url"].optional and by["image_url"].label == "image"
    assert by["duration"].options == ["5", "10"] and not by["duration"].advanced
    assert [o.type for o in kling.outputs] == ["VIDEO"]
    hailuo = {i.name: i for i in spec("fal-ai/minimax/hailuo-02/standard/image-to-video").inputs}
    assert hailuo["end_image_url"].type == "IMAGE" and hailuo["end_image_url"].optional

    tts = spec("fal-ai/elevenlabs/tts/turbo-v2.5")
    assert tts.inputs[0].name == "text" and tts.inputs[0].extra["multiline"]
    assert [o.type for o in tts.outputs] == ["AUDIO"]


def test_names_of_the_nodes():
    assert fal.node_id(KLING) == "LoomFalKlingVideoV21StandardImageToVideo"
    assert fal.node_id("fal-ai/flux/dev") == "LoomFalFluxDev"
    assert fal.node_id("bytedance/seedream/v4") == "LoomFalBytedanceSeedreamV4"
    assert fal.title(KLING) == "Kling Video v2.1 Standard Image to Video"
    assert fal.title("fal-ai/nano-banana/edit") == "Nano Banana Edit"


@pytest.fixture
async def falenv(monkeypatch, tmp_path):
    import respx

    from tests.conftest import Env
    client = await start(monkeypatch, tmp_path, CONFIG + FAL)
    monkeypatch.setattr(fal, "POLL_SECONDS", 0.01)
    with respx.mock(assert_all_called=False) as mock:
        schemas = mock.get(fal.SCHEMA_URL).mock(side_effect=lambda request: httpx.Response(
            200, json=schema(request.url.params["endpoint_id"])))
        env = Env(client, mock)
        env.schemas = schemas
        yield env
    await client.aclose()
    nodes.set_dynamic([])
    await db.dispose()


async def test_endpoints_are_nodes_of_the_editor(falenv):
    info = (await falenv.client.get("/api/object_info")).json()
    kling = info["LoomFalKlingVideoV21StandardImageToVideo"]
    assert kling["category"] == "Generate/fal/image-to-video" and kling["display_name"].startswith("Kling Video v2.1")
    assert kling["input"]["required"]["image_url"] == ["IMAGE", {"display_name": "image",
                                                                "tooltip": "The image to be used for the video"}]
    assert info["LoomFalElevenlabsTtsTurboV25"]["display_name"] == "Voice"
    assert info["LoomFalNanoBananaEdit"]["input"]["optional"]["image_urls"][0] == "COMFY_AUTOGROW_V3"
    status = (await falenv.client.get("/api/nodeloom/status")).json()["providers"]
    assert {"name": "fal", "type": "fal", "models": 0, "nodes": 5} in status
    # the schemas are kept in the data directory: a new process does not ask fal again
    asked = falenv.schemas.call_count
    assert asked == 5
    providers.registry()["fal"]._nodes = None
    from nodeloom import config
    config.get_settings.cache_clear()
    assert len(await providers.registry()["fal"].nodes()) == 5 and falenv.schemas.call_count == asked
    ru = (await falenv.client.get("/api/i18n")).json()["ru"]["nodeCategories"]
    assert ru["image-to-video"] == "картинка в видео"


def kling_mocks(env, request_id="r1"):
    submit = env.mock.post(f"{QUEUE}/{KLING}").respond(json={"request_id": request_id, "status": "IN_QUEUE"})
    env.mock.get(f"{QUEUE}/fal-ai/kling-video/requests/{request_id}/status").mock(side_effect=[
        httpx.Response(200, json={"status": "IN_QUEUE", "queue_position": 2}),
        httpx.Response(200, json={"status": "IN_PROGRESS"}),
        httpx.Response(200, json={"status": "COMPLETED"})])
    env.mock.get(f"{QUEUE}/fal-ai/kling-video/requests/{request_id}").respond(json={
        "video": {"url": "https://v3.fal.media/files/koala/out.mp4", "content_type": "video/mp4"}})
    env.mock.get("https://v3.fal.media/files/koala/out.mp4").respond(content=b"\x00\x00\x00\x18ftypmp42",
                                                                    headers={"content-type": "video/mp4"})
    return submit


async def test_a_run_goes_through_the_queue(falenv):
    pic = await upload(falenv)
    submit = kling_mocks(falenv)
    graph = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
             "2": {"class_type": "LoomFalKlingVideoV21StandardImageToVideo", "inputs": {
                 "prompt": "the cat waves", "image_url": ["1", 0], "duration": "5"}}}
    r = await queue(falenv, graph)
    assert r.status_code == 200, r.text
    await run_all(falenv)
    j = await job(falenv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert j["outputs"]["2"]["animated"] == [True]
    sent = submit.calls[0].request
    assert sent.headers["authorization"] == "Key fk"
    body = json.loads(sent.content)
    assert body["prompt"] == "the cat waves" and body["duration"] == "5" and body["cfg_scale"] == 0.5
    assert body["image_url"].startswith("data:image/png;base64,") and "negative_prompt" in body
    async with db.sessionmaker()() as s:
        from nodeloom.models import Run
        assert (await s.get(Run, r.json()["prompt_id"])).extra["loom_costs"]["2"] == 280_000  # $0.28 a request


async def test_required_links_and_fals_own_errors(falenv):
    graph = {"1": {"class_type": "LoomFalNanoBananaEdit", "inputs": {"prompt": "make it night"}}}
    assert await error_of(falenv, graph) == "Connect “image”"
    falenv.mock.post(f"{QUEUE}/fal-ai/flux/dev").respond(422, json={"detail": [
        {"loc": ["body", "prompt"], "msg": "String should have at least 3 characters", "type": "string_too_short"}]})
    graph = {"1": {"class_type": "LoomFalFluxDev", "inputs": {"prompt": "ab"}}}
    assert await error_of(falenv, graph) == "fal: prompt: String should have at least 3 characters"


async def test_a_resumed_run_follows_its_request(falenv):
    from datetime import timedelta

    from nodeloom import engine
    from nodeloom.auth import LOCAL_USER
    from nodeloom.models import NodeResult, Run
    await (falenv.client.get("/api/object_info"))  # the nodes, as a fresh process makes them
    submit = kling_mocks(falenv, "r9")
    pic = await upload(falenv)
    graph = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
             "2": {"class_type": "LoomFalKlingVideoV21StandardImageToVideo", "inputs": {
                 "prompt": "waves", "image_url": ["1", 0]}}}
    run_id = (await queue(falenv, graph)).json()["prompt_id"]
    node = nodes.BY_ID["LoomFalKlingVideoV21StandardImageToVideo"]
    async with db.sessionmaker()() as s:
        from nodeloom import files
        f = await files.get(s, LOCAL_USER, "input", pic)
    up = engine._sig(nodes.BY_ID["LoomLoadImage"], {"image": pic, "image#file": [f.id, f.size]}, {})
    given = engine.grouped(node, graph["2"]["inputs"])
    sig = engine._sig(node, {k: v for k, v in given.items() if k != "image_url"}, {"image_url": [up, 0]})
    async with db.sessionmaker()() as s:
        run = await s.get(Run, run_id)
        run.status, run.worker, run.heartbeat_at = "running", "dead:1", db.utcnow() - timedelta(minutes=5)
        s.add(NodeResult(user_id=LOCAL_USER, sig=sig, node_type=node.id, task_id="r9"))
        await s.commit()
    from nodeloom import runs
    await runs.dispatcher.recover()
    await run_all(falenv)
    j = await job(falenv, run_id)
    assert j["status"] == "completed", j.get("execution_error")
    assert not submit.called  # followed, not paid again


async def test_endpoints_that_cannot_be_nodes(monkeypatch, tmp_path):
    import respx
    client = await start(monkeypatch, tmp_path, '[providers.fal]\ntype = "fal"\nendpoints = ["fal-ai/flux/dev", '
                                                '"not an endpoint", "fal-ai/gone"]\n')
    with respx.mock:
        respx.get(fal.SCHEMA_URL, params={"endpoint_id": "fal-ai/flux/dev"}).respond(json=schema("fal-ai/flux/dev"))
        respx.get(fal.SCHEMA_URL, params={"endpoint_id": "fal-ai/gone"}).respond(404)
        made = await providers.registry()["fal"].nodes()
    assert [n.id for n in made] == ["LoomFalFluxDev"]
    await client.aclose()
    nodes.set_dynamic([])
    await db.dispose()


def test_fal_key_alone_gives_popular_endpoints(monkeypatch, tmp_path):
    from tests.conftest import configure
    configure(monkeypatch, tmp_path, None, fal_key="fk")
    p = providers.registry()["fal"]
    assert p.headers() == {"Authorization": "Key fk"} and "fal-ai/flux/dev" in p.endpoints()


@pytest.mark.parametrize("path", sorted(DATA.glob("*.json")), ids=lambda p: p.stem)
def test_every_saved_schema_makes_a_node(path):
    endpoint = json.loads(path.read_text())["info"]["x-fal-metadata"]["endpointId"]
    node = fal.FalProvider("fal", providers.ProviderSettings(type="fal")).node(
        endpoint, json.loads(path.read_text()), fal.ModelSettings())
    assert node.inputs and node.outputs and node.id.startswith("LoomFal")
    assert PNG and SAME  # (the helpers above use them)
