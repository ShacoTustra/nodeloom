"""Replicate: nodes made of the models' version schemas (real ones in tests/data/replicate), runs as predictions
(mocked): official models by name, community ones by version, files as data URIs or through the Files API, outputs
of every shape."""

import json
from pathlib import Path

import httpx
import pytest

from nodeloom import db, nodes, providers, schema_nodes
from nodeloom.providers import replicate
from nodeloom.providers.base import _within
from tests.conftest import CONFIG, error_of, job, queue, run_all, start, upload

DATA = Path(__file__).parent / "data" / "replicate"
API = "https://api.replicate.com/v1"
OUT = "https://replicate.delivery/xezq/abc"
KLING = "kwaivgi/kling-v2.5-turbo-pro"
REPLICATE = """
[providers.replicate]
type = "replicate"
api_key = "r8_test"

[providers.replicate.endpoints."black-forest-labs/flux-schnell"]
price = { image = 0.003 }
[providers.replicate.endpoints."kwaivgi/kling-v2.5-turbo-pro"]
price = { request = 0.35 }
[providers.replicate.endpoints."openai/whisper"]
price = { second = 0.001 }
[providers.replicate.endpoints."minimax/speech-02-turbo"]
[providers.replicate.endpoints."meta/meta-llama-3-8b-instruct"]
"""


def model(name: str) -> dict:
    return json.loads((DATA / (name.replace("/", "_") + ".json")).read_text())


def spec(name: str) -> schema_nodes.Spec:
    m = model(name)
    api = m["latest_version"]["openapi_schema"]
    media = replicate.guess_media(schema_nodes.properties(api, "Input"), m["description"])
    return schema_nodes.read(api, "Input", "Output", root_media=media)


def test_schemas_become_widgets_links_and_outputs():
    flux = spec("black-forest-labs/flux-schnell")
    by = {i.name: i for i in flux.inputs}
    assert flux.inputs[0].name == "prompt" and not by["prompt"].optional and by["prompt"].extra["multiline"]
    assert by["aspect_ratio"].type == "COMBO" and by["aspect_ratio"].default == "1:1" and not by["aspect_ratio"].advanced
    assert by["num_outputs"].type == "INT" and by["num_outputs"].extra == {"min": 1, "max": 4}
    assert by["seed"].extra["control_after_generate"] == "fixed" and by["go_fast"].type == "BOOLEAN"
    assert [(o.type, o.label) for o in flux.outputs] == [("IMAGE", "images")] and flux.out_fields[0].many

    banana = spec("google/nano-banana-pro")
    images = banana.inputs[1]
    assert (images.name, images.type, images.label, images.grow) == ("image_input", "IMAGE", "image", 10)
    assert [(o.type, o.label) for o in banana.outputs] == [("IMAGE", "image")]
    assert replicate.category(banana) == "text-to-image"

    kling = spec(KLING)
    by = {i.name: i for i in kling.inputs}
    assert "image" not in by and "guidance_scale" not in by  # deprecated
    assert by["start_image"].type == "IMAGE" and by["start_image"].optional and by["start_image"].label == "start image"
    assert by["duration"].options == ["5", "10"]
    assert next(f for f in kling.fields if f.name == "duration").cast == "int"  # sent back as a number
    assert [o.type for o in kling.outputs] == ["VIDEO"] and replicate.category(kling) == "text-to-video"

    speech = spec("minimax/speech-02-turbo")
    assert speech.inputs[0].name == "text" and [o.type for o in speech.outputs] == ["AUDIO"]
    assert replicate.category(speech) == "text-to-speech"

    whisper = spec("openai/whisper")
    assert whisper.inputs[0].name == "audio" and whisper.inputs[0].type == "AUDIO" and not whisper.inputs[0].optional
    assert [(o.type, o.label) for o in whisper.outputs] == [("STRING", "transcription"), ("STRING", "translation")]
    assert replicate.category(whisper) == "speech-to-text"  # srt_file and txt_file are no pictures: left out

    llama = spec("meta/meta-llama-3-8b-instruct")
    assert [(o.type, o.label) for o in llama.outputs] == [("STRING", "text")] and replicate.category(llama) == "llm"


def test_enums_behind_refs():
    # the API's own schemas keep enums as components: {"allOf": [{"$ref": ...}], "default": ...}
    api = {"components": {"schemas": {
        "Input": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string", "x-order": 0},
            "duration": {"allOf": [{"$ref": "#/components/schemas/duration"}], "default": 8, "x-order": 1},
            "first_frame_image": {"type": "string", "format": "uri", "x-order": 2},
            "lora_weights": {"type": "string", "format": "uri", "x-order": 3, "default": ""}}},
        "duration": {"enum": [4, 6, 8], "type": "integer", "title": "duration"},
        "Output": {"type": "string", "format": "uri"}}}}
    s = schema_nodes.read(api, "Input", "Output", root_media="VIDEO")
    by = {i.name: i for i in s.inputs}
    assert by["duration"].options == ["4", "6", "8"] and by["duration"].default == "8"
    assert by["first_frame_image"].type == "IMAGE" and by["first_frame_image"].label == "first frame image"
    assert by["lora_weights"].type == "STRING"  # a link to weights, not a file of the user
    assert replicate.guess_media(schema_nodes.properties(api, "Input")) == "VIDEO"


def test_names_of_the_nodes():
    assert replicate.node_id("black-forest-labs/flux-schnell") == "LoomReplicateBlackForestLabsFluxSchnell"
    assert replicate.node_id("openai/whisper:" + "a" * 64) == "LoomReplicateOpenaiWhisper"
    assert replicate.title(KLING) == "Kling v2.5 Turbo Pro"
    assert replicate.ENDPOINT.match("openai/whisper:" + "8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e")
    assert not replicate.ENDPOINT.match("whisper") and not replicate.ENDPOINT.match("a/b/c")


def test_file_hosts_are_whole_names():
    assert _within("v3.fal.media", ["fal.media"]) and _within("fal.media", [".fal.media"])
    assert not _within("evilfal.media", ["fal.media"]) and not _within("replicate.delivery.evil.com", ["replicate.delivery"])


@pytest.fixture
async def repenv(monkeypatch, tmp_path):
    import respx

    from tests.conftest import Env
    client = await start(monkeypatch, tmp_path, CONFIG + REPLICATE)
    monkeypatch.setattr(replicate, "POLL_SECONDS", 0.01)
    with respx.mock(assert_all_called=False) as mock:
        def answer(request):
            name = request.url.path.removeprefix("/v1/models/")
            path = DATA / (name.replace("/", "_") + ".json")
            return httpx.Response(200, json=json.loads(path.read_text())) if path.is_file() else httpx.Response(
                404, json={"title": "Not found", "detail": "Model not found", "status": 404})
        models = mock.get(url__regex=rf"{API}/models/[^/]+/[^/]+$").mock(side_effect=answer)
        env = Env(client, mock)
        env.models = models
        yield env
    await client.aclose()
    nodes.set_dynamic([])
    await db.dispose()


async def test_models_are_nodes_of_the_editor(repenv):
    info = (await repenv.client.get("/api/object_info")).json()
    flux = info["LoomReplicateBlackForestLabsFluxSchnell"]
    assert flux["category"] == "Generate/Replicate/text-to-image" and flux["display_name"] == "Flux Schnell"
    assert flux["description"].startswith("The fastest image generation model")
    assert info["LoomReplicateKwaivgiKlingV25TurboPro"]["category"] == "Generate/Replicate/text-to-video"
    assert info["LoomReplicateMinimaxSpeech02Turbo"]["output"] == ["AUDIO"]
    assert info["LoomReplicateOpenaiWhisper"]["output"] == ["STRING", "STRING"]
    status = (await repenv.client.get("/api/nodeloom/status")).json()["providers"]
    assert {"name": "replicate", "type": "replicate", "models": 0, "nodes": 5} in status
    assert repenv.models.calls[0].request.headers["authorization"] == "Bearer r8_test"
    ru = (await repenv.client.get("/api/i18n")).json()["ru"]["nodeCategories"]
    assert ru["Replicate"] == "Replicate" and ru["llm"] == "языковые модели"


def prediction(env, pid="p1", output=None, metrics=None, statuses=("starting", "processing")):
    create = env.mock.post(f"{API}/predictions").respond(201, json={"id": pid, "status": "starting"})
    env.mock.get(f"{API}/predictions/{pid}").mock(side_effect=[
        *[httpx.Response(200, json={"id": pid, "status": s}) for s in statuses],
        httpx.Response(200, json={"id": pid, "status": "succeeded", "output": output,
                                  "metrics": metrics or {"predict_time": 12.5}})])
    return create


async def test_an_official_model_runs_by_its_name(repenv):
    pic = await upload(repenv)
    create = prediction(repenv, output=f"{OUT}/out.mp4")
    video = repenv.mock.get(f"{OUT}/out.mp4").respond(content=b"\x00\x00\x00\x18ftypmp42",
                                                      headers={"content-type": "video/mp4"})
    graph = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
             "2": {"class_type": "LoomReplicateKwaivgiKlingV25TurboPro", "inputs": {
                 "prompt": "the cat waves", "start_image": ["1", 0], "duration": "10"}}}
    r = await queue(repenv, graph)
    assert r.status_code == 200, r.text
    await run_all(repenv)
    j = await job(repenv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert j["outputs"]["2"]["animated"] == [True]
    body = json.loads(create.calls[0].request.content)
    assert body["version"] == KLING  # official: by its name
    assert body["input"]["duration"] == 10 and body["input"]["prompt"] == "the cat waves"
    assert body["input"]["start_image"].startswith("data:image/png;base64,")
    assert video.calls[0].request.headers["authorization"] == "Bearer r8_test"
    async with db.sessionmaker()() as s:
        from nodeloom.models import Run
        assert (await s.get(Run, r.json()["prompt_id"])).extra["loom_costs"]["2"] == 350_000


async def test_a_community_model_runs_by_its_version_and_large_files_are_uploaded(repenv):
    sound = await upload(repenv, "voice.mp3", b"ID3" + b"\x00" * 300_000, "audio/mpeg")
    files_api = repenv.mock.post(f"{API}/files").respond(201, json={
        "id": "f1", "urls": {"get": f"{API}/files/f1"}, "content_type": "audio/mpeg", "size": 300_003})
    create = prediction(repenv, output={"transcription": "hello there", "translation": None, "segments": [],
                                        "detected_language": "english"}, metrics={"predict_time": 4.0})
    graph = {"1": {"class_type": "LoomLoadAudio", "inputs": {"audio": sound}},
             "2": {"class_type": "LoomReplicateOpenaiWhisper", "inputs": {"audio": ["1", 0]}}}
    r = await queue(repenv, graph)
    await run_all(repenv)
    j = await job(repenv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert j["outputs"]["2"]["text"] == ["hello there"]
    body = json.loads(create.calls[0].request.content)
    assert body["version"] == "openai/whisper:" + model("openai/whisper")["latest_version"]["id"]
    assert body["input"]["audio"] == f"{API}/files/f1" and body["input"]["transcription"] == "plain text"
    sent = files_api.calls[0].request
    assert b'name="content"' in sent.content and sent.headers["content-type"].startswith("multipart/form-data")
    async with db.sessionmaker()() as s:
        from nodeloom.models import Run
        assert (await s.get(Run, r.json()["prompt_id"])).extra["loom_costs"]["2"] == 4_000  # 4 s at $0.001


async def test_outputs_of_every_shape(repenv):
    prediction(repenv, "p2", output=[f"{OUT}/a.webp", f"{OUT}/b.webp"])
    for name in ("a", "b"):
        repenv.mock.get(f"{OUT}/{name}.webp").respond(content=b"RIFF\x00\x00\x00\x00WEBPVP8 ",
                                                      headers={"content-type": "image/webp"})
    r = await queue(repenv, {"1": {"class_type": "LoomReplicateBlackForestLabsFluxSchnell", "inputs": {
        "prompt": "two cats", "num_outputs": 2}}})
    await run_all(repenv)
    j = await job(repenv, r.json()["prompt_id"])
    assert j["status"] == "completed", j.get("execution_error")
    assert len(j["outputs"]["1"]["images"]) == 2
    async with db.sessionmaker()() as s:
        from nodeloom.models import Run
        assert (await s.get(Run, r.json()["prompt_id"])).extra["loom_costs"]["1"] == 6_000  # 2 pictures at $0.003

    prediction(repenv, "p3", output=["Once ", "upon ", "a time"])  # an LLM's tokens
    r = await queue(repenv, {"1": {"class_type": "LoomReplicateMetaMetaLlama38bInstruct", "inputs": {
        "prompt": "a story"}}})
    await run_all(repenv)
    assert (await job(repenv, r.json()["prompt_id"]))["outputs"]["1"]["text"] == ["Once upon a time"]


async def test_failures_say_why(repenv):
    repenv.mock.post(f"{API}/predictions").respond(201, json={"id": "p4", "status": "starting"})
    repenv.mock.get(f"{API}/predictions/p4").respond(json={
        "id": "p4", "status": "failed", "error": "NSFW content detected. Try a different prompt."})
    graph = {"1": {"class_type": "LoomReplicateBlackForestLabsFluxSchnell", "inputs": {"prompt": "x"}}}
    assert await error_of(repenv, graph) == "replicate: NSFW content detected. Try a different prompt."
    prediction(repenv, "p5", output="https://169.254.169.254/latest/meta-data")
    graph = {"1": {"class_type": "LoomReplicateBlackForestLabsFluxSchnell", "inputs": {"prompt": "y"}}}
    assert await error_of(repenv, graph) == "Invalid file address"
    r = await queue(repenv, {"1": {"class_type": "LoomReplicateOpenaiWhisper", "inputs": {}}})
    assert r.json()["node_errors"]["1"]["errors"][0]["type"] == "required_input_missing"


async def test_a_resumed_run_follows_its_prediction(repenv):
    from nodeloom.auth import LOCAL_USER
    from nodeloom.nodes import Ctx
    await repenv.client.get("/api/object_info")
    create = prediction(repenv, "p9", output="hello", statuses=())
    node = nodes.BY_ID["LoomReplicateMetaMetaLlama38bInstruct"]

    async def progress(_):
        pass
    ctx = Ctx(LOCAL_USER, "1", db.sessionmaker(), progress, pending_task="p9")
    values, ui = await node.run(ctx, prompt="hi")
    assert values == ["hello"] and ui == {"text": ["hello"]} and not create.called


async def test_models_that_cannot_be_nodes(repenv):
    p = providers.registry()["replicate"]
    p.settings.endpoints = ["openai/whisper", "no-such/model", "not a model"]
    p._nodes = None
    made = await p.nodes()
    assert [n.id for n in made] == ["LoomReplicateOpenaiWhisper"]


def test_the_token_alone_gives_popular_models(monkeypatch, tmp_path):
    from tests.conftest import configure
    configure(monkeypatch, tmp_path, None, replicate_api_token="r8_x")
    p = providers.registry()["replicate"]
    assert p.headers() == {"Authorization": "Bearer r8_x"} and "black-forest-labs/flux-schnell" in p.endpoints()
