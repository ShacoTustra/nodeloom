"""The backend as ComfyUI's frontend sees it: nodes, uploads, runs through the providers (mocked), the result cache,
resuming after a restart, cancelling, userdata, costs, the Host and origin checks."""

import asyncio
import shutil
import subprocess
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import select

from nodeloom import bus, db, engine, nodes, runs
from nodeloom.auth import LOCAL_USER
from nodeloom.models import NodeResult, Run, StoredFile
from tests.conftest import (
    ARK,
    IMAGE_GRAPH,
    OPENAI,
    PNG,
    SAME,
    TOS,
    error_of,
    image_reply,
    job,
    queue,
    run_all,
    start,
    upload,
    video_mocks,
)


async def test_host_and_origin_checks(env):
    assert (await env.client.get("/api/object_info")).status_code == 200
    # a page of another site that points its name at 127.0.0.1 (DNS rebinding) gives itself away by its Host
    r = await env.client.get("/api/object_info", headers={"Host": "evil.example"})
    assert r.status_code == 403 and "ALLOWED_HOSTS" in r.text
    assert (await env.client.get("/api/object_info", headers={"Host": "127.0.0.1:8188"})).status_code == 200
    assert (await env.client.get("/api/object_info", headers={"Host": "[::1]:8188"})).status_code == 200
    # state-changing requests only from our own pages
    r = await env.client.post("/api/prompt", headers={"Sec-Fetch-Site": "cross-site"}, json={"prompt": {}})
    assert r.status_code == 403
    r = await env.client.post("/api/prompt", headers={"Origin": "https://evil.example"}, json={"prompt": {}})
    assert r.status_code == 403


async def test_object_info_lists_our_nodes_and_models(env):
    info = (await env.client.get("/api/object_info")).json()
    assert {"LoomText", "LoomSeedreamImage", "LoomSeedanceVideo", "LoomGPTImage", "LoomLoadImage",
            "LoomJoinVideos"} <= set(info)
    assert "LoadImage" not in info and "KSampler" not in info
    video = info["LoomSeedanceVideo"]
    assert video["display_name"] == "Seedance video" and video["category"] == "Generate/ByteDance"
    assert video["input"]["required"]["model"][1]["options"] == ["ark/dreamina-seedance-2-0", "ark/dreamina-seedance-2-5"]
    assert video["input"]["optional"]["first_frame"] == ["IMAGE", {"display_name": "first frame"}]
    grow = video["input"]["optional"]["references"]
    assert grow[0] == "COMFY_AUTOGROW_V3" and grow[1]["template"]["names"][:2] == ["reference 1", "reference 2"]
    # the text model node takes every provider's chat models
    assert info["LoomTextModel"]["input"]["required"]["model"][1]["options"] == [
        "ark/seed-2-0-lite", "ark/deepseek-v4-flash", "openai/gpt-5-mini"]
    assert info["LoomGPTImage"]["input"]["required"]["model"][1]["options"] == ["openai/gpt-image-1"]
    # the editor's audio upload button needs its player; "Show" plays an audio result in it
    assert list(info["LoomLoadAudio"]["input"]["required"]) == ["audio", "audioUI"]
    assert info["LoomShow"]["input"]["required"]["audioUI"] == ["AUDIO_UI", {}]
    # the editor starts with our pages: settings in the browser's language, no local models
    assert (await env.client.get("/api/settings")).json()["Comfy.Locale"] == "en"
    assert (await env.client.get("/api/experiment/models")).json() == []
    assert "/extensions/nodeloom/nodeloom.js" in (await env.client.get("/api/extensions")).json()
    page = await env.client.get("/")
    assert 'id="loom-splash"' in page.text and "splash/splash.js" in page.text
    assert "<title>Nodeloom</title>" in page.text and 'name="application-name"' in page.text
    assert page.headers["content-security-policy"].startswith("default-src 'self'")
    status = (await env.client.get("/api/nodeloom/status")).json()
    assert status == {"providers": [{"name": "ark", "type": "ark", "models": 5},
                                    {"name": "openai", "type": "openai", "models": 2}]}


async def test_nodes_without_models_are_not_offered(monkeypatch, tmp_path):
    only_openai = f'[providers.openai]\ntype = "openai"\nbase_url = "{OPENAI}"\nmodels = ["gpt-5-mini", "gpt-image-1"]\n'
    client = await start(monkeypatch, tmp_path, only_openai)
    info = (await client.get("/api/object_info")).json()
    assert {"LoomTextModel", "LoomGPTImage", "LoomLoadImage", "LoomShow"} <= set(info)
    assert not {"LoomSeedreamImage", "LoomSeedanceVideo", "LoomSeedanceCharacter"} & set(info)
    graph = {"1": {"class_type": "LoomSeedanceVideo", "inputs": {"model": "ark/x", "prompt": "sea"}}}
    r = await client.post("/api/prompt", headers=SAME, json={"prompt": graph})
    assert r.status_code == 400 and "configure a provider" in r.json()["error"]["message"]
    await client.aclose()
    # no provider at all: the editor still opens, with uploads and tools
    client = await start(monkeypatch, tmp_path / "none", None)
    info = (await client.get("/api/object_info")).json()
    assert "LoomLoadImage" in info and "LoomTextModel" not in info
    assert (await client.get("/api/nodeloom/status")).json() == {"providers": []}
    await client.aclose()
    await db.dispose()


async def test_upload_view_and_choices(env):
    name = await upload(env)
    assert name == "frame.png"
    assert await upload(env) == "frame (1).png"  # same name, no overwrite
    info = (await env.client.get("/api/object_info")).json()
    assert info["LoomLoadImage"]["input"]["required"]["image"][1]["options"][:2] == ["frame (1).png", "frame.png"]
    r = await env.client.get("/api/view", params={"filename": "frame.png", "type": "input"})
    assert r.status_code == 200 and r.content == PNG and r.headers["content-type"] == "image/png"
    assert r.headers["x-content-type-options"] == "nosniff" and "sandbox" in r.headers["content-security-policy"]
    for bad in (("x.exe", b"MZ", "application/x-msdownload"), ("x.svg", b"<svg onload=alert(1)/>", "image/svg+xml"),
                ("fake.png", b"<html><script>alert(1)</script>", "image/png")):
        r = await env.client.post("/api/upload/image", headers=SAME, files={"image": bad})
        assert r.status_code == 400, bad
    assert (await env.client.get("/internal/files/input")).json() == ["frame (1).png", "frame.png"]


async def test_validation_errors_in_comfy_shape(env):
    r = await queue(env, {"1": {"class_type": "KSampler", "inputs": {}}})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_prompt"
    r = await queue(env, {"1": {"class_type": "LoomText", "inputs": {"value": "x"}}})
    assert r.json()["error"]["type"] == "prompt_no_outputs"
    r = await queue(env, {"2": {"class_type": "LoomSeedreamImage", "inputs": {"model": "nope", "size": "2K"}}})
    body = r.json()
    assert body["error"]["type"] == "prompt_outputs_failed_validation"
    errors = {e["extra_info"]["input_name"]: e["type"] for e in body["node_errors"]["2"]["errors"]}
    assert errors == {"model": "value_not_in_list"}  # missing settings take their defaults
    r = await queue(env, {"2": {"class_type": "LoomSeedanceVideo", "inputs": {"model": "ark/dreamina-seedance-2-0",
                                                                              "first_frame": "a.png"}}})
    errors = {e["extra_info"]["input_name"]: e["type"] for e in r.json()["node_errors"]["2"]["errors"]}
    assert errors == {"first_frame": "bad_linked_input"}


async def test_image_run_is_cached_and_rerun_on_change(env):
    route = env.mock.post(f"{ARK}/images/generations").respond(json=image_reply())
    with bus.subscribe(LOCAL_USER) as events:
        r = await queue(env, IMAGE_GRAPH)
        assert r.status_code == 200, r.text
        run_id = r.json()["prompt_id"]
        await run_all(env)
        seen = []
        while not events.empty():
            seen.append(events.get_nowait())
    types = [m["type"] for m in seen]
    assert types.index("execution_start") < types.index("executed") < types.index("execution_success")
    assert all(m["sid"] in (None, "tab1") for m in seen)
    assert route.call_count == 1
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer ark-key"
    assert '"model":"dola-seedream-5-0-pro"' in sent.content.decode() and "a cat in a hat" in sent.content.decode()

    h = (await env.client.get(f"/api/history/{run_id}")).json()[run_id]
    assert h["status"]["status_str"] == "success"
    item = h["outputs"]["2"]["images"][0]
    assert item["type"] == "output"
    assert (await env.client.get("/api/view", params={"filename": item["filename"], "type": "output"})).content == PNG
    j = await job(env, run_id)
    assert j["status"] == "completed" and j["preview_output"]["mediaType"] == "images"

    # the same graph again: the result is reused, nothing is paid, and the results gallery doesn't repeat it
    again = (await queue(env, IMAGE_GRAPH)).json()["prompt_id"]
    await run_all(env)
    assert route.call_count == 1
    j = await job(env, again)
    assert j["status"] == "completed" and j["outputs_count"] == 0 and "preview_output" not in j
    assert j["outputs"]["2"]["images"][0]["filename"] == item["filename"]  # the run itself still shows it
    # another prompt: generated again
    changed = {**IMAGE_GRAPH, "1": {"class_type": "LoomText", "inputs": {"value": "a dog"}}}
    await queue(env, changed)
    await run_all(env)
    assert route.call_count == 2


async def test_failed_node_shows_the_providers_message(env):
    env.mock.post(f"{ARK}/images/generations").respond(429, json={"error": {
        "code": "AccountOverdueError", "message": "Your account balance is insufficient"}})
    r = await queue(env, IMAGE_GRAPH)
    run_id = r.json()["prompt_id"]
    await run_all(env)
    j = await job(env, run_id)
    assert j["status"] == "failed" and j["execution_error"]["node_id"] == "2"
    assert j["execution_error"]["exception_message"] == ("ark: too many requests or out of credit: "
                                                         "Your account balance is insufficient")
    env.mock.post(f"{ARK}/images/generations").respond(401, json={"error": {"message": "invalid key"}})
    assert await error_of(env, {**IMAGE_GRAPH, "1": {"class_type": "LoomText", "inputs": {"value": "b"}}}) == (
        "ark refused the key: invalid key")
    env.mock.post(f"{ARK}/images/generations").mock(side_effect=httpx.ConnectError("down"))
    assert await error_of(env, {**IMAGE_GRAPH, "1": {"class_type": "LoomText", "inputs": {"value": "c"}}}) == (
        "ark did not answer: ConnectError")


async def test_video_from_first_frame(env):
    frame = await upload(env)
    create = video_mocks(env)
    graph = {
        "1": {"class_type": "LoomLoadImage", "inputs": {"image": frame}},
        "2": {"class_type": "LoomSeedanceVideo", "inputs": {
            "model": "ark/dreamina-seedance-2-0", "prompt": "bring it to life", "first_frame": ["1", 0],
            "resolution": "720p", "ratio": "adaptive", "duration": 5, "sound": True, "seed": 1, "watermark": False}},
    }
    run_id = (await queue(env, graph)).json()["prompt_id"]
    await run_all(env)
    j = await job(env, run_id)
    assert j["status"] == "completed", j
    body = create.calls[0].request.content.decode()
    assert '"role":"first_frame"' in body and "data:image/png;base64," in body
    out = j["outputs"]["2"]
    assert out["animated"] == [True] and out["images"][0]["filename"].endswith(".mp4")
    async with db.sessionmaker()() as s:
        kinds = sorted(f.content_type for f in (await s.execute(select(StoredFile))).scalars())
        assert kinds == ["image/png", "image/png", "video/mp4"]  # the upload, the last frame, the video
        # the price: 1000 video tokens at $7 per million
        assert (await s.get(Run, run_id)).extra["loom_costs"]["2"] == 7000


async def test_result_links_only_from_the_providers_storage(env):
    env.mock.post(f"{ARK}/contents/generations/tasks").respond(json={"id": "cgt-2"})
    env.mock.get(f"{ARK}/contents/generations/tasks/cgt-2").respond(json={
        "id": "cgt-2", "status": "succeeded", "content": {"video_url": "https://169.254.169.254/latest/meta-data"}})
    graph = {"2": {"class_type": "LoomSeedanceVideo", "inputs": {"model": "ark/dreamina-seedance-2-0", "prompt": "x"}}}
    assert await error_of(env, graph) == "Invalid file address"


async def test_reference_slots_and_frames_do_not_mix(env):
    a = await upload(env, "a.png")
    graph = {
        "1": {"class_type": "LoomLoadImage", "inputs": {"image": a}},
        "2": {"class_type": "LoomSeedanceVideo", "inputs": {
            "model": "ark/dreamina-seedance-2-0", "prompt": "x", "first_frame": ["1", 0],
            "references.reference 1": ["1", 0], "references.reference 2": ["1", 0],
            "resolution": "720p", "ratio": "adaptive", "duration": 5, "sound": True, "seed": 0, "watermark": False}},
    }
    assert "do not go with the first and last frames" in await error_of(env, graph)


async def test_run_resumes_after_its_process_died(env):
    """A run left `running` by a dead process goes back to the queue; the video task it started is followed, not
    started again."""
    create = video_mocks(env, "cgt-7")
    graph = {"2": {"class_type": "LoomSeedanceVideo", "inputs": {
        "model": "ark/dreamina-seedance-2-0", "prompt": "the sea", "resolution": "720p", "ratio": "16:9",
        "duration": 5, "sound": False, "seed": 3, "watermark": False}}}
    run_id = (await queue(env, graph)).json()["prompt_id"]
    # what the dead process left: the run claimed long ago, the task id remembered for the node's signature
    sig = engine._sig(nodes.BY_ID["LoomSeedanceVideo"], dict(graph["2"]["inputs"]), {})
    async with db.sessionmaker()() as s:
        run = await s.get(Run, run_id)
        run.status, run.worker, run.heartbeat_at = "running", "dead:1", db.utcnow() - timedelta(minutes=5)
        s.add(NodeResult(user_id=LOCAL_USER, sig=sig, node_type="LoomSeedanceVideo", task_id="cgt-7"))
        await s.commit()
    await runs.dispatcher.recover()
    await run_all(env)
    j = await job(env, run_id)
    assert j["status"] == "completed", j.get("execution_error", {}).get("exception_message")
    assert create.call_count == 0  # followed, not created again


async def test_cancel_pending_and_queue_view(env):
    r = await queue(env, IMAGE_GRAPH)
    run_id = r.json()["prompt_id"]
    q = (await env.client.get("/api/queue")).json()
    assert [x[1] for x in q["queue_pending"]] == [run_id]
    r = await env.client.post(f"/api/jobs/{run_id}/cancel", headers=SAME)
    assert r.json()["cancelled"] is True
    assert (await job(env, run_id))["status"] == "cancelled"
    assert (await env.client.get("/api/queue")).json()["queue_pending"] == []


async def test_userdata_workflows(env):
    wf = b'{"nodes": [], "links": []}'
    r = await env.client.post("/api/userdata/workflows%2Fmy.json?overwrite=false&full_info=true", headers=SAME,
                              content=wf)
    assert r.status_code == 200 and r.json()["path"] == "workflows/my.json"
    r = await env.client.post("/api/userdata/workflows%2Fmy.json?overwrite=false", headers=SAME, content=wf)
    assert r.status_code == 409
    listed = (await env.client.get("/api/userdata", params={"dir": "workflows", "recurse": "true",
                                                             "full_info": "true"})).json()
    assert [x["path"] for x in listed] == ["my.json"]
    r = await env.client.post("/api/userdata/workflows%2Fmy.json/move/workflows%2Fnew.json", headers=SAME)
    assert r.json()["path"] == "workflows/new.json"
    assert (await env.client.get("/api/userdata/workflows%2Fnew.json")).content == wf
    assert (await env.client.delete("/api/userdata/workflows%2Fnew.json", headers=SAME)).status_code == 204
    assert (await env.client.get("/api/userdata/workflows%2Fnew.json")).status_code == 404
    await env.client.post("/api/settings/Comfy.Minimap.Visible", headers=SAME, json=True)
    assert (await env.client.get("/api/settings/Comfy.Minimap.Visible")).json() is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_join_videos_and_take_a_frame(env, tmp_path):
    clips = []
    for i, color in enumerate(("red", "blue")):
        p = tmp_path / f"c{i}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"color={color}:s=320x240:d=1",
                        "-pix_fmt", "yuv420p", str(p)], check=True)
        clips.append(await upload(env, f"c{i}.mp4", p.read_bytes(), "video/mp4"))
    graph = {
        "1": {"class_type": "LoomLoadVideo", "inputs": {"file": clips[0]}},
        "2": {"class_type": "LoomLoadVideo", "inputs": {"file": clips[1]}},
        "3": {"class_type": "LoomJoinVideos", "inputs": {"videos.video 1": ["1", 0], "videos.video 2": ["2", 0]}},
        "4": {"class_type": "LoomVideoFrame", "inputs": {"video": ["3", 0], "which": "last"}},
    }
    run_id = (await queue(env, graph)).json()["prompt_id"]
    await run_all(env)
    j = await job(env, run_id)
    assert j["status"] == "completed", j
    joined = j["outputs"]["3"]["images"][0]["filename"]
    data = (await env.client.get("/api/view", params={"filename": joined, "type": "output"})).content
    p = tmp_path / "out.mp4"
    p.write_bytes(data)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout
    assert 1.8 < float(dur) < 2.3
    assert j["outputs"]["4"]["images"][0]["filename"].endswith(".png")


def test_worker_id_fits_the_column():
    w = runs.worker_id("node-a.europe-west4-a.c.example-project.internal", 4194304)
    assert w.startswith("node-a:4194304:") and len(w) <= 64
    assert len(runs.worker_id("x" * 300, 1)) <= 64


async def test_no_run_when_the_files_quota_is_used_up(env, monkeypatch):
    from nodeloom import config
    await upload(env)
    monkeypatch.setenv("STORAGE_LIMIT_MB", "1")
    config.get_settings.cache_clear()
    monkeypatch.setattr("nodeloom.files.limit", lambda: 10)  # 10 bytes, already used by the upload
    r = await queue(env, IMAGE_GRAPH)
    assert r.status_code == 400 and r.json()["error"]["type"] == "quota_exceeded"
    assert "No room left for files" in r.json()["error"]["message"]


async def test_run_cost_and_deleting_results(env):
    route = env.mock.post(f"{ARK}/images/generations").respond(json=image_reply())
    with bus.subscribe(LOCAL_USER) as events:
        run_id = (await queue(env, IMAGE_GRAPH)).json()["prompt_id"]
        await run_all(env)
        seen = []
        while not events.empty():
            seen.append(events.get_nowait())
    cost = next(m["data"] for m in seen if m["type"] == "nodeloom.cost")
    assert cost["paid"] and cost["cost_micro"] == 40_000 and cost["text"] == "$0.04" and cost["unpriced"] == 0
    assert any(m["type"] == "__text" and m["data"]["text"] == "Done · $0.04" for m in seen)
    async with db.sessionmaker()() as s:
        run = await s.get(Run, run_id)
        assert run.extra["loom_cost_micro"] == cost["cost_micro"] == run.extra["loom_costs"]["2"]
        out = (await s.execute(select(StoredFile).where(StoredFile.kind == "output"))).scalar_one()
        assert out.run_id == run_id

    # deleting the picture in the gallery deletes its run and the file
    r = await env.client.post("/api/history", headers=SAME, json={"delete": [run_id]})
    assert r.status_code == 200
    assert (await env.client.get(f"/api/jobs/{run_id}")).status_code == 404
    assert (await env.client.get("/api/view", params={"filename": out.name, "type": "output"})).status_code == 404
    # the same graph again: the result it would reuse is gone, so it is made again (not a broken picture)
    await queue(env, IMAGE_GRAPH)
    await run_all(env)
    assert route.call_count == 2


async def test_a_model_without_a_price_makes_the_cost_unknown(env):
    env.mock.post(f"{ARK}/chat/completions").respond(json={
        "choices": [{"message": {"role": "assistant", "content": "hi"}}], "usage": {"prompt_tokens": 9}})
    graph = {"t": {"class_type": "LoomTextModel", "inputs": {"model": "ark/deepseek-v4-flash", "prompt": "hi"}},
             **IMAGE_GRAPH}
    env.mock.post(f"{ARK}/images/generations").respond(json=image_reply())
    with bus.subscribe(LOCAL_USER) as events:
        await queue(env, graph)
        await run_all(env)
        seen = []
        while not events.empty():
            seen.append(events.get_nowait())
    cost = next(m["data"] for m in seen if m["type"] == "nodeloom.cost")
    assert cost["nodes"] == {"2": 40_000} and cost["unpriced"] == 1  # the picture's price is known, the text's not


async def test_stopping_process_hands_runs_over_without_losing_the_video(env):
    """A deploy stops the process mid-video: the task is not cancelled upstream, the run goes back to the queue at once,
    and the next process follows the same task."""
    done = {"now": False}
    create = env.mock.post(f"{ARK}/contents/generations/tasks").respond(json={"id": "cgt-9"})
    delete_ = env.mock.delete(f"{ARK}/contents/generations/tasks/cgt-9").respond(json={})

    def status(request):
        if not done["now"]:
            return httpx.Response(200, json={"id": "cgt-9", "status": "running"})
        return httpx.Response(200, json={"id": "cgt-9", "status": "succeeded", "resolution": "720p",
                                         "content": {"video_url": f"{TOS}/v.mp4", "last_frame_url": f"{TOS}/last.png"},
                                         "usage": {"completion_tokens": 1000, "total_tokens": 1000}})
    env.mock.get(f"{ARK}/contents/generations/tasks/cgt-9").mock(side_effect=status)
    env.mock.get(f"{TOS}/v.mp4").respond(content=b"\x00\x00\x00\x18ftypmp42", headers={"content-type": "video/mp4"})
    env.mock.get(f"{TOS}/last.png").respond(content=PNG, headers={"content-type": "image/png"})
    image = env.mock.post(f"{ARK}/images/generations").respond(json=image_reply())
    video = {"class_type": "LoomSeedanceVideo", "inputs": {"model": "ark/dreamina-seedance-2-0", "prompt": "waves",
                                                           "resolution": "720p", "duration": 5, "sound": False}}
    graph = {"v": video, **IMAGE_GRAPH}  # a video and, alongside, a picture
    run_id = (await queue(env, graph)).json()["prompt_id"]
    await runs.dispatcher.claim()
    for _ in range(300):
        async with db.sessionmaker()() as s:
            made = ((await s.get(Run, run_id)).extra or {}).get("loom_new", [])
        if create.called and "2" in made:  # the picture done before the deploy
            break
        await asyncio.sleep(0.01)
    await runs.dispatcher.stop()  # the deploy
    async with db.sessionmaker()() as s:
        run = await s.get(Run, run_id)
        assert run.status == "pending" and run.worker is None
    assert not delete_.called

    runs.dispatcher.stopping = False  # the next process
    done["now"] = True
    await run_all(env)
    j = await job(env, run_id)
    assert j["status"] == "completed", j.get("execution_error")
    assert create.call_count == 1 and image.call_count == 1
    # the picture made before the handover is still this run's: in the gallery and in its cost
    assert j["outputs_count"] == 2
    async with db.sessionmaker()() as s:
        costs = (await s.get(Run, run_id)).extra["loom_costs"]
    assert set(costs) == {"2", "v"} and all(v > 0 for v in costs.values())


async def test_a_character_goes_to_video_by_id(env):
    create = video_mocks(env)
    graph = {
        "1": {"class_type": "LoomSeedanceCharacter", "inputs": {"character": "asset://asset-1"}},
        "2": {"class_type": "LoomSeedanceVideo", "inputs": {
            "model": "ark/dreamina-seedance-2-0", "prompt": "she waves", "first_frame": ["1", 0],
            "resolution": "720p", "ratio": "adaptive", "duration": 5, "sound": True, "seed": 1, "watermark": False}},
    }
    run_id = (await queue(env, graph)).json()["prompt_id"]
    await run_all(env)
    j = await job(env, run_id)
    assert j["status"] == "completed", j
    sent = create.calls[0].request.content.decode()
    assert '"url":"asset://asset-1"' in sent and "base64" not in sent
    assert "1" not in (j["outputs"] or {})  # the node shows its preview itself (nodeloom.js)
    # a picture model cannot take a character: the node says what to do
    graph = {"1": {"class_type": "LoomSeedanceCharacter", "inputs": {"character": "asset-1"}},
             "2": {"class_type": "LoomSeedreamImage", "inputs": {"model": "ark/dola-seedream-5-0-pro",
                                                                 "prompt": "a portrait",
                                                                 "references.reference 1": ["1", 0]}}}
    assert "works only in Seedance video" in await error_of(env, graph)
    bad = {"1": {"class_type": "LoomSeedanceCharacter", "inputs": {"character": "../etc"}}, "2": graph["2"]}
    assert await error_of(env, bad) == "Choose a character"


async def test_character_library_of_a_gateway(env):
    """A gateway in front of ModelArk may list ready characters (GET /assets): the picker and the previews go through
    the backend; without one the picker says so."""
    env.mock.get(f"{ARK}/assets").respond(json={"object": "list", "total": 1, "items": [
        {"id": "asset-1", "gender": "female", "age": 30, "country": "Australia", "profession": "Student",
         "title": "Woman, 30", "subtitle": "Student, Australia"}]})
    env.mock.get(f"{ARK}/assets/asset-1/preview").respond(content=PNG, headers={"content-type": "image/png"})
    r = (await env.client.get("/api/nodeloom/assets", params={"gender": "female"})).json()
    assert r["available"] and r["total"] == 1 and r["items"][0]["id"] == "asset-1"
    pic = await env.client.get("/api/view", params={"filename": "asset-1", "type": "asset"})
    assert pic.status_code == 200 and pic.content == PNG and "sandbox" in pic.headers["content-security-policy"]
    assert (await env.client.get("/api/view", params={"filename": "../x", "type": "asset"})).status_code == 404


async def test_no_character_library_without_a_gateway(env):
    env.mock.get(f"{ARK}/assets").respond(404, json={"error": {"message": "not found"}})
    assert (await env.client.get("/api/nodeloom/assets")).json() == {"total": 0, "items": [], "available": False}


def test_frontend_strings_are_replaced(tmp_path):
    from nodeloom import api
    f = tmp_path / "settingStore-x.js"
    f.write_bytes(b"let n=getUnconflictedPath(GI.basePath+(e??`Unsaved Workflow.json`));")
    assert api._patched(str(f)) == ("let n=getUnconflictedPath(GI.basePath+(e??`${document.documentElement.lang==='en'"
                                     "?'Untitled':'Без названия'}.json`));").encode()
    g = tmp_path / "other.js"
    g.write_bytes(b"nothing to change")
    assert api._patched(str(g)) is None


def test_a_character_shown_is_not_a_job_preview():
    """The editor's job list takes only files (input/output/temp) and drops the whole list over a character."""
    from nodeloom.api import _preview
    character = {"filename": "asset-1", "subfolder": "", "type": "asset"}
    assert _preview({"7": {"images": [character]}}) == (0, None)
    count, preview = _preview({"7": {"images": [character]}, "9": {"images": [{**character, "type": "output"}]}})
    assert count == 1 and preview["nodeId"] == "9" and preview["type"] == "output"
