"""Sound: "Audio file" goes to "Seedance video" as Seedance's reference_audio and to "Text model" as input_audio; any
format is converted to mp3 and the limits are checked before any money is spent."""

import base64
import json
import shutil
import subprocess

import pytest

from tests.conftest import ARK, chat_reply, error_of, upload, video_mocks

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def tone(tmp_path, seconds: float, name="tone.wav") -> bytes:
    p = tmp_path / name
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    str(p)], check=True)
    return p.read_bytes()


def video(model="ark/dreamina-seedance-2-0", **inputs):
    return {"class_type": "LoomSeedanceVideo", "inputs": {
        "model": model, "prompt": "she sings the song from @Audio 1", "resolution": "480p", "ratio": "16:9",
        "duration": 5, "sound": True, "seed": 0, "watermark": False, **inputs}}


async def test_sound_goes_to_video_as_reference_audio(env, tmp_path):
    pic = await upload(env)
    wav = await upload(env, "voice.wav", tone(tmp_path, 3), "audio/wav")
    create = video_mocks(env)
    graph = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
             "2": {"class_type": "LoomLoadAudio", "inputs": {"audio": wav}},
             "3": video(**{"references.reference 1": ["1", 0], "audio.audio 1": ["2", 0]})}
    assert await error_of(env, graph) == "completed"
    content = json.loads(create.calls[0].request.content)["content"]
    sound = [c for c in content if c["type"] == "audio_url"]
    assert len(sound) == 1 and sound[0]["role"] == "reference_audio"
    url = sound[0]["audio_url"]["url"]
    assert url.startswith("data:audio/mp3;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")  # wav became mp3


async def test_video_sound_limits(env, tmp_path):
    pic = await upload(env)
    short = await upload(env, "short.wav", tone(tmp_path, 1, "s.wav"), "audio/wav")
    long_ = await upload(env, "long.wav", tone(tmp_path, 9, "l.wav"), "audio/wav")
    ok = await upload(env, "ok.wav", tone(tmp_path, 3, "o.wav"), "audio/wav")
    base = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
            "2": {"class_type": "LoomLoadAudio", "inputs": {"audio": short}},
            "3": {"class_type": "LoomLoadAudio", "inputs": {"audio": long_}},
            "4": {"class_type": "LoomLoadAudio", "inputs": {"audio": ok}}}
    ref = {"references.reference 1": ["1", 0]}
    tasks = env.mock.post(f"{ARK}/contents/generations/tasks").respond(json={"id": "cgt-never"})
    assert "it should be 2 to 15 s" in await error_of(env, {**base, "9": video(**ref, **{"audio.audio 1": ["2", 0]})})
    two_long = {"audio.audio 1": ["3", 0], "audio.audio 2": ["3", 0]}
    assert "takes up to 15 s" in await error_of(env, {**base, "9": video(**ref, **two_long)})
    four = {f"audio.audio {k}": ["4", 0] for k in range(1, 5)}
    assert "takes at most 3 sounds" in await error_of(env, {**base, "9": video(**ref, **four)})
    assert "together with a reference" in await error_of(env, {**base, "9": video(**{"audio.audio 1": ["4", 0]})})
    framed = video(ratio="adaptive", first_frame=["1", 0], **{"audio.audio 1": ["4", 0]})
    assert "do not go with the first and last frames" in await error_of(env, {**base, "9": framed})
    silent = video(sound=False, **ref, **{"audio.audio 1": ["4", 0]})
    assert "turn on “sound”" in await error_of(env, {**base, "9": silent})
    assert "4 to 15 s" in await error_of(env, {**base, "9": video(duration=20)})  # 30 s is for Seedance 2.5
    assert not tasks.called  # every mistake is caught before the model is paid


async def test_text_model_hears_the_sound(env, tmp_path):
    wav = await upload(env, "note.wav", tone(tmp_path, 2), "audio/wav")
    chat = env.mock.post(f"{ARK}/chat/completions").respond(json=chat_reply("A cat on a windowsill", {
        "prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128, "prompt_tokens_details": {"audio_tokens": 100}}))
    graph = {"1": {"class_type": "LoomLoadAudio", "inputs": {"audio": wav}},
             "2": {"class_type": "LoomTextModel", "inputs": {
                 "model": "ark/seed-2-0-lite", "prompt": "Turn the voice message into an image prompt",
                 "audio.audio 1": ["1", 0], "temperature": 1.0, "max_tokens": 0}}}
    assert await error_of(env, graph) == "completed"
    parts = json.loads(chat.calls[0].request.content)["messages"][-1]["content"]
    assert [p["type"] for p in parts] == ["input_audio", "text"]
    assert parts[0]["input_audio"]["format"] == "mp3" and len(parts[0]["input_audio"]["data"]) > 1000
    # a model that does not hear: refused before the call
    graph["2"]["inputs"]["model"] = "ark/deepseek-v4-flash"
    assert "does not understand sound" in await error_of(env, graph)
    assert chat.call_count == 1
