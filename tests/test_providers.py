"""Providers: who they are (config file or environment), what models they list and what kind, what a call costs, and
the OpenAI-specific nodes' requests."""

import json

import httpx
import pytest
import respx

from nodeloom import config, providers
from nodeloom.providers.pricing import cost_micro
from tests.conftest import OPENAI, PNG, SAME, configure, error_of, start, upload

GATEWAY = "https://gw.test/v1"


def test_common_providers_come_from_the_environment(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, None, openai_api_key="sk-1", openrouter_api_key="or-1", ark_api_key="ark-1")
    reg = providers.registry()
    assert {n: (p.type, p.base_url) for n, p in reg.items()} == {
        "openai": ("openai", "https://api.openai.com/v1"),
        "openrouter": ("openai", "https://openrouter.ai/api/v1"),
        "ark": ("ark", "https://ark.ap-southeast.bytepluses.com/api/v3")}
    assert reg["openai"].headers() == {"Authorization": "Bearer sk-1"}
    assert reg["openai"].max_tokens_key == "max_completion_tokens" and reg["openrouter"].max_tokens_key == "max_tokens"


def test_bad_provider_config_says_what_is_wrong(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, '[providers."my/gw"]\ntype = "openai"\n')
    with pytest.raises(ValueError, match="lowercase letters"):
        providers.registry()
    configure(monkeypatch, tmp_path / "b", '[providers.gw]\ntype = "replicate"\nbase_url = "https://x"\n')
    with pytest.raises(ValueError, match="unknown type 'replicate'"):
        providers.registry()


async def test_models_of_openai_and_openrouter(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, None, openai_api_key="sk-1", openrouter_api_key="or-1")
    with respx.mock:
        respx.get("https://api.openai.com/v1/models").respond(json={"object": "list", "data": [
            {"id": "gpt-4o-mini", "created": 1}, {"id": "gpt-5", "created": 3}, {"id": "gpt-image-1", "created": 2},
            {"id": "text-embedding-3-small", "created": 4}, {"id": "whisper-1", "created": 5},
            {"id": "gpt-4o-audio-preview", "created": 0}, {"id": "o3-pro", "created": 6}]})
        respx.get("https://openrouter.ai/api/v1/models").respond(json={"data": [
            {"id": "google/gemini-2.5-flash", "architecture": {"input_modalities": ["text", "image", "audio", "video"],
                                                               "output_modalities": ["text"]},
             "pricing": {"prompt": "0.0000003", "completion": "0.0000025"}},
            {"id": "openrouter/auto", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
             "pricing": {"prompt": "-1", "completion": "-1"}},
            {"id": "black-forest-labs/flux", "architecture": {"output_modalities": ["image"]}}]})
        models = {m.value: m for m in await providers.all_models()}
        ch = await providers.choices()
    assert ch["models:chat"] == ["openai/gpt-5", "openai/gpt-4o-mini", "openai/gpt-4o-audio-preview",
                                 "openrouter/google/gemini-2.5-flash", "openrouter/openrouter/auto"]  # newest first
    assert ch["models:openai-image"] == ["openai/gpt-image-1"]
    assert models["openai/gpt-5"].inputs == {"image"} and models["openai/gpt-4o-audio-preview"].inputs == {"image", "audio"}
    gemini = models["openrouter/google/gemini-2.5-flash"]
    assert gemini.inputs == {"image", "audio", "video"} and gemini.price == pytest.approx({"input": 0.3, "output": 2.5})
    assert models["openrouter/openrouter/auto"].price is None
    p, m = await providers.resolve("openrouter/google/gemini-2.5-flash")  # ids with a slash of their own
    assert p.name == "openrouter" and m.id == "google/gemini-2.5-flash"


async def test_models_of_modelark_and_of_a_gateway(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, f'[providers.gw]\ntype = "ark"\nbase_url = "{GATEWAY}"\n'
                                     '[providers.gw.extra_models."seed-2-0-lite"]\nprice = {input = 0.25, output = 2}\n'
                                     '[providers.gw.extra_models."ep-123"]\nkind = "video"\n',
              ark_api_key="ark-1")
    with respx.mock:
        respx.get("https://ark.ap-southeast.bytepluses.com/api/v3/models").respond(json={"data": [
            {"id": "seed-2-0-lite-260428", "domain": "VLM", "status": "Active",
             "modalities": {"input_modalities": ["text", "image", "video"], "output_modalities": ["text"]}},
            {"id": "dola-seedream-5-0-pro-260628", "domain": "ImageGeneration"},
            {"id": "dreamina-seedance-2-0-260128", "domain": "VideoGeneration"},
            {"id": "seedance-1-0-lite-250428", "domain": "VideoGeneration", "status": "Shutdown"},
            {"id": "doubao-embedding-vision", "domain": "Embedding"}]})
        respx.get(f"{GATEWAY}/models").respond(json={"data": [  # the gateway's: kind and input_modalities
            {"id": "seed-2-0-lite", "kind": "chat", "input_modalities": ["text", "image", "audio"]},
            {"id": "dola-seedream-5-0-pro", "kind": "image"}, {"id": "bge", "kind": "embedding"}]})
        ch = await providers.choices()
        models = {m.value: m for m in await providers.all_models()}
    assert ch["models:chat"] == ["gw/seed-2-0-lite", "ark/seed-2-0-lite-260428"]
    assert ch["models:seedream"] == ["gw/dola-seedream-5-0-pro", "ark/dola-seedream-5-0-pro-260628"]
    assert ch["models:seedance"] == ["gw/ep-123", "ark/dreamina-seedance-2-0-260128"]
    assert models["gw/seed-2-0-lite"].inputs == {"image", "audio"}
    assert models["gw/seed-2-0-lite"].price == {"input": 0.25, "output": 2}  # the config's over the gateway's list
    assert models["ark/seed-2-0-lite-260428"].inputs == {"image", "video"}


async def test_a_provider_away_is_asked_again_soon(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, None, openai_api_key="sk-1")
    with respx.mock:
        route = respx.get("https://api.openai.com/v1/models")
        route.side_effect = [httpx.Response(500), httpx.Response(200, json={"data": [{"id": "gpt-5"}]})]
        assert await providers.all_models() == []
        assert await providers.all_models() == []  # not asked again at once
        p = providers.get("openai")
        p._models_at -= 61
        assert [m.id for m in await providers.all_models()] == ["gpt-5"]


def test_costs():
    assert cost_micro(None, {"prompt_tokens": 1000}) is None
    assert cost_micro({"input": 0.25, "output": 2.0}, None) is None
    assert cost_micro({"input": 0.25, "output": 2.0}, {"prompt_tokens": 1000, "completion_tokens": 500}) == 1250
    assert cost_micro(None, {"cost": 0.0123}) == 12300  # OpenRouter's own
    assert cost_micro({"image": 0.04}, {"generated_images": 3, "output_tokens": 49152}) == 120_000
    gpt_image = {"input": 5.0, "image_input": 10.0, "output": 40.0}
    usage = {"input_tokens": 1300, "output_tokens": 4160, "input_tokens_details": {"text_tokens": 50, "image_tokens": 1250}}
    assert cost_micro(gpt_image, usage) == 50 * 5 + 1250 * 10 + 4160 * 40
    assert cost_micro({"output": 7.0}, {"completion_tokens": 48437, "total_tokens": 48437}) == 339_059


async def test_gpt_image_draws_and_edits(env):
    b64 = __import__("base64").b64encode(PNG).decode()
    usage = {"input_tokens": 20, "output_tokens": 1000, "input_tokens_details": {"text_tokens": 20, "image_tokens": 0}}
    gen = env.mock.post(f"{OPENAI}/images/generations").respond(json={"data": [{"b64_json": b64}], "usage": usage})
    graph = {"1": {"class_type": "LoomGPTImage", "inputs": {"model": "openai/gpt-image-1", "prompt": "a red fox",
                                                            "size": "1024x1536", "quality": "high"}}}
    assert await error_of(env, graph) == "completed"
    body = json.loads(gen.calls[0].request.content)
    assert body == {"model": "gpt-image-1", "prompt": "a red fox", "n": 1, "size": "1024x1536", "quality": "high",
                    "background": "auto"}
    assert gen.calls[0].request.headers["authorization"] == "Bearer sk-test"
    # with references: /images/edits, the pictures as image[] files
    pic = await upload(env)
    edit = env.mock.post(f"{OPENAI}/images/edits").respond(json={"data": [{"b64_json": b64}, {"b64_json": b64}]})
    graph = {"1": {"class_type": "LoomLoadImage", "inputs": {"image": pic}},
             "2": {"class_type": "LoomGPTImage", "inputs": {"model": "openai/gpt-image-1", "prompt": "the same, at night",
                                                            "references.reference 1": ["1", 0], "count": 2}}}
    r = await env.client.post("/api/prompt", headers=SAME, json={"prompt": graph})
    run_id = r.json()["prompt_id"]
    from tests.conftest import job, run_all
    await run_all(env)
    j = await job(env, run_id)
    assert j["status"] == "completed" and len(j["outputs"]["2"]["images"]) == 2
    form = edit.calls[0].request.content
    assert b'name="image[]"; filename="reference-1.png"' in form and b'name="n"\r\n\r\n2' in form
    # the seed is ours, not the model's: another number draws again, the same one reuses the picture
    assert await error_of(env, {"1": {**graph["1"]}, "2": {**graph["2"], "inputs": {**graph["2"]["inputs"],
                                                                                    "seed": 5}}}) == "completed"
    assert edit.call_count == 2


async def test_text_model_on_openai(env):
    chat = env.mock.post(f"{OPENAI}/chat/completions").respond(json={
        "choices": [{"message": {"role": "assistant", "content": "Hi"}}], "usage": {"prompt_tokens": 5}})
    graph = {"1": {"class_type": "LoomTextModel", "inputs": {"model": "openai/gpt-5-mini", "prompt": "Say hi",
                                                             "max_tokens": 50, "temperature": 1.0}}}
    assert await error_of(env, graph) == "completed"
    body = json.loads(chat.calls[0].request.content)
    # the base URL is not api.openai.com, so the length goes as max_tokens; temperature 1 is not sent
    assert body == {"model": "gpt-5-mini", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 50}


async def test_a_provider_removed_from_the_config(monkeypatch, tmp_path):
    client = await start(monkeypatch, tmp_path, f'[providers.openai]\ntype = "openai"\nbase_url = "{OPENAI}"\n'
                                                'models = ["gpt-5-mini"]\n')
    graph = {"1": {"class_type": "LoomTextModel", "inputs": {"model": "ark/seed-2-0-lite", "prompt": "hi"}}}
    r = await client.post("/api/prompt", headers=SAME, json={"prompt": graph})
    assert r.status_code == 400 and r.json()["node_errors"]["1"]["errors"][0]["type"] == "value_not_in_list"
    with pytest.raises(providers.ProviderError, match="Provider “ark” is not configured"):
        await providers.resolve("ark/seed-2-0-lite")
    await client.aclose()
    config.get_settings.cache_clear()
