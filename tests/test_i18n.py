"""Russian: the user's language (the editor's Comfy.Locale, else LANGUAGE, else the browser's) picks the editor's
locale; the nodes' Russian comes through /api/i18n in the editor's own format while /object_info keeps the English
names (saved graphs stay valid in both languages), list words are accepted in both, messages follow the language."""

import re

from nodeloom import engine, nodes
from tests.conftest import SAME, error_of, queue

CYR = re.compile("[А-Яа-яЁё]")
RU = {"Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"}


def test_every_node_has_russian():
    payload = nodes.i18n_ru()
    defs = payload["nodeDefs"]
    assert set(defs) == {n.id for n in nodes.NODES}
    for n in nodes.NODES:
        d = defs[n.id]
        assert not CYR.search(n.title + n.description + n.category), n.id  # the node's own words are English
        assert CYR.search(d["description"]), n.id
        for i in n.inputs:
            assert i.label_ru and not CYR.search(i.label), (n.id, i.name)
            assert not i.tooltip or (i.tip_ru and not CYR.search(i.tooltip)), (n.id, i.name)
            assert not i.extra.get("placeholder") or (i.hint_ru and not CYR.search(i.extra["placeholder"])), (n.id, i.name)
            assert d["inputs"][i.name]["name"] == i.label_ru
        for k, o in enumerate(n.outputs):
            assert d["outputs"][str(k)]["name"] == o.label_ru and not CYR.search(o.label), (n.id, k)
    assert payload["nodeCategories"]["Generate"] == "Генерация"
    assert defs["LoomSeedanceVideo"]["inputs"]["reference 2"] == {"name": "референс 2"}


async def test_the_language_follows_the_browser_then_the_users_choice(env):
    en_info = (await env.client.get("/api/object_info")).json()
    assert (await env.client.get("/api/settings", headers=RU)).json()["Comfy.Locale"] == "ru"
    assert (await env.client.get("/api/settings/Comfy.Locale", headers=RU)).json() == "ru"
    page = await env.client.get("/", headers=RU)
    assert '<html lang="ru">' in page.text
    info = (await env.client.get("/api/object_info", headers=RU)).json()
    # names, labels and slot names stay English (the translation is /api/i18n); words in lists become Russian
    assert info["LoomSeedanceVideo"]["display_name"] == en_info["LoomSeedanceVideo"]["display_name"] == "Seedance video"
    assert info["LoomSeedanceVideo"]["input"]["optional"]["references"] == en_info["LoomSeedanceVideo"]["input"][
        "optional"]["references"]
    which = info["LoomVideoFrame"]["input"]["required"]["which"][1]
    assert which["options"] == ["последний", "первый"] and which["default"] == "последний"
    assert en_info["LoomVideoFrame"]["input"]["required"]["which"][1]["options"] == ["last", "first"]
    assert info["LoomText"]["input"]["required"]["value"][1]["placeholder"] == "Напишите текст"
    strings = (await env.client.get("/api/i18n")).json()
    assert strings["ru"]["nodeDefs"]["LoomSeedanceVideo"]["display_name"] == "Видео Seedance"
    assert strings["ru"]["sideToolbar"]["workflows"] == "Воркфлоу"
    # the user's own choice (the language menu saves the editor's setting) wins over the browser
    await env.client.post("/api/settings/Comfy.Locale", headers=SAME, json="en")
    assert (await env.client.get("/api/settings", headers=RU)).json()["Comfy.Locale"] == "en"


def test_list_words_are_accepted_in_both_languages():
    frame = nodes.BY_ID["LoomVideoFrame"]
    assert engine.grouped(frame, {"which": "первый"})["which"] == "first"
    assert engine.grouped(frame, {"which": "first"})["which"] == "first"
    prompt = {"1": {"class_type": "LoomLoadVideo", "inputs": {"file": "a.mp4"}},
              "2": {"class_type": "LoomVideoFrame", "inputs": {"video": ["1", 0], "which": "первый"}}}
    assert engine.validate(prompt, {"files:video": ["a.mp4"]}) == ["1", "2"]


async def test_messages_in_russian(env):
    await env.client.post("/api/settings/Comfy.Locale", headers=SAME, json="ru")
    bad = {"1": {"class_type": "LoomSeedreamImage", "inputs": {"model": "nope", "prompt": "x", "size": "2K",
                                                               "count": 99, "seed": 0, "watermark": False}}}
    r = await queue(env, bad)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["message"] == "Граф не прошёл проверку"
    assert "Такого значения нет в списке" in err["details"] and "Больше допустимого (15)" in err["details"]
    empty = {"1": {"class_type": "LoomSeedreamImage", "inputs": {"model": "ark/dola-seedream-5-0-pro", "prompt": " ",
                                                                 "size": "2K", "count": 1, "seed": 0,
                                                                 "watermark": False}}}
    # the run itself (another task, outside the request) speaks the language it was queued in
    assert await error_of(env, empty) == "Напишите описание"
    r = await env.client.post("/api/nodeloom/agent", headers=SAME, json={"messages": []})
    assert r.text.strip().endswith('"message": "Напишите, что собрать"}')
