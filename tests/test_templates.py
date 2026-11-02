"""Templates and the starter graph: complete, consistent with the nodes, offered only with the models they need, given
the user's models and language."""

import json
from pathlib import Path

import pytest

from nodeloom import nodes
from tests.conftest import OPENAI, start

WEB = Path(nodes.__file__).parent / "web"
TEMPLATES = WEB / "templates"
GRAPHS = sorted(p for p in TEMPLATES.glob("*.json") if p.name not in ("index.json", "ru.json")) + [
    WEB / "starter.json", WEB / "starter-openai.json", WEB / "starter-empty.json"]


def test_every_template_is_listed_with_its_picture():
    listed = {t["name"] for group in json.loads((TEMPLATES / "index.json").read_text()) for t in group["templates"]}
    assert listed == {p.stem for p in GRAPHS if p.parent == TEMPLATES}
    for name in listed:
        assert (TEMPLATES / f"{name}-1.jpg").is_file(), name


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.name)
def test_graphs_match_the_nodes(path):
    """Node types exist, links join outputs and inputs of the right types, slot names are the nodes' own, and the
    model is the first widget (api._fit_models swaps it by position)."""
    graph = json.loads(path.read_text())
    by_id = {n["id"]: n for n in graph["nodes"]}
    for n in graph["nodes"]:
        node = nodes.BY_ID.get(n["type"])
        assert node is not None or n["type"] == "Note", n["type"]
        if node is None:
            continue
        for i in n.get("inputs") or []:
            if i["name"] in ("upload", "audioUI"):  # the editor's own widgets of upload nodes
                continue
            base, dot, slot = i["name"].partition(".")
            spec = node.input(base)
            assert spec is not None, (path.name, n["type"], i["name"])
            if dot:
                assert slot in [nodes.slot_name(spec, k) for k in range(1, spec.grow + 1)], (path.name, i["name"])
        assert [o["name"] for o in n.get("outputs") or []] == [o.label for o in node.outputs]
        if node.input("model"):
            widgets = [i["name"] for i in n["inputs"] if i.get("widget")]
            assert widgets[0] == "model", (path.name, n["type"])
            assert n["widgets_values"][0] == n["widgets_values_named"]["model"]
    for link_id, src, src_slot, dst, dst_slot, kind in graph["links"]:
        out = by_id[src]["outputs"][src_slot]
        inp = by_id[dst]["inputs"][dst_slot]
        assert inp["link"] == link_id and link_id in (out.get("links") or []), (path.name, link_id)
        assert out["type"] == kind and inp["type"] in (kind, "*", "COMBO" if kind == "STRING" else kind), \
            (path.name, link_id)


def test_russian_table_has_no_conflicts_with_values():
    """Translating a template never touches what is not text for people (types, media kinds, widget names)."""
    table = json.loads((TEMPLATES / "ru.json").read_text())
    assert "image" in table["strings"]  # a label...
    assert "image" not in table["values"]  # ...but not the upload widget's value, which stays "image"


async def test_templates_follow_the_users_models_and_language(env):
    index = (await env.client.get("/templates/index.json")).json()
    names = {t["name"] for group in index for t in group["templates"]}
    assert {"idea-image-video", "gpt-text-to-image", "describe-image"} <= names
    graph = (await env.client.get("/templates/idea-image-video.json")).json()
    models = {n["type"]: n["widgets_values"][0] for n in graph["nodes"] if nodes.BY_ID[n["type"]].input("model")}
    # the template's own where the user has them (ark/deepseek-v4-flash), else the first of the user's list
    assert models == {"LoomTextModel": "ark/deepseek-v4-flash", "LoomSeedreamImage": "ark/dola-seedream-5-0-pro",
                      "LoomSeedanceVideo": "ark/dreamina-seedance-2-0"}
    assert graph["nodes"][0]["title"] == "Idea"
    ru = (await env.client.get("/templates/index.ru.json", headers={"Accept-Language": "ru"})).json()
    assert ru[0]["title"] == "Начало" and ru[0]["templates"][0]["title"] == "Идея, картинка, видео"
    assert ru[0]["templates"][0]["mediaType"] == "image"  # not a word to translate
    graph = (await env.client.get("/templates/idea-image-video.json", headers={"Accept-Language": "ru"})).json()
    text = json.dumps(graph, ensure_ascii=False)
    assert "Уютная кофейня" in text and "Идея" in text
    refs = json.dumps((await env.client.get("/templates/references-to-image.json",
                                            headers={"Accept-Language": "ru"})).json(), ensure_ascii=False)
    assert '"name": "references.reference 1"' in refs and '"localized_name": "референс 1"' in refs
    assert '"upload": "image"' in refs  # the upload widget's value is not a word
    assert (await env.client.get("/templates/ru.json")).status_code == 404
    starter = (await env.client.get("/extensions/nodeloom/starter.json")).json()
    assert {n["type"] for n in starter["nodes"]} >= {"LoomSeedanceVideo"}


async def test_an_openai_user_gets_openai_templates(monkeypatch, tmp_path):
    only_openai = f'[providers.mine]\ntype = "openai"\nbase_url = "{OPENAI}"\nmodels = ["gpt-4.1", "gpt-image-1.5"]\n' \
                  '[providers.mine.extra_models."gpt-image-1.5"]\nkind = "image"\n'
    client = await start(monkeypatch, tmp_path, only_openai)
    index = (await client.get("/templates/index.json")).json()
    names = {t["name"] for group in index for t in group["templates"]}
    assert names == {"describe-image", "gpt-text-to-image", "gpt-references-to-image"}
    starter = (await client.get("/extensions/nodeloom/starter.json")).json()
    assert {n["type"] for n in starter["nodes"]} == {"LoomText", "LoomTextModel", "LoomJoinText", "LoomGPTImage"}
    models = [n["widgets_values_named"]["model"] for n in starter["nodes"] if "model" in n["widgets_values_named"]]
    assert models == ["mine/gpt-4.1", "mine/gpt-image-1.5"]
    await client.aclose()
    # no provider: an empty canvas, not one of missing nodes
    client = await start(monkeypatch, tmp_path / "none", None)
    assert (await client.get("/extensions/nodeloom/starter.json")).json()["nodes"] == []
    assert (await client.get("/templates/index.json")).json() == []
    await client.aclose()
