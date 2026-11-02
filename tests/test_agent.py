"""The Builder: the model's graph becomes the editor's API prompt and back, it is checked by the engine's validation
(file pickers left for the user), laid out in columns, and a turn streams steps and the graph (the LLM mocked)."""

import json

import httpx

from nodeloom import agent
from tests.conftest import ARK, SAME

CHOICES = {"models:chat": ["ark/seed-2-0-lite"], "models:seedream": ["ark/dola-seedream-5-0-pro"],
           "models:seedance": ["ark/dreamina-seedance-2-0"], "models:openai-image": [],
           "files:image": [], "files:video": [], "files:audio": []}

GRAPH = [  # a picture from an uploaded photo and a text model's prompt, then a video with a character
    {"id": "1", "type": "LoomLoadImage", "title": "Product photo"},
    {"id": "2", "type": "LoomTextModel", "values": {"model": "ark/seed-2-0-lite", "prompt": "Write an ad prompt"}},
    {"id": "3", "type": "LoomSeedreamImage", "values": {"model": "ark/dola-seedream-5-0-pro"},
     "links": {"prompt": "2.0", "references": ["1.0"]}},
    {"id": "4", "type": "LoomSeedanceCharacter", "values": {"character": "asset-1"}},
    {"id": "5", "type": "LoomSeedanceVideo", "values": {"model": "ark/dreamina-seedance-2-0", "prompt": "she smiles",
                                                        "duration": 5},
     "links": {"references": ["4.0", "3"]}},
]


def test_graph_to_prompt_and_back():
    prompt, titles, problems = agent.to_prompt(GRAPH)
    assert problems == []
    assert titles == {"1": "Product photo"}
    assert prompt["3"] == {"class_type": "LoomSeedreamImage", "inputs": {
        "model": "ark/dola-seedream-5-0-pro", "prompt": ["2", 0], "references.reference 1": ["1", 0]}}
    assert prompt["5"]["inputs"]["references.reference 2"] == ["3", 0]
    back = {n["id"]: n for n in agent.compact(prompt)}
    assert back["5"]["links"] == {"references": ["4.0", "3.0"]}
    assert back["3"]["links"] == {"prompt": "2.0", "references": ["1.0"]}
    assert back["2"]["values"] == {"model": "ark/seed-2-0-lite", "prompt": "Write an ad prompt"}

    errors, notes = agent.check(prompt, CHOICES)
    assert errors == [] and notes == ["node 1 “Image”: the user picks the file"]
    pos = agent.layout(prompt)
    assert pos["1"][0] == pos["2"][0] == pos["4"][0] < pos["3"][0] < pos["5"][0]  # columns by depth
    assert pos["2"][1] > pos["1"][1]  # stacked in a column


def test_mistakes_come_back_to_the_model():
    bad = [{"id": "1", "type": "LoomMagic"},
           {"id": "2", "type": "LoomSeedreamImage", "values": {"model": "gpt-image", "prompt": "x", "colour": "red"},
            "links": {"references": ["9.0"]}},
           {"id": "3", "type": "LoomSeedanceVideo", "values": {"model": "ark/dreamina-seedance-2-0", "duration": 99},
            "links": {"first_frame": "2.5"}}]
    prompt, _, problems = agent.to_prompt(bad)
    assert problems == ["node 1: no node type 'LoomMagic' in the catalog",
                        "node 2 (LoomSeedreamImage): no setting 'colour'",
                        "node 2: 'references.reference 1' links to node 9, which is not in the graph"]
    errors, _ = agent.check(prompt, CHOICES)
    assert any("This value is not in the list" in e and "gpt-image" in e for e in errors)
    assert any("Above the maximum" in e for e in errors)
    assert "ark/seed-2-0-lite" in agent.catalog(CHOICES) and "file the user uploads" in agent.catalog(CHOICES)
    assert "LoomGPTImage" not in agent.catalog(CHOICES)  # no models for it: not offered
    mixed = {"1": {"class_type": "LoomSeedreamImage", "inputs": {"model": "ark/dola-seedream-5-0-pro", "prompt": "cup"}},
             "2": {"class_type": "LoomSeedanceVideo", "inputs": {
                 "model": "ark/dreamina-seedance-2-0", "prompt": "steam", "ratio": "9:16", "first_frame": ["1", 0],
                 "references.reference 1": ["1", 0]}}}
    errors, _ = agent.check(mixed, CHOICES)  # the node refuses these only when it runs: the agent learns now
    assert [e.split(": ", 1)[1][:30] for e in errors] == ["references cannot be combined ",
                                                          "with a first frame the ratio m"]


def reply(content="", calls=()):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                             for i, (n, a) in enumerate(calls)]
    return httpx.Response(200, json={"id": "c", "object": "chat.completion", "model": "seed-2-0-lite-260428",
                                     "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})


async def turn(env, messages, graph=None):
    r = await env.client.post("/api/nodeloom/agent", headers=SAME, json={"messages": messages, "graph": graph})
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line]


async def test_a_turn_builds_the_graph(env):
    env.mock.get(f"{ARK}/assets").respond(json={"total": 2, "items": [
        {"id": "asset-1", "gender": "female", "age": 30, "country": "Australia", "profession": "Student"},
        {"id": "asset-2", "gender": "female", "age": 41, "country": "Spain", "profession": "Chef"}]})
    video = [{"id": "1", "type": "LoomSeedanceCharacter", "values": {"character": "asset-1"}},
             {"id": "2", "type": "LoomSeedanceVideo", "values": {"model": "ark/dreamina-seedance-2-0",
                                                                 "prompt": "she waves"},
              "links": {"references": ["1.0"]}}]
    llm = env.mock.post(f"{ARK}/chat/completions").mock(side_effect=[
        reply(calls=[("search_characters", {"gender": "female"})]),
        reply(calls=[("set_graph", {"nodes": [{**video[0], "values": {"character": "asset-nope"}}, video[1]]})]),
        reply(calls=[("set_graph", {"nodes": video})]),
        reply("Done: a character and a video. Press Run."),
    ])
    events = await turn(env, [{"role": "user", "text": "A video: a woman waves"}])
    assert [e["type"] for e in events] == ["step", "step", "step", "done"]
    assert [e["label"] for e in events[:3]] == ["Choosing characters: 2 found", "Graph: fixing mistakes",
                                                "Graph: 2 nodes"]
    done = events[-1]
    assert done["text"].startswith("Done")
    assert done["graph"]["prompt"]["2"]["inputs"]["references.reference 1"] == ["1", 0]
    assert set(done["graph"]["layout"]) == {"1", "2"}
    sent = [json.loads(c.request.content) for c in llm.calls]
    assert sent[0]["model"] == "seed-2-0-lite"  # the first chat model, without its provider
    system = sent[0]["messages"][0]["content"]
    assert "LoomSeedanceVideo “Seedance video”" in system and "ark/dreamina-seedance-2-0" in system
    assert "search_characters" in system and {t["function"]["name"] for t in sent[0]["tools"]} == {
        "set_graph", "search_characters"}
    assert "not in the library" in json.loads(sent[2]["messages"][-1]["content"])["errors"][0]

    # a change: the canvas goes with the request; an answer without saving is checked once
    prompt = done["graph"]["prompt"]
    llm = env.mock.post(f"{ARK}/chat/completions").mock(side_effect=[reply("Done, changed it"),
                                                                     reply("I changed nothing.")])
    events = await turn(env, [{"role": "user", "text": "A video: a woman waves"},
                              {"role": "assistant", "text": "Done"},
                              {"role": "user", "text": "What is on the canvas now?"}], graph=prompt)
    assert events == [{"type": "done", "text": "I changed nothing."}]  # no graph: the canvas stays
    first = json.loads(llm.calls[-2].request.content)["messages"]
    assert first[-1]["role"] == "user" and '"type": "LoomSeedanceVideo"' in first[-1]["content"]
    assert json.loads(llm.calls[-1].request.content)["messages"][-1]["role"] == "system"


async def test_the_agent_model_and_its_errors(env, monkeypatch):
    from nodeloom import config
    env.mock.get(f"{ARK}/assets").respond(404)
    llm = env.mock.post("https://openai.test/v1/chat/completions").respond(402, json={"error": {
        "message": "You exceeded your current quota"}})
    monkeypatch.setenv("AGENT_MODEL", "openai/gpt-5-mini")
    config.get_settings.cache_clear()
    events = await turn(env, [{"role": "user", "text": "a picture of a cat"}])
    assert events == [{"type": "error", "message": "openai: You exceeded your current quota"}]
    sent = json.loads(llm.calls[0].request.content)
    assert sent["model"] == "gpt-5-mini" and [t["function"]["name"] for t in sent["tools"]] == ["set_graph"]
    assert (await turn(env, []))[0] == {"type": "error", "message": "Write what to build"}
    r = await env.client.post("/api/nodeloom/agent", json={"messages": []}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403  # pages of other sites cannot spend the user's money


def test_sound_rules_for_the_agent():
    rules = agent.video_rules
    assert rules({"model": "ark/dreamina-seedance-2-0", "audio.audio 1": ["1", 0]}) == [
        "sound needs at least one reference (a picture or a character) on this model"]
    assert rules({"model": "ark/dreamina-seedance-2-5", "audio.audio 1": ["1", 0]}) == []  # 2.5 takes sound alone
    assert "sound cannot be combined" in rules({"first_frame": ["1", 0], "audio.audio 1": ["2", 0]})[0]
    assert rules({"references.reference 1": ["1", 0], "audio.audio 1": ["2", 0], "sound": False}) == [
        "with sound linked, the value 'sound' must be true"]
    assert rules({"model": "ark/dreamina-seedance-2-0", "duration": 20}) == [
        "duration must be 4 to 15 seconds on this model, or -1"]
    assert rules({"model": "ark/dreamina-seedance-2-5", "duration": 30}) == []
    assert "“audio”" in agent.catalog(CHOICES) and "@Audio 1" in agent.catalog(CHOICES)
