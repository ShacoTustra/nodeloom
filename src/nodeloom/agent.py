"""The Builder: an agent that builds a graph of our nodes from a request.

The editor's side panel sends the conversation and the graph on the canvas (the API form the editor runs,
`app.graphToPrompt().output`). A chat model with function calling (AGENT_MODEL, else the first chat model) sees the
catalog of the nodes the user has models for, may look up ready characters (where a provider lists them), and saves
the whole graph with `set_graph`. The graph is checked by the engine's own validation (engine.validate), so what the
agent saves is what the editor can run; empty file pickers are left for the user. The answer goes back as NDJSON
events (a step, then the end with the graph and its layout); the editor rebuilds the canvas. Nothing is run: the user
presses Run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from nodeloom import engine, i18n, providers
from nodeloom.config import get_settings
from nodeloom.nodes import BY_ID, In, all_nodes, available, slot_name, video_problems
from nodeloom.providers import ProviderError
from nodeloom.providers.ark import ArkProvider

log = logging.getLogger(__name__)

MAX_STEPS = 10
MAX_NODES = 40
MAX_HISTORY = 20
COLUMN = 480  # layout: one column per depth of the graph
HEIGHT = {"LoomText": 220, "LoomLoadImage": 360, "LoomSeedanceCharacter": 420, "LoomLoadVideo": 360,
          "LoomLoadAudio": 160, "LoomTextModel": 420, "LoomGPTImage": 520, "LoomSeedreamImage": 520,
          "LoomSeedanceVideo": 620, "LoomVideo": 600, "LoomGeminiImage": 480, "LoomGeminiVideo": 640, "LoomVideoFrame": 360,
          "LoomJoinVideos": 420, "LoomJoinText": 260, "LoomShow": 260}
CHARACTER, VIDEO, ROUTER_VIDEO = "LoomSeedanceCharacter", "LoomSeedanceVideo", "LoomVideo"
CAPS = (("supported_durations", "seconds"), ("supported_resolutions", "quality"),
        ("supported_aspect_ratios", "aspect ratio"), ("supported_frame_images", "frames"))
FILE_INPUTS = ("files:image", "files:video", "files:audio")


# ---------------------------------------------------------------- the catalog the model sees


def _describe_input(i: In, choices: dict[str, list]) -> str:
    kind = i.type
    if i.type == "COMBO":
        options = choices.get(i.options, []) if isinstance(i.options, str) else list(i.options or [])
        if isinstance(i.options, str) and i.options.startswith("files:"):
            kind = "file the user uploads (leave empty)"
        else:
            kind = "one of: " + ", ".join(map(str, options[:40])) if options else "COMBO (no options available)"
    lo, hi = i.extra.get("min"), i.extra.get("max")
    if lo is not None or hi is not None:
        kind += f", {lo} to {hi}"
    parts = [f"{i.name} {_name(i.label, i.label_ru)}: {kind}"]
    if i.default not in (None, ""):
        parts.append(f"default {i.default!r}")
    if i.optional:
        parts.append("optional")
    if i.advanced:
        parts.append("rarely needed")
    if i.tooltip:
        parts.append(i.tooltip)
    return "; ".join(parts)


def _name(en: str, ru: str) -> str:
    """A node's or an input's name: the English one the rules below use, and the Russian one a Russian user sees."""
    return f"“{en}” (in Russian «{ru}»)" if i18n.current() == "ru" and ru else f"“{en}”"


def _caps_line(value: str, caps: dict) -> str:
    said = [f"{name} {', '.join(map(str, caps[key]))}" for key, name in CAPS if caps.get(key)]
    return f"    - {value}: " + "; ".join(said)


def catalog(choices: dict[str, list], caps: dict[str, dict] | None = None) -> str:
    """The available nodes for a model to read; `caps`: what the video models can do, by model value."""
    caps = caps or {}
    lines = []
    for n in all_nodes():
        if not available(n, choices):
            continue
        lines.append(f"\n{n.id} {_name(n.title, n.title_ru)} ({n.category}): {n.description}")
        values = [i for i in n.inputs if i.widget and not i.grow]
        links = [i for i in n.inputs if not i.widget or i.grow]
        if values:
            lines.append("  values:")
            lines += [f"    - {_describe_input(i, choices)}" for i in values]
        if links:
            lines.append("  link inputs:")
            for i in links:
                many = f", a list of up to {i.grow} links" if i.grow else ""
                opt = "" if many or not i.optional else ", optional"
                tip = f"; {i.tooltip}" if i.tooltip else ""
                lines.append(f"    - {i.name} {_name(i.label, i.label_ru)}: {i.type}{many}{opt}{tip}")
        model = n.input("model")
        known = [v for v in choices.get(model.options, []) if v in caps] if model and isinstance(model.options, str) else []
        if known:
            lines.append("  what each model can do (use only these values):")
            lines += [_caps_line(v, caps[v]) for v in known]
        if n.outputs:
            lines.append("  outputs: " + ", ".join(f"{k}: {o.type} {_name(o.label, o.label_ru)}"
                                                   for k, o in enumerate(n.outputs)))
        if n.output_node:
            lines.append("  shows its result on the canvas")
    return "\n".join(lines)


SYSTEM = """You are the Builder, the agent of a node editor where the user builds graphs of nodes that generate
text, pictures and video with paid models. You build or change the graph for the user's request.

The nodes (the only ones that exist):
{catalog}

The graph you save with set_graph is a list of nodes: {{"id": "1", "type": "LoomTextModel", "title": optional,
"values": {{input: value}}, "links": {{input: "<node id>.<output index>"}}}}. A link input that takes a list gets a list
of links. Any STRING value can instead be linked from a STRING output, for example a prompt from “Text model”.

How to work:
- Make the smallest graph that does the job; put prompts straight into the generating node's values unless a text
  is reused or comes from a text model. Write prompts for the models in English unless the user wants text in
  their language; spoken lines in video prompts go in quotes in the user's language.
- Use the user's models from the lists above: a model value is "<provider>/<model>", exactly as listed.
- “Video” (a router's video models): set seconds, quality and aspect ratio only to values the chosen model lists
  under the node, or leave 0 / auto for the model's default.
- “Seedance video” takes a first (and a last) frame, or up to 9 references, not both. With a first frame its ratio
  stays adaptive: for a vertical video make the picture vertical (its size). Characters together with a picture: all
  go in as references.
{characters}- File nodes (“Image”, “Video file”, “Audio file”): leave the file empty, the user picks it.
- Sound: “Audio file” goes to “Seedance video” as “audio” (a voice, music or sounds for the video, named @Audio 1 in
  the prompt, with at least one reference picture or character on Seedance 2.0) or to “Text model” (to transcribe or
  describe it, e.g. a voice message turned into a prompt).
- set_graph checks the graph like the editor does before a run and returns errors: fix them and save again.
- When a graph is already on the canvas, it is given below the request. Change it as asked and save the whole new
  graph: keep the nodes that do not change with the same ids and values, so their results are reused for free.
- Saving does not run anything: the user presses Run. Never say you ran it.
- Answer in the user's language in 1 to 3 sentences: what the graph does and what the user has to fill in
  (files to upload) before Run. Never say you built or changed the graph unless set_graph returned "saved": true in
  this turn."""

CHARACTERS = """- People on screen in a Seedance video: a “Seedance character” node whose character value is an asset id
  from search_characters, linked to “Seedance video” as a reference or the first frame. Characters work only with
  Seedance video. Photos of real people are refused by the models.
"""


def tools(characters: bool) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "set_graph",
            "description": "Save the whole graph: it replaces the canvas. Returns errors to fix, if any.",
            "parameters": {"type": "object", "required": ["nodes"], "properties": {
                "nodes": {"type": "array", "items": {"type": "object", "required": ["id", "type"], "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "description": "A node type from the catalog, e.g. LoomTextModel"},
                    "title": {"type": "string", "description": "Optional title on the canvas, user's language"},
                    "values": {"type": "object", "description": "Settings by input name"},
                    "links": {"type": "object", "description": "Link inputs: \"<node id>.<output index>\" "
                                                               "or a list of them"},
                }}}}}}},
    ] + ([{"type": "function", "function": {
            "name": "search_characters",
            "description": "Search the library of ready characters (photorealistic people) for “Seedance character” "
                           "nodes. Returns asset ids with gender, age, country and profession.",
            "parameters": {"type": "object", "properties": {
                "gender": {"type": "string", "enum": ["female", "male"]},
                "age_min": {"type": "integer"}, "age_max": {"type": "integer"},
                "q": {"type": "string", "description": "A profession or a country, in English"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12}}}}}] if characters else [])


# ---------------------------------------------------------------- graphs: the model's form and the editor's


def _link(ref) -> list | None:
    """'3.1' / '3' / {"node": "3", "output": 1} / ["3", 1] → ["3", 1]."""
    if isinstance(ref, dict):
        ref = f"{ref.get('node')}.{ref.get('output', 0)}"
    if isinstance(ref, list) and len(ref) == 2:
        ref = f"{ref[0]}.{ref[1]}"
    if not isinstance(ref, str) or not ref.strip():
        return None
    node, _, out = ref.strip().partition(".")
    try:
        return [node, int(out or 0)]
    except ValueError:
        return None


def to_prompt(nodes: list[dict]) -> tuple[dict, dict, list[str]]:
    """The model's node list → the editor's API prompt, titles, and problems the engine would not report."""
    prompt, titles, problems = {}, {}, []
    if not nodes:
        return {}, {}, ["the graph is empty"]
    if len(nodes) > MAX_NODES:
        problems.append(f"at most {MAX_NODES} nodes")
    for n in nodes[:MAX_NODES]:
        nid, node = str(n.get("id") or "").strip(), BY_ID.get(str(n.get("type") or ""))
        if not nid or nid in prompt:
            problems.append(f"node ids must be unique and not empty: '{nid}'")
            continue
        if node is None:
            problems.append(f"node {nid}: no node type '{n.get('type')}' in the catalog")
            continue
        inputs: dict = {}
        for name, value in (n.get("values") or {}).items():
            spec = node.input(name)
            if spec is None or not spec.widget or spec.grow:
                problems.append(f"node {nid} ({node.id}): no setting '{name}'")
                continue
            inputs[name] = value
        for name, ref in (n.get("links") or {}).items():
            spec = node.input(name)
            if spec is None:
                problems.append(f"node {nid} ({node.id}): no input '{name}'")
                continue
            refs = ref if isinstance(ref, list) and not (len(ref) == 2 and isinstance(ref[1], int)) else [ref]
            if not spec.grow and len(refs) > 1:
                problems.append(f"node {nid}: input '{name}' takes one link")
            links = [lk for r in refs if (lk := _link(r)) is not None]
            if len(links) != len(refs):
                problems.append(f"node {nid}: links of '{name}' must look like \"<node id>.<output index>\"")
            if spec.grow:
                if len(links) > spec.grow:
                    problems.append(f"node {nid}: '{name}' takes at most {spec.grow} links")
                for k, lk in enumerate(links[:spec.grow], 1):
                    inputs[f"{name}.{slot_name(spec, k)}"] = lk
            elif links:
                inputs[name] = links[0]
        prompt[nid] = {"class_type": node.id, "inputs": inputs}
        if title := str(n.get("title") or "").strip():
            titles[nid] = title[:80]
    for nid, n in prompt.items():
        for name, v in n["inputs"].items():
            if engine._is_link(v) and v[0] not in prompt:
                problems.append(f"node {nid}: '{name}' links to node {v[0]}, which is not in the graph")
    return prompt, titles, problems


def compact(prompt: dict) -> list[dict]:
    """The editor's API prompt → the model's node list (the graph on the canvas, for a change)."""
    out = []
    for nid, n in (prompt or {}).items():
        node = BY_ID.get((n or {}).get("class_type"))
        if node is None:
            continue
        values, links = {}, {}
        for name, v in (n.get("inputs") or {}).items():
            base, dot, _ = name.partition(".")
            spec = node.input(base)
            if spec is None:
                continue
            if engine._is_link(v):
                ref = f"{v[0]}.{v[1]}"
                if spec.grow:
                    links.setdefault(base, []).append(ref)
                else:
                    links[base] = ref
            elif not dot:
                values[base] = v
        item = {"id": str(nid), "type": node.id, "values": values}
        if links:
            item["links"] = links
        out.append(item)
    return out


def check(prompt: dict, choices: dict[str, list], caps: dict[str, dict] | None = None) -> tuple[list[str], list[str]]:
    """Errors the editor would refuse the run with (or the model would, by what it can do: `caps` by model value),
    and notes (files for the user to pick)."""
    errors: list[str] = []
    notes: list[str] = []
    try:
        engine.validate(prompt, choices)
    except engine.PromptInvalid as e:
        node_errors = e.body.get("node_errors") or {}
        if not node_errors:
            errors.append(e.body["error"]["message"] + (f": {e.body['error']['details']}"
                                                        if e.body["error"].get("details") else ""))
        for nid, ne in node_errors.items():
            node = BY_ID[prompt[nid]["class_type"]]
            for err in ne["errors"]:
                spec = node.input((err.get("extra_info") or {}).get("input_name") or "")
                if spec is not None and isinstance(spec.options, str) and spec.options in FILE_INPUTS:
                    notes.append(f"node {nid} “{node.title}”: the user picks the file")
                    continue
                errors.append(f"node {nid} ({node.id}): {err['message']} ({err.get('details') or ''})")
    for nid, n in prompt.items():
        if n["class_type"] == CHARACTER and not str(n["inputs"].get("character") or "").strip():
            notes.append(f"node {nid} “Seedance character”: no character chosen")
        if n["class_type"] == VIDEO:
            errors += [f"node {nid} ({VIDEO}): {e}" for e in video_rules(n["inputs"])]
        if n["class_type"] == ROUTER_VIDEO and (c := (caps or {}).get(str(n["inputs"].get("model")))):
            i = n["inputs"]
            duration = i.get("duration", 0)
            errors += [f"node {nid} ({ROUTER_VIDEO}, {i.get('model')}): {e}" for e in video_problems(
                c, duration if isinstance(duration, int) else 0, str(i.get("resolution", "auto")),
                str(i.get("aspect_ratio", "auto")), "first_frame" in i, "last_frame" in i)]
    return errors, notes


def video_rules(inputs: dict) -> list[str]:
    """What “Seedance video” refuses only when it runs (nodes.run_seedance) and the video API requires."""
    first, last = "first_frame" in inputs, "last_frame" in inputs
    refs = any(k.startswith("references.") for k in inputs)
    sounds = sum(k.startswith("audio.") for k in inputs)
    v25 = "seedance-2-5" in str(inputs.get("model", ""))
    out = []
    if refs and (first or last):
        out.append("references cannot be combined with the first or last frame: use one or the other (a picture "
                   "can be a reference too)")
    if sounds and (first or last):
        out.append("sound cannot be combined with the first or last frame: link the picture as a reference instead")
    if sounds and not refs and not v25:
        out.append("sound needs at least one reference (a picture or a character) on this model")
    if sounds > (10 if v25 else 3):
        out.append(f"this model takes at most {10 if v25 else 3} sounds")
    duration = inputs.get("duration", 5)
    if isinstance(duration, int) and duration != -1 and not 4 <= duration <= (30 if v25 else 15):
        out.append(f"duration must be 4 to {30 if v25 else 15} seconds on this model, or -1")
    if sounds and inputs.get("sound") is False:
        out.append("with sound linked, the value 'sound' must be true")
    if last and not first:
        out.append("a last frame needs a first frame")
    if first and inputs.get("ratio", "adaptive") != "adaptive":
        out.append("with a first frame the ratio must be adaptive: the video takes the frame's proportions, so "
                   "set the size of the picture instead")
    return out


def characters_in(prompt: dict | None) -> set[str]:
    return {str(n["inputs"].get("character")).removeprefix("asset://") for n in (prompt or {}).values()
            if isinstance(n, dict) and n.get("class_type") == CHARACTER and (n.get("inputs") or {}).get("character")}


def _check_characters(prompt: dict, known: set[str]) -> list[str]:
    """Characters the agent did not find this turn (and that were not on the canvas) are made up."""
    return [f"character '{a}' is not in the library: take ids from search_characters"
            for a in sorted(characters_in(prompt) - known)]


async def _library() -> ArkProvider | None:
    """The first ark provider that lists ready characters."""
    for p in providers.registry().values():
        if isinstance(p, ArkProvider) and await p.has_assets():
            return p
    return None


async def search_characters(p: ArkProvider, args: dict) -> dict:
    params = {k: args[k] for k in ("gender", "age_min", "age_max", "q") if args.get(k) not in (None, "")}
    params["limit"] = max(1, min(int(args.get("limit") or 8), 12))
    try:
        found = await p.assets(params)
    except ProviderError as e:
        return {"error": str(e)}
    return {"characters": [{k: x.get(k) for k in ("id", "gender", "age", "country", "profession")}
                           for x in found["items"]]}


async def model_caps() -> dict[str, dict]:
    """What each model with known abilities can do (routers' video models), by model value."""
    return {m.value: m.caps for m in await providers.all_models() if m.caps}


def agent_model(chat_models: list[str]) -> str | None:
    return get_settings().agent_model or (chat_models[0] if chat_models else None)


def layout(prompt: dict) -> dict[str, list[int]]:
    """Columns by depth (sources on the left), nodes stacked in each column."""
    deps = {nid: [v[0] for v in n["inputs"].values() if engine._is_link(v) and v[0] in prompt]
            for nid, n in prompt.items()}
    depth: dict[str, int] = {}

    def d(nid: str, seen: frozenset = frozenset()) -> int:
        if nid in depth:
            return depth[nid]
        if nid in seen:  # a cycle: the engine refuses it anyway
            return 0
        depth[nid] = 1 + max((d(x, seen | {nid}) for x in deps[nid]), default=-1)
        return depth[nid]

    for nid in prompt:
        d(nid)
    pos, y = {}, {}
    for nid in sorted(prompt, key=lambda k: (depth[k], int(k) if k.isdigit() else 0, k)):
        col = depth[nid]
        top = y.get(col, 60)
        pos[nid] = [60 + col * COLUMN, top]
        y[col] = top + HEIGHT.get(prompt[nid]["class_type"], 360) + 40
    return pos


# ---------------------------------------------------------------- the turn


def _label(name: str, args: dict, result: dict) -> str:
    if name == "search_characters":
        found = len(result.get('characters') or [])
        return i18n.tr(f"Подбор персонажей: найдено {found}", f"Choosing characters: {found} found")
    if result.get("saved"):
        return i18n.tr(f"Схема: {result.get('nodes')} нод", f"Graph: {result.get('nodes')} nodes")
    return i18n.tr("Схема: исправляю ошибки", "Graph: fixing mistakes")


async def run(choices: dict[str, list], messages: list[dict], graph: dict | None) -> AsyncIterator[dict]:
    """One turn: step events, then {"type": "done", text, graph?} or {"type": "error", message}."""
    turns = [t for t in messages if t.get("role") in ("user", "assistant") and str(t.get("text") or "").strip()]
    turns = turns[-MAX_HISTORY:]
    if not turns or turns[-1]["role"] != "user":
        yield {"type": "error", "message": i18n.tr("Напишите, что собрать", "Write what to build")}
        return
    model = agent_model(choices.get("models:chat") or [])
    if model is None:
        yield {"type": "error", "message": i18n.tr("Нет текстовых моделей: настройте провайдера",
                                                   "No text models: configure a provider")}
        return
    try:
        p, m = await providers.resolve(model)
    except ProviderError as e:
        yield {"type": "error", "message": str(e)}
        return
    library = await _library() if choices.get("models:seedance") else None
    known = characters_in(graph)
    caps = await model_caps()
    convo = [{"role": "system", "content": SYSTEM.format(catalog=catalog(choices, caps),
                                                           characters=CHARACTERS if library else "")}]
    for i, t in enumerate(turns):
        text = str(t["text"])[:8000]
        if i == len(turns) - 1 and graph:
            text += "\n\nThe graph on the canvas now:\n" + json.dumps(compact(graph), ensure_ascii=False)
        convo.append({"role": t["role"], "content": text})
    saved = None
    nudged = False
    text = ""
    for _ in range(MAX_STEPS):
        try:
            answer = await p.request("POST", p.chat_path, json={
                "model": m.id, "messages": convo, "tools": tools(library is not None)})
        except ProviderError as e:
            log.warning("agent: %s", e)
            yield {"type": "error", "message": str(e)}
            return
        msg = (answer.get("choices") or [{}])[0].get("message") or {}
        calls = msg.get("tool_calls") or []
        convo.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_content")})
        if not calls:
            text = (msg.get("content") or "").strip()
            if saved is None and not nudged:
                nudged = True
                convo.append({"role": "system", "content": "Check before you answer: you did not save a graph in "
                              "this turn, so the canvas has not changed. If the user asked for a graph or a change, "
                              "call set_graph now. If not, give your answer again."})
                continue
            break
        for tc in calls:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            if fn.get("name") == "set_graph":
                prompt, titles, problems = to_prompt(args.get("nodes") or [])
                errors, notes = check(prompt, choices, caps) if prompt else ([], [])
                errors = problems + errors + _check_characters(prompt, known)
                if errors:
                    result = {"saved": False, "errors": errors}
                else:
                    saved = {"prompt": prompt, "titles": titles, "layout": layout(prompt)}
                    result = {"saved": True, "nodes": len(prompt), **({"notes": notes} if notes else {})}
            elif fn.get("name") == "search_characters" and library is not None:
                result = await search_characters(library, args)
                known |= {x["id"] for x in result.get("characters") or []}
            else:
                result = {"error": "unknown tool"}
            yield {"type": "step", "label": _label(fn.get("name") or "", args, result)}
            convo.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result, ensure_ascii=False)})
    else:
        text = text or (i18n.tr("Схема готова.", "The graph is ready.") if saved else "")
    if not text and saved is None:
        yield {"type": "error", "message": i18n.tr("Агент не справился, попробуйте переформулировать",
                                                   "The agent could not do it, try putting it differently")}
        return
    yield {"type": "done", "text": text or i18n.tr("Схема готова.", "The graph is ready."),
           **({"graph": saved} if saved else {})}

