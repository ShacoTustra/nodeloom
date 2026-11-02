"""Nodes made from an API's OpenAPI schema, for providers whose every model has its own inputs (fal, Replicate): one
node per model, its widgets and links read off the model's input schema, its outputs off the output schema. The
provider runs it (providers/schemas.py); this module only reads schemas.

What becomes what:
- `prompt`-like strings: a multiline STRING; other strings, numbers and booleans: STRING, INT, FLOAT, BOOLEAN with the
  schema's default and bounds; an enum (or an enum next to a custom object, like fal's image_size): COMBO, its value
  sent back as the schema's type (Replicate's durations are integers);
- files: fal's `<x>_url` strings and Replicate's `format: uri` ones are a link input of a picture, a video or a sound
  (by the name); lists of them (`<x>_urls`, arrays of uris) are a growing input;
- `seed`: INT with the editor's "fixed / randomize" control (fixed: a run with nothing changed reuses the result);
- outputs: fal's files ({url}) named images / image / video / audio, Replicate's uris (the whole output, of the kind
  the provider says), and text (`description`, `transcription`..., or a list of strings, an LLM's tokens): IMAGE,
  VIDEO, AUDIO, STRING.
Fields the editor cannot set (objects, lists of objects) and deprecated ones are left out; a model that requires one
is not offered.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from nodeloom.nodes import SEED, In, Out

log = logging.getLogger(__name__)

SKIP = {"sync_mode"}  # transport details of the API, not settings of the model
MAIN = {"prompt", "text", "image_size", "aspect_ratio", "duration", "resolution", "num_images", "num_outputs",
        "number_of_images", "generate_audio", "voice", "voice_id", "size"}  # shown without "Show advanced inputs"
MULTILINE = {"prompt", "negative_prompt", "text", "lyrics", "style", "system_prompt"}
MEDIA_WORDS = (("video", "VIDEO"), ("audio", "AUDIO"), ("voice", "AUDIO"), ("speech", "AUDIO"), ("sound", "AUDIO"),
               ("music", "AUDIO"), ("song", "AUDIO"))
OUT_MEDIA_WORDS = ("image", "mask", "frame", "video", "audio", "voice", "speech", "sound", "music", "song")
NOT_FILES = ("lora", "weights", "webhook", "model")  # uris that are not files of the user
TEXT_OUT = ("description", "text", "output", "transcript", "transcription", "translation", "caption")
GROW_MAX = 10
ROOT = ""  # an output field: the whole result (Replicate's output is a uri, a list of uris or text itself)


class Unsupported(Exception):
    """The schema asks for something the editor cannot give (a required object, a list of objects)."""


@dataclass
class Field_:
    """How a node's input goes into the API request."""

    name: str  # the API's field
    kind: str  # value | media | media_list
    media: str = ""  # IMAGE | VIDEO | AUDIO
    optional: bool = True
    cast: str = ""  # int | float: a COMBO of numbers goes back as numbers


@dataclass
class OutField:
    name: str  # the result's field, or ROOT
    type: str  # IMAGE | VIDEO | AUDIO | STRING
    many: bool = False  # a list of files


@dataclass
class Spec:
    inputs: list[In] = field(default_factory=list)
    fields: list[Field_] = field(default_factory=list)
    outputs: list[Out] = field(default_factory=list)
    out_fields: list[OutField] = field(default_factory=list)


def _deref(schema: dict, node: dict) -> dict:
    while isinstance(node, dict) and "$ref" in node:
        node = schema["components"]["schemas"].get(node["$ref"].split("/")[-1], {})
    return node or {}


def _unwrap(schema: dict, prop: dict) -> tuple[dict, bool, list | None]:
    """The property's own schema, whether it may be null, and an enum next to other options (image_size)."""
    prop = _deref(schema, prop)
    if prop.get("allOf"):  # Replicate's enums: {"allOf": [{"$ref": ...}], "default": ...}
        merged: dict = {}
        for part in prop["allOf"]:
            merged.update(_deref(schema, part))
        prop = {**merged, **{k: v for k, v in prop.items() if k != "allOf"}}
    options = prop.get("anyOf") or prop.get("oneOf")
    if not options:
        return prop, bool(prop.get("nullable")), None
    real = [_deref(schema, o) for o in options if o.get("type") != "null"]
    nullable = len(real) < len(options)
    enums = [o for o in real if o.get("enum") and o.get("type", "string") == "string"]
    if enums and len(real) > 1:  # an enum or a custom object: the enum
        return {**prop, **enums[0], "type": "string"}, nullable, enums[0]["enum"]
    base = {k: v for k, v in prop.items() if k not in ("anyOf", "oneOf")}
    return ({**real[0], **base} if real else base), nullable, None


def _words(name: str) -> str:
    return name.replace("_", " ").strip()


def _tip(prop: dict) -> str | None:
    text = re.sub(r"\s+", " ", str(prop.get("description") or "")).strip()
    return (text[:400] + "…" if len(text) > 400 else text) or None


def _media_tip(tip: str | None) -> str | None:
    """"URL of the image to be used for the video" -> "The image to be used for the video": here a file is linked."""
    if not tip:
        return tip
    tip = re.sub(r"^(the |a |an )?(urls?|uris?) (of|for|to) ", "", tip, flags=re.IGNORECASE)
    return tip[:1].upper() + tip[1:]


def _media(name: str) -> str:
    low = name.lower()
    return next((t for word, t in MEDIA_WORDS if word in low), "IMAGE")


def _media_label(name: str, many: bool) -> str:
    """image_url -> image, image_input -> image, input_images -> image, start_image -> start image."""
    words = [w for w in re.sub(r"_urls?$", "", name).split("_") if w and w != "input"] or [name]
    if many and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def _file(name: str, prop: dict) -> bool:
    """A file of the user: fal's `<x>_url`, Replicate's `format: uri`."""
    if any(w in name.lower() for w in NOT_FILES):
        return False
    return name.endswith(("_url", "_urls")) or prop.get("format") == "uri"


def _order(props: dict, order: list | None) -> list[str]:
    if order:  # fal's x-fal-order-properties
        return [n for n in order if n in props] + [n for n in props if n not in order]
    return sorted(props, key=lambda n: props[n].get("x-order", 1_000))  # Replicate's x-order


def _ordered(schema: dict, props: dict, order: list | None) -> list[str]:
    names = _order(props, order)
    first = [n for n in names if n in ("prompt", "text")]

    def media(n: str) -> bool:
        p, _, _ = _unwrap(schema, props[n])
        items = _deref(schema, p.get("items") or {})
        return _file(n, p) or (p.get("type") == "array" and _file(n, items))
    files = [n for n in names if n not in first and media(n)]
    return first + files + [n for n in names if n not in first and n not in files]


def properties(schema: dict, name: str) -> dict[str, dict]:
    """A component's properties, each one's own schema (refs and allOf resolved)."""
    comp = _deref(schema, {"$ref": f"#/components/schemas/{name}"})
    return {n: _unwrap(schema, p)[0] for n, p in (comp.get("properties") or {}).items()}


def read(schema: dict, input_name: str, output_name: str, root_media: str = "IMAGE") -> Spec:
    """The node's inputs and outputs from the schemas of its request and its result. `root_media`: what a bare uri
    output is (the schema says only "uri"; the provider knows the model)."""
    s = Spec()
    inp = _deref(schema, {"$ref": f"#/components/schemas/{input_name}"})
    required = set(inp.get("required") or [])
    props = inp.get("properties") or {}
    for name in _ordered(schema, props, inp.get("x-fal-order-properties")):
        if name in SKIP:
            continue
        prop, nullable, enum = _unwrap(schema, props[name])
        need = name in required
        if prop.get("deprecated") and not need:
            continue
        label, tip = _words(name), _tip(prop)
        typ = prop.get("type")
        items = _deref(schema, prop.get("items") or {})
        if typ == "array" and items.get("type") == "string" and _file(name, items):
            media = _media(name)
            grow = min(int(prop.get("maxItems") or GROW_MAX), GROW_MAX)
            s.inputs.append(In(name, media, _media_label(name, True), optional=True, grow=grow,
                               tooltip=_media_tip(tip)))
            s.fields.append(Field_(name, "media_list", media, optional=not need))
            continue
        if typ == "string" and not enum and not prop.get("enum") and _file(name, prop):
            media = _media(name)
            s.inputs.append(In(name, media, _media_label(name, False), optional=not need, tooltip=_media_tip(tip)))
            s.fields.append(Field_(name, "media", media, optional=not need))
            continue
        default = prop.get("default")
        advanced = not need and name not in MAIN
        cast = ""
        if name == "seed" and typ in ("integer", None):
            s.inputs.append(In(name, "INT", "seed", default=0, advanced=True, extra=dict(SEED), tooltip=tip))
        elif enum or prop.get("enum"):
            values = [v for v in (enum or prop["enum"]) if v is not None]
            options = [str(v) for v in values]
            if typ in ("integer", "number") and all(isinstance(v, (int, float)) for v in values):
                cast = "int" if typ == "integer" else "float"
            if default is not None and str(default) not in options:
                default = None
            s.inputs.append(In(name, "COMBO", label, default=str(default) if default is not None else options[0],
                               options=options, advanced=advanced, tooltip=tip))
        elif typ == "boolean":
            s.inputs.append(In(name, "BOOLEAN", label, default=bool(default), advanced=advanced, tooltip=tip))
        elif typ in ("integer", "number"):
            if default is None and not need:
                continue  # nothing to send unless set, and a widget always has a value
            extra = {}
            lo = prop.get("minimum", prop.get("exclusiveMinimum"))
            if lo is not None:
                extra["min"] = lo + (1 if "minimum" not in prop and typ == "integer" else 0)
            if prop.get("maximum") is not None:
                extra["max"] = prop["maximum"]
            if typ == "number":
                extra["step"] = 0.01 if (extra.get("max", 10) - extra.get("min", 0)) <= 2 else 0.1
            value = default if default is not None else extra.get("min", 0)
            s.inputs.append(In(name, "INT" if typ == "integer" else "FLOAT", label,
                               default=int(value) if typ == "integer" else float(value), advanced=advanced,
                               extra=extra, tooltip=tip))
        elif typ == "string" or (typ is None and "default" in prop and isinstance(default, str)):
            extra = {"multiline": True} if name in MULTILINE or (prop.get("maxLength") or 0) > 300 else {}
            s.inputs.append(In(name, "STRING", label, default=default or "", optional=not need and name != "prompt",
                               advanced=advanced and name not in MULTILINE, extra=extra, tooltip=tip))
        else:
            if need:
                raise Unsupported(f"required field {name!r} of type {typ or 'object'}")
            continue  # objects, lists of objects (LoRAs, ...): left out
        s.fields.append(Field_(name, "value", optional=not need, cast=cast))

    out = _deref(schema, {"$ref": f"#/components/schemas/{output_name}"})
    if out.get("properties"):
        _outputs_of_object(schema, out, s)
    else:
        _output_of_root(schema, out, s, root_media)
    if not s.out_fields:
        raise Unsupported("no image, video, sound or text in its result")
    return s


def _media_out(name: str) -> str | None:
    low = name.lower()
    if not any(w in low for w in OUT_MEDIA_WORDS):
        return None  # a file of another kind (a mesh, a zip, subtitles): left out
    return "IMAGE" if any(w in low for w in ("image", "mask", "frame")) else _media(name)


def _outputs_of_object(schema: dict, out: dict, s: Spec) -> None:
    """fal's results and some of Replicate's: {"images": [{url}], "video": {url}, "description": "..."}."""
    oprops = out["properties"]
    names = _ordered(schema, oprops, out.get("x-fal-order-properties"))
    # files in the schema's order, then texts by importance (Whisper: the transcription before the translation)
    names.sort(key=lambda n: TEXT_OUT.index(n) if n in TEXT_OUT else -1)
    for name in names:
        prop, _, _ = _unwrap(schema, oprops[name])
        typ = prop.get("type")
        item = _deref(schema, prop.get("items") or {}) if typ == "array" else {}
        many = typ == "array"
        one = item if many else prop
        is_file = "url" in (one.get("properties") or {}) or (one.get("type") == "string" and one.get("format") == "uri")
        if is_file:
            media = _media_out(name)
            if media is None:
                continue
            s.outputs.append(Out(media, _words(name)))
            s.out_fields.append(OutField(name, media, many=many))
        elif typ == "string" and name in TEXT_OUT:
            s.outputs.append(Out("STRING", _words(name)))
            s.out_fields.append(OutField(name, "STRING"))


def _output_of_root(schema: dict, out: dict, s: Spec, media: str) -> None:
    """Replicate's output itself: a uri, a list of uris, a string, or a list of strings (an LLM's tokens)."""
    typ = out.get("type")
    item = _deref(schema, out.get("items") or {}) if typ == "array" else {}
    if typ == "string" and out.get("format") == "uri":
        s.outputs.append(Out(media, media.lower()))
        s.out_fields.append(OutField(ROOT, media))
    elif typ == "array" and item.get("type") == "string" and item.get("format") == "uri":
        s.outputs.append(Out(media, media.lower() + ("s" if media == "IMAGE" else "")))
        s.out_fields.append(OutField(ROOT, media, many=True))
    elif typ == "string" or (typ == "array" and item.get("type") == "string"):
        s.outputs.append(Out("STRING", "text"))
        s.out_fields.append(OutField(ROOT, "STRING"))
