"""Nodes made from an API's OpenAPI schema, for providers whose every model has its own inputs (fal, and later
Replicate): one node per model, its widgets and links read off the model's input schema, its outputs off the output
schema. The provider runs it (providers/fal.py); this module only reads schemas.

What becomes what:
- `prompt`-like strings: a multiline STRING; other strings, numbers and booleans: STRING, INT, FLOAT, BOOLEAN with the
  schema's default and bounds; a string enum (or an enum next to a custom object, like fal's image_size): COMBO;
- `<x>_url` strings: a link input of a picture, a video or a sound (by the name); `<x>_urls` arrays: a growing input;
- `seed`: INT with the editor's "fixed / randomize" control (fixed: a run with nothing changed reuses the result);
- output `images` / `image` / `video` / `audio` files: IMAGE, VIDEO, AUDIO; a `description` or `text`: STRING.
Fields the editor cannot set (objects, lists of objects) are left out; a model that requires one is not offered.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from nodeloom.nodes import SEED, In, Out

log = logging.getLogger(__name__)

SKIP = {"sync_mode"}  # transport details of the API, not settings of the model
MAIN = {"prompt", "text", "image_size", "aspect_ratio", "duration", "resolution", "num_images", "generate_audio",
        "voice"}  # settings shown without "Show advanced inputs"
MULTILINE = {"prompt", "negative_prompt", "text", "lyrics", "style"}
MEDIA_WORDS = (("video", "VIDEO"), ("audio", "AUDIO"), ("voice", "AUDIO"), ("sound", "AUDIO"), ("music", "AUDIO"))
TEXT_OUT = ("description", "text", "output", "transcript")
GROW_MAX = 10


class Unsupported(Exception):
    """The schema asks for something the editor cannot give (a required object, a list of objects)."""


@dataclass
class Field_:
    """How a node's input goes into the API request."""

    name: str  # the API's field
    kind: str  # value | media | media_list
    media: str = ""  # IMAGE | VIDEO | AUDIO
    optional: bool = True


@dataclass
class OutField:
    name: str
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
    options = prop.get("anyOf") or prop.get("oneOf")
    if not options:
        return prop, False, None
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
    tip = re.sub(r"^(the |a )?urls? (of|for|to) ", "", tip, flags=re.IGNORECASE)
    return tip[:1].upper() + tip[1:]


def _media(name: str) -> str:
    low = name.lower()
    return next((t for word, t in MEDIA_WORDS if word in low), "IMAGE")


def _ordered(props: dict, order: list | None) -> list[str]:
    names = [n for n in (order or []) if n in props] + [n for n in props if n not in (order or [])]
    first = [n for n in names if n in ("prompt", "text")]
    media = [n for n in names if n not in first and (n.endswith("_url") or n.endswith("_urls"))]
    return first + media + [n for n in names if n not in first and n not in media]


def read(schema: dict, input_name: str, output_name: str) -> Spec:
    """The node's inputs and outputs from the schemas of its request and its result."""
    s = Spec()
    inp = _deref(schema, {"$ref": f"#/components/schemas/{input_name}"})
    required = set(inp.get("required") or [])
    props = inp.get("properties") or {}
    for name in _ordered(props, inp.get("x-fal-order-properties")):
        if name in SKIP:
            continue
        prop, nullable, enum = _unwrap(schema, props[name])
        need = name in required
        label, tip = _words(name), _tip(prop)
        typ = prop.get("type")
        if name.endswith("_urls") and typ == "array" and _deref(schema, prop.get("items") or {}).get("type") == "string":
            media = _media(name)
            grow = min(int(prop.get("maxItems") or GROW_MAX), GROW_MAX)
            s.inputs.append(In(name, media, _words(name.removesuffix("_urls")), optional=True, grow=grow,
                               tooltip=_media_tip(tip)))
            s.fields.append(Field_(name, "media_list", media, optional=not need))
            continue
        if name.endswith("_url") and typ == "string" and not enum:
            media = _media(name)
            s.inputs.append(In(name, media, _words(name.removesuffix("_url")), optional=not need,
                               tooltip=_media_tip(tip)))
            s.fields.append(Field_(name, "media", media, optional=not need))
            continue
        default = prop.get("default")
        advanced = not need and name not in MAIN
        if name == "seed" and typ in ("integer", None):
            s.inputs.append(In(name, "INT", "seed", default=0, advanced=True, extra=dict(SEED), tooltip=tip))
        elif enum or prop.get("enum"):
            options = [str(v) for v in (enum or prop["enum"])]
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
        s.fields.append(Field_(name, "value", optional=not need))

    out = _deref(schema, {"$ref": f"#/components/schemas/{output_name}"})
    oprops = out.get("properties") or {}
    for name in _ordered(oprops, out.get("x-fal-order-properties")):
        prop, _, _ = _unwrap(schema, oprops[name])
        typ = prop.get("type")
        if typ == "array":
            item = _deref(schema, prop.get("items") or {})
            if "url" in (item.get("properties") or {}):
                media = "IMAGE" if "image" in name else _media(name)
                s.outputs.append(Out(media, _words(name)))
                s.out_fields.append(OutField(name, media, many=True))
        elif "url" in (prop.get("properties") or {}):
            media = _media(name) if name != "image" and "image" not in name else "IMAGE"
            s.outputs.append(Out(media, _words(name)))
            s.out_fields.append(OutField(name, media))
        elif typ == "string" and name in TEXT_OUT:
            s.outputs.append(Out("STRING", _words(name)))
            s.out_fields.append(OutField(name, "STRING"))
    if not s.out_fields:
        raise Unsupported("no image, video, sound or text in its result")
    return s
