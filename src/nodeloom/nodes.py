"""The nodes: what the editor offers and how each one runs.

Kept deliberately small and plain for people who don't know ComfyUI: text, uploads, generations through the
providers and a few video tools. Values between nodes are plain: text (STRING), a list of images (IMAGE), one video
(VIDEO), one audio track (AUDIO); a picture or a clip is a file ref (files.py), never a tensor.

A node is described once (`Node`) and serialised for the editor in ComfyUI's /object_info format. Names, labels and
the slot names of growing inputs are English: they are what saved graphs keep. Russian is a translation (`*_ru`),
served through /api/i18n in the editor's own format. `run` gets the run context and the input values by name and
returns the outputs and what the node shows (ComfyUI's "ui"). Nodes whose work costs money or time are `cached`: the
engine reuses their result while their inputs stay the same.

Generation nodes are per API (the text model node takes every provider's chat models; GPT Image, Seedream and
Seedance have nodes of their own, see providers/), and a node whose models no provider has is not offered.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nodeloom import files, i18n, providers, video_tools
from nodeloom.providers import Model, Provider, ProviderError
from nodeloom.providers.pricing import cost_micro

log = logging.getLogger(__name__)

SEED_MAX = 2147483647
MIN_VIDEO_IMAGE_SIDE = 300  # ModelArk refuses smaller input frames ("expected the height to be at least 300px")
MAX_REFERENCES = 10
GPT_IMAGE_REFERENCES = 16
# Sound for Seedance (reference_audio, wav/mp3): 2.0 models take up to 3 clips of 2-15 s, 15 s in all, and only with a
# reference picture or video; 2.5 takes up to 10 clips of 2-30 s, 30 s in all, sound alone too.
VIDEO_AUDIO = {"count": 3, "min": 2.0, "max": 15.0, "total": 15.0}
VIDEO_AUDIO_25 = {"count": 10, "min": 2.0, "max": 30.0, "total": 30.0}
TEXT_AUDIO_COUNT, TEXT_AUDIO_MAX = 3, 300.0  # a text model hears up to 3 clips of up to 5 minutes
TEXT_VIDEO_COUNT = 2  # and watches up to 2 videos of up to 5 minutes (video_tools.MODEL_VIDEO_MAX_S)
POLL_SECONDS = 5.0
TASK_TIMEOUT = 45 * 60
TASK_STATUS = {"queued": ("в очереди", "queued"), "running": ("генерация", "generating"),
               "succeeded": ("готово", "done"), "failed": ("ошибка", "failed"), "cancelled": ("отменено", "cancelled"),
               "expired": ("истекло", "expired")}
ASSET_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def _status(status: str) -> str:
    ru, en = TASK_STATUS.get(status, (status, status))
    return i18n.tr(ru, en)


AUTOGROW = "COMFY_AUTOGROW_V3"


class NodeError(Exception):
    """A message for the user: shown in the editor's error dialog as is."""


class Suspended(Exception):
    """The process is stopping: the node lets go without losing anything (a video task in flight is followed by the
    process that resumes the run)."""


@dataclass
class In:
    name: str
    type: str  # STRING, INT, FLOAT, BOOLEAN, COMBO, IMAGE, VIDEO, AUDIO, "*"
    label: str
    optional: bool = False
    default: Any = None
    options: list | str | None = None  # COMBO choices, or a key of the per-user choices ("models:chat")
    tooltip: str | None = None
    advanced: bool = False
    extra: dict = field(default_factory=dict)  # multiline, min, max, step, image_upload, control_after_generate
    grow: int = 0  # > 0: a growing list of this input ("<label> 1", "<label> 2", ...), at most `grow`
    # Russian: the label, the tooltip, the placeholder and COMBO values that are words (same order as `options`).
    # The editor shows them to a Russian user; the English label stays the input's own (slot names of growing inputs
    # come from it), and a value in either language is accepted (engine.grouped makes it the English one).
    label_ru: str = ""
    tip_ru: str | None = None
    hint_ru: str | None = None
    options_ru: list | None = None

    @property
    def widget(self) -> bool:
        return self.type in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO")


@dataclass
class Out:
    type: str
    label: str
    label_ru: str = ""


@dataclass
class Node:
    id: str
    title: str
    category: str
    description: str
    inputs: list[In]
    outputs: list[Out]
    run: Callable[..., Awaitable[tuple[list, dict | None]]]
    output_node: bool = False
    cached: bool = False
    # the editor's audio player: it plays an uploaded file or an audio result. The editor adds it only to its own
    # audio nodes (by name), and its audio upload button fails to create a node without it. Not part of the prompt.
    audio_player: bool = False
    version: int = 1  # bump when the same inputs would now give a different result
    search: list[str] = field(default_factory=list)
    title_ru: str = ""
    about_ru: str = ""  # the description in Russian
    needs: str | None = None  # a list of choices that must not be empty for the node to be offered ("models:seedance")

    def input(self, name: str) -> In | None:
        return next((i for i in self.inputs if i.name == name), None)


# ---------------------------------------------------------------- run context


@dataclass
class Ctx:
    """What a running node may use: the providers, the user's files, progress and the run's task memo."""

    user_id: str
    node_id: str
    session_factory: Callable
    progress: Callable[[str], Awaitable[None]]
    pending_task: str | None = None  # an upstream task this node started before a restart
    remember_task: Callable[[str], Awaitable[None]] | None = None
    run_id: str | None = None
    suspending: Callable[[], bool] = lambda: False  # the process is stopping: let go at the next safe point
    paid: bool = False  # the node made a paid call
    cost: int = 0  # what its paid calls cost, micro-dollars (when known)
    cost_known: bool = True

    async def model(self, value: str) -> tuple[Provider, Model]:
        try:
            return await providers.resolve(value)
        except ProviderError as e:
            raise NodeError(str(e)) from None

    async def call(self, p: Provider, method: str, path: str, body: dict | None = None, *, data: dict | None = None,
                   upload=None) -> dict:
        try:
            return await p.request(method, path, json=body, data=data, files=upload)
        except ProviderError as e:
            raise NodeError(str(e)) from None

    def charge(self, m: Model, usage) -> None:
        """A paid call succeeded: its cost from the usage it reported and the model's price."""
        self.paid = True
        c = cost_micro(m.price, usage)
        if c is None:
            self.cost_known = False
        else:
            self.cost += c

    async def save(self, data: bytes | Path, content_type: str, kind: str = "output", meta: dict | None = None) -> dict:
        async with self.session_factory() as s:
            f = await files.save(s, self.user_id, kind, data, content_type, meta=meta,
                                 run_id=self.run_id if kind == "output" else None)
            await s.commit()
            return files.ref(f)

    async def read(self, ref: dict) -> bytes:
        return await files.read(self.user_id, ref)

    async def data_uri(self, ref: dict, min_side: int = 0) -> str:
        if ref.get("asset"):  # only Seedance takes a character, and only by id
            raise NodeError(i18n.tr("Персонаж работает только в видео Seedance: подключите его к ноде «Видео Seedance»",
                                    "A character works only in Seedance video: connect it to “Seedance video”"))
        data, ct = await self.read(ref), ref.get("ct") or "image/png"
        if min_side and min(ref.get("w") or min_side, ref.get("h") or min_side) < min_side:
            data, ct = _upscale(data, min_side), "image/png"
        return f"data:{ct};base64," + base64.b64encode(data).decode()

    async def download(self, p: Provider, url: str) -> dict:
        """Copy a generated file from the provider's storage (its links expire) into the user's results."""
        try:
            async with p.download(url) as (tmp, ct):
                meta = files.image_meta(tmp.read_bytes()) if files.media_of(ct) == "image" else {}
                return await self.save(tmp, ct, meta=meta)
        except ProviderError as e:
            raise NodeError(str(e)) from None


async def _audio(ctx: Ctx, ref: dict) -> tuple[str, float]:
    """An uploaded sound as the models take it: mp3 (128 kbit/s, any format and the sound of a video converted),
    base64, and its length in seconds."""
    tmp = Path(tempfile.mkdtemp(prefix="loom-"))
    try:
        src, dst = tmp / "in", tmp / "out.mp3"
        src.write_bytes(await ctx.read(ref))
        await _ffmpeg("-i", str(src), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(dst))
        seconds = float(((await video_tools.probe(dst)).get("format") or {}).get("duration") or 0)
        return base64.b64encode(dst.read_bytes()).decode(), seconds
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _video(ctx: Ctx, ref: dict) -> dict:
    """A video as a chat model takes it: a video_url content part with the mp4 in base64."""
    tmp = Path(tempfile.mkdtemp(prefix="loom-"))
    try:
        src, dst = tmp / "in", tmp / "out.mp4"
        src.write_bytes(await ctx.read(ref))
        try:
            seconds = await video_tools.for_model(src, dst)
        except video_tools.VideoError as e:
            raise NodeError(_video_error(e)) from None
        part = {"url": "data:video/mp4;base64," + base64.b64encode(dst.read_bytes()).decode()}
        if seconds <= 60:  # short clips: two frames a second (the default is one), motion matters
            part["fps"] = 2
        return {"type": "video_url", "video_url": part}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _upscale(data: bytes, min_side: int) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        scale = min_side / min(im.size)
        im = im.convert("RGB").resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def _png(data: bytes) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def _flat(values) -> list:
    """Images of a growing input: each slot holds a list of images (a generation may give several)."""
    out = []
    for v in values or []:
        out.extend(v if isinstance(v, list) else [v])
    return [v for v in out if v]


def _prompt(text: str) -> str:
    if not (text or "").strip():
        raise NodeError(i18n.tr("Напишите описание", "Write a description"))
    return text


def show_images(refs: list[dict]) -> dict:
    return {"images": [files.view_item(r) for r in refs]}


def show_video(r: dict) -> dict:
    return {"images": [files.view_item(r)], "animated": [True]}


async def _images_of(ctx: Ctx, p: Provider, data: dict) -> list[dict]:
    """The pictures of an images API answer (base64 or links), saved as results."""
    out, errors = [], []
    for item in data.get("data") or []:
        if item.get("b64_json"):
            raw = base64.b64decode(item["b64_json"])
            out.append(await ctx.save(raw, _sniff(raw)))
        elif item.get("url"):
            out.append(await ctx.download(p, item["url"]))
        elif item.get("error"):
            errors.append((item["error"] or {}).get("message", i18n.tr("ошибка", "error")))
    if not out:
        raise NodeError(i18n.tr("Картинка не получена", "No image came back") + (f": {errors[0]}" if errors else ""))
    return out


def _sniff(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


# ---------------------------------------------------------------- inputs


async def run_text(ctx: Ctx, value: str = "") -> tuple[list, dict | None]:
    return [value or ""], None


async def _load(ctx: Ctx, name: str, media: str) -> dict:
    async with ctx.session_factory() as s:
        f = await files.get(s, ctx.user_id, "input", name or "")
    if f is None or files.media_of(f.content_type) != media:
        raise NodeError(i18n.tr("Выберите или загрузите файл", "Choose or upload a file"))
    return files.ref(f)


async def run_load_image(ctx: Ctx, image: str) -> tuple[list, dict | None]:
    return [[await _load(ctx, image, "image")]], None


async def run_load_video(ctx: Ctx, file: str) -> tuple[list, dict | None]:
    return [await _load(ctx, file, "video")], None


async def run_load_audio(ctx: Ctx, audio: str) -> tuple[list, dict | None]:
    return [await _load(ctx, audio, "audio")], None


async def run_character(ctx: Ctx, character: str = "") -> tuple[list, dict | None]:
    """A ready character for Seedance: an IMAGE value that names it (asset://<id> in a video request), no file."""
    asset = (character or "").strip().removeprefix("asset://")
    if not ASSET_ID.match(asset):
        raise NodeError(i18n.tr("Выберите персонажа", "Choose a character"))
    return [[{"asset": asset, "kind": "asset", "name": asset, "media": "image"}]], None  # the node shows its preview


# ---------------------------------------------------------------- generation


async def run_text_model(ctx: Ctx, model: str, prompt: str, system: str = "", images=None, audio=None, video=None,
                         temperature: float = 1.0, max_tokens: int = 0) -> tuple[list, dict | None]:
    _prompt(prompt)
    p, m = await ctx.model(model)
    imgs, sounds, videos = _flat(images), _flat(audio), _flat(video)
    if imgs and not m.takes("image"):
        raise NodeError(i18n.tr(f"Модель {m.id} не понимает картинки: выберите другую или уберите картинки",
                                f"Model {m.id} does not understand images: choose another one or remove the images"))
    if sounds and not m.takes("audio"):
        raise NodeError(i18n.tr(f"Модель {m.id} не понимает звук: выберите другую или уберите звук",
                                f"Model {m.id} does not understand sound: choose another one or remove the sound"))
    if videos and not m.takes("video"):
        raise NodeError(i18n.tr(f"Модель {m.id} не понимает видео: выберите другую или уберите видео",
                                f"Model {m.id} does not understand video: choose another one or remove the video"))
    if len(sounds) > TEXT_AUDIO_COUNT:
        raise NodeError(i18n.tr(f"Не больше {TEXT_AUDIO_COUNT} звуков", f"At most {TEXT_AUDIO_COUNT} sounds"))
    if len(videos) > TEXT_VIDEO_COUNT:
        raise NodeError(i18n.tr(f"Не больше {TEXT_VIDEO_COUNT} видео", f"At most {TEXT_VIDEO_COUNT} videos"))
    if imgs or sounds or videos:
        content = [{"type": "image_url", "image_url": {"url": await ctx.data_uri(i)}} for i in imgs]
        for k, a in enumerate(sounds, 1):
            data, seconds = await _audio(ctx, a)
            if seconds > TEXT_AUDIO_MAX:
                raise NodeError(i18n.tr(f"Звук {k} длиннее {int(TEXT_AUDIO_MAX // 60)} минут: укоротите его",
                                        f"Sound {k} is longer than {int(TEXT_AUDIO_MAX // 60)} minutes: shorten it"))
            content.append({"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}})
        clips = [await _video(ctx, v) for v in videos]
        if sum(len(c["video_url"]["url"]) for c in clips) > video_tools.MODEL_VIDEO_BUDGET:
            raise NodeError(i18n.tr("Видео слишком большие для одного запроса: укоротите их",
                                    "The videos are too large for one request: shorten them"))
        content += clips
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt
    messages = [{"role": "system", "content": system}] if (system or "").strip() else []
    messages.append({"role": "user", "content": content})
    body: dict = {"model": m.id, "messages": messages}
    if temperature != 1.0:  # 1 is everyone's default, and reasoning models refuse any other
        body["temperature"] = temperature
    if max_tokens:
        body[p.max_tokens_key] = max_tokens
    await ctx.progress(i18n.tr("Пишет ответ…", "Writing the answer…"))
    data = await ctx.call(p, "POST", "/chat/completions", body)
    ctx.charge(m, data.get("usage"))
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise NodeError(i18n.tr("Модель не вернула ответ", "The model returned no answer")) from None
    return [text], {"text": [text]}


async def run_gpt_image(ctx: Ctx, model: str, prompt: str, references=None, size: str = "auto", quality: str = "auto",
                        background: str = "auto", count: int = 1, seed: int = 0) -> tuple[list, dict | None]:
    """OpenAI's images API: /images/generations, or /images/edits with the references. The model has no seed:
    `seed` only makes the node draw again (another number, another signature)."""
    _prompt(prompt)
    p, m = await ctx.model(model)
    refs = _flat(references)
    if len(refs) > GPT_IMAGE_REFERENCES:
        raise NodeError(i18n.tr(f"Не больше {GPT_IMAGE_REFERENCES} референсов",
                                f"At most {GPT_IMAGE_REFERENCES} references"))
    params = {"model": m.id, "prompt": prompt, "n": count, "size": size, "quality": quality, "background": background}
    await ctx.progress(i18n.tr("Рисует…", "Drawing…"))
    if refs:
        upload = []
        for k, r in enumerate(refs, 1):
            if r.get("asset"):
                await ctx.data_uri(r)  # says what a character is for
            data, ct = await ctx.read(r), r.get("ct") or "image/png"
            if ct not in ("image/png", "image/jpeg", "image/webp"):
                data, ct = _png(data), "image/png"
            upload.append(("image[]", (f"reference-{k}.{ct.split('/')[1]}", data, ct)))
        answer = await ctx.call(p, "POST", "/images/edits", data={k: str(v) for k, v in params.items()}, upload=upload)
    else:
        answer = await ctx.call(p, "POST", "/images/generations", params)
    ctx.charge(m, answer.get("usage"))
    out = await _images_of(ctx, p, answer)
    return [out], show_images(out)


async def run_seedream(ctx: Ctx, model: str, prompt: str, size: str = "2K", references=None, count: int = 1,
                       seed: int = 0, watermark: bool = False) -> tuple[list, dict | None]:
    _prompt(prompt)
    p, m = await ctx.model(model)
    refs = _flat(references)
    if len(refs) > MAX_REFERENCES:
        raise NodeError(i18n.tr(f"Не больше {MAX_REFERENCES} референсов", f"At most {MAX_REFERENCES} references"))
    body = {"model": m.id, "prompt": prompt, "size": size, "seed": seed, "watermark": watermark,
            "response_format": "b64_json"}
    if refs:
        uris = [await ctx.data_uri(r) for r in refs]
        body["image"] = uris if len(uris) > 1 else uris[0]
    if count > 1:
        body["sequential_image_generation"] = "auto"
        body["sequential_image_generation_options"] = {"max_images": count}
    await ctx.progress(i18n.tr("Рисует…", "Drawing…"))
    data = await ctx.call(p, "POST", "/images/generations", body)
    ctx.charge(m, data.get("usage"))
    out = await _images_of(ctx, p, data)
    return [out], show_images(out)


async def run_seedance(ctx: Ctx, model: str, prompt: str, resolution: str = "720p", ratio: str = "adaptive",
                       duration: int = 5, first_frame=None, last_frame=None, references=None, audio=None,
                       sound: bool = True, seed: int = 0, watermark: bool = False) -> tuple[list, dict | None]:
    p, m = await ctx.model(model)
    first, last, refs, sounds = _flat(first_frame)[:1], _flat(last_frame)[:1], _flat(references), _flat(audio)
    if (refs or sounds) and (first or last):
        raise NodeError(i18n.tr("Референсы и звук не сочетаются с первым и последним кадром: оставьте что-то одно",
                                "References and sound do not go with the first and last frames: keep one or the other"))
    if last and not first:
        raise NodeError(i18n.tr("Последний кадр задаётся только вместе с первым", "A last frame needs a first frame"))
    if len(refs) > 9:
        raise NodeError(i18n.tr("Не больше 9 референсов для видео", "At most 9 references for a video"))
    longest = 30 if "seedance-2-5" in m.id else 15
    if duration != -1 and not 4 <= duration <= longest:
        raise NodeError(i18n.tr(f"Длительность у {m.id}: от 4 до {longest} с (или -1, тогда выберет модель)",
                                f"Length on {m.id}: 4 to {longest} s (or -1 to let the model choose)"))
    task_id = ctx.pending_task
    if task_id is None:
        _prompt(prompt)
        audio_parts = await _video_audio(ctx, m, sounds, refs, sound) if sounds else []
        content: list[dict] = [{"type": "text", "text": prompt}]

        async def url(r: dict) -> str:
            return f"asset://{r['asset']}" if r.get("asset") else await ctx.data_uri(r, MIN_VIDEO_IMAGE_SIDE)
        for role, imgs in (("first_frame", first), ("last_frame", last)):
            for r in imgs:
                content.append({"type": "image_url", "role": role, "image_url": {"url": await url(r)}})
        for r in refs:
            content.append({"type": "image_url", "role": "reference_image", "image_url": {"url": await url(r)}})
        content += audio_parts
        body = {"model": m.id, "content": content, "resolution": resolution, "ratio": ratio, "duration": duration,
                "seed": seed, "generate_audio": sound, "watermark": watermark, "return_last_frame": True}
        task = await ctx.call(p, "POST", "/contents/generations/tasks", body)
        task_id = task.get("id")
        if not task_id:
            raise NodeError(i18n.tr(f"{p.name} не вернул номер задачи", f"{p.name} returned no task id"))
        if ctx.remember_task:
            await ctx.remember_task(task_id)
    started = time.monotonic()
    try:
        while True:
            task = await ctx.call(p, "GET", f"/contents/generations/tasks/{task_id}")
            status = task.get("status", "")
            await ctx.progress(i18n.tr(f"Видео: {_status(status)}, {int(time.monotonic() - started)} с",
                                        f"Video: {_status(status)}, {int(time.monotonic() - started)} s"))
            if status == "succeeded":
                break
            if status in ("failed", "cancelled", "expired"):
                err = (task.get("error") or {}).get("message") or _status(status)
                raise NodeError(i18n.tr(f"Видео не получилось: {err}", f"The video failed: {err}"))
            if time.monotonic() - started > TASK_TIMEOUT:
                raise NodeError(i18n.tr(f"Видео не готово за {TASK_TIMEOUT // 60} минут",
                                        f"The video is not ready after {TASK_TIMEOUT // 60} minutes"))
            if ctx.suspending():
                raise Suspended  # the task id is remembered: the resumed run follows this very task
            await asyncio.sleep(POLL_SECONDS)
    except asyncio.CancelledError:
        # the user cancelled: a queued task is cancelled upstream; a running one can't be and just finishes (and is
        # billed). A stopping process leaves the task alone for the run's next process.
        if not ctx.suspending():
            asyncio.get_running_loop().create_task(_cancel_quietly(ctx, p, task_id))
        raise
    ctx.charge(m, task.get("usage"))
    content = task.get("content") or {}
    if not content.get("video_url"):
        raise NodeError(i18n.tr("В готовой задаче нет видео", "The finished task has no video"))
    video = await ctx.download(p, content["video_url"])
    if content.get("last_frame_url"):
        frame = await ctx.download(p, content["last_frame_url"])
    else:
        frame = await _frame(ctx, video, last=True)
    return [video, [frame]], show_video(video)


async def _video_audio(ctx: Ctx, m: Model, sounds: list[dict], refs: list, sound: bool) -> list[dict]:
    """Sound references of the Seedance node, checked against what the model takes before any money is spent."""
    if not sound:
        raise NodeError(i18n.tr("Аудио на входе задаёт звук ролика: включите «звук» или уберите аудио",
                                "Audio on the input sets the video's sound: turn on “sound” or remove the audio"))
    lim = VIDEO_AUDIO_25 if "seedance-2-5" in m.id else VIDEO_AUDIO
    if lim is VIDEO_AUDIO and not refs:
        raise NodeError(i18n.tr("Звук работает вместе с референсом: подключите картинку или персонажа к «референс»",
                                "Sound works together with a reference: connect an image or a character to “reference”"))
    if len(sounds) > lim["count"]:
        raise NodeError(i18n.tr(f"У {m.id} не больше {lim['count']} звуков", f"{m.id} takes at most {lim['count']} sounds"))
    parts, total = [], 0.0
    for k, a in enumerate(sounds, 1):
        data, seconds = await _audio(ctx, a)
        if not lim["min"] <= seconds <= lim["max"] + 0.2:
            raise NodeError(i18n.tr(f"Звук {k} длится {seconds:.1f} с, а нужно от {lim['min']:.0f} до {lim['max']:.0f} с",
                                    f"Sound {k} lasts {seconds:.1f} s, it should be {lim['min']:.0f} to {lim['max']:.0f} s"))
        total += seconds
        parts.append({"type": "audio_url", "role": "reference_audio",
                      "audio_url": {"url": "data:audio/mp3;base64," + data}})
    if total > lim["total"] + 0.2:
        raise NodeError(i18n.tr(f"Звуки вместе длятся {total:.1f} с, а у {m.id} можно до {lim['total']:.0f} с",
                                f"The sounds last {total:.1f} s together, {m.id} takes up to {lim['total']:.0f} s"))
    return parts


async def _cancel_quietly(ctx: Ctx, p: Provider, task_id: str) -> None:
    try:
        await ctx.call(p, "DELETE", f"/contents/generations/tasks/{task_id}")
    except NodeError:
        pass


# ---------------------------------------------------------------- video tools (ffmpeg)


def _video_error(e: Exception) -> str:
    """ffmpeg failed (video_tools says it in Russian): the same in the user's language."""
    return i18n.tr(str(e), getattr(e, "en", "Could not process the video"))


async def _ffmpeg(*args: str) -> None:
    try:
        await video_tools.ffmpeg(*args)
    except video_tools.VideoError as e:
        raise NodeError(_video_error(e)) from None


async def _frame(ctx: Ctx, video: dict, last: bool) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="loom-"))
    try:
        src = tmp / "in.mp4"
        src.write_bytes(await ctx.read(video))
        dst = tmp / "frame.png"
        if last:
            await _ffmpeg("-sseof", "-0.2", "-i", str(src), "-update", "1", "-frames:v", "1", str(dst))
        else:
            await _ffmpeg("-i", str(src), "-frames:v", "1", str(dst))
        raw = dst.read_bytes()
        return await ctx.save(raw, "image/png", meta=files.image_meta(raw))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_video_frame(ctx: Ctx, video: dict, which: str = "last") -> tuple[list, dict | None]:
    frame = await _frame(ctx, video, last=which != "first")
    return [[frame]], show_images([frame])


async def run_join_videos(ctx: Ctx, videos=None) -> tuple[list, dict | None]:
    clips = [v for v in (videos or []) if v]
    if len(clips) < 2:
        raise NodeError(i18n.tr("Подключите хотя бы два видео", "Connect at least two videos"))
    tmp = Path(tempfile.mkdtemp(prefix="loom-"))
    try:
        paths = []
        for i, v in enumerate(clips):
            p = tmp / f"in{i}.mp4"
            p.write_bytes(await ctx.read(v))
            paths.append(p)
        out = tmp / "out.mp4"
        await ctx.progress(i18n.tr("Склеивает…", "Joining…"))
        try:
            await video_tools.join(paths, out)
        except video_tools.VideoError as e:
            raise NodeError(_video_error(e)) from None
        video = await ctx.save(out, "video/mp4")
        return [video], show_video(video)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- text tools and display


async def run_join_text(ctx: Ctx, texts=None, separator: str = "\\n\\n") -> tuple[list, dict | None]:
    sep = (separator or "").replace("\\n", "\n")
    text = sep.join(t for t in (texts or []) if t)
    return [text], {"text": [text]}


async def run_show(ctx: Ctx, value=None) -> tuple[list, dict | None]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [], show_images([r for r in value if r.get("media") == "image"])
    if isinstance(value, dict) and value.get("media") == "video":
        return [], show_video(value)
    if isinstance(value, dict) and value.get("media") == "audio":
        return [], {"audio": [files.view_item(value)]}
    return [], {"text": ["" if value is None else str(value)]}


# ---------------------------------------------------------------- the list

INPUT, GENERATE, OPENAI, BYTEDANCE, VIDEO_TOOLS, TEXT_TOOLS = (
    "Input", "Generate", "Generate/OpenAI", "Generate/ByteDance", "Video", "Text")
SEEDREAM_SIZES = ["2K", "4K", "2048x2048", "2304x1728", "1728x2304", "2560x1440", "1440x2560",
                  "2496x1664", "1664x2496", "3024x1296", "1024x1024"]
GPT_IMAGE_SIZES = ["auto", "1024x1024", "1536x1024", "1024x1536"]
# "fixed": a run with nothing changed reuses the results instead of paying for new ones; another variant is a click
SEED = dict(min=0, max=SEED_MAX, control_after_generate="fixed")


def _model(family: str) -> In:
    return In("model", "COMBO", "model", options=f"models:{family}", label_ru="модель")


def _variant(**kw) -> In:
    return In("seed", "INT", "variant", default=0, advanced=True, extra=SEED,
              tooltip="Another number: another variant with the same settings",
              label_ru="вариант", tip_ru="Другое число: другой вариант при тех же настройках", **kw)


NODES: list[Node] = [
    Node("LoomText", "Text", INPUT, "Text for other nodes: a description, a task, an instruction.",
         [In("value", "STRING", "text", default="", extra={"multiline": True, "placeholder": "Write the text"},
             label_ru="текст", hint_ru="Напишите текст")],
         [Out("STRING", "text", "текст")], run_text, search=["prompt", "промпт", "описание"],
         title_ru="Текст", about_ru="Текст для других нод: описание, задание, инструкция."),
    Node("LoomLoadImage", "Image", INPUT, "Upload an image from your computer: a frame, a reference, a sample.",
         [In("image", "COMBO", "image", options="files:image", extra={"image_upload": True},
             label_ru="картинка")],
         [Out("IMAGE", "image", "картинка")], run_load_image, search=["upload", "photo", "загрузить", "фото"],
         title_ru="Картинка", about_ru="Загрузите картинку с компьютера: кадр, референс, образец."),
    Node("LoomLoadVideo", "Video file", INPUT, "Upload a video from your computer.",
         [In("file", "COMBO", "video", options="files:video", extra={"video_upload": True}, label_ru="видео")],
         [Out("VIDEO", "video", "видео")], run_load_video, search=["upload", "clip", "загрузить", "ролик"],
         title_ru="Видео-файл", about_ru="Загрузите видео с компьютера."),
    Node("LoomLoadAudio", "Audio file", INPUT,
         "Upload a voice, music or sounds: for “Seedance video” (the video's sound) and “Text model” (to transcribe).",
         [In("audio", "COMBO", "audio", options="files:audio", extra={"audio_upload": True}, label_ru="аудио")],
         [Out("AUDIO", "audio", "аудио")], run_load_audio, audio_player=True,
         search=["upload", "sound", "music", "загрузить", "звук", "музыка"],
         title_ru="Аудио-файл", about_ru="Загрузите голос, музыку или звуки: для «Видео Seedance» (звук ролика) и "
                                         "«Текстовая модель» (расшифровать)."),

    Node("LoomTextModel", "Text model", GENERATE,
         "A language model's answer: write or improve a description, describe an image, translate.",
         [_model("chat"),
          In("prompt", "STRING", "task", default="", extra={
              "multiline": True, "placeholder": "What to do: write a description, describe an image, translate…"},
             label_ru="задание", hint_ru="Что сделать: придумать описание, описать картинку, перевести…"),
          In("images", "IMAGE", "image", optional=True, grow=MAX_REFERENCES,
             tooltip="Images for the task (not every model understands them)",
             label_ru="картинка", tip_ru="Картинки к заданию (понимают не все модели)"),
          In("audio", "AUDIO", "audio", optional=True, grow=TEXT_AUDIO_COUNT,
             tooltip="A voice or a sound for the task: transcribe it, describe it, turn it into a description for an "
                     "image or a video (not every model hears; up to 5 minutes)",
             label_ru="аудио", tip_ru="Голос или звук к заданию: расшифровать, описать, превратить в описание для "
                                      "картинки или видео (слышат не все модели; до 5 минут)"),
          In("video", "VIDEO", "video", optional=True, grow=TEXT_VIDEO_COUNT,
             tooltip="A video for the task: describe it, retell it, find a moment, write a description for a similar "
                     "clip (not every model watches video; up to 5 minutes)",
             label_ru="видео", tip_ru="Видео к заданию: описать, пересказать, найти момент, придумать описание для "
                                      "похожего ролика (видео понимают не все модели; до 5 минут)"),
          In("system", "STRING", "role", default="", optional=True, advanced=True,
             extra={"multiline": True, "placeholder": "For example: you write scripts for commercials"},
             tooltip="Who the model should be and how to answer (optional)",
             label_ru="роль", hint_ru="Например: ты сценарист рекламных роликов",
             tip_ru="Кем модели быть и как отвечать (необязательно)"),
          In("temperature", "FLOAT", "freedom", default=1.0, advanced=True, extra={"min": 0, "max": 2, "step": 0.05},
             tooltip="Lower: more precise and uniform, higher: more varied (reasoning models take only 1)",
             label_ru="свобода", tip_ru="Меньше: точнее и однообразнее, больше: разнообразнее (модели с рассуждением "
                                        "принимают только 1)"),
          In("max_tokens", "INT", "answer length", default=0, advanced=True, extra={"min": 0, "max": 131072},
             tooltip="The longest answer in tokens; 0: up to the model",
             label_ru="длина ответа", tip_ru="Предел длины ответа в токенах; 0: как решит модель")],
         [Out("STRING", "text", "текст")], run_text_model, output_node=True, cached=True,
         search=["llm", "gpt", "chat", "prompt", "текст", "улучшить промпт", "описать"], needs="models:chat",
         title_ru="Текстовая модель", about_ru="Ответ языковой модели: придумать или улучшить описание, описать "
                                               "картинку, перевести."),
    Node("LoomGPTImage", "GPT Image", OPENAI,
         "An image from a description with OpenAI's GPT Image. With references: editing, from a sample, the same "
         "object or character.",
         [_model("openai-image"),
          In("prompt", "STRING", "description", default="", extra={
              "multiline": True, "placeholder": "What the image should show"},
             label_ru="описание", hint_ru="Что должно быть на картинке"),
          In("references", "IMAGE", "reference", optional=True, grow=GPT_IMAGE_REFERENCES,
             tooltip="Images to edit or take from: an object, a character, a style (up to 16)",
             label_ru="референс", tip_ru="Картинки для правки или образцы: предмет, персонаж, стиль (до 16)"),
          In("size", "COMBO", "size", default="auto", options=GPT_IMAGE_SIZES,
             tooltip="auto: the model picks; otherwise width×height in pixels",
             label_ru="размер", tip_ru="auto: выбирает модель; иначе ширина×высота в пикселях"),
          In("quality", "COMBO", "quality", default="auto", options=["auto", "low", "medium", "high"],
             tooltip="Higher quality costs more and takes longer",
             label_ru="качество", tip_ru="Чем выше качество, тем дороже и дольше"),
          In("background", "COMBO", "background", default="auto", options=["auto", "transparent", "opaque"],
             advanced=True, tooltip="transparent: a PNG with a transparent background",
             label_ru="фон", tip_ru="transparent: PNG с прозрачным фоном"),
          In("count", "INT", "how many images", default=1, extra={"min": 1, "max": 10},
             label_ru="сколько картинок"),
          _variant()],
         [Out("IMAGE", "images", "картинки")], run_gpt_image, output_node=True, cached=True,
         search=["openai", "gpt-image", "dall-e", "image", "edit", "картинка", "рисунок"], needs="models:openai-image",
         title_ru="GPT Image", about_ru="Картинка по описанию в GPT Image от OpenAI. С референсами: правка, по "
                                        "образцу, тот же предмет или персонаж."),
    Node("LoomSeedreamImage", "Seedream image", BYTEDANCE,
         "An image from a description with ByteDance's Seedream. With references: from a sample, editing, the same "
         "character.",
         [_model("seedream"),
          In("prompt", "STRING", "description", default="", extra={
              "multiline": True, "placeholder": "What the image should show"},
             label_ru="описание", hint_ru="Что должно быть на картинке"),
          In("references", "IMAGE", "reference", optional=True, grow=MAX_REFERENCES,
             tooltip="Samples: a character, a style, an object (up to 10)",
             label_ru="референс", tip_ru="Образцы: персонаж, стиль, предмет (до 10)"),
          In("size", "COMBO", "size", default="2K", options=SEEDREAM_SIZES,
             tooltip="2K or 4K: the model picks the proportions; otherwise width×height in pixels",
             label_ru="размер", tip_ru="2K или 4K: пропорции выбирает модель; иначе ширина×высота в пикселях"),
          In("count", "INT", "how many images", default=1, extra={"min": 1, "max": 15},
             tooltip="More than one: a series of related images",
             label_ru="сколько картинок", tip_ru="Больше одной: серия связанных картинок"),
          _variant(),
          In("watermark", "BOOLEAN", "watermark", default=False, advanced=True, label_ru="водяной знак")],
         [Out("IMAGE", "images", "картинки")], run_seedream, output_node=True, cached=True,
         search=["seedream", "bytedance", "image", "картинка", "рисунок", "фото"], needs="models:seedream",
         title_ru="Картинка Seedream", about_ru="Картинка по описанию в Seedream от ByteDance. С референсами: по "
                                                "образцу, редактирование, тот же персонаж."),
    Node("LoomSeedanceVideo", "Seedance video", BYTEDANCE,
         "A video with ByteDance's Seedance: from a description, from the first (and last) frame or from references: "
         "images, characters, sound. Takes minutes.",
         [_model("seedance"),
          In("prompt", "STRING", "description", default="", extra={
              "multiline": True, "placeholder": "What happens in the shot, camera movement, sound"},
             label_ru="описание", hint_ru="Что происходит в кадре, движение камеры, звук"),
          In("first_frame", "IMAGE", "first frame", optional=True, label_ru="первый кадр"),
          In("last_frame", "IMAGE", "last frame", optional=True, label_ru="последний кадр"),
          In("references", "IMAGE", "reference", optional=True, grow=9,
             tooltip="Samples of a character, an object, a style; not together with frames. In the description: "
                     "@Image 1, @Image 2",
             label_ru="референс", tip_ru="Образцы персонажа, предмета, стиля; не вместе с кадрами. В описании: "
                                         "@Image 1, @Image 2"),
          In("audio", "AUDIO", "audio", optional=True, grow=VIDEO_AUDIO_25["count"],
             tooltip="A voice, music or sounds for the video: wav or mp3 (other formats are converted). Seedance 2.0: "
                     "up to 3 clips of 2–15 s, 15 s in all, only with a reference; Seedance 2.5: up to 10 clips of "
                     "2–30 s, 30 s in all. Not together with frames. In the description: @Audio 1",
             label_ru="аудио", tip_ru="Голос, музыка или звуки для ролика: wav или mp3 (другие форматы переводятся "
                                      "сами). Seedance 2.0: до 3 звуков по 2–15 с, всего до 15 с, только вместе с "
                                      "референсом; Seedance 2.5: до 10 по 2–30 с, всего до 30 с. Не вместе с кадрами. "
                                      "В описании: @Audio 1"),
          In("resolution", "COMBO", "quality", default="720p", options=["480p", "720p", "1080p", "4k"],
             tooltip="1080p: Seedance 2.0 and 2.5; 4k: Seedance 2.0 only; fast and mini: up to 720p",
             label_ru="качество", tip_ru="1080p: Seedance 2.0 и 2.5; 4k: только Seedance 2.0; fast и mini: до 720p"),
          In("ratio", "COMBO", "aspect ratio", default="adaptive",
             options=["adaptive", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
             tooltip="adaptive: from the frame or as the model decides",
             label_ru="пропорции", tip_ru="adaptive: по кадру или как решит модель"),
          In("duration", "INT", "seconds", default=5, extra={"min": -1, "max": 30},
             tooltip="Seedance 2.0: 4–15 s, Seedance 2.5: 4–30 s; -1: the model picks the length",
             label_ru="секунд", tip_ru="Seedance 2.0: 4–15 с, Seedance 2.5: 4–30 с; -1: длительность выбирает модель"),
          In("sound", "BOOLEAN", "sound", default=True, label_ru="звук"),
          _variant(),
          In("watermark", "BOOLEAN", "watermark", default=False, advanced=True, label_ru="водяной знак")],
         [Out("VIDEO", "video", "видео"), Out("IMAGE", "last frame", "последний кадр")], run_seedance,
         output_node=True, cached=True, search=["seedance", "bytedance", "video", "animate", "ролик", "оживить"],
         needs="models:seedance",
         title_ru="Видео Seedance", about_ru="Видео в Seedance от ByteDance: по описанию, по первому (и последнему) "
                                             "кадру или по референсам: картинки, персонажи, звук. Занимает минуты."),
    Node("LoomSeedanceCharacter", "Seedance character", BYTEDANCE,
         "A ready character for Seedance video: the same face in every clip. Connect it to the first frame or a "
         "reference of “Seedance video”.",
         [In("character", "STRING", "character", default="", extra={"placeholder": "asset id, or press “Pick a character”"},
             label_ru="персонаж", hint_ru="id ассета или кнопка «Выбрать персонажа»")],
         [Out("IMAGE", "character", "персонаж")], run_character,
         search=["character", "actor", "face", "asset", "персонаж", "актёр", "лицо"], needs="models:seedance",
         title_ru="Персонаж Seedance", about_ru="Готовый персонаж для видео Seedance: одно лицо во всех роликах. "
                                                "Подключите к первому кадру или референсу ноды «Видео Seedance»."),

    Node("LoomVideoFrame", "Frame from video", VIDEO_TOOLS,
         "The first or the last frame of a video: for example, to continue it with the next clip.",
         [In("video", "VIDEO", "video", label_ru="видео"),
          In("which", "COMBO", "frame", default="last", options=["last", "first"],
             label_ru="кадр", options_ru=["последний", "первый"])],
         [Out("IMAGE", "frame", "кадр")], run_video_frame, output_node=True, cached=True,
         search=["frame", "last frame", "continue", "последний кадр", "продолжить"],
         title_ru="Кадр из видео", about_ru="Первый или последний кадр видео: например, чтобы продолжить ролик "
                                            "следующим."),
    Node("LoomJoinVideos", "Join videos", VIDEO_TOOLS, "Clips one after another in one video.",
         [In("videos", "VIDEO", "video", grow=10, label_ru="видео")],
         [Out("VIDEO", "video", "видео")], run_join_videos, output_node=True, cached=True,
         search=["concat", "edit", "монтаж", "объединить"],
         title_ru="Склеить видео", about_ru="Ролики друг за другом в одно видео."),

    Node("LoomJoinText", "Join text", TEXT_TOOLS, "Several texts in one, with a separator.",
         [In("texts", "STRING", "text", grow=10, label_ru="текст"),
          In("separator", "STRING", "separator", default="\\n\\n", advanced=True,
             tooltip="\\n: a line break", label_ru="разделитель", tip_ru="\\n: перенос строки")],
         [Out("STRING", "text", "текст")], run_join_text, output_node=True, search=["concat", "объединить"],
         title_ru="Собрать текст", about_ru="Несколько текстов в один, через разделитель."),
    Node("LoomShow", "Show", TEXT_TOOLS, "Shows what comes in: text, images, video, sound.",
         [In("value", "*", "what to show", label_ru="что показать")], [], run_show, output_node=True,
         audio_player=True, search=["preview", "просмотр"],
         title_ru="Показать", about_ru="Показывает то, что пришло на вход: текст, картинки, видео, звук."),
]

BY_ID = {n.id: n for n in NODES}  # with the providers' own nodes (set_dynamic)
_dynamic: list[Node] = []


def set_dynamic(made: list[Node]) -> None:
    """The nodes the providers made of their APIs' schemas (fal), in place of the ones before."""
    global _dynamic
    for n in _dynamic:
        BY_ID.pop(n.id, None)
    _dynamic = [n for n in made if n.id not in BY_ID]
    BY_ID.update({n.id: n for n in _dynamic})


def all_nodes() -> list[Node]:
    return NODES + _dynamic


def available(node: Node, choices: dict[str, list]) -> bool:
    """Offered to the user: no provider has its models, no node (a graph that has one says it is missing)."""
    return node.needs is None or bool(choices.get(node.needs))


# ---------------------------------------------------------------- /object_info


def _input_spec(i: In, choices: dict[str, list]) -> list:
    opts: dict[str, Any] = {"display_name": i.label}
    if i.tooltip:
        opts["tooltip"] = i.tooltip
    if i.advanced:
        opts["advanced"] = True
    if i.default is not None:
        opts["default"] = i.default
    opts.update(i.extra)
    ru = i18n.current() == "ru"
    if ru and i.hint_ru and "placeholder" in opts:
        opts["placeholder"] = i.hint_ru
    if i.type == "COMBO":
        options = choices.get(i.options, []) if isinstance(i.options, str) else list(i.options or [])
        if ru and i.options_ru:  # values that are words: Russian ones for a Russian user (both are accepted)
            opts["default"] = i.options_ru[options.index(i.default)] if i.default in options else i.options_ru[0]
            options = list(i.options_ru)
        opts["options"] = options
        opts["multiselect"] = False
        if "default" not in opts and options:
            opts["default"] = options[0]
    return [i.type, opts]


def _grow_spec(i: In, choices: dict[str, list]) -> list:
    """A list input that grows by a slot as the last one is connected; the editor names the slots, so the names
    are the labels ("reference 1", "reference 2", ...) and the graph's keys are "<input>.<slot name>"."""
    one = In(i.name, i.type, i.label, optional=True, tooltip=i.tooltip, extra={"forceInput": True} if i.widget else {},
             label_ru=i.label_ru, tip_ru=i.tip_ru)
    names = [slot_name(i, k) for k in range(1, i.grow + 1)]
    return [AUTOGROW, {"display_name": i.label, "template": {
        "input": {"optional": {i.name: _input_spec(one, choices)}}, "names": names, "min": 1}}]


def slot_name(i: In, k: int) -> str:
    return f"{i.label} {k}"


def object_info(choices: dict[str, list]) -> dict:
    out = {}
    for n in all_nodes():
        if not available(n, choices):
            continue
        required, optional = {}, {}
        for i in n.inputs:
            spec = _grow_spec(i, choices) if i.grow else _input_spec(i, choices)
            (optional if i.optional or i.grow else required)[i.name] = spec
        if n.audio_player:
            required["audioUI"] = ["AUDIO_UI", {}]
        out[n.id] = {
            "input": {"required": required, "optional": optional},
            "input_order": {"required": list(required), "optional": list(optional)},
            "is_input_list": False,
            "output": [o.type for o in n.outputs],
            "output_is_list": [False] * len(n.outputs),
            "output_name": [o.label for o in n.outputs],
            "name": n.id,
            "display_name": n.title,
            "description": n.description,
            "python_module": "nodes",  # shown with the editor's own nodes, not as an "extension"
            "category": n.category,
            "output_node": n.output_node,
            "has_intermediate_output": False,
            "search_aliases": n.search + ([n.title_ru] if i18n.current() == "ru" and n.title_ru else []),
        }
    return out


# ---------------------------------------------------------------- the editor's Russian (GET /api/i18n)

CATEGORY_RU = {"Input": "Ввод", "Generate": "Генерация", "OpenAI": "OpenAI", "ByteDance": "ByteDance",
               "Video": "Видео", "Text": "Текст",
               # fal's categories of models
               "text-to-image": "текст в картинку", "image-to-image": "картинка в картинку",
               "text-to-video": "текст в видео", "image-to-video": "картинка в видео",
               "video-to-video": "видео в видео", "text-to-speech": "текст в речь", "text-to-audio": "текст в звук",
               "speech-to-text": "речь в текст", "audio-to-audio": "звук в звук"}


def i18n_ru() -> dict:
    """The nodes in Russian, in the editor's own translation format: a Russian user sees these names, tooltips and
    categories while /object_info keeps the English ones (saved graphs and the slot names of growing inputs stay the
    same in both languages). The slots of growing inputs ("reference 1") are labelled by nodeloom.js."""
    defs = {}
    for n in all_nodes():
        inputs: dict[str, dict] = {}
        for i in n.inputs:
            inputs[i.name] = {"name": i.label_ru or i.label, **({"tooltip": i.tip_ru} if i.tip_ru else {})}
            for k in range(1, i.grow + 1):
                inputs[slot_name(i, k)] = {"name": f"{i.label_ru or i.label} {k}"}
        defs[n.id] = {"display_name": n.title_ru or n.title, "description": n.about_ru or n.description,
                      "inputs": inputs,
                      "outputs": {str(k): {"name": o.label_ru or o.label} for k, o in enumerate(n.outputs)}}
    return {"nodeDefs": defs, "nodeCategories": dict(CATEGORY_RU)}
