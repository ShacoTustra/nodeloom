"""ffmpeg helpers of the video nodes: a video for a chat model, the sound of a clip, joining clips."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class VideoError(Exception):
    """The message is Russian; `en` is the same for an English user."""

    def __init__(self, message: str, en: str = "Could not process the video"):
        super().__init__(message)
        self.en = en


async def ffmpeg(*args: str, timeout: float = 600) -> None:
    proc = await asyncio.create_subprocess_exec("ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except BaseException:
        proc.kill()
        raise
    if proc.returncode != 0:
        log.warning("ffmpeg %s: %s", args, err.decode(errors="replace")[-500:])
        raise VideoError("Не удалось обработать видео")


async def probe(path: Path) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json",
        str(path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    try:
        return json.loads(out or b"{}")
    except ValueError:
        return {}


MODEL_VIDEO_MAX_S = 300.0  # a chat model watches a video of up to 5 minutes (our limit, like the sound's)
MODEL_VIDEO_BUDGET = 45 * 1024 * 1024  # base64 of the videos in one chat request (ModelArk takes a body of up to 64 MB)


def duration(info: dict) -> float:
    try:
        return float((info.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        return 0.0


async def for_model(src: Path, dst: Path) -> float:
    """A video as a chat model takes it: H.264/AAC mp4, the long side at most 1280 px (the model looks at a frame a
    second, a smaller file keeps the request under the body limit). Returns its length in seconds."""
    info = await probe(src)
    if not any(st.get("codec_type") == "video" for st in info.get("streams") or []):
        raise VideoError("В файле нет видео", "The file has no video")
    if duration(info) > MODEL_VIDEO_MAX_S:
        minutes = int(MODEL_VIDEO_MAX_S // 60)
        raise VideoError(f"Видео длиннее {minutes} минут: укоротите его",
                         f"The video is longer than {minutes} minutes: shorten it")
    await ffmpeg("-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                 "-vf", "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dst))
    return duration(await probe(dst))


async def join(paths: list[Path], out: Path) -> None:
    """Clips one after another into an H.264/AAC mp4 with the first clip's frame size; a clip without sound gets
    silence, so the audio track runs through the whole video."""
    infos = [await probe(p) for p in paths]
    streams = [{s.get("codec_type"): s for s in (info.get("streams") or [])} for info in infos]
    first = streams[0].get("video") or {}
    w, h = first.get("width") or 1280, first.get("height") or 720
    args, parts, filters = [], [], []
    for p in paths:
        args += ["-i", str(p)]
    for i, st in enumerate(streams):
        filters.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                       f"setsar=1,fps=24,format=yuv420p[v{i}]")
        if "audio" in st:
            filters.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[a{i}]")
        else:
            dur = float((infos[i].get("format") or {}).get("duration") or 5)
            filters.append(f"anullsrc=r=44100:cl=stereo,atrim=duration={dur:.3f}[a{i}]")
        parts.append(f"[v{i}][a{i}]")
    filters.append(f"{''.join(parts)}concat=n={len(paths)}:v=1:a=1[v][a]")
    await ffmpeg(*args, "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart",
                 str(out))
