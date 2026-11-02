# Nodeloom

*aka ComfyUI Lite*

**ComfyUI's node editor for models behind APIs.** Build workflows that chain text, image, video and voice models (OpenAI, OpenRouter, fal.ai, BytePlus ModelArk and anything that speaks the OpenAI API) on the canvas you know from ComfyUI, without local inference: no torch, no CUDA, no model downloads, no GPU. It installs in seconds and runs on a laptop or a small server.

Think of it as "ComfyUI Lite". It is an independent project, not affiliated with Comfy Org. The editor in the browser is the official ComfyUI frontend ([ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend), the `comfyui-frontend-package` wheel, unchanged); the backend is a new, small server that speaks the part of ComfyUI's protocol the frontend uses.

## Why

ComfyUI is a great editor, but it comes with a local inference stack. If all you need is to wire up hosted models (an LLM writes the prompt, an image model draws it, a video model animates it) you still install gigabytes of torch, and its API nodes only work through Comfy's own cloud account. This project keeps the editor and replaces everything behind it:

- **Your providers, your keys.** Configure OpenAI, OpenRouter, fal.ai, ModelArk, a local Ollama or LM Studio, or your own gateway. Model lists come from each provider's `GET /models`; every fal.ai model you list becomes a node of its own, built from the model's API schema (Flux, Kling, Veo, Wan, ElevenLabs and hundreds more).
- **Parallel by default.** Independent branches of a graph run at the same time (two videos from two frames generate together).
- **Paid results are never paid twice.** Every generation is cached by its inputs: run the graph again and only what changed is regenerated. Change the prompt of the last node and the upstream picture is reused for free.
- **Robust queue.** Runs wait in a database (SQLite, or PostgreSQL for several processes). A restart in the middle of a 5-minute video generation does not lose it: the run resumes and follows the same upstream task.
- **Costs.** After a run, the editor tells you what it cost, per node (from the provider's usage and the model's price).
- **The Builder.** A side panel where you describe what you want ("a product shot on white, then a 5-second video of it") and an LLM builds the graph for you. Nothing runs until you press Run.
- **English and Russian** UI, including the nodes.

## Quick start

Requires Python 3.12+ and, for the video tools, `ffmpeg` on the PATH.

```bash
pip install git+https://github.com/ShacoTustra/nodeloom    # not on PyPI yet
export OPENAI_API_KEY=sk-...        # and/or OPENROUTER_API_KEY, FAL_KEY, ARK_API_KEY
nodeloom --open                     # http://127.0.0.1:8188
```

or, without installing: `uvx --from git+https://github.com/ShacoTustra/nodeloom nodeloom --open`.

Or with Docker:

```bash
git clone https://github.com/ShacoTustra/nodeloom && cd nodeloom
docker build -t nodeloom .
docker run -p 8188:8188 -e OPENAI_API_KEY=sk-... -v nodeloom:/data nodeloom
```

The database and your files live in your user data directory (`~/.local/share/nodeloom` on Linux, `~/Library/Application Support/nodeloom` on macOS, `%LOCALAPPDATA%\nodeloom` on Windows), or wherever `--data-dir` / `DATA_DIR` points.

## Providers

The common ones need only an environment variable:

| Variable | Provider | Nodes |
|---|---|---|
| `OPENAI_API_KEY` (`OPENAI_BASE_URL`) | OpenAI | Text model, GPT Image |
| `OPENROUTER_API_KEY` | OpenRouter: hundreds of chat models, with prices | Text model |
| `FAL_KEY` | fal.ai | a node per model: Flux Dev, Flux Pro Kontext, Nano Banana (+ Edit), Qwen Image, Kling 2.1, Veo 3 Fast, Veo 3 Image to Video, Hailuo 02, Wan 2.2, ElevenLabs TTS |
| `ARK_API_KEY` (`ARK_BASE_URL`) | BytePlus ModelArk (Volcengine Ark with its own base URL) | Text model, Seedream image, Seedance video, Seedance character |

Anything else goes into `config.toml` (in the working directory or the data directory, or `--config path`). See [`config.example.toml`](config.example.toml):

```toml
[providers.local]
type = "openai"                       # the OpenAI API: Ollama, LM Studio, vLLM, Groq, DeepSeek, Together...
base_url = "http://localhost:11434/v1"
models = ["llama3.2", "qwen2.5vl"]    # optional: otherwise GET /models

[providers.local.extra_models."qwen2.5vl"]
inputs = ["image"]                    # what a chat model takes besides text

[providers.openai.extra_models."gpt-image-1"]
price = { input = 5.0, image_input = 10.0, output = 40.0 }   # US$ per million tokens, for the cost shown

[providers.fal]
type = "fal"
api_key_env = "FAL_KEY"
[providers.fal.endpoints."fal-ai/flux/dev"]                  # any model of fal.ai/models, by its endpoint id
price = { request = 0.025 }
[providers.fal.endpoints."fal-ai/kling-video/v2.1/master/image-to-video"]
title = "Kling 2.1 Master"
```

In the nodes a model is `<provider>/<model>`, so the same graph can mix providers. A node whose models no provider has is not offered (and a template that needs it is not listed).

A fal.ai endpoint becomes a node under *Generate / fal / <category>*: its settings are widgets (the common ones in front, the rest under "Show advanced inputs"), its image, video and audio inputs are links, its results are outputs. Its schema is fetched once and kept in the data directory.

## Nodes

| Category | Nodes |
|---|---|
| Input | Text, Image, Video file, Audio file |
| Generate | **Text model**: any provider's chat model; images, sound and video as inputs where the model takes them |
| Generate / OpenAI | **GPT Image**: generation, or editing with up to 16 reference images |
| Generate / ByteDance | **Seedream image** (up to 10 references, series of images), **Seedance video** (text, first/last frame, references, sound), **Seedance character** (ready faces for video) |
| Generate / fal / … | one node per fal.ai model you configure: images, video, speech |
| Video | Frame from video (first or last: continue a clip), Join videos |
| Text | Join text, Show (text, images, video, sound) |

Images, video and sound travel between nodes as files; a generating node shows its result on the canvas, and every result lands in the editor's assets panel.

## Security

There is one local user and no sign-in, as in ComfyUI. The server listens on `127.0.0.1` by default and accepts only requests whose `Host` is a local name (`ALLOWED_HOSTS`), so a web page cannot reach it through DNS rebinding; requests that change anything must come from the editor's own pages. To use it over a network, put it behind a reverse proxy with authentication, start it with `--host 0.0.0.0` and add your host name to `ALLOWED_HOSTS`. Your provider keys stay on the server.

## Settings

| Setting | Default | |
|---|---|---|
| `DATA_DIR` | user data directory | database and files |
| `DATABASE_URL` | SQLite in `DATA_DIR` | `postgresql+asyncpg://...` with `pip install nodeloom[postgres]` |
| `REDIS_URL` | none | events across several processes |
| `MEDIA_DIR` / `MEDIA_S3_*` | `DATA_DIR/media` | files on disk or in S3-compatible storage (`nodeloom[s3]`) |
| `STORAGE_LIMIT_MB` | 0 (no limit) | room for a user's files |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1,::1` | `*` behind your own proxy |
| `LANGUAGE` | the browser's | `en` or `ru` (also switchable in the editor's menu) |
| `AGENT_MODEL` | the first chat model | the Builder's model, `<provider>/<model>` with function calling |
| `RUNS_PER_USER`, `QUEUE_PER_USER`, `RUNS_PER_PROCESS`, `GENERATIONS_PER_PROCESS` | 2, 20, 200, 100 | concurrency |

## Development

```bash
uv sync
uv run pytest          # providers are mocked; the video tests need ffmpeg
uv run nodeloom
```

How it works inside: [docs/architecture.md](docs/architecture.md).

## Status

Alpha. Working: the editor, the queue, caching and resuming, the nodes above, fal.ai models, the Builder, costs, managing uploads. Next: Replicate (its models' schemas, like fal's), Google's Gemini API, several users.

## License

GPL-3.0, like the ComfyUI frontend it serves. "ComfyUI" is a name of Comfy Org, used here only to say what this works with.
