# Nodeloom

*aka ComfyUI Lite*

**ComfyUI's node editor for models behind APIs.** Build workflows that chain text, image and video models (OpenAI, OpenRouter, BytePlus ModelArk and anything that speaks the OpenAI API) on the canvas you know from ComfyUI, without local inference: no torch, no CUDA, no model downloads, no GPU. It installs in seconds and runs on a laptop or a small server.

Think of it as "ComfyUI Lite". It is an independent project, not affiliated with Comfy Org. The editor in the browser is the official ComfyUI frontend ([ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend), the `comfyui-frontend-package` wheel, unchanged); the backend is a new, small server that speaks the part of ComfyUI's protocol the frontend uses.

## Why

ComfyUI is a great editor, but it comes with a local inference stack. If all you need is to wire up hosted models (an LLM writes the prompt, an image model draws it, a video model animates it) you still install gigabytes of torch, and its API nodes only work through Comfy's own cloud account. This project keeps the editor and replaces everything behind it:

- **Your providers, your keys.** Configure OpenAI, OpenRouter, ModelArk, a local Ollama or LM Studio, or your own gateway. Model lists come from each provider's `GET /models`.
- **Parallel by default.** Independent branches of a graph run at the same time (two videos from two frames generate together).
- **Paid results are never paid twice.** Every generation is cached by its inputs: run the graph again and only what changed is regenerated. Change the prompt of the last node and the upstream picture is reused for free.
- **Robust queue.** Runs wait in a database (SQLite, or PostgreSQL for several processes). A restart in the middle of a 5-minute video generation does not lose it: the run resumes and follows the same upstream task.
- **Costs.** After a run, the editor tells you what it cost, per node (from the provider's usage and the model's price).
- **The Builder.** A side panel where you describe what you want ("a product shot on white, then a 5-second video of it") and an LLM builds the graph for you. Nothing runs until you press Run.
- **English and Russian** UI, including the nodes.

## Quick start

Requires Python 3.12+ and, for the video tools, `ffmpeg` on the PATH.

```bash
pip install nodeloom                # or: uvx nodeloom
export OPENAI_API_KEY=sk-...        # and/or OPENROUTER_API_KEY, ARK_API_KEY
nodeloom --open                     # http://127.0.0.1:8188
```

Or with Docker:

```bash
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
```

In the nodes a model is `<provider>/<model>`, so the same graph can mix providers. A node whose models no provider has is not offered (and a template that needs it is not listed).

## Nodes

| Category | Nodes |
|---|---|
| Input | Text, Image, Video file, Audio file |
| Generate | **Text model**: any provider's chat model; images, sound and video as inputs where the model takes them |
| Generate / OpenAI | **GPT Image**: generation, or editing with up to 16 reference images |
| Generate / ByteDance | **Seedream image** (up to 10 references, series of images), **Seedance video** (text, first/last frame, references, sound), **Seedance character** (ready faces for video) |
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

Alpha. Working: the editor, the queue, caching and resuming, the nodes above, the Builder, costs. Next: more providers and nodes for them (fal, Replicate, Google Veo and Gemini, Kling, Runway), several users, managing uploads in the editor.

## License

GPL-3.0, like the ComfyUI frontend it serves. "ComfyUI" is a name of Comfy Org, used here only to say what this works with.
