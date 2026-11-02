# How it works

The browser runs the stock ComfyUI frontend (`comfyui-frontend-package`, served unchanged from the wheel). The
backend is ours: it answers the part of ComfyUI's HTTP and WebSocket protocol that the frontend uses, runs graphs of
our own nodes, and calls model providers. There is no torch and no local model.

```
browser: ComfyUI frontend + web/nodeloom.js
   │  /api/object_info, /api/prompt, /api/jobs, /api/view, /ws, ...
   ▼
api.py ── runs.py (queue in the DB, dispatcher) ── engine.py (graph, cache) ── nodes.py ── providers/ ── APIs
   │                                                                  │
   └── files.py (uploads, results: media_store.py, disk or S3)  ◄─────┘
```

## Modules

| Module | What |
|---|---|
| `app.py` | the FastAPI app: routes, the dispatcher and the hourly cleanup in its lifespan, the Host check |
| `cli.py` | `nodeloom`: host, port, config, data directory |
| `config.py` | settings from the environment and `config.toml` |
| `auth.py` | the user: one local user for now, and their language |
| `api.py` | ComfyUI's routes: the editor's page and files, `/api/*`, `/ws`, templates; our `/api/nodeloom/*` |
| `nodes.py` | the nodes: their description for `/object_info` and how each runs |
| `providers/` | who runs the models: `openai`, `google`, `ark`, `fal` and `replicate` types, model lists, costs |
| `schema_nodes.py` | nodes made of a model's OpenAPI schema (fal's endpoints, Replicate's models) |
| `engine.py` | checking a graph (ComfyUI's error format) and running it |
| `runs.py` | the run queue: submit, claim, heartbeat, recover, cancel, cleanup |
| `bus.py` | run events to the user's open tabs (Redis pub/sub, or within the process) |
| `files.py`, `media_store.py` | the user's files and where they are stored |
| `agent.py` | the Builder: an LLM with `set_graph` that builds graphs |
| `mcp.py` | MCP for agents outside the editor: the catalog, checks, runs, uploads, the editor's canvas |
| `web/` | our extension of the editor (`nodeloom.js`, `nodeloom.css`), the loading screen, templates, the starter graph |

## Runs

- `POST /api/prompt` checks the graph (`engine.validate`) and stores a `pending` run (`runs` table), in the language
  of the request.
- Every process has a dispatcher that claims pending runs (`FOR UPDATE SKIP LOCKED` on PostgreSQL), at most
  `RUNS_PER_USER` per user and `RUNS_PER_PROCESS` per process, and runs them. A running run's process refreshes its
  heartbeat; a run whose process went silent goes back to `pending` and is resumed by another process.
- On stop (a deploy, Ctrl+C) generations in flight get `STOP_GRACE_S` to finish, video tasks are let go (their ids
  are remembered), and the runs go back to the queue at once.
- Events go to the browser tab that queued the run, in ComfyUI's format (`execution_start`, `executing`, `executed`,
  `progress_state`, binary progress text, ...), and at the end our `nodeloom.cost` with what the run cost.

## The engine

Unlike ComfyUI, nodes run concurrently: each starts as soon as its inputs are ready. A node marked `cached` is skipped
when its result for the same inputs is known: its signature hashes the node type and version, its settings, the
signatures of what feeds it, and the id and size of uploaded files. The result is stored in `node_results` per user;
a generation that is reused costs nothing. `seed` widgets default to "fixed", so a run with nothing changed reuses
everything; another variant is one click.

When a node fails, nodes that have not started are dropped, the ones already running finish (their results are paid
for and kept), and the run ends with the error.

A node that starts an upstream task (Seedance video) stores the task id under its signature before polling, so a run
resumed after a restart follows the same task instead of paying for a new one.

## Nodes and the protocol

Values between nodes are plain: text (`STRING`), a list of images (`IMAGE`), one video (`VIDEO`), one sound
(`AUDIO`); a picture or a clip is a file ref `{"kind", "name", "media", "ct", "w", "h"}`, never a tensor.

Inputs that take a list grow by a slot as the last one is connected (`COMFY_AUTOGROW_V3`); their slots are named
after the label (`references.reference 1`, `references.reference 2`): those names are what saved graphs keep.

Node names, labels and slot names are English. Russian comes through the frontend's own mechanism for custom nodes:
`GET /api/i18n` returns `nodeDefs` and `nodeCategories` for `ru`. Placeholders and list values that are words
(`last` / `first`) are given in the user's language in `/object_info` itself, and `engine.grouped` accepts both.
Slot labels of growing inputs, which the frontend's translation does not reach, are set by `nodeloom.js`.

Quirks of the frontend the server relies on:
- An audio upload button needs the node to have an `audioUI` player (`["AUDIO_UI", {}]` in `/object_info`), or the
  node cannot be created.
- `/api/jobs`: `preview_output.type` must be `input`, `output` or `temp`; one other value drops the whole list. A
  character shown by "Show" (type `asset`) is therefore not a job preview.
- Validation errors are `400` in ComfyUI's shape (`error.type`, `node_errors[id].errors[].extra_info.input_name`),
  so the editor highlights the node and the input.

Empty answers for what ComfyUI has and this does not: local models, embeddings, blueprints, node replacements, logs.
The page is served with a CSP that allows only its own origin: no Comfy cloud, registry or CDN.

## Providers

A provider is a base URL, a key and a type. `openai` speaks the OpenAI API: `/chat/completions` for the text model
node, `/images/generations` and `/images/edits` for GPT Image, and, where the server is a router with a video API
(OpenRouter, ProxyAPI), `/videos` for the Video node (its models from `GET /videos/models`). `google` speaks Google's
Gemini API: text and Nano Banana through `generateContent` (the text model node's request is turned into Google's by
the provider's `chat`), the Builder through the OpenAI-compatible `/openai/chat/completions` (the provider's
`chat_path`, for function calling; the key as a bearer token there and as `x-goog-api-key` elsewhere), Omni video
through the Interactions API (`POST /interactions` in the background, its interaction id remembered and polled). `ark` speaks BytePlus ModelArk: chat, Seedream
(`/images/generations` with its own parameters), Seedance (`/contents/generations/tasks`), and ready characters from a
gateway's `/assets`.

Models come from `GET /models` (cached for 10 minutes), completed by the config (`models`, `extra_models`). Each
model has a kind (chat, image, video) and a family that says which node lists it: every chat model goes to the text
model node; `openai` images to GPT Image, a router's videos to Video; `google` images to Gemini image, its videos to Gemini video; `ark` images to
Seedream, `ark` videos to Seedance. A node whose family has
no models is not offered, and a template that needs one is not listed.

`fal` and `replicate` are different: each model has inputs of its own, so each becomes a node of its own
(`LoomFal<Endpoint>`, `LoomReplicate<Owner><Name>`), made of its OpenAPI schema (`schema_nodes.py`: strings, numbers,
enums and booleans are widgets, fal's `*_url` fields and Replicate's `format: uri` ones are links of an image, a video
or a sound, lists of them growing inputs; files and text in the result are outputs). What they share is in
`providers/schemas.py`: the schemas are kept in the data directory and fetched again in the background once a day,
the nodes are registered next to the static ones (`nodes.set_dynamic`) whenever the model lists are asked for, and the
request body and the outputs are made the same way. A run goes through fal's queue or as a Replicate prediction
(official models by name, others by the version their schema came from; files over 256 KB through Replicate's Files
API): the request or prediction id is remembered under the node's signature, so a resumed run follows it. A
Replicate output that is just a uri is a picture, a video or a sound as `replicate.guess_media` reads the model's
settings and description (the config's `kind` overrides it).

Result files are downloaded only from the provider's own hosts (or any public host where the type has no list),
each redirect checked again; the key goes along only to the API's own host and the type's `auth_file_hosts`.

The cost of a node is computed from the `usage` the provider returned and the model's `price` (`providers/pricing.py`):
OpenRouter's `usage.cost` as it is, otherwise tokens, images, requests and seconds of run time at the configured
prices. A node that paid without a
known price makes the run's total a lower bound, and the editor says so.

## MCP

`POST /mcp` speaks MCP's Streamable HTTP without a stream or sessions: each JSON-RPC message (or batch) gets a JSON
answer, notifications a 202. It is guarded like the editor (the Host check, no cross-site browser requests, JSON
only) and acts as the local user. The tools take graphs in the Builder's form and reuse its code: `agent.catalog`
(with uploads named instead of left to the user), `agent.to_prompt` and `agent.check` (the engine's validation and
the models' abilities), `runs.submit`, then the run's row for its state and outputs; file outputs are links to
`/api/view` and pictures also come back as small JPEG previews. A call waits for a run up to `wait_seconds` (at most
300; clients drop calls that take longer than a minute or so), then the agent follows it with `get_run`.

`open_in_editor` and `get_editor_graph` ask the open editor through the run events (`nodeloom.graph`,
`nodeloom.ask_graph`, with a request id): one editor tab answers at `/api/nodeloom/editor/reply/{request}` (the
visible one; the others stand back through a BroadcastChannel). It opens the graph with the Builder's `applyGraph` in
a new tab, renamed as asked, or sends `app.graphToPrompt()` of its canvas. No answer in 5 s: no editor is open.
The answer must reach the process that asked, as it does with one process.

`nodeloom mcp` (cli.py) is a stdio bridge for clients that start a command: each line goes to `/mcp` (side by side,
so a long run does not hold up a ping), each answer to stdout; it ignores proxy variables to reach the local server.

## The Builder

`POST /api/nodeloom/agent` with the conversation and the graph on the canvas (`app.graphToPrompt().output`) streams NDJSON:
`{"type": "step", "label"}` while it works, then `{"type": "done", "text", "graph"?}` or `{"type": "error"}`. The model
(`AGENT_MODEL`, else the first chat model) sees the catalog of available nodes with the user's models, and saves the
whole graph with `set_graph`; the graph is checked with `engine.validate` plus what the video node would refuse only
when it runs, and the errors go back to the model. Characters come from `search_characters` where a provider has a
library; ids it did not find are refused. The panel rebuilds the canvas with the same node ids; nothing is run.

## Files

Uploads (`input`), results (`output`) and intermediate files (`temp`) live in the media store under
`<user>/<kind>/<name>`, with a row in `files` (size for the optional quota, `run_id` for results). Deleting a run
from the editor's history deletes its results; a cached result whose files are gone is generated again. Files are
served with `nosniff` and a sandbox CSP: an upload is never run as a page.

## Updating the ComfyUI frontend

1. `comfyui-frontend-package==<version>` in `pyproject.toml`; `FRONTEND_VERSION` and `COMFYUI_VERSION` (the ComfyUI
   version that frontend expects) in `api.py`.
2. Open the editor and watch the browser console: new routes the frontend asks for answer 404 until they are added
   (or answered empty).
3. Look through the menus, the node library, the run bar and the side panels: entries of new features that make no
   sense here go into the lists of `nodeloom.css` / `nodeloom.js`. Check the patched strings (`api.FRONTEND_PATCHES`).
4. `uv run pytest`.
