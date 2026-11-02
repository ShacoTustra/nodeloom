"""Nodeloom (aka ComfyUI Lite): ComfyUI's node editor for models behind APIs (text, images, video), no local inference.

The editor in the browser is ComfyUI's frontend (the `comfyui-frontend-package` wheel, unchanged, extended by
`web/nodeloom.js`). The backend is ours: it speaks the part of ComfyUI's HTTP and WebSocket protocol that the frontend
uses, without ComfyUI's single queue, shared folders and torch:

- api.py        the editor's files, /api/* and /ws, as ComfyUI serves them;
- nodes.py      the nodes the editor offers: simple inputs, generations through the providers, a few video tools;
- providers/    who runs the models: OpenAI-compatible APIs, Gemini, ModelArk, fal.ai, Replicate;
- engine.py     checks a graph and runs it: independent branches in parallel, results reused while inputs stay the same;
- runs.py       the run queue in the database: any process claims pending runs, a dead process's runs resume;
- bus.py        run events to the user's open editors, across processes (Redis) or in one process;
- files.py      the user's files (uploads, results) in the media store;
- agent.py      the Builder: a chat model that builds graphs from a request;
- mcp.py        MCP for agents outside the editor: the same nodes, runs and canvas over /mcp.
"""

__version__ = "0.1.0"
