# Changelog

## 0.2.0

- **MCP for agents.** Nodeloom is an MCP server at `/mcp`: Claude Code, Cursor, Claude Desktop or any MCP client can
  read the catalog of nodes and models, check and run graphs, follow runs, upload input files, open a graph in the
  editor (a new tab) and read the one on the canvas. `nodeloom mcp` is a stdio bridge for clients that start a
  command.
- **Video models say what they can do.** The durations, resolutions, aspect ratios and frames of a router's video
  models (from `/videos/models`) reach the Builder, which keeps to them, and the Video node, which checks them
  before any request.

## 0.1.0

The first release: ComfyUI's node editor for models behind APIs. Providers: OpenAI and OpenAI-compatible servers,
OpenRouter (chat and video), Google's Gemini API, fal.ai and Replicate (a node per model, made of its schema),
BytePlus ModelArk. A queue that survives restarts, results cached by their inputs, costs per run, the Builder,
English and Russian.
