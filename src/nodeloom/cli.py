"""The command: `nodeloom [--host 127.0.0.1] [--port 8188] [--config config.toml] [--data-dir DIR] [--open]`, and
`nodeloom mcp [--url http://127.0.0.1:8188]`: MCP over stdio, forwarded to a running server (mcp.py)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor

from nodeloom import __version__
from nodeloom.config import APP_NAME


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["mcp"]:
        return mcp_bridge(argv[1:])
    parser = argparse.ArgumentParser(prog="nodeloom", description=f"{APP_NAME}: ComfyUI's editor for API models",
                                     epilog="nodeloom mcp [--url URL]: MCP over stdio for agents, forwarded to a "
                                            "running server (see nodeloom mcp --help)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default 127.0.0.1; 0.0.0.0 for the network: also set "
                             "ALLOWED_HOSTS and put it behind a proxy with auth)")
    parser.add_argument("--port", type=int, default=8188, help="port (default 8188, as ComfyUI)")
    parser.add_argument("--config", help="the TOML config file (default: config.toml here or in the data directory)")
    parser.add_argument("--data-dir", help="where the database and the files live")
    parser.add_argument("--open", action="store_true", help="open the editor in the browser")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    if args.config:
        os.environ["CONFIG_FILE"] = args.config
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from nodeloom.app import create_app

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    if args.open:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    logging.getLogger(__name__).info("%s on %s", APP_NAME, url)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info", ws_ping_interval=20)


def mcp_bridge(argv: list[str]) -> None:
    """MCP over stdio for clients that only start a command (Claude Desktop, ...): each JSON-RPC line from stdin goes
    to the server's /mcp, its answer to stdout. Requests are forwarded side by side, so a long run_graph does not hold
    up the rest. Only the protocol goes to stdout."""
    import httpx

    from nodeloom.mcp import WAIT_MAX

    parser = argparse.ArgumentParser(prog="nodeloom mcp", description=f"MCP over stdio, forwarded to a running "
                                                                      f"{APP_NAME} (start it with `nodeloom`)")
    parser.add_argument("--url", default=os.environ.get("NODELOOM_URL", "http://127.0.0.1:8188"),
                        help="the running server (default http://127.0.0.1:8188, or NODELOOM_URL)")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")
    url = base if base.endswith("/mcp") else base + "/mcp"
    lock = threading.Lock()

    def out(obj) -> None:
        with lock:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    # trust_env off: a proxy from the environment must not get between the bridge and the local server
    client = httpx.Client(timeout=httpx.Timeout(WAIT_MAX + 30, connect=5), trust_env=False)

    def forward(line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            out({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            return
        ask = isinstance(msg, dict) and msg.get("id") is not None and msg.get("method")
        try:
            r = client.post(url, content=line.encode(), headers={"Content-Type": "application/json",
                                                                 "Accept": "application/json, text/event-stream"})
        except httpx.HTTPError as e:
            if ask:
                out({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32603, "message":
                     f"{APP_NAME} is not running at {base}: start it with `nodeloom` ({type(e).__name__})"}})
            return
        if r.status_code == 202 or not r.content:
            return
        try:
            out(r.json())
        except ValueError:
            if ask:
                out({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32603, "message": f"HTTP {r.status_code}"}})

    with ThreadPoolExecutor(8) as pool:
        for line in sys.stdin:
            if line.strip():
                pool.submit(forward, line.strip())


if __name__ == "__main__":
    main()
