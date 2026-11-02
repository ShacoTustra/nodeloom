"""The command: `nodeloom [--host 127.0.0.1] [--port 8188] [--config config.toml] [--data-dir DIR] [--open]`."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import webbrowser

from nodeloom import __version__
from nodeloom.config import APP_NAME


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nodeloom", description=f"{APP_NAME}: ComfyUI's editor for API models")
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


if __name__ == "__main__":
    main()
