"""Make the GitHub release of every version tag that has none: its notes from CHANGELOG.md, its files as PyPI has
them (the release is made once PyPI has the version). `--dry-run` says what it would do, without `gh`."""

import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PACKAGE = "nodeloom"
DRY = "--dry-run" in sys.argv


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def notes(version: str) -> str:
    text = Path("CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(version)}[ \t]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    # one line per paragraph and list item: GitHub would keep the file's line breaks
    body = re.sub(r"\n(?!\n|- |#)[ \t]*", " ", m.group(1).strip()) if m else ""
    return (body + "\n\n" if body else "") + (f"PyPI: https://pypi.org/project/{PACKAGE}/{version}/ · "
                                              f"`pip install {PACKAGE}=={version}`")


def pypi_files(version: str, attempts: int = 20) -> list[dict] | None:
    """The version's files on PyPI; None when it is not there (yet: a new upload takes a minute to show)."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{PACKAGE}/{version}/json", timeout=30) as r:
                return [{"name": u["filename"], "url": u["url"]} for u in json.load(r)["urls"]]
        except OSError:
            if i + 1 < attempts:
                time.sleep(15)
    return None


def main() -> None:
    tags = sorted((t for t in sh("git", "tag", "--list", "v*").split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)),
                  key=lambda t: tuple(int(x) for x in t[1:].split(".")))
    made = set() if DRY else {r["tagName"] for r in json.loads(sh("gh", "release", "list", "--limit", "200",
                                                                       "--json", "tagName"))}
    for tag in tags:
        if tag in made:
            continue
        version = tag[1:]
        found = pypi_files(version, attempts=1 if DRY else 20)
        if found is None:
            print(f"{tag}: not on PyPI, left for the next run")
            continue
        latest = "true" if tag == tags[-1] else "false"
        if DRY:
            print(f"{tag}: would release (latest {latest}) with {[f['name'] for f in found]}\n{notes(version)}\n")
            continue
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for f in found:
                path = Path(d) / f["name"]
                urllib.request.urlretrieve(f["url"], path)
                paths.append(str(path))
            sh("gh", "release", "create", tag, *paths, "--verify-tag", "--title", f"Nodeloom {version}",
               "--notes", notes(version), f"--latest={latest}")
        print(f"{tag}: released")


if __name__ == "__main__":
    main()
