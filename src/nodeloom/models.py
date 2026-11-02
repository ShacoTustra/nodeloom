"""The tables: runs, reusable node results, files and the editor's userdata.

Everything belongs to a user (`user_id`): one local user for now (auth.py), more later without a schema change.
"""

import secrets
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from nodeloom.db import Base, utcnow


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class Run(Base):
    """A run of a workflow (a "prompt" in ComfyUI's terms; `id` is its prompt_id). Runs wait as `pending` and any
    process claims them (see runs.py); `outputs` is what the nodes showed, by node id."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    client_id: Mapped[str | None] = mapped_column(String(64))  # the browser tab that queued it (its events)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|running|success|error|cancelled
    prompt: Mapped[dict] = mapped_column(default=dict)  # the graph in ComfyUI's API format
    extra: Mapped[dict] = mapped_column(default=dict)  # extra_data (the editor's workflow is in it)
    targets: Mapped[list | None] = mapped_column(default=None)  # partial execution: output nodes to run
    outputs: Mapped[dict] = mapped_column(default=dict)
    messages: Mapped[list] = mapped_column(default=list)  # [[event, data], ...] as ComfyUI's history keeps them
    error: Mapped[dict | None] = mapped_column(default=None)
    worker: Mapped[str | None] = mapped_column(String(64))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NodeResult(Base):
    """What a node produced for a given input (its signature): reused instead of paying for the same generation
    again. While an upstream task runs, `task_id` is set and `value` is empty, so a run resumed after a restart
    follows the task instead of starting a new one."""

    __tablename__ = "node_results"
    __table_args__ = (UniqueConstraint("user_id", "sig"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    sig: Mapped[str] = mapped_column(String(64))
    node_type: Mapped[str] = mapped_column(String(64))
    value: Mapped[list | None] = mapped_column(default=None)  # the node's outputs
    ui: Mapped[dict | None] = mapped_column(default=None)  # what the node shows
    task_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class StoredFile(Base):
    """A user's file in the media store: uploaded (`input`), generated (`output`) or intermediate (`temp`).
    The editor addresses it by (kind, name), as ComfyUI's /view does by (type, filename)."""

    __tablename__ = "files"
    __table_args__ = (UniqueConstraint("user_id", "kind", "name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("f"))
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(8))
    name: Mapped[str] = mapped_column(String(255))
    key: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(100))
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    meta: Mapped[dict] = mapped_column(default=dict)  # width, height, duration
    # The run that made a result: deleting the run from the editor's history deletes its files.
    run_id: Mapped[str | None] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Userdata(Base):
    """The editor's own files (ComfyUI's /userdata): saved workflows, settings, templates."""

    __tablename__ = "userdata"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(String(512), primary_key=True)
    content: Mapped[bytes] = mapped_column(LargeBinary)
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
