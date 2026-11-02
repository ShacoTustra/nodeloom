"""The run queue: runs wait in the database, any process runs them.

- /prompt checks the graph and stores a `pending` run; the process wakes its dispatcher.
- Every process's dispatcher claims pending runs (FOR UPDATE SKIP LOCKED on PostgreSQL), at most
  RUNS_PER_USER at a time for a user and RUNS_PER_PROCESS for the process, and runs them.
- A running run's process refreshes `heartbeat_at`. A run whose process stopped beating (crash, deploy) goes back to
  `pending` and is run again: finished nodes come from the cache, a video task in flight is followed, not restarted.
- Cancelling marks the run; its process notices on its next pass and stops it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import socket
import struct
from datetime import timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nodeloom import bus, engine, files, i18n, providers
from nodeloom.config import get_settings
from nodeloom.db import is_sqlite, sessionmaker, utcnow
from nodeloom.models import NodeResult, Run

log = logging.getLogger(__name__)

PENDING, RUNNING, CANCELLING = "pending", "running", "cancelling"
FINAL = ("success", "error", "cancelled")
HEARTBEAT_S = 15
STALE_S = 60
MAX_ATTEMPTS = 3
TEXT_EVENT = 3  # ComfyUI's binary event type for progress text



def worker_id(hostname: str, pid: int) -> str:
    """This process in `runs.worker` (64 characters): a hostname may be a long FQDN, its first label is enough."""
    return f"{hostname.split('.')[0][:40]}:{pid}:{secrets.token_hex(3)}"


WORKER = worker_id(socket.gethostname(), os.getpid())


# ---------------------------------------------------------------- choices: models and files per user


async def choices(session: AsyncSession, user_id: str) -> dict[str, list]:
    """The values of the per-user lists: the providers' models by node family, the user's uploads by type."""
    out: dict[str, list] = await providers.choices()
    for media in files.MEDIA:
        out[f"files:{media}"] = await files.names(session, user_id, "input", media)
    return out


# ---------------------------------------------------------------- queueing


async def queue_remaining(session: AsyncSession, user_id: str) -> int:
    return (await session.execute(select(func.count()).select_from(Run).where(
        Run.user_id == user_id, Run.status.in_((PENDING, RUNNING, CANCELLING))))).scalar_one()


async def send_status(user_id: str) -> None:
    async with sessionmaker()() as s:
        n = await queue_remaining(s, user_id)
    await bus.publish(user_id, "status", {"status": {"exec_info": {"queue_remaining": n}}})


class QueueFull(Exception):
    pass


async def submit(session: AsyncSession, user_id: str, *, prompt: dict, extra: dict, client_id: str | None,
                 targets: list[str] | None, run_id: str | None = None) -> Run:
    """Check the graph and queue it (commits); the run will speak the current language. Raises engine.PromptInvalid
    or QueueFull."""
    import uuid

    ch = await choices(session, user_id)
    engine.validate(prompt, ch, targets)
    try:  # checked before paying, so the last run may go over by its own files
        await files.check_room(session, user_id)
    except files.FileError as e:
        raise engine.PromptInvalid({"type": "quota_exceeded", "message": str(e), "details": "",
                                    "extra_info": {}}) from None
    if await queue_remaining(session, user_id) >= get_settings().queue_per_user:
        raise QueueFull
    run = Run(id=run_id or str(uuid.uuid4()), user_id=user_id, client_id=client_id, prompt=prompt,
              extra={**(extra or {}), "loom_lang": i18n.current()}, targets=targets or None)
    session.add(run)
    await session.commit()
    dispatcher.wake()
    await send_status(user_id)
    return run


async def cancel(session: AsyncSession, user_id: str, run_ids: list[str] | None = None) -> list[str]:
    """Cancel the user's runs (all running ones when no ids are given); returns the ids touched."""
    q = select(Run).where(Run.user_id == user_id)
    q = q.where(Run.id.in_(run_ids)) if run_ids else q.where(Run.status == RUNNING)
    touched = []
    for run in (await session.execute(q)).scalars():
        if run.status == PENDING:
            run.status, run.finished_at = "cancelled", utcnow()
            run.messages = [["execution_interrupted", {"prompt_id": run.id, "timestamp": engine.now_ms()}]]
            touched.append(run.id)
        elif run.status == RUNNING:
            run.status = CANCELLING
            touched.append(run.id)
    await session.commit()
    for rid in touched:
        dispatcher.cancel_local(rid)
    await send_status(user_id)
    return touched


# ---------------------------------------------------------------- the dispatcher


class Dispatcher:
    def __init__(self):
        self.tasks: dict[str, asyncio.Task] = {}
        self._wake: asyncio.Event | None = None  # created in the running loop
        self._loop_task: asyncio.Task | None = None
        self.stopping = False
        self.parallel: asyncio.Semaphore | None = None

    def wake(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def cancel_local(self, run_id: str) -> None:
        t = self.tasks.get(run_id)
        if t is not None:
            t.cancel()

    def start(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._wake = asyncio.Event()
            self.stopping = False
            self.parallel = asyncio.Semaphore(get_settings().generations_per_process)
            self._loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop for a deploy or a restart: no new runs or nodes; generations in flight get STOP_GRACE_S to
        finish (a picture paid for is kept, not paid for again), video tasks are let go and followed by the run's
        next process; the runs go back to the queue at once, so another node goes on with them."""
        self.stopping = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        tasks = list(self.tasks.values())
        if tasks:
            _, late = await asyncio.wait(tasks, timeout=get_settings().stop_grace_s)
            for t in late:  # a generation that did not finish in time: it is redone where the run resumes
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _loop(self) -> None:
        last_beat = 0.0
        loop = asyncio.get_running_loop()
        while True:
            try:
                await self.claim()
                await self.check_cancelled()
                if loop.time() - last_beat > HEARTBEAT_S:
                    last_beat = loop.time()
                    await self.heartbeat()
                    await self.recover()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  the database may be away for a moment
                log.exception("dispatcher")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=1.0)
            self._wake.clear()

    async def claim(self) -> list[str]:
        s_ = get_settings()
        room = s_.runs_per_process - len(self.tasks)
        if room <= 0:
            return []
        claimed = []
        async with sessionmaker()() as s:
            q = select(Run).where(Run.status == PENDING).order_by(Run.created_at).limit(room * 4)
            if not is_sqlite():
                q = q.with_for_update(skip_locked=True)
            pending = list((await s.execute(q)).scalars())
            busy: dict[str, int] = {}
            for run in pending:
                if run.user_id not in busy:
                    busy[run.user_id] = (await s.execute(select(func.count()).select_from(Run).where(
                        Run.user_id == run.user_id, Run.status.in_((RUNNING, CANCELLING))))).scalar_one()
                if busy[run.user_id] >= s_.runs_per_user or len(claimed) >= room:
                    continue
                busy[run.user_id] += 1
                run.status, run.worker = RUNNING, WORKER
                run.started_at = run.started_at or utcnow()
                run.heartbeat_at = utcnow()
                claimed.append(run.id)
            await s.commit()
        for rid in claimed:
            self.tasks[rid] = asyncio.create_task(self._run(rid))
        return claimed

    async def check_cancelled(self) -> None:
        if not self.tasks:
            return
        async with sessionmaker()() as s:
            ids = (await s.execute(select(Run.id).where(
                Run.id.in_(list(self.tasks)), Run.status == CANCELLING))).scalars()
            for rid in ids:
                self.cancel_local(rid)

    async def heartbeat(self) -> None:
        if not self.tasks:
            return
        async with sessionmaker()() as s:
            await s.execute(update(Run).where(Run.id.in_(list(self.tasks)), Run.worker == WORKER)
                            .values(heartbeat_at=utcnow()))
            await s.commit()

    async def recover(self) -> None:
        """Runs whose process went silent: back to the queue (or failed after MAX_ATTEMPTS)."""
        cutoff = utcnow() - timedelta(seconds=STALE_S)
        async with sessionmaker()() as s:
            q = select(Run).where(Run.status.in_((RUNNING, CANCELLING)), Run.heartbeat_at < cutoff)
            if not is_sqlite():
                q = q.with_for_update(skip_locked=True)
            for run in (await s.execute(q)).scalars():
                if run.id in self.tasks:
                    continue
                attempts = int((run.extra or {}).get("loom_attempts", 0)) + 1
                run.extra = {**(run.extra or {}), "loom_attempts": attempts}
                if run.status == CANCELLING:
                    run.status, run.finished_at = "cancelled", utcnow()
                elif attempts >= MAX_ATTEMPTS:
                    run.status, run.finished_at = "error", utcnow()
                    run.error = {"prompt_id": run.id, "exception_message": i18n.tr("Запуск прервался, попробуйте ещё раз",
                                                                       "The run was interrupted, try again"),
                                 "exception_type": i18n.tr("Ошибка", "Error"), "traceback": [], "node_id": None, "node_type": None,
                                 "executed": [], "timestamp": engine.now_ms()}
                    run.messages = [*(run.messages or []), ["execution_error", run.error]]
                else:
                    run.status, run.worker = PENDING, None
                    log.warning("run %s resumes (attempt %s)", run.id, attempts + 1)
            await s.commit()

    async def _run(self, run_id: str) -> None:
        try:
            await execute(run_id, self.parallel)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("run %s", run_id)
        finally:
            self.tasks.pop(run_id, None)
            self.wake()


dispatcher = Dispatcher()


# ---------------------------------------------------------------- one run


async def execute(run_id: str, parallel: asyncio.Semaphore | None = None) -> None:
    sf = sessionmaker()
    async with sf() as s:
        run = await s.get(Run, run_id)
        i18n.use((run.extra or {}).get("loom_lang"))  # the run's messages (node errors, progress) in its language
        order = engine.plan(run.prompt, run.targets)
        user_id, client_id, prompt = run.user_id, run.client_id, run.prompt
    history: list = []
    lock = asyncio.Lock()

    async def node_done(node_id: str, cost: int | None, paid: bool) -> None:
        """Kept with the run as each node finishes, so a run resumed by another process still knows what it made and
        what that cost (its own reused results count as its own)."""
        async with lock, sf() as s:
            row = await s.get(Run, run_id)
            extra = dict(row.extra or {})
            extra["loom_new"] = sorted({*extra.get("loom_new", []), node_id})
            if cost is not None:
                extra["loom_costs"] = {**extra.get("loom_costs", {}), node_id: cost}
            elif paid:
                extra["loom_unpriced"] = sorted({*extra.get("loom_unpriced", []), node_id})
            row.extra = extra
            await s.commit()

    async def send(event: str, data: dict) -> None:
        if event in ("execution_start", "execution_cached", "execution_success", "execution_error",
                     "execution_interrupted"):
            history.append([event, data])
        await bus.publish(user_id, event, data, sid=client_id)

    async def text(node_id: str, message: str) -> None:
        await bus.publish(user_id, "__text", {"node": node_id, "text": message}, sid=client_id)

    ex = engine.Execution(run_id=run_id, prompt=prompt, order=order, user_id=user_id, session_factory=sf,
                          events=engine.Events(send, text), parallel=parallel, suspend=lambda: dispatcher.stopping,
                          on_done=node_done)
    outcome = await ex.run()
    if outcome.status == "suspended" or (outcome.status == "cancelled" and dispatcher.stopping):
        # the process stops, not the user: back to the queue for another process (finished nodes are reused)
        async with sf() as s:
            await s.execute(update(Run).where(Run.id == run_id, Run.status == RUNNING)
                            .values(status=PENDING, worker=None))
            await s.commit()
        return
    async with sf() as s:
        run = await s.get(Run, run_id)
        run.status = outcome.status
        run.outputs = outcome.outputs
        run.error = outcome.error
        run.messages = [*(run.messages or []), *history]
        run.finished_at = utcnow()
        # the nodes that really ran (not reused), in this attempt and before a handover: what the run made, for the
        # results gallery, and what it cost
        extra = dict(run.extra or {})
        new = sorted({*extra.get("loom_new", []), *outcome.executed})
        costs = {**extra.get("loom_costs", {}), **outcome.costs}
        unpriced = sorted({*extra.get("loom_unpriced", []), *outcome.unpriced})
        run.extra = {**extra, "loom_new": new, "loom_costs": costs, "loom_unpriced": unpriced,
                     "loom_cost_micro": sum(costs.values())}
        await s.commit()
    total = sum(costs.values())
    # what the run cost, for the editor's notice (our event, shown by nodeloom.js); "unpriced": nodes whose provider and
    # config say no price, so the total is not the whole cost
    await bus.publish(user_id, "nodeloom.cost", {"prompt_id": run_id, "cost_micro": total, "text": engine.money(total),
                                             "nodes": costs, "paid": bool(costs), "unpriced": len(unpriced),
                                             "reused": len(ex.reused)}, sid=client_id)
    await bus.publish(user_id, "executing", {"node": None, "prompt_id": run_id}, sid=client_id)
    await send_status(user_id)


# ---------------------------------------------------------------- housekeeping

RESULTS_DAYS = 30  # the reuse of a node's result lasts this long; the files themselves stay with the user
RUNS_DAYS = 90  # the editor's queue history


async def cleanup() -> None:
    """Drop old reuse records and old finished runs (an hourly job of every process; deleting twice is harmless)."""
    from sqlalchemy import delete

    now = utcnow()
    async with sessionmaker()() as s:
        r1 = await s.execute(delete(NodeResult).where(NodeResult.created_at < now - timedelta(days=RESULTS_DAYS)))
        r2 = await s.execute(delete(Run).where(Run.status.in_(FINAL),
                                                     Run.created_at < now - timedelta(days=RUNS_DAYS)))
        await s.commit()
    if r1.rowcount or r2.rowcount:
        log.info("cleanup: %s results, %s runs", r1.rowcount, r2.rowcount)


def encode_text(node_id: str, text: str) -> bytes:
    """ComfyUI's binary progress text: event type, then the node id with its length, then the text."""
    nid = node_id.encode()
    return struct.pack(">I", TEXT_EVENT) + struct.pack(">I", len(nid)) + nid + text.encode()
