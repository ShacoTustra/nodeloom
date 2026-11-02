"""Checking and running a graph in ComfyUI's API format: {node id: {"class_type", "inputs": {name: value | [src, i]}}}.

Unlike ComfyUI, nodes run concurrently: each starts as soon as its inputs are ready, so independent branches (two
videos from two frames) generate at the same time. A node marked `cached` is skipped when the user already has its
result for the same inputs (its signature: the node, its settings and the signatures of what feeds it); a changed
prompt reruns only what depends on it.

When a node fails, nodes that have not started are dropped and the ones already running finish (their results are
kept for the next run, since they are paid for); then the run ends with the error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from nodeloom import files, i18n
from nodeloom.models import NodeResult
from nodeloom.nodes import BY_ID, Ctx, In, Node, NodeError, Suspended, available

log = logging.getLogger(__name__)

LITERAL_TYPES = {"STRING": str, "INT": int, "FLOAT": (int, float), "BOOLEAN": bool}


def now_ms() -> int:
    return int(time.time() * 1000)


def money(micro: int) -> str:
    """Dollars for the editor: 0,045 $; 1,20 $; small sums keep their digits (0,0023 $)."""
    v = micro / 1_000_000
    digits = 2 if v >= 1 else 3 if v >= 0.01 else 4
    text = f"{v:.{digits}f}"
    if digits > 2:
        text = text.rstrip("0")
        if len(text.split(".")[1]) < 2:
            text = f"{v:.2f}"
    return f"${text}" if i18n.current() == "en" else text.replace(".", ",") + " $"


# ---------------------------------------------------------------- validation


class PromptInvalid(Exception):
    """400 for /prompt, in ComfyUI's shape: {"error": {...}, "node_errors": {...}}."""

    def __init__(self, error: dict, node_errors: dict | None = None):
        super().__init__(error.get("message"))
        self.body = {"error": error, "node_errors": node_errors or {}}


def _err(type_: str, message: str, details: str = "", input_name: str | None = None) -> dict:
    extra = {"input_name": input_name} if input_name else {}
    return {"type": type_, "message": message, "details": details, "extra_info": extra}


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def grouped(node: Node, inputs: dict) -> dict:
    """Inputs by the node's input names; a growing input's slots ("references.reference 1", ...) become one list,
    and a missing setting takes its default."""
    out: dict = {}
    for name, v in inputs.items():
        base, dot, slot = name.partition(".")
        spec = node.input(base)
        if spec is None:
            continue
        if spec.grow:
            if dot:
                out.setdefault(base, []).append((slot, v))
        elif not dot:
            if spec.options_ru and v in spec.options_ru and isinstance(spec.options, list):
                v = spec.options[spec.options_ru.index(v)]  # a value chosen in Russian means the English one
            out[name] = v
    for i in node.inputs:
        if i.grow and i.name in out:
            out[i.name] = [v for _, v in sorted(out[i.name], key=lambda sv: _slot_index(sv[0]))]
        elif i.widget and out.get(i.name) is None and i.default is not None:
            out[i.name] = i.default  # a setting the graph does not have yet (saved before it was added)
    return out


def _slot_index(slot: str) -> int:
    digits = "".join(ch for ch in slot if ch.isdigit())
    return int(digits) if digits else 0


def _output_type(prompt: dict, link: list) -> str | None:
    src = prompt.get(link[0])
    node = BY_ID.get((src or {}).get("class_type"))
    if node is None or not (0 <= link[1] < len(node.outputs)):
        return None
    return node.outputs[link[1]].type


def _label(spec: In) -> str:
    return i18n.tr(spec.label_ru or spec.label, spec.label)


def _check_input(prompt: dict, spec: In, value, choices: dict[str, list]) -> dict | None:
    if _is_link(value):
        t = _output_type(prompt, value)
        if t is None:
            return _err("bad_linked_input", i18n.tr("Вход подключён к несуществующему выходу",
                                                    "The input is linked to an output that does not exist"),
                        _label(spec), spec.name)
        if spec.type != "*" and t != spec.type and not (spec.type == "COMBO" and t == "STRING"):
            return _err("return_type_mismatch", i18n.tr("Не тот тип данных на входе", "Wrong type of data on the input"),
                        f"{_label(spec)}: {t}", spec.name)
        return None
    if spec.type in ("IMAGE", "VIDEO", "AUDIO", "*"):
        return _err("bad_linked_input", i18n.tr("Этот вход подключается только связью", "This input takes only a link"),
                    _label(spec), spec.name)
    if spec.type == "COMBO":
        options = choices.get(spec.options, []) if isinstance(spec.options, str) else list(spec.options or [])
        if value not in options:
            return _err("value_not_in_list", i18n.tr("Такого значения нет в списке", "This value is not in the list"),
                        f"{_label(spec)}: {value}", spec.name)
        return None
    want = LITERAL_TYPES.get(spec.type)
    if want and (not isinstance(value, want) or (spec.type in ("INT", "FLOAT") and isinstance(value, bool))):
        return _err("invalid_input_type", i18n.tr("Неверное значение", "Invalid value"), f"{_label(spec)}: {value!r}",
                    spec.name)
    lo, hi = spec.extra.get("min"), spec.extra.get("max")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if lo is not None and value < lo:
            return _err("value_smaller_than_min", i18n.tr(f"Меньше допустимого ({lo})", f"Below the minimum ({lo})"),
                        _label(spec), spec.name)
        if hi is not None and value > hi:
            return _err("value_bigger_than_max", i18n.tr(f"Больше допустимого ({hi})", f"Above the maximum ({hi})"),
                        _label(spec), spec.name)
    return None


def plan(prompt: dict, targets: list[str] | None = None) -> list[str]:
    """The node ids to run: the output nodes (or the given targets) and everything they depend on."""
    outputs = [nid for nid, n in prompt.items() if BY_ID.get(n.get("class_type")) and BY_ID[n["class_type"]].output_node]
    if targets:
        outputs = [t for t in targets if t in prompt]
    need, stack = set(), list(outputs)
    while stack:
        nid = stack.pop()
        if nid in need or nid not in prompt:
            continue
        need.add(nid)
        stack.extend(v[0] for v in (prompt[nid].get("inputs") or {}).values() if _is_link(v))
    return [nid for nid in prompt if nid in need]


def validate(prompt: dict, choices: dict[str, list], targets: list[str] | None = None) -> list[str]:
    """Raise PromptInvalid like ComfyUI does, or return the nodes to run."""
    if not isinstance(prompt, dict) or not prompt:
        raise PromptInvalid(_err("invalid_prompt", i18n.tr("Пустой граф", "The graph is empty")))
    for nid, n in prompt.items():
        if not isinstance(n, dict) or n.get("class_type") not in BY_ID:
            ct = n.get("class_type") if isinstance(n, dict) else None
            raise PromptInvalid(_err("invalid_prompt", i18n.tr(f"Ноды {ct or nid} здесь нет",
                                                               f"There is no node {ct or nid} here"),
                                     f"Node ID '#{nid}'"))
        if not available(BY_ID[n["class_type"]], choices):
            node = BY_ID[n["class_type"]]
            raise PromptInvalid(_err("invalid_prompt", i18n.tr(
                f"Для ноды «{node.title_ru or node.title}» нет моделей: настройте провайдера",
                f"No models for the node “{node.title}”: configure a provider"), f"Node ID '#{nid}'"))
    order = plan(prompt, targets)
    if not order:
        raise PromptInvalid(_err("prompt_no_outputs", i18n.tr("Нечего запускать", "Nothing to run"),
                                 i18n.tr("Добавьте ноду, которая показывает результат (например, «Показать»)",
                                         "Add a node that shows a result (for example, “Show”)")))
    node_errors: dict = {}
    for nid in order:
        n = prompt[nid]
        node = BY_ID[n["class_type"]]
        given = grouped(node, n.get("inputs") or {})
        errors = []
        for spec in node.inputs:
            if spec.grow:
                for v in given.get(spec.name, []):
                    if (e := _check_input(prompt, spec, v, choices)) is not None:
                        errors.append(e)
                continue
            if spec.name not in given or given[spec.name] is None:
                if not spec.optional:
                    errors.append(_err("required_input_missing", i18n.tr("Не заполнен обязательный вход",
                                                                         "A required input is empty"),
                                       _label(spec), spec.name))
                continue
            if (e := _check_input(prompt, spec, given[spec.name], choices)) is not None:
                errors.append(e)
        if errors:
            node_errors[nid] = {"errors": errors, "dependent_outputs": [], "class_type": node.id}
    if node_errors:
        raise PromptInvalid(_err("prompt_outputs_failed_validation", i18n.tr("Граф не прошёл проверку",
                                                                             "The graph did not pass the check"),
                                 "; ".join(e["message"] for v in node_errors.values() for e in v["errors"])),
                            node_errors)
    return order


# ---------------------------------------------------------------- execution


@dataclass
class Events:
    """Where the engine reports; runs.py turns these into the editor's WebSocket messages and history."""

    send: Callable[[str, dict], Awaitable[None]]  # (event, data) to the run's browser tab
    text: Callable[[str, str], Awaitable[None]]  # (node id, progress text)


@dataclass
class Outcome:
    status: str  # success | error | cancelled | suspended (the process stops; the run goes back to the queue)
    outputs: dict = field(default_factory=dict)  # node id -> ui
    error: dict | None = None
    executed: list[str] = field(default_factory=list)
    costs: dict[str, int] = field(default_factory=dict)  # node id -> micro-dollars, nodes that paid a provider
    unpriced: list[str] = field(default_factory=list)  # nodes that paid a provider, at a cost not known

    @property
    def cost(self) -> int:
        return sum(self.costs.values())


def _sig(node: Node, values: dict, upstream: dict) -> str:
    payload = [node.id, node.version, {k: values[k] for k in sorted(values)}, upstream]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class Execution:
    def __init__(self, *, run_id: str, prompt: dict, order: list[str], user_id: str, session_factory, events: Events,
                 parallel: asyncio.Semaphore | None = None, suspend=lambda: False, on_done=None):
        self.run_id, self.prompt, self.order, self.user_id = run_id, prompt, order, user_id
        self.sf, self.events = session_factory, events
        self.parallel = parallel
        self.suspend = suspend
        self.suspended = False
        # async (node id, cost or None, paid): a node really ran (kept with the run at once)
        self.on_done = on_done
        self.results: dict[str, list] = {}
        self.sigs: dict[str, str] = {}
        self.outputs: dict[str, dict] = {}
        self.executed: list[str] = []
        self.states: dict[str, str] = {}
        self.costs: dict[str, int] = {}
        self.unpriced: list[str] = []
        self.reused: list[str] = []
        self.futures: dict[str, asyncio.Future] = {}
        self.failed: asyncio.Event = asyncio.Event()

    # ---- progress

    async def _state(self, nid: str, state: str) -> None:
        self.states[nid] = state
        nodes = {k: {"value": 1.0 if s == "finished" else 0.0, "max": 1.0, "state": s, "node_id": k,
                     "prompt_id": self.run_id, "display_node_id": k, "parent_node_id": None, "real_node_id": k}
                 for k, s in self.states.items()}
        await self.events.send("progress_state", {"prompt_id": self.run_id, "nodes": nodes})

    # ---- signatures and cache

    async def _signature(self, nid: str, node: Node, given: dict) -> str:
        values, upstream = {}, {}
        for name, v in given.items():
            if node.input(name) and node.input(name).grow:
                upstream[name] = [[self.sigs[x[0]], x[1]] if _is_link(x) else x for x in v]
            elif _is_link(v):
                upstream[name] = [self.sigs[v[0]], v[1]]
            else:
                values[name] = v
                spec = node.input(name)
                if spec and isinstance(spec.options, str) and spec.options.startswith("files:"):
                    async with self.sf() as s:  # a re-uploaded file with the same name is a different input
                        f = await files.get(s, self.user_id, "input", v)
                    values[name + "#file"] = [f.id, f.size] if f else None
        return _sig(node, values, upstream)

    async def _cached(self, sig: str) -> NodeResult | None:
        async with self.sf() as s:
            return (await s.execute(select(NodeResult).where(
                NodeResult.user_id == self.user_id, NodeResult.sig == sig))).scalar_one_or_none()

    async def _store(self, sig: str, node: Node, *, value=None, ui=None, task_id=None) -> None:
        async with self.sf() as s:
            row = (await s.execute(select(NodeResult).where(
                NodeResult.user_id == self.user_id, NodeResult.sig == sig))).scalar_one_or_none()
            if row is None:
                row = NodeResult(user_id=self.user_id, sig=sig, node_type=node.id)
                s.add(row)
            if task_id is not None:
                row.task_id = task_id
            if value is not None:
                row.value, row.ui = value, ui
            try:
                await s.commit()
            except IntegrityError:  # the same result stored by a concurrent run
                await s.rollback()

    # ---- running

    def _resolve(self, v):
        if _is_link(v):
            return self.results[v[0]][v[1]]
        return v

    async def _node(self, nid: str) -> None:
        n = self.prompt[nid]
        node = BY_ID[n["class_type"]]
        given = grouped(node, n.get("inputs") or {})
        deps = {v[0] for v in (n.get("inputs") or {}).values() if _is_link(v)}
        await asyncio.gather(*(self.futures[d] for d in deps if d in self.futures))
        if self.failed.is_set():
            raise asyncio.CancelledError
        if self.suspend():
            raise Suspended  # no new work while the process stops; what's done is kept for the resumed run
        sig = await self._signature(nid, node, given)
        self.sigs[nid] = sig
        row = await self._cached(sig) if node.cached else None
        if row is not None and row.value is not None:
            async with self.sf() as s:  # its files may have been deleted with their run since
                if not await files.all_exist(s, self.user_id, files.refs_in(row.value)):
                    await s.delete(await s.merge(row))
                    await s.commit()
                    row = None
        if row is not None and row.value is not None:
            self.results[nid] = row.value
            if row.ui:
                self.outputs[nid] = row.ui
                await self.events.send("executed", {"node": nid, "display_node": nid, "output": row.ui,
                                                    "prompt_id": self.run_id})
            self.reused.append(nid)
            await self.events.text(nid, i18n.tr("Готово раньше, повтор бесплатно", "Done before, the repeat is free"))
            await self._state(nid, "finished")
            return
        kwargs = {}
        for name, v in given.items():
            spec = node.input(name)
            kwargs[name] = [self._resolve(x) for x in v] if spec and spec.grow else self._resolve(v)

        async def progress(text: str) -> None:
            await self.events.text(nid, text)

        async def remember(task_id: str) -> None:
            await self._store(sig, node, task_id=task_id)

        ctx = Ctx(user_id=self.user_id, node_id=nid, session_factory=self.sf, progress=progress,
                  pending_task=row.task_id if row is not None else None, remember_task=remember,
                  run_id=self.run_id, suspending=self.suspend)
        await self._state(nid, "running")
        await self.events.send("executing", {"node": nid, "display_node": nid, "prompt_id": self.run_id})
        try:
            if self.parallel is not None and node.cached:
                async with self.parallel:
                    value, ui = await node.run(ctx, **kwargs)
            else:
                value, ui = await node.run(ctx, **kwargs)
        finally:
            if ctx.paid and ctx.cost_known:
                self.costs[nid] = ctx.cost
            elif ctx.paid:
                self.unpriced.append(nid)
        self.results[nid] = value
        if node.cached:
            await self._store(sig, node, value=value, ui=ui)
        self.executed.append(nid)
        if self.on_done is not None:
            await self.on_done(nid, self.costs.get(nid), ctx.paid)
        if ui:
            self.outputs[nid] = ui
            await self.events.send("executed", {"node": nid, "display_node": nid, "output": ui,
                                                "prompt_id": self.run_id})
        if nid in self.costs:
            await self.events.text(nid, i18n.tr("Готово", "Done") + f" · {money(self.costs[nid])}")
        await self._state(nid, "finished")

    async def run(self) -> Outcome:
        await self.events.send("execution_start", {"prompt_id": self.run_id, "timestamp": now_ms()})
        await self.events.send("execution_cached", {"nodes": [], "prompt_id": self.run_id, "timestamp": now_ms()})
        first_error: dict | None = None
        loop = asyncio.get_running_loop()
        tasks: dict[str, asyncio.Task] = {}
        for nid in self.order:
            self.futures[nid] = loop.create_future()

        async def one(nid: str) -> None:
            nonlocal first_error
            fut = self.futures[nid]
            try:
                await self._node(nid)
                fut.set_result(None)
            except asyncio.CancelledError:
                fut.cancel()
                raise
            except Suspended:
                self.suspended = True
                for other, t in tasks.items():
                    if other != nid and self.states.get(other) is None:
                        t.cancel()
                fut.cancel()
            except Exception as e:  # noqa: BLE001  any node failure ends the run with its message
                if first_error is None:
                    first_error = self._error(nid, e)
                    self.failed.set()
                    for other, t in tasks.items():  # drop what hasn't started; running generations finish
                        if other != nid and self.states.get(other) is None:
                            t.cancel()
                fut.cancel()

        for nid in self.order:
            tasks[nid] = asyncio.create_task(one(nid))
        try:
            await asyncio.gather(*tasks.values(), return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks.values():
                t.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            running = next((k for k, s in self.states.items() if s == "running"), None)
            data = {"prompt_id": self.run_id, "node_id": running,
                    "node_type": self.prompt.get(running, {}).get("class_type") if running else None,
                    "executed": self.executed, "timestamp": now_ms()}
            await self.events.send("execution_interrupted", data)
            return Outcome("cancelled", self.outputs, data, self.executed, self.costs, self.unpriced)
        if self.suspended and first_error is None:
            return Outcome("suspended", self.outputs, None, self.executed, self.costs, self.unpriced)
        if first_error is not None:
            await self.events.send("execution_error", first_error)
            return Outcome("error", self.outputs, first_error, self.executed, self.costs, self.unpriced)
        await self.events.send("execution_success", {"prompt_id": self.run_id, "timestamp": now_ms()})
        return Outcome("success", self.outputs, None, self.executed, self.costs, self.unpriced)

    def _error(self, nid: str, e: Exception) -> dict:
        user_facing = isinstance(e, (NodeError, files.FileError))
        if not user_facing:
            log.exception("run %s node %s", self.run_id, nid)
        return {"prompt_id": self.run_id, "node_id": nid, "node_type": self.prompt[nid]["class_type"],
                "executed": self.executed,
                "exception_message": str(e) if user_facing else i18n.tr("Внутренняя ошибка", "Internal error"),
                "exception_type": i18n.tr("Ошибка", "Error") if user_facing else type(e).__name__,
                "traceback": [] if user_facing else traceback.format_exception_only(e),
                "current_inputs": {}, "current_outputs": [], "timestamp": now_ms()}
