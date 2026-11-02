// Our extension of the stock ComfyUI frontend: the parts of the UI left out, a starter graph of our nodes, text results
// on the nodes, the character picker, the Builder's panel, the language switch and our name and icon.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BASE = new URL(".", import.meta.url);
// The editor's language: the server puts it into <html lang> and the editor's own locale (Comfy.Locale). Our texts
// are written in both: tr("Русский", "English").
const RU = document.documentElement.lang === "ru";
const tr = (ru, en) => (RU ? ru : en);
const TITLE = document.querySelector('meta[name="application-name"]')?.content || "Nodeloom";
// Sidebar tabs that make no sense here: local models, app mode.
const HIDDEN_TABS = ["model-library", "apps"];
// Commands that lead to parts of the stock UI left out here: settings (Comfy account, server options), Comfy
// sign-in, help centre and links, extension/node managers. They become no-ops, so keyboard shortcuts and menu
// entries do nothing; the menu entries themselves are hidden in nodeloom.css.
const DISABLED_COMMANDS = [
  /^Comfy\.ShowSettingsDialog$/, /^Comfy\.User\./, /^Comfy\.Help\./, /^Comfy\.ToggleHelpCenter$/,
  /^Comfy\.OpenManagerDialog$/, /^Comfy\.Manager\./, /^Comfy\.BrowseModelAssets$/, /^Comfy\.Dev\./,
  /^Workspace\.ToggleSidebarTab\.(model-library|apps)$/,
  // parts left out (hidden in nodeloom.css): bypass, pinning, subgraphs, app mode, the bottom panel
  // (server logs, shortcuts), canvas performance info, local memory, the clipspace, API export, Nodes 2.0 switch
  /^Comfy\.Canvas\.ToggleSelected(Nodes)?\.(Bypass|Pin)$/, /^Comfy\.Graph\.(ConvertToSubgraph|UnpackSubgraph|EditSubgraphWidgets)$/,
  /^Comfy\.(PublishSubgraph|Subgraph\..*)$/, /^Comfy\.ToggleLinear$/, /^Workspace\.ToggleBottomPanel/,
  /^Comfy\.ToggleCanvasInfo$/, /^Comfy\.Memory\./, /^Comfy\.OpenClipspace$/, /^Comfy\.ExportWorkflowAPI$/,
  /^Experimental\.ToggleVueNodes$/,
];
// Nodes of the library besides ours: only the note (handy for comments), under "Text".
const KEEP_FRONTEND_NODES = new Set(["Note"]);
// Canvas menu entries for ComfyUI's node templates (group nodes): not used here.
const CANVAS_MENU_OUT = new Set(["Save Selected as Template", "Node Templates", "Сохранить выбранное как шаблон",
  "Шаблоны узлов"]);
// The growing inputs ("reference 1", "image 2", ...): the server names their slots in English and the editor's
// translations do not reach them, so in Russian they get a label. Their names stay: graphs saved in either language
// keep their links.
const GROW_RU = { reference: "референс", image: "картинка", video: "видео", text: "текст", audio: "аудио" };

function labelSlots(node) {
  for (const input of node.inputs ?? []) {
    const m = /^[^.]+\.(\S+) (\d+)$/.exec(input.name ?? "");
    if (!m || !GROW_RU[m[1]]) continue;
    const ru = `${GROW_RU[m[1]]} ${m[2]}`;
    if (RU) input.label = ru;
    else if (input.label === ru) delete input.label; // a graph saved in Russian, opened in English
  }
}

function setupSlotLabels(nodeType) {
  const created = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    created?.apply(this, []);
    labelSlots(this);
  };
  const configured = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (info) {
    configured?.apply(this, [info]);
    labelSlots(this);
  };
  const added = nodeType.prototype.onInputAdded;
  nodeType.prototype.onInputAdded = function (input) {
    added?.apply(this, [input]);
    labelSlots(this);
  };
}

function disableCommands() {
  for (const cmd of app.extensionManager.command?.commands ?? []) {
    if (DISABLED_COMMANDS.some((re) => re.test(cmd.id))) cmd.function = () => {};
  }
}

function addStylesheet() {
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("nodeloom.css", BASE).href;
  document.head.appendChild(link);
}

function setIdentity() {
  // Our icon, and it stays: while a prompt runs the frontend swaps in its own progress frames, from an
  // absolute /assets path that would not exist if the editor were served under a prefix.
  const ours = new URL("favicon.ico", BASE).href;
  const pin = () => {
    for (const el of document.querySelectorAll("link[rel~='icon']")) if (el.href !== ours) el.href = ours;
  };
  if (!document.querySelector("link[rel~='icon']")) {
    const icon = document.createElement("link");
    icon.rel = "icon";
    document.head.appendChild(icon);
  }
  pin();
  new MutationObserver(pin).observe(document.head, { subtree: true, childList: true, attributes: true, attributeFilter: ["href"] });
  // The frontend keeps rewriting the title ("[progress] workflow - ComfyUI"); swap the product name (our own name
  // may contain the word, so what already reads TITLE stays).
  const title = document.querySelector("title");
  const fix = () => {
    const t = document.title.split(TITLE).map((part) => part.replace(/ComfyUI/g, TITLE)).join(TITLE);
    if (t !== document.title) document.title = t;
  };
  new MutationObserver(fix).observe(title, { childList: true, characterData: true, subtree: true });
  fix();
}

// The ComfyUI mark in the top-left menu button becomes ours.
function swapLogo() {
  const place = () => {
    for (const svg of document.querySelectorAll("svg.comfyui-logo")) {
      const img = document.createElement("img");
      img.src = new URL("mark.svg", BASE).href;
      img.alt = TITLE;
      img.className = "loom-mark " + svg.getAttribute("class").replace("comfyui-logo", "");
      svg.replaceWith(img);
    }
  };
  new MutationObserver(place).observe(document.body, { childList: true, subtree: true });
  place();
}

// Nodes whose result is text show it on the node, the way the stock "Preview as Text" does.
const TEXT_NODES = new Set(["LoomTextModel", "LoomJoinText", "LoomShow"]);
const preview = () => globalThis.comfyAPI?.textPreviewWidgets;

// Entries of the workflow menu (next to the tab) that carry no aria-label for nodeloom.css to match.
const WORKFLOW_MENU_OUT = new Set(["Добавить в закладки", "Удалить из закладок", "Экспорт (API)",
  "Войти в режим приложения", "Войти в режим конструктора приложения",
  "Add to Bookmarks", "Remove from Bookmarks", "Export (API)", "Enter app mode", "Build app"]);

function trimWorkflowMenu() {
  const hide = (root) => {
    for (const item of root.querySelectorAll?.('div[role="menuitem"]') ?? []) {
      if (WORKFLOW_MENU_OUT.has(item.textContent.trim())) item.style.display = "none";
    }
  };
  new MutationObserver((records) => {
    for (const r of records) for (const n of r.addedNodes) if (n.nodeType === 1) hide(n);
  }).observe(document.body, { childList: true, subtree: true });
}

// What a run cost (the backend's "nodeloom.cost" event when it ends): a notice like ComfyUI's own ones. A model without
// a price (the provider says none and the config gives none) makes the total a lower bound, or unknown.
function showRunCost() {
  api.addEventListener("nodeloom.cost", ({ detail }) => {
    let text;
    if (detail.paid && detail.unpriced) {
      text = tr(`Запуск стоил не меньше ${detail.text}: у части моделей не задана цена`,
        `The run cost at least ${detail.text}: some models have no price set`);
    } else if (detail.paid) {
      text = tr(`Запуск стоил ${detail.text}`, `The run cost ${detail.text}`);
    } else if (!detail.unpriced && detail.reused) {
      text = tr("Запуск ничего не стоил: всё взято из прошлых результатов",
        "The run cost nothing: everything came from earlier results");
    }
    if (text) app.extensionManager.toast?.add({ severity: "info", summary: TITLE, detail: text, life: 6000 });
  });
}

// No provider, or no models from the ones there are: say so once, the node library would just look empty.
async function checkProviders() {
  try {
    const { providers } = await (await fetch(api.apiURL("/nodeloom/status"))).json();
    let detail = null;
    if (!providers.length) {
      detail = tr("Провайдеры моделей не настроены: задайте OPENAI_API_KEY, OPENROUTER_API_KEY или ARK_API_KEY " +
        "(или config.toml) и перезапустите. Пока доступны только загрузки и инструменты.",
      "No model providers are configured: set OPENAI_API_KEY, OPENROUTER_API_KEY or ARK_API_KEY (or a config.toml) " +
        "and restart. Until then only uploads and tools are available.");
    } else if (providers.every((p) => !p.models && !p.nodes)) {
      detail = tr(`Провайдеры (${providers.map((p) => p.name).join(", ")}) не вернули ни одной модели: проверьте ключи`,
        `The providers (${providers.map((p) => p.name).join(", ")}) returned no models: check the keys`);
    }
    if (detail) app.extensionManager.toast?.add({ severity: "warn", summary: TITLE, detail });
  } catch {
    // the editor works without it
  }
}

// The language of the editor and of what the server says (node labels, messages): a menu entry for each, saved as the
// editor's own Comfy.Locale; the page reloads, since the server's strings come with the page and the node list.
async function setLanguage(lang) {
  await app.extensionManager.setting.set("Comfy.Locale", lang);
  location.reload();
}

function pinia() {
  return document.querySelector("#vue-app")?.__vue_app__?.config.globalProperties.$pinia;
}

// The node library: our nodes and the note; the note goes to "Text" instead of "utilities".
function filterNodeLibrary() {
  const store = pinia()?._s.get("nodeDef");
  if (!store?.registerNodeDefFilter) return;
  store.registerNodeDefFilter({
    id: "nodeloom.nodes", name: TITLE, description: tr("Только ноды этого редактора", "This editor's nodes only"),
    predicate: (def) => def.name.startsWith("Loom") || KEEP_FRONTEND_NODES.has(def.name),
  });
  const note = store.nodeDefsByName?.Note;
  // the editor keeps our nodes' category already translated (nodeCategories of /api/i18n): the note joins it
  if (note) note.category = tr("Текст", "Text");
}

// Settings whose switches are hidden: users who opened the editor before keep them in their saved settings.
const FIXED_SETTINGS = { "Comfy.Graph.CanvasInfo": false, "Comfy.VueNodes.Enabled": true };

async function fixSettings() {
  const settings = app.extensionManager.setting;
  for (const [id, value] of Object.entries(FIXED_SETTINGS)) {
    if (settings?.get(id) !== value) await settings?.set(id, value);
  }
}

// "Run (on change)" / "Run (instant)" are hidden; a mode chosen before still applies, so back to plain runs.
function plainRuns() {
  const queue = pinia()?._s.get("queueSettingsStore");
  if (queue && "mode" in queue) queue.mode = "disabled";
}

function trimCanvasMenu() {
  const LGraphCanvas = globalThis.LGraphCanvas;
  const original = LGraphCanvas?.prototype.getCanvasMenuOptions;
  if (!original) return;
  LGraphCanvas.prototype.getCanvasMenuOptions = function () {
    const items = original.apply(this, arguments).filter((o) => !(o && CANVAS_MENU_OUT.has(o.content)));
    while (items.length && items.at(-1) === null) items.pop();
    return items;
  };
}

// The stock default graph needs local models. The frontend exports it as
// comfyAPI.defaultGraph.defaultGraph and loads that same object on first start, for "load default
// workflow" and when the last tab is closed; it is rewritten in place with the starter graph.
async function replaceDefaultGraph() {
  const exported = globalThis.comfyAPI?.defaultGraph;
  const res = await fetch(new URL("starter.json", BASE));
  if (!exported?.defaultGraph || !res.ok) return;
  const starter = await res.json();
  for (const k of Object.keys(exported.defaultGraph)) delete exported.defaultGraph[k];
  Object.assign(exported.defaultGraph, starter);
  exported.defaultGraphJSON = JSON.stringify(starter);
}

// ---------------------------------------------------------------- "Seedance character": a ready character
// The node keeps the character's asset id in its text widget; the button opens a picker over the library of the first
// provider that lists one (the server's /api/nodeloom/assets) and the node shows the character's preview.
const PRESET_AGES = [["", tr("Любой возраст", "Any age")], ["18-25", "18–25"], ["26-35", "26–35"], ["36-50", "36–50"], ["51-65", "51–65"], ["66-120", "66+"]];
const presetPreview = (id) => api.apiURL(`/view?type=asset&filename=${encodeURIComponent(id)}`);
const presetTitle = (p) => p.title || [p.gender === "female" ? tr("Женщина", "Woman") : tr("Мужчина", "Man"), p.age].filter(Boolean).join(", ");
const presetSub = (p) => p.subtitle || [p.profession, p.country].filter(Boolean).join(", ");

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

// The preview is a DOM widget of the node: the editor (Vue nodes) draws output images from its own store only.
function showCharacter(node, id, title) {
  if (title) node.title = tr(`Персонаж: ${title}`, `Character: ${title}`);
  const box = node.loomPreview;
  if (!box) return;
  const empty = (text) => el("div", { class: "loom-pr-node-empty" }, text);
  // no library to show it from (a provider without /assets): the id still works in the video
  const noPreview = () => box.replaceChildren(empty(tr("Превью нет", "No preview")));
  box.replaceChildren(id
    ? el("img", { src: presetPreview(id), alt: title || "", class: "loom-pr-node-img", onerror: noPreview })
    : empty(tr("Персонаж не выбран", "No character selected")));
  app.graph?.setDirtyCanvas(true, true);
}

function openPresetPicker(node, widget) {
  const state = { gender: "", age: "", q: "", items: [], total: 0, seq: 0 };
  const grid = el("div", { class: "loom-pr-grid" });
  const foot = el("div", { class: "loom-pr-foot" });
  const close = () => { overlay.remove(); document.removeEventListener("keydown", onKey, true); };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  const query = (offset) => {
    const p = new URLSearchParams({ offset: String(offset), limit: "60" });
    if (state.gender) p.set("gender", state.gender);
    if (state.age) { const [a, b] = state.age.split("-"); p.set("age_min", a); p.set("age_max", b); }
    if (state.q) p.set("q", state.q);
    return p;
  };
  const card = (p) => el("button", {
    type: "button", class: "loom-pr-card", title: `${presetTitle(p)}. ${presetSub(p)}`,
    onclick: () => {
      widget.value = p.id;
      widget.callback?.(p.id);
      showCharacter(node, p.id, `${presetTitle(p)}, ${presetSub(p)}`);
      close();
    },
  }, el("img", { src: presetPreview(p.id), loading: "lazy", alt: "" }),
     el("span", { class: "loom-pr-title" }, presetTitle(p)), el("span", { class: "loom-pr-sub" }, presetSub(p)));
  const renderFoot = () => {
    foot.replaceChildren(el("span", {}, state.total
      ? tr(`Показано ${state.items.length} из ${state.total}`, `Showing ${state.items.length} of ${state.total}`)
      : tr("Никого не нашлось", "No one found")));
    if (state.items.length < state.total) foot.append(el("button", { type: "button", class: "loom-pr-more", onclick: () => load(true) }, tr("Показать ещё", "Show more")));
  };
  const load = async (more = false) => {
    const n = ++state.seq;
    if (!more) { state.items = []; grid.replaceChildren(el("div", { class: "loom-pr-note" }, tr("Загрузка…", "Loading…"))); }
    try {
      const r = await (await fetch(api.apiURL(`/nodeloom/assets?${query(more ? state.items.length : 0)}`), { credentials: "same-origin" })).json();
      if (n !== state.seq) return;
      if (!r.available) {
        grid.replaceChildren(el("div", { class: "loom-pr-note" }, tr(
          "Ни один провайдер не даёт библиотеку персонажей: впишите id ассета в ноду вручную.",
          "No provider lists ready characters: type an asset id into the node.")));
        foot.replaceChildren();
        return;
      }
      if (!more) grid.replaceChildren();
      state.items.push(...r.items);
      state.total = r.total;
      grid.append(...r.items.map(card));
      renderFoot();
    } catch {
      grid.replaceChildren(el("div", { class: "loom-pr-note" }, tr("Не удалось загрузить персонажей", "Could not load the characters")));
    }
  };
  const chip = (value, label) => el("button", {
    type: "button", class: `loom-pr-chip${state.gender === value ? " is-on" : ""}`, "data-g": value,
    onclick: (e) => {
      state.gender = value;
      for (const c of e.target.parentNode.querySelectorAll(".loom-pr-chip")) c.classList.toggle("is-on", c.dataset.g === value);
      load();
    },
  }, label);
  const ages = el("select", { class: "loom-pr-select", "aria-label": tr("Возраст", "Age"), onchange: (e) => { state.age = e.target.value; load(); } },
    ...PRESET_AGES.map(([v, l]) => el("option", { value: v }, l)));
  let timer;
  const search = el("input", { class: "loom-pr-search", placeholder: tr("Профессия или страна", "Profession or country"), "aria-label": tr("Поиск", "Search"),
    oninput: (e) => { clearTimeout(timer); timer = setTimeout(() => { state.q = e.target.value.trim(); load(); }, 300); } });
  const overlay = el("div", { class: "loom-pr-overlay", onclick: (e) => { if (e.target === overlay) close(); } },
    el("div", { class: "loom-pr-dialog", role: "dialog", "aria-modal": "true", "aria-label": tr("Персонажи", "Characters") },
      el("div", { class: "loom-pr-head" },
        el("div", {}, el("h2", {}, tr("Персонажи", "Characters")),
          el("p", {}, tr("Готовые персонажи для видео Seedance: лицо и образ сохраняются во всех роликах. Подключите ноду к первому кадру или референсу ноды «Видео Seedance».",
            "Ready-made characters for Seedance video: the face and look stay the same in every clip. Connect the node to the first frame or a reference of “Seedance video”."))),
        el("button", { type: "button", class: "loom-pr-close", "aria-label": tr("Закрыть", "Close"), onclick: close }, "×")),
      el("div", { class: "loom-pr-filters" }, chip("", tr("Все", "All")), chip("female", tr("Женщины", "Women")), chip("male", tr("Мужчины", "Men")), ages, search),
      el("div", { class: "loom-pr-scroll" }, grid, foot)));
  document.body.append(overlay);
  document.addEventListener("keydown", onKey, true);
  search.focus();
  load();
}

// "Show" plays an audio result in the editor's audio player (audioUI), which keeps its 70px while it has
// nothing to play: a picture or a text would sit under an empty band. Empty, it takes no room; when a sound comes,
// the node grows to fit it.
function setupShowNode(nodeType) {
  const created = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    created?.apply(this, []);
    const player = this.widgets?.find((w) => w.name === "audioUI");
    if (!player?.computeLayoutSize) return;
    const size = player.computeLayoutSize.bind(player);
    player.computeLayoutSize = (node) => player.element?.classList.contains("empty-audio-widget")
      ? { minHeight: 0, maxHeight: 0, minWidth: 0 } : size(node);
  };
  const executed = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (output) {
    executed?.apply(this, [output]);
    if (!output?.audio?.length) return;
    requestAnimationFrame(() => {
      const h = this.computeSize?.()[1] || 0;
      if (h > this.size[1]) this.setSize([this.size[0], h]);
      app.graph?.setDirtyCanvas(true, true);
    });
  };
}

// "Audio file": the editor puts the chosen file into its player while the node is being created, before it is on
// the canvas, and that value is lost: a new node showed no player until another file was picked. Once the node is
// on the canvas, the player is set again the way the editor does it when the file changes.
function setupLoadAudioNode(nodeType) {
  const added = nodeType.prototype.onAdded;
  nodeType.prototype.onAdded = function (graph) {
    added?.apply(this, [graph]);
    requestAnimationFrame(() => this.widgets?.find((w) => w.name === "audio")?.callback?.());
  };
}

function setupCharacterNode(nodeType) {
  const created = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    created?.apply(this, []);
    const widget = this.widgets?.find((w) => w.name === "character");
    // the name stays (saved graphs keep widget values by name); the label is in the editor's language
    const pick = this.addWidget("button", "Pick a character", null, () => openPresetPicker(this, widget));
    if (pick) pick.label = tr("Выбрать персонажа", "Pick a character");
    this.loomPreview = el("div", { class: "loom-pr-node" });
    this.addDOMWidget?.("preview", "loom-preview", this.loomPreview,
      { serialize: false, getMinHeight: () => 240, getMaxHeight: () => 240 });
    showCharacter(this, widget?.value);
    this.setSize?.([Math.max(this.size?.[0] || 0, 260), this.computeSize?.()[1] || 330]);
  };
  const configured = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (info) {
    configured?.apply(this, [info]);
    const id = this.widgets?.find((w) => w.name === "character")?.value;
    if (id) showCharacter(this, id);
  };
}

// ---------------------------------------------------------------- "Uploads": the user's uploaded files
// The editor's assets panel lists uploads but deletes them only in Comfy's cloud: this tab shows them with their size
// and the room they take, deletes them (/api/nodeloom/uploads) and refreshes the file lists of the nodes.
function formatSize(n) {
  const units = tr(["Б", "КБ", "МБ", "ГБ"], ["B", "KB", "MB", "GB"]);
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${i ? n.toFixed(n < 10 ? 1 : 0) : n} ${units[i]}`;
}

function uploadThumb(item) {
  const src = api.apiURL(`/view?type=input&filename=${encodeURIComponent(item.name)}`);
  if (item.media === "image") return el("img", { class: "loom-up-thumb", src, loading: "lazy", alt: "" });
  if (item.media === "video") return el("video", { class: "loom-up-thumb", src, preload: "metadata", muted: "" });
  return el("div", { class: "loom-up-thumb loom-up-icon" }, el("i", { class: "pi pi-volume-up" }));
}

async function renderUploads(root) {
  const box = el("div", { class: "loom-up" });
  root.replaceChildren(box);
  let data;
  try {
    data = await (await fetch(api.apiURL("/nodeloom/uploads"))).json();
  } catch {
    box.replaceChildren(el("div", { class: "loom-up-note" }, tr("Не удалось загрузить список", "Could not load the list")));
    return;
  }
  const used = data.limit
    ? tr(`${formatSize(data.used)} из ${formatSize(data.limit)}`, `${formatSize(data.used)} of ${formatSize(data.limit)}`)
    : tr(`Занято ${formatSize(data.used)}`, `${formatSize(data.used)} used`);
  const remove = async (item) => {
    if (!confirm(tr(`Удалить «${item.name}»? Результаты, сделанные из него, останутся.`,
      `Delete “${item.name}”? The results made from it stay.`))) return;
    const r = await fetch(api.apiURL(`/nodeloom/uploads/${encodeURIComponent(item.name)}`), { method: "DELETE" });
    if (!r.ok) app.extensionManager.toast?.add({ severity: "error", summary: TITLE, detail: tr("Файл не удалён", "The file was not deleted"), life: 5000 });
    await renderUploads(root);
    await app.refreshComboInNodes?.();
  };
  const rows = data.items.map((item) => el("div", { class: "loom-up-item" }, uploadThumb(item),
    el("div", { class: "loom-up-meta" }, el("div", { class: "loom-up-name", title: item.name }, item.name),
      el("div", { class: "loom-up-size" }, formatSize(item.size))),
    el("button", { type: "button", class: "loom-up-delete", title: tr("Удалить", "Delete"), "aria-label": tr("Удалить", "Delete"),
      onclick: () => remove(item) }, el("i", { class: "pi pi-trash" }))));
  box.replaceChildren(
    el("div", { class: "loom-up-head" }, el("span", { class: "loom-up-title" }, tr("Загрузки", "Uploads")),
      el("span", { class: "loom-up-used" }, used)),
    el("div", { class: "loom-up-list" }, ...(rows.length ? rows : [el("div", { class: "loom-up-note" },
      tr("Загрузок нет: перетащите файл на холст или выберите его в ноде «Картинка», «Видео-файл» или «Аудио-файл».",
        "No uploads yet: drop a file on the canvas or pick one in an Image, Video file or Audio file node."))])));
}

function setupUploads() {
  app.extensionManager.registerSidebarTab?.({
    id: "loom-uploads", icon: "pi pi-upload", title: tr("Загрузки", "Uploads"), label: tr("Загрузки", "Uploads"),
    tooltip: tr("Загруженные файлы: сколько места и удаление", "Uploaded files: the room they take, deleting"),
    type: "custom", render: (root) => { renderUploads(root); },
  });
}

// ---------------------------------------------------------------- errors in plain words
// The stock editor shows a run's error as a generic phrase, a "ComfyUI Error Report" (versions, arguments, the
// workflow JSON) and links to Comfy's GitHub and support. Here the generic phrases become the node's own message,
// the report and the links go away.
const GENERIC_ERROR = [/^Этот узел вызвал ошибку во время выполнения/, /^Узел вызвал ошибку во время выполнения/,
  /^Node threw an error/i, /^This node threw an error during execution/i];
const COMFY_HELP = /^(Получить помощь|Найти на GitHub|Найти проблемы|создайте issue на GitHub|свяжитесь с нашей поддержкой|Get Help|Find Issues|Find on GitHub|submit a GitHub issue|contact our support)$/i;
let lastError = null;

// where the editor shows errors: the properties panel, the run's error overlay, dialogs, toasts
const ERROR_PLACES = '[data-testid="properties-panel"], [data-testid="error-overlay"], [role="dialog"], [role="alertdialog"], .p-toast, [role="alert"]';

function cleanErrors() {
  for (const root of document.querySelectorAll(ERROR_PLACES)) {
    for (const e of root.querySelectorAll('[data-testid="runtime-error-panel"], [data-testid="error-card-copy"]')) {
      e.style.display = "none";
    }
    for (const e of root.querySelectorAll("p, span, div, pre, textarea")) {
      if (e.children.length) continue;
      const t = (e.value ?? e.textContent ?? "").trim();
      if (lastError && GENERIC_ERROR.some((r) => r.test(t))) e.textContent = lastError;
      else if (t.startsWith("# ComfyUI Error Report")) e.style.display = "none";
    }
    for (const b of root.querySelectorAll("button, a")) {
      if (COMFY_HELP.test(b.textContent.trim())) b.style.display = "none";
    }
    // "Для получения дополнительной помощи создайте issue на GitHub или свяжитесь с нашей поддержкой" ("For more help, …")
    for (const e of root.querySelectorAll("p, div")) {
      if (e.childElementCount <= 4 && /^(Для получения дополнительной помощи|For more help)/.test(e.textContent.trim())) e.style.display = "none";
    }
  }
}

function plainErrors() {
  api.addEventListener("execution_error", (e) => { lastError = e.detail?.exception_message || null; });
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; cleanErrors(); });
  }).observe(document.body, { childList: true, subtree: true, characterData: true });
}

// Nodes that take a video and show a picture (the editor would put their output in a video player).
const IMAGE_PREVIEW_NODES = new Set(["LoomVideoFrame"]);

// ---------------------------------------------------------------- the Builder: the agent
// A side panel with a chat (agent.py): the request and the graph on the canvas go to the agent, the graph it saves
// replaces the canvas (nodes, values, links); positions of nodes that stay are kept. Nothing is run.
const AGENT = { messages: [], busy: false, view: null };

function graphPrompt() {
  return app.graphToPrompt().then((p) => (Object.keys(p.output || {}).length ? p.output : null)).catch(() => null);
}

function applyGraph(g) {
  const graph = app.graph;
  const keep = {};
  for (const n of graph._nodes ?? []) keep[String(n.id)] = [...n.pos];
  graph.clear();
  const made = {};
  for (const [id, n] of Object.entries(g.prompt)) {
    const node = globalThis.LiteGraph.createNode(n.class_type);
    if (!node) continue;
    if (/^\d+$/.test(id)) node.id = Number(id);
    graph.add(node);
    node.pos = keep[id] ?? g.layout[id] ?? [60, 60];
    if (g.titles?.[id]) node.title = g.titles[id];
    made[id] = node;
  }
  const isLink = (v) => Array.isArray(v) && v.length === 2 && typeof v[1] === "number";
  for (const [id, n] of Object.entries(g.prompt)) {
    const node = made[id];
    if (!node) continue;
    for (const [name, value] of Object.entries(n.inputs)) {
      if (isLink(value) || name.includes(".")) continue;
      const widget = node.widgets?.find((w) => w.name === name);
      if (!widget) continue;
      widget.value = value;
      if (name === "character") showCharacter(node, value);
    }
    // links in slot order: a growing input adds its next slot once the previous one is connected
    const slotNo = (k) => Number((k.match(/(\d+)$/) || [0, 0])[1]);
    const links = Object.entries(n.inputs).filter(([, v]) => isLink(v)).sort(([a], [b]) => slotNo(a) - slotNo(b));
    for (const [name, [src, out]] of links) {
      const from = made[String(src)];
      const slot = node.inputs?.findIndex((i) => i.name === name) ?? -1;
      if (from && slot >= 0) from.connect(out, node, slot);
    }
  }
  graph.setDirtyCanvas(true, true);
  app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
  // fit once the editor has drawn the new nodes (their sizes are known only then)
  setTimeout(() => { try { app.extensionManager?.command?.execute("Comfy.Canvas.FitView"); } catch { /* none */ } }, 400);
}

function agentRender() {
  const v = AGENT.view;
  if (!v) return;
  v.list.replaceChildren(...(AGENT.messages.length ? AGENT.messages.map((m) => el("div", { class: `loom-ag-msg is-${m.role}` },
    ...(m.steps?.length ? [el("div", { class: "loom-ag-steps" }, ...m.steps.map((s) => el("div", {}, "· " + s)))] : []),
    m.text ? el("div", { class: "loom-ag-text" }, m.text) : null,
    m.error ? el("div", { class: "loom-ag-error" }, m.error) : null)) : [el("div", { class: "loom-ag-hint" },
    tr("Опишите, что нужно сделать, и Конструктор соберёт схему из нод. Например: «Картинка товара на белом фоне, " +
    "потом из неё видео на 5 секунд» или «Придумай рекламный слоган и нарисуй постер с ним». Схему можно " +
    "менять сообщениями. Запуск по кнопке «Запустить», как обычно.",
    "Describe what you need, and the Builder puts together a graph of nodes. For example: \"A product picture on a " +
    "white background, then a 5-second video from it\" or \"Come up with an ad slogan and draw a poster with it\". " +
    "Change the graph with messages. Run it with the Run button as usual."))]));
  v.send.disabled = AGENT.busy;
  v.input.disabled = AGENT.busy;
  v.list.scrollTop = v.list.scrollHeight;
}

async function agentSend() {
  const v = AGENT.view;
  const text = v?.input.value.trim();
  if (!text || AGENT.busy) return;
  v.input.value = "";
  AGENT.messages.push({ role: "user", text });
  const reply = { role: "assistant", text: "", steps: [tr("Думает…", "Thinking…")] };
  AGENT.messages.push(reply);
  AGENT.busy = true;
  agentRender();
  try {
    const r = await fetch(api.apiURL("/nodeloom/agent"), {
      method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: AGENT.messages.slice(0, -1).filter((m) => m.text).map(({ role, text }) => ({ role, text })),
        graph: await graphPrompt() }),
    });
    if (!r.ok || !r.body) throw new Error(tr(`Ошибка ${r.status}`, `Error ${r.status}`));
    reply.steps = [];
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "step") reply.steps.push(ev.label);
        else if (ev.type === "error") reply.error = ev.message;
        else if (ev.type === "done") {
          reply.text = ev.text;
          if (ev.graph) applyGraph(ev.graph);
        }
        agentRender();
      }
    }
  } catch (e) {
    reply.error = tr("Конструктор не ответил: ", "The Builder did not answer: ") + (e?.message || tr("сбой сети", "network error"));
  } finally {
    if (!reply.text && !reply.error) reply.error = tr("Ответа нет, попробуйте ещё раз", "No answer, please try again");
    AGENT.busy = false;
    agentRender();
  }
}

function setupAgent() {
  app.extensionManager.registerSidebarTab?.({
    id: "loom-agent", icon: "pi pi-sparkles", title: tr("Конструктор", "Builder"), label: tr("Конструктор", "Builder"),
    tooltip: tr("Конструктор: соберёт схему по описанию", "Builder: puts together a graph from a description"), type: "custom",
    render: (root) => {
      const list = el("div", { class: "loom-ag-list" });
      const input = el("textarea", { class: "loom-ag-input", rows: "3", placeholder: tr("Что собрать или поменять?", "What should I build or change?") });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); agentSend(); }
        e.stopPropagation(); // the canvas has its own shortcuts
      });
      const send = el("button", { class: "loom-ag-send", type: "button", onclick: agentSend }, tr("Отправить", "Send"));
      const reset = el("button", { class: "loom-ag-reset", type: "button",
        title: tr("Начать заново: схема на холсте останется", "Start over: the graph on the canvas stays"),
        onclick: () => { if (!AGENT.busy) { AGENT.messages = []; agentRender(); } } }, tr("Новая беседа", "New chat"));
      root.replaceChildren(el("div", { class: "loom-ag" },
        el("div", { class: "loom-ag-head" }, el("span", { class: "loom-ag-title" }, tr("Конструктор", "Builder")), reset),
        list, el("div", { class: "loom-ag-compose" }, input, send)));
      AGENT.view = { list, input, send };
      agentRender();
    },
  });
}

app.registerExtension({
  name: "nodeloom",
  commands: [
    { id: "Nodeloom.Language.English", label: "English", function: () => setLanguage("en") },
    { id: "Nodeloom.Language.Russian", label: "Русский", function: () => setLanguage("ru") },
  ],
  menuCommands: [{ path: [tr("Язык", "Language")], commands: ["Nodeloom.Language.English", "Nodeloom.Language.Russian"] }],
  async init() {
    addStylesheet();
    setIdentity();
    await replaceDefaultGraph();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name.startsWith("Loom")) setupSlotLabels(nodeType);
    if (nodeData.name === "LoomSeedanceCharacter") setupCharacterNode(nodeType);
    if (nodeData.name === "LoomShow") setupShowNode(nodeType);
    if (nodeData.name === "LoomLoadAudio") setupLoadAudioNode(nodeType);
    if (IMAGE_PREVIEW_NODES.has(nodeData.name)) {
      // A node with a video input shows its output in a video player unless it says otherwise: this one shows a picture.
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        created?.apply(this, []);
        this.previewMediaType = "image";
      };
    }
    if (!TEXT_NODES.has(nodeData.name)) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, []);
      preview()?.addTextPreviewWidgets(this);
      // the stock preview adds a "Plain text / Markdown" switch with an internal label: plain text is enough here
      const toggle = this.widgets?.find((w) => w.options?.on === "Markdown" || w.options?.label_on === "Markdown");
      if (toggle) this.removeWidget ? this.removeWidget(toggle) : this.widgets.splice(this.widgets.indexOf(toggle), 1);
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      executed?.apply(this, [output]);
      if (output?.text) preview()?.updateTextPreviewWidgets(this, output);
    };
  },
  onNodeOutputsUpdated(outputs) {
    for (const [id, output] of Object.entries(outputs)) {
      const node = app.graph?.getNodeById?.(id);
      if (node && TEXT_NODES.has(node.type) && output?.text) preview()?.updateTextPreviewWidgets(node, output);
    }
  },
  async setup() {
    swapLogo();
    for (const id of HIDDEN_TABS) app.extensionManager.unregisterSidebarTab?.(id);
    disableCommands();
    filterNodeLibrary();
    plainRuns();
    await fixSettings();
    trimCanvasMenu();
    trimWorkflowMenu();
    showRunCost();
    setupUploads();
    setupAgent();
    plainErrors();
    checkProviders();
  },
});
