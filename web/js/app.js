// PGNM Studio: pick a look, see it on your photo, save it.
import { KERNEL_LENGTH, LOOKUP_LENGTH } from "./engine.js";
import { Pool } from "./pool.js";
import { Viewer } from "./viewer.js";

const MAX_PIXELS = 4096 * 4096;      // the largest canvas every browser accepts
const SAMPLE = { width: 1536, height: 1024 };
const ICONS = { camera: "#i-camera", film: "#i-film" };

const $ = (selector) => document.querySelector(selector);
const pool = new Pool();
const viewer = new Viewer($("#viewer"), { onZoom: showZoom });
const presets = new Map();           // id -> manifest entry
const loaded = new Map();            // id -> Promise<{ id, kernels, g, blur }>
const state = { source: null, after: null, look: null, chosen: {}, seed: 1, serial: 0, idle: Promise.resolve(), finish: null };

// Looks ------------------------------------------------------------------------------

async function start() {
  const manifest = await (await fetch("presets.json")).json();
  for (const group of manifest.groups) {
    const tab = document.createElement("button");
    tab.setAttribute("role", "tab");
    tab.dataset.group = group.id;
    tab.innerHTML = `<svg><use href="${ICONS[group.id]}"/></svg><span>${group.label}</span>`;
    tab.addEventListener("click", () => choose(state.chosen[group.id]));
    $("#tabs").append(tab);

    const section = document.createElement("section");
    section.className = "group";
    section.dataset.group = group.id;
    for (const row of group.rows) {
      const label = Object.assign(document.createElement("div"), { className: "row-label", textContent: row.label });
      const chips = Object.assign(document.createElement("div"), { className: "chips" });
      chips.style.setProperty("--cols", group.id === "camera" ? 6 : 4);
      for (const preset of row.presets) {
        presets.set(preset.id, { ...preset, group: group.id });
        const chip = Object.assign(document.createElement("button"), { className: "chip", title: preset.title });
        chip.dataset.id = preset.id;
        chip.setAttribute("aria-pressed", "false");
        chip.innerHTML = `<span class="swatch" style="background-image:url(swatches/${preset.id}.png)"></span><span class="cap">${preset.label}</span>`;
        chip.addEventListener("click", () => choose(preset.id));
        chips.append(chip);
      }
      section.append(label, chips);
    }
    $("#looks").append(section);
  }
  for (const group of manifest.groups) state.chosen[group.id] = group.default;
  choose(manifest.groups[0].default);

  const image = await pool.scene(SAMPLE.width, SAMPLE.height);
  if (image && !state.source) setSource("studio", image);
}

function choose(id) {
  const preset = presets.get(id);
  if (preset === state.look) return;
  state.look = preset;
  state.chosen[preset.group] = id;
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.group === preset.group));
  }
  for (const section of document.querySelectorAll(".group")) section.hidden = section.dataset.group !== preset.group;
  for (const chip of document.querySelectorAll(".chip")) chip.setAttribute("aria-pressed", String(chip.dataset.id === id));
  $("#look").textContent = preset.title;
  render();
}

function loadPreset(id) {
  if (!loaded.has(id)) {
    loaded.set(id, fetch(`presets/${id}.bin`).then(async (response) => {
      if (!response.ok) throw new Error(`preset ${id}: ${response.status}`);
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength !== 4 * (KERNEL_LENGTH + LOOKUP_LENGTH)) throw new Error(`preset ${id}: wrong size`);
      const { blur } = presets.get(id);
      return {
        id,
        kernels: new Float32Array(buffer, 0, KERNEL_LENGTH),
        g: new Float32Array(buffer, 4 * KERNEL_LENGTH, LOOKUP_LENGTH),
        blur: blur ? Float64Array.from(blur.flat()) : null,
      };
    }).catch((error) => { loaded.delete(id); throw error; }));
  }
  return loaded.get(id);
}

// Rendering --------------------------------------------------------------------------

function setSource(name, image) {
  const canvas = (width, height) => Object.assign(document.createElement("canvas"), { width, height });
  const before = canvas(image.width, image.height), after = canvas(image.width, image.height);
  before.getContext("2d").putImageData(image, 0, 0);
  after.getContext("2d").drawImage(before, 0, 0);
  state.source = { name, width: image.width, height: image.height, data: image.data, before };
  state.after = after;
  viewer.show(before, after);
  render();
}

async function render() {
  const { source, look, seed } = state;
  if (!source || !look) return;
  const serial = ++state.serial;
  if (!state.finish) state.idle = new Promise((resolve) => { state.finish = resolve; });
  document.body.classList.add("busy");
  progress(0);
  try {
    const preset = await loadPreset(look.id);
    if (serial !== state.serial) return;
    const context = state.after.getContext("2d");
    const done = await pool.render({
      source, preset, seed, center: viewer.center(),
      onTile: (tile, fraction) => {
        context.putImageData(new ImageData(tile.pixels, tile.w, tile.h), tile.x0, tile.y0);
        viewer.invalidate();
        progress(fraction);
      },
    });
    if (done && serial === state.serial) settle();
  } catch (error) {
    if (serial !== state.serial) return;
    settle();
    toast("Something went wrong rendering this look");
    console.error(error);
  }
}

function settle() {
  document.body.classList.remove("busy");
  const finish = state.finish;
  state.finish = null;
  if (finish) finish();
}

function progress(fraction) {
  $(".progress i").style.transform = `scaleX(${fraction})`;
}

// Images in and out ----------------------------------------------------------------

async function open(file) {
  if (!file || !file.type.startsWith("image/")) return toast("That isn't an image");
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, Math.sqrt(MAX_PIXELS / (bitmap.width * bitmap.height)));
    const width = Math.max(1, Math.round(bitmap.width * scale)), height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = Object.assign(document.createElement("canvas"), { width, height });
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    setSource(file.name.replace(/\.[^.]+$/, "") || "image", context.getImageData(0, 0, width, height));
    if (scale < 1) toast(`Resized to ${width} × ${height}`);
  } catch (error) {
    toast("Couldn't open that image");
    console.error(error);
  }
}

async function save() {
  if (!state.source) return;
  await state.idle;
  const name = `${state.source.name}-${state.look.id}.png`;
  state.after.toBlob((blob) => {
    const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
  }, "image/png");
}

function reroll() {
  state.seed = (state.seed * 747796405 + 2891336453) >>> 0;
  const button = $("#reroll");
  button.classList.remove("spin");
  void button.offsetWidth;
  button.classList.add("spin");
  render();
}

// Chrome -----------------------------------------------------------------------------

function showZoom(zoom) {
  for (const button of document.querySelectorAll(".zoom button")) {
    const value = button.dataset.zoom;
    const active = value === "fit" ? zoom === "fit" : zoom !== "fit" && Math.abs(zoom - Number(value)) < 1e-3;
    button.setAttribute("aria-pressed", String(active));
  }
}

function setZoom(value) {
  if (value === "fit") viewer.fit();
  else viewer.zoomTo(Number(value) * viewer.dpr);
}

let toastTimer = 0;
function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), 2600);
}

$("#reroll").addEventListener("click", reroll);
$("#open").addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", (event) => { open(event.target.files[0]); event.target.value = ""; });
$("#save").addEventListener("click", save);
for (const button of document.querySelectorAll(".zoom button")) {
  button.addEventListener("click", () => setZoom(button.dataset.zoom));
}

let dragDepth = 0;
window.addEventListener("dragenter", (event) => {
  if (!event.dataTransfer?.types.includes("Files")) return;
  dragDepth++;
  document.body.classList.add("dragging");
});
window.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); }
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  open(event.dataTransfer?.files[0]);
});
window.addEventListener("paste", (event) => {
  const item = [...(event.clipboardData?.items || [])].find((entry) => entry.type.startsWith("image/"));
  if (item) open(item.getAsFile());
});
window.addEventListener("keydown", (event) => {
  if (event.altKey || event.target instanceof HTMLInputElement) return;
  const key = event.key.toLowerCase();
  if ((event.metaKey || event.ctrlKey) && key === "s") { event.preventDefault(); return save(); }
  if (event.metaKey || event.ctrlKey) return;
  const actions = { r: reroll, o: () => $("#file").click(), s: save, f: () => setZoom("fit"),
    1: () => setZoom(1), 2: () => setZoom(2) };
  if (actions[key]) { event.preventDefault(); actions[key](); }
});

start().catch((error) => {
  document.body.classList.remove("busy");
  toast("Couldn't start the studio");
  console.error(error);
});
