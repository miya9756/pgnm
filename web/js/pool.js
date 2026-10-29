// Worker pool. One job at a time: starting a job drops the previous one, and
// its late results are ignored. Each worker holds the spectra of its three most
// recent presets; the pool mirrors that cache so it knows what to send.
import { extractTile, tiles } from "./engine.js";

const CACHE = 3;

export class Pool {
  constructor(size = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1))) {
    this.serial = 0;
    this.job = null;
    this.slots = Array.from({ length: size }, () => {
      const slot = { worker: new Worker(new URL("./worker.js", import.meta.url), { type: "module" }), busy: false, cache: [] };
      slot.worker.onmessage = ({ data }) => this.receive(slot, data);
      slot.worker.onerror = (event) => this.fail(event.message || "worker failed to start");
      return slot;
    });
  }

  /** Noisy tiles of `source` ({ width, height, data }); resolves false if superseded. */
  render({ source, preset, seed, center, onTile }) {
    const halo = preset.blur ? 3 : 0;
    return this.start(tiles(source.width, source.height, center.x, center.y), (slot, tile, id) => {
      if (!this.touch(slot, preset.id)) {
        slot.worker.postMessage({ type: "preset", id: preset.id, kernels: preset.kernels, g: preset.g, blur: preset.blur });
      }
      const pixels = extractTile(source.data, source.width, source.height, tile.x0, tile.y0, tile.w, tile.h, halo);
      slot.worker.postMessage({ type: "tile", job: id, preset: preset.id, seed, width: source.width,
        height: source.height, x0: tile.x0, y0: tile.y0, w: tile.w, h: tile.h, halo, pixels }, [pixels.buffer]);
    }, onTile);
  }

  /** The procedural sample scene as ImageData, rendered in bands. */
  async scene(width, height) {
    const image = new ImageData(width, height), bands = [];
    for (let y0 = 0; y0 < height; y0 += 32) bands.push({ y0, y1: Math.min(height, y0 + 32) });
    const done = await this.start(bands, (slot, band, id) => {
      slot.worker.postMessage({ type: "scene", job: id, width, height, ...band });
    }, (result) => image.data.set(result.pixels, result.y0 * width * 4));
    return done ? image : null;
  }

  start(tasks, post, onResult) {
    if (this.job) this.job.resolve(false);
    return new Promise((resolve, reject) => {
      this.job = { id: ++this.serial, tasks, post, onResult, resolve, reject, done: 0, total: tasks.length };
      for (const slot of this.slots) if (!slot.busy) this.dispatch(slot);
    });
  }

  /** Mark a preset most recent in a worker's cache; false if it was missing. */
  touch(slot, id) {
    const at = slot.cache.indexOf(id);
    if (at >= 0) slot.cache.splice(at, 1);
    slot.cache.push(id);
    if (slot.cache.length > CACHE) slot.cache.shift();
    return at >= 0;
  }

  dispatch(slot) {
    const job = this.job;
    slot.busy = Boolean(job && job.tasks.length);
    if (slot.busy) job.post(slot, job.tasks.shift(), job.id);
  }

  receive(slot, data) {
    const job = this.job;
    if (job && data.job === job.id) {
      if (data.type === "error") return this.fail(data.message);
      job.done++;
      job.onResult(data, job.done / job.total);
      if (job.done === job.total) {
        this.job = null;
        job.resolve(true);
      }
    }
    this.dispatch(slot);
  }

  fail(message) {
    const job = this.job;
    this.job = null;
    // Forgetting is safe: what the pool remembers stays a suffix of each worker's cache.
    for (const slot of this.slots) Object.assign(slot, { busy: false, cache: [] });
    if (job) job.reject(new Error(message));
  }
}
