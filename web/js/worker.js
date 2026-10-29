// Render worker: caches filter spectra per preset and renders tiles on demand.
import { FieldBuilder, kernelSpectra, renderTile } from "./engine.js";
import { renderScene } from "./scene.js";

const builder = new FieldBuilder();
const presets = new Map();   // id -> { spectra, g, blur }, most recent last

function remember(id, preset) {
  presets.delete(id);
  presets.set(id, preset);
  while (presets.size > 3) presets.delete(presets.keys().next().value);
}

self.onmessage = ({ data }) => {
  try {
    if (data.type === "preset") {
      remember(data.id, { spectra: kernelSpectra(data.kernels, builder.fft), g: data.g, blur: data.blur });
    } else if (data.type === "tile") {
      const preset = presets.get(data.preset);
      remember(data.preset, preset);
      const pixels = renderTile(builder, preset, data);
      self.postMessage({ type: "tile", job: data.job, x0: data.x0, y0: data.y0, w: data.w, h: data.h, pixels },
        [pixels.buffer]);
    } else if (data.type === "scene") {
      const pixels = renderScene(data.width, data.height, data.y0, data.y1);
      self.postMessage({ type: "scene", job: data.job, y0: data.y0, y1: data.y1, pixels }, [pixels.buffer]);
    }
  } catch (error) {
    self.postMessage({ type: "error", job: data.job, message: String(error && error.message || error) });
  }
};
