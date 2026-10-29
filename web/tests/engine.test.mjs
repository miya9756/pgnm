// node --test web/tests — the browser engine against pgnm/ on identical noise.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  BLOCK, FieldBuilder, KERNEL, extractTile, gaussianNoise, kernelSpectra, renderTile, tiles,
} from "../js/engine.js";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const PAD = KERNEL - 1;

const reference = (() => {
  const out = mkdtempSync(join(tmpdir(), "pgnm-web-"));
  const python = process.env.PYTHON || "python3";
  const run = spawnSync(python, [join(ROOT, "web/tests/reference.py"), out], { encoding: "utf8" });
  if (run.error || run.status !== 0) throw new Error(`reference.py failed: ${run.error || run.stderr}`);
  const meta = JSON.parse(readFileSync(join(out, "meta.json"), "utf8"));
  const floats = (name) => {
    const bytes = readFileSync(join(out, name));
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  };
  const bytes = (name) => new Uint8Array(readFileSync(join(out, name)));
  return { meta, floats, bytes };
})();

function preset(key) {
  const params = JSON.parse(readFileSync(join(ROOT, reference.meta.presets[key]), "utf8"));
  const kernels = Float32Array.from(params.kernels.flat(3));
  const g = Float32Array.from(params.marginal.g.flat(2));
  const blur = key in reference.meta.blur
    ? Float64Array.from(JSON.parse(readFileSync(join(ROOT, "presets/film/blur.json"), "utf8"))[reference.meta.blur[key]].flat())
    : null;
  return { spectra: kernelSpectra(kernels), g, blur };
}

/** Noise source reading PGNM's own padded noise array. */
function fixtureNoise(noise, width) {
  const fieldWidth = width + PAD;
  return (x0, y0, w, h, planes, stride) => {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const from = ((y0 + y) * fieldWidth + x0 + x) * 3;
        for (let c = 0; c < 3; c++) planes[c][y * stride + x] = noise[from + c];
      }
    }
  };
}

for (const key of ["camera", "film"]) {
  test(`${key}: latent field matches pgnm on the same noise`, () => {
    const { width, height } = reference.meta;
    const expected = reference.floats(`field_${key}`);
    const noise = fixtureNoise(reference.floats(`noise_${key}`), width);
    const { spectra } = preset(key), builder = new FieldBuilder();
    let worst = 0;
    for (const { x0, y0, w, h } of tiles(width, height)) {
      const field = builder.build(spectra, x0, y0, w, h, noise);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w * 3; x++) {
          worst = Math.max(worst, Math.abs(field[y * w * 3 + x] - expected[((y0 + y) * width + x0) * 3 + x]));
        }
      }
    }
    assert.ok(worst < 1e-5, `largest field difference ${worst}`);
  });

  test(`${key}: delivered image matches pgnm on the same noise`, () => {
    const { width, height } = reference.meta;
    const rgb = reference.bytes("clean"), expected = reference.bytes(`noisy_${key}`);
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let p = 0; p < width * height; p++) {
      rgba.set(rgb.subarray(3 * p, 3 * p + 3), 4 * p);
      rgba[4 * p + 3] = 255;
    }
    const params = preset(key), builder = new FieldBuilder();
    const noise = fixtureNoise(reference.floats(`noise_${key}`), width);
    const halo = params.blur ? 3 : 0;
    let differing = 0, worst = 0;
    for (const tile of tiles(width, height)) {
      const pixels = extractTile(rgba, width, height, tile.x0, tile.y0, tile.w, tile.h, halo);
      const out = renderTile(builder, params, { ...tile, width, height, halo, pixels }, noise);
      for (let y = 0; y < tile.h; y++) {
        for (let x = 0; x < tile.w; x++) {
          for (let c = 0; c < 3; c++) {
            const diff = Math.abs(out[(y * tile.w + x) * 4 + c] - expected[((tile.y0 + y) * width + tile.x0 + x) * 3 + c]);
            if (diff) differing++;
            worst = Math.max(worst, diff);
          }
          assert.equal(out[(y * tile.w + x) * 4 + 3], 255);
        }
      }
    }
    // Float32 storage of g may flip a rare value sitting on a rounding boundary.
    assert.ok(worst <= 1 && differing <= width * height * 3 * 1e-4, `${differing} codes differ, by up to ${worst}`);
  });
}

test("seeded noise is standard, independent and seamless", () => {
  const n = 256, size = n * n;
  const planes = [new Float64Array(size), new Float64Array(size), new Float64Array(size)];
  gaussianNoise(3, 1000, 10, 20, n, n, planes, n);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const corr = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s / a.length;
  };
  for (const z of planes) {
    assert.ok(Math.abs(mean(z)) < 0.01);
    assert.ok(Math.abs(corr(z, z) - 1) < 0.02);
    assert.ok(Math.abs(corr(z.subarray(1), z.subarray(0, size - 1))) < 0.01);
    assert.ok(Math.abs(corr(z.subarray(n), z.subarray(0, size - n))) < 0.01);
  }
  assert.ok(Math.abs(corr(planes[0], planes[1])) < 0.01);
  assert.ok(Math.abs(corr(planes[0], planes[2])) < 0.01);

  // A sub-window reproduces the same values; another seed does not.
  const window = [new Float64Array(16), new Float64Array(16), new Float64Array(16)];
  gaussianNoise(3, 1000, 15, 25, 4, 4, window, 4);
  assert.equal(window[2][5], planes[2][6 * n + 6]);
  gaussianNoise(4, 1000, 15, 25, 4, 4, window, 4);
  assert.notEqual(window[2][5], planes[2][6 * n + 6]);
});

test("tiles cover the image once, nearest first", () => {
  const width = 3 * BLOCK + 5, height = BLOCK + 1, seen = new Uint8Array(width * height);
  const list = tiles(width, height, 0, 0);
  for (const { x0, y0, w, h } of list) {
    for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) seen[y * width + x]++;
  }
  assert.ok(seen.every((count) => count === 1));
  assert.deepEqual([list[0].x0, list[0].y0], [0, 0]);
});
