// The sample image: a procedural studio still life, smooth enough that every
// tone from shadow to highlight shows its noise. Rendered by the workers in
// bands, deterministic, with no assets to download.

const SPHERES = [
  // center, radius, linear albedo, gloss
  { c: [-1.08, 0.84, -0.25], r: 0.84, albedo: [0.78, 0.76, 0.72], gloss: 0.1 },
  { c: [0.52, 0.54, 0.55], r: 0.54, albedo: [0.72, 0.18, 0.06], gloss: 0.6 },
  { c: [1.58, 0.37, -0.4], r: 0.37, albedo: [0.04, 0.11, 0.4], gloss: 0.75 },
  { c: [-0.08, 0.21, 1.55], r: 0.21, albedo: [0.82, 0.58, 0.1], gloss: 0.6 },
  { c: [-0.66, 0.13, 1.98], r: 0.13, albedo: [0.02, 0.02, 0.022], gloss: 0.9 },
];
const KEY = normalize([-0.55, 0.8, 0.42]);
const KEY_COLOR = [2.5, 2.38, 2.2];
const RIM = normalize([0.75, 0.35, -0.6]);
const RIM_COLOR = [0.55, 0.62, 0.75];
const FLOOR_ALBEDO = [0.4, 0.385, 0.37];
const EYE = [0.25, 1.22, 6.5];
const TARGET = [0.2, 0.52, 0.2];
const FOV = 27 * Math.PI / 180;

function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const smoothstep = (a, b, x) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

function hitSphere(o, d, s) {
  const ox = o[0] - s.c[0], oy = o[1] - s.c[1], oz = o[2] - s.c[2];
  const b = ox * d[0] + oy * d[1] + oz * d[2];
  const h = b * b - (ox * ox + oy * oy + oz * oz - s.r * s.r);
  if (h < 0) return Infinity;
  const t = -b - Math.sqrt(h);
  return t > 1e-4 ? t : Infinity;
}

function nearest(o, d) {
  let t = Infinity, id = -1;
  for (let i = 0; i < SPHERES.length; i++) {
    const ti = hitSphere(o, d, SPHERES[i]);
    if (ti < t) { t = ti; id = i; }
  }
  return { t, id };
}

/** Soft shadow toward a distant light: penumbra widens with distance. */
function shadow(p, l, skip) {
  let light = 1;
  for (let i = 0; i < SPHERES.length; i++) {
    if (i === skip) continue;
    const s = SPHERES[i];
    const cx = s.c[0] - p[0], cy = s.c[1] - p[1], cz = s.c[2] - p[2];
    const along = cx * l[0] + cy * l[1] + cz * l[2];
    if (along <= 0) continue;
    const dx = cx - along * l[0], dy = cy - along * l[1], dz = cz - along * l[2];
    const spread = 0.16 * along;
    light *= smoothstep(s.r - spread, s.r + spread, Math.hypot(dx, dy, dz));
  }
  return light;
}

/** Ambient occlusion of the spheres seen from p with normal n. */
function occlusion(p, n, skip) {
  let open = 1;
  for (let i = 0; i < SPHERES.length; i++) {
    if (i === skip) continue;
    const s = SPHERES[i];
    const rx = s.c[0] - p[0], ry = s.c[1] - p[1], rz = s.c[2] - p[2];
    const l = Math.hypot(rx, ry, rz);
    const amount = (n[0] * rx + n[1] * ry + n[2] * rz) * s.r * s.r / (l * l * l);
    open *= 1 - Math.min(Math.max(amount, 0), 1);
  }
  return open;
}

function backdrop(d) {
  const rise = smoothstep(-0.04, 0.3, d[1]);
  const glow = Math.exp(-((d[0] + 0.04) ** 2) / 0.05);
  const tone = (0.06 + 0.1 * glow) * (1 - 0.8 * rise) + 0.01;
  return [tone * 1.02, tone, tone * 0.98];
}

function floorColor(p, d, t, shadows = true) {
  const n = [0, 1, 0];
  const pool = Math.exp(-((p[0] - 0.1) ** 2 + (p[2] - 0.6) ** 2) / (2 * 2 * 2));
  const key = KEY[1] * (shadows ? shadow(p, KEY, -1) : 1) * pool;
  const ambient = 0.1 * (shadows ? occlusion(p, n, -1) : 1);
  const lit = [0, 1, 2].map((c) => FLOOR_ALBEDO[c] * (KEY_COLOR[c] * key + ambient));
  const far = backdrop(d), fade = smoothstep(7, 15, t);
  return lit.map((v, c) => v + (far[c] - v) * fade);
}

function environment(o, d) {
  if (d[1] < 0) {
    const t = -o[1] / d[1];
    return floorColor([o[0] + t * d[0], 0, o[2] + t * d[2]], d, t, false);
  }
  const softbox = 4 * smoothstep(0.9, 0.975, dot(d, KEY));
  return backdrop(d).map((v) => v + softbox);
}

function shade(o, d) {
  const { t, id } = nearest(o, d);
  if (id < 0) {
    if (d[1] >= 0) return backdrop(d);
    const tf = -o[1] / d[1];
    return floorColor([o[0] + tf * d[0], 0, o[2] + tf * d[2]], d, tf);
  }
  const s = SPHERES[id];
  const p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
  const n = normalize([p[0] - s.c[0], p[1] - s.c[1], p[2] - s.c[2]]);
  const v = [-d[0], -d[1], -d[2]];
  const nv = Math.max(dot(n, v), 0);
  const lit = Math.max((dot(n, KEY) + 0.3) / 1.3, 0) ** 1.5 * shadow(p, KEY, id);
  const ao = occlusion(p, n, id) * (0.6 + 0.4 * smoothstep(-0.2, 1, n[1]));
  const sky = 0.1 + 0.06 * n[1];
  const fresnel = (0.04 + 0.96 * (1 - nv) ** 5) * s.gloss;
  const h = normalize([KEY[0] + v[0], KEY[1] + v[1], KEY[2] + v[2]]);
  const spec = s.gloss * (8 + 300 * s.gloss) / 60 * Math.max(dot(n, h), 0) ** (8 + 500 * s.gloss * s.gloss) * lit;
  const rim = 0.8 * (1 - nv) ** 3 * Math.max(dot(n, RIM), 0);
  const r = [d[0] - 2 * dot(d, n) * n[0], d[1] - 2 * dot(d, n) * n[1], d[2] - 2 * dot(d, n) * n[2]];
  const env = s.gloss > 0.2 ? environment(p, r) : [0, 0, 0];
  return [0, 1, 2].map((c) =>
    s.albedo[c] * (KEY_COLOR[c] * lit + sky * ao) * (1 - fresnel)
    + fresnel * env[c] + spec * KEY_COLOR[c] + rim * RIM_COLOR[c]);
}

function camera(width, height) {
  const f = normalize([TARGET[0] - EYE[0], TARGET[1] - EYE[1], TARGET[2] - EYE[2]]);
  const r = normalize([f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0]);
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  const scale = Math.tan(FOV / 2) * 2 / height;
  return (x, y) => {
    const a = (x - width / 2) * scale, b = (height / 2 - y) * scale;
    return normalize([f[0] + a * r[0] + b * u[0], f[1] + a * r[1] + b * u[1], f[2] + a * r[2] + b * u[2]]);
  };
}

function encode(v) {
  // ACES fit, then the sRGB transfer curve.
  const x = v * 0.9;
  const tone = Math.min(Math.max((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0), 1);
  const srgb = tone <= 0.0031308 ? 12.92 * tone : 1.055 * tone ** (1 / 2.4) - 0.055;
  return Math.round(srgb * 255);
}

/** RGBA bytes of rows [y0, y1) of a width×height render. */
export function renderScene(width, height, y0 = 0, y1 = height) {
  const ray = camera(width, height);
  // Sphere ids at pixel corners mark the edges that need supersampling.
  const ids = new Int8Array((width + 1) * (y1 - y0 + 1));
  for (let y = y0; y <= y1; y++) {
    for (let x = 0; x <= width; x++) ids[(y - y0) * (width + 1) + x] = nearest(EYE, ray(x, y)).id;
  }
  const out = new Uint8ClampedArray(width * (y1 - y0) * 4);
  const offsets = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
  for (let y = y0; y < y1; y++) {
    for (let x = 0; x < width; x++) {
      const at = (y - y0) * (width + 1) + x;
      const edge = ids[at] !== ids[at + 1] || ids[at] !== ids[at + width + 1] || ids[at] !== ids[at + width + 2];
      const samples = edge ? offsets : [[0.5, 0.5]];
      let r = 0, g = 0, b = 0;
      for (const [dx, dy] of samples) {
        const c = shade(EYE, ray(x + dx, y + dy));
        r += c[0]; g += c[1]; b += c[2];
      }
      const nx = (x + 0.5) / width - 0.5, ny = (y + 0.5) / height - 0.5;
      const lens = (1 - 0.5 * (nx * nx + ny * ny)) / samples.length;
      const q = ((y - y0) * width + x) * 4;
      out[q] = encode(r * lens); out[q + 1] = encode(g * lens); out[q + 2] = encode(b * lens); out[q + 3] = 255;
    }
  }
  return out;
}
