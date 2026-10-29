// PGNM inference in the browser: the two equations of pgnm/model.py.
//
//   t_c(p) = sum_i sum_u A_ci(u) z_i(p - u)              latent field
//   y_c(p) = clip(x_c(p) + g_c(x_c(p), t_c(p)), 0, 1)     noisy image
//
// The latent field is a valid convolution of padded Gaussian noise, computed
// tile by tile with 256-point FFTs (overlap-save). The Gaussian noise is a hash
// of (seed, pixel), so any tile can be rendered alone and tiles agree at their
// seams. Nothing here touches the DOM: workers and tests import it directly.

export const KERNEL = 25;                       // kernels[25][25][3 input][3 output]
export const FFT_SIZE = 256;
export const BLOCK = FFT_SIZE - KERNEL + 1;     // 232 output pixels per tile side
export const KNOTS = 25;                        // marginal.t = -3, -2.75, ..., 3
export const KNOT_START = -3;
export const KNOT_STEP = 0.25;
export const CODES = 256;                       // marginal.g[3][256][25]
export const KERNEL_LENGTH = KERNEL * KERNEL * 9;
export const LOOKUP_LENGTH = 3 * CODES * KNOTS;

// Fourier transform ---------------------------------------------------------------

export class FFT {
  constructor(n) {
    this.n = n;
    this.reverse = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.reverse[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos(2 * Math.PI * i / n);
      this.sin[i] = Math.sin(2 * Math.PI * i / n);
    }
  }

  /** In-place radix-2 transform of re/im[offset, offset + n); unscaled inverse. */
  transform(re, im, offset, inverse) {
    const { n, reverse, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = reverse[i];
      if (j > i) {
        const a = offset + i, b = offset + j;
        let t = re[a]; re[a] = re[b]; re[b] = t;
        t = im[a]; im[a] = im[b]; im[b] = t;
      }
    }
    const sign = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let start = offset; start < offset + n; start += size) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = cos[t], wi = sign * sin[t];
          const a = start + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  /**
   * Column transforms of an n×n array, restricted to columns [from, to).
   * Butterflies combine whole rows, so the inner loop runs along memory.
   */
  columns(re, im, inverse, from = 0, to = this.n) {
    const { n, reverse, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = reverse[i];
      if (j > i) {
        for (let x = from; x < to; x++) {
          const a = i * n + x, b = j * n + x;
          let t = re[a]; re[a] = re[b]; re[b] = t;
          t = im[a]; im[a] = im[b]; im[b] = t;
        }
      }
    }
    const sign = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = cos[t], wi = sign * sin[t];
          const rowA = (start + k) * n, rowB = rowA + half * n;
          for (let x = from; x < to; x++) {
            const a = rowA + x, b = rowB + x;
            const xr = re[b] * wr - im[b] * wi;
            const xi = re[b] * wi + im[b] * wr;
            re[b] = re[a] - xr; im[b] = im[a] - xi;
            re[a] += xr; im[a] += xi;
          }
        }
      }
    }
  }

  /** Forward 2-D transform; rows at or past `filledRows` must be zero. */
  forward2d(re, im, filledRows = this.n) {
    for (let y = 0; y < filledRows; y++) this.transform(re, im, y * this.n, false);
    this.columns(re, im, false);
  }

  /** Unscaled inverse 2-D transform, exact only inside rows/columns [from, to). */
  inverse2d(re, im, from = 0, to = this.n) {
    for (let y = 0; y < this.n; y++) this.transform(re, im, y * this.n, true);
    this.columns(re, im, true, from, to);
  }
}

// Gaussian noise ------------------------------------------------------------------

function hash(x) {
  // Wellons' triple32: a full-avalanche bijection on 32-bit integers.
  x ^= x >>> 17; x = Math.imul(x, 0xed5ad4bb);
  x ^= x >>> 11; x = Math.imul(x, 0xac4c1b51);
  x ^= x >>> 15; x = Math.imul(x, 0x31848bab);
  x ^= x >>> 14;
  return x >>> 0;
}

const TWO_PI = 2 * Math.PI;
const UNIT = 1 / 4294967296;

/**
 * Fill three channel planes with independent standard Gaussian noise for the
 * padded-field window rows [y0, y0 + h), columns [x0, x0 + w) of a field
 * `fieldWidth` wide. Plane layout is row-major with the given row stride.
 */
export function gaussianNoise(seed, fieldWidth, x0, y0, w, h, planes, stride) {
  const salt = hash((seed ^ 0x9e3779b9) >>> 0);
  const [z0, z1, z2] = planes;
  for (let y = 0; y < h; y++) {
    let index = (y0 + y) * fieldWidth + x0;
    let at = y * stride;
    for (let x = 0; x < w; x++, index++, at++) {
      const a = hash(hash(index) ^ salt);
      const b = hash(a ^ 0x632be5ab);
      const c = hash(b ^ 0x85ebca6b);
      const d = hash(c ^ 0xc2b2ae35);
      const r1 = Math.sqrt(-2 * Math.log((a + 0.5) * UNIT));
      const r2 = Math.sqrt(-2 * Math.log((c + 0.5) * UNIT));
      const p1 = TWO_PI * b * UNIT, p2 = TWO_PI * d * UNIT;
      z0[at] = r1 * Math.cos(p1);
      z1[at] = r1 * Math.sin(p1);
      z2[at] = r2 * Math.cos(p2);
    }
  }
}

// Latent field --------------------------------------------------------------------

/**
 * Spectra of the nine filters for overlap-save convolution. Outputs 0 and 1
 * are packed as one complex signal (M_i = K_0i + j K_1i), output 2 alone.
 */
export function kernelSpectra(kernels, fft = new FFT(FFT_SIZE)) {
  const n = fft.n, size = n * n;
  const spectrum = (input, output) => {
    const re = new Float64Array(size), im = new Float64Array(size);
    for (let u = 0; u < KERNEL; u++) {
      for (let v = 0; v < KERNEL; v++) {
        re[u * n + v] = kernels[((u * KERNEL + v) * 3 + input) * 3 + output];
      }
    }
    fft.forward2d(re, im, KERNEL);
    return { re, im };
  };
  const packed = [], third = [];
  for (let input = 0; input < 3; input++) {
    const k0 = spectrum(input, 0), k1 = spectrum(input, 1);
    const re = new Float64Array(size), im = new Float64Array(size);
    for (let k = 0; k < size; k++) {
      re[k] = k0.re[k] - k1.im[k];
      im[k] = k0.im[k] + k1.re[k];
    }
    packed.push({ re, im });
    third.push(spectrum(input, 2));
  }
  return { n, packed, third };
}

/** Scratch buffers and transforms for one worker. */
export class FieldBuilder {
  constructor(n = FFT_SIZE) {
    const size = n * n;
    this.fft = new FFT(n);
    this.n = n;
    this.pRe = new Float64Array(size); this.pIm = new Float64Array(size);
    this.qRe = new Float64Array(size); this.qIm = new Float64Array(size);
    this.sRe = new Float64Array(size); this.sIm = new Float64Array(size);
  }

  /**
   * Latent field for output pixels [x0, x0 + w) × [y0, y0 + h) of an image
   * `width` wide, as interleaved RGB float32. `noise(x, y, w, h, planes, stride)`
   * fills padded-field noise; the default is the seeded hash.
   */
  build(spectra, x0, y0, w, h, noise) {
    const { n, fft, pRe, pIm, qRe, qIm, sRe, sIm } = this;
    const pad = KERNEL - 1;
    if (w > n - pad || h > n - pad) throw new RangeError("tile larger than the FFT block");
    pRe.fill(0); pIm.fill(0); qRe.fill(0); qIm.fill(0);
    noise(x0, y0, w + pad, h + pad, [pRe, pIm, qRe], n);
    fft.forward2d(pRe, pIm, h + pad);
    fft.forward2d(qRe, qIm, h + pad);

    const [m0, m1, m2] = spectra.packed, [k0, k1, k2] = spectra.third;
    const mask = n - 1;
    for (let ky = 0; ky < n; ky++) {
      const row = ky * n, mirrorRow = ((n - ky) & mask) * n;
      for (let kx = 0; kx < n; kx++) {
        const k = row + kx, mirror = mirrorRow + ((n - kx) & mask);
        // Unpack the spectra of the two real inputs sharing one transform.
        const pr = pRe[k], pi = pIm[k], nr = pRe[mirror], ni = pIm[mirror];
        const x0r = 0.5 * (pr + nr), x0i = 0.5 * (pi - ni);
        const x1r = 0.5 * (pi + ni), x1i = 0.5 * (nr - pr);
        const x2r = qRe[k], x2i = qIm[k];
        sRe[k] = m0.re[k] * x0r - m0.im[k] * x0i + m1.re[k] * x1r - m1.im[k] * x1i + m2.re[k] * x2r - m2.im[k] * x2i;
        sIm[k] = m0.re[k] * x0i + m0.im[k] * x0r + m1.re[k] * x1i + m1.im[k] * x1r + m2.re[k] * x2i + m2.im[k] * x2r;
        qRe[k] = k0.re[k] * x0r - k0.im[k] * x0i + k1.re[k] * x1r - k1.im[k] * x1i + k2.re[k] * x2r - k2.im[k] * x2i;
        qIm[k] = k0.re[k] * x0i + k0.im[k] * x0r + k1.re[k] * x1i + k1.im[k] * x1r + k2.re[k] * x2i + k2.im[k] * x2r;
      }
    }
    fft.inverse2d(sRe, sIm, pad, pad + w);
    fft.inverse2d(qRe, qIm, pad, pad + w);

    // Circular samples at offsets >= KERNEL - 1 equal the valid convolution.
    const scale = 1 / (n * n);
    const field = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const k = (y + pad) * n + x + pad, out = (y * w + x) * 3;
        field[out] = sRe[k] * scale;
        field[out + 1] = sIm[k] * scale;
        field[out + 2] = qRe[k] * scale;
      }
    }
    return field;
  }
}

// Residual lookup -----------------------------------------------------------------

/** g_c(x, t): bilinear in clean intensity and latent value, linear beyond ±3. */
export function residual(lookup, channel, intensity, latent) {
  const ip = Math.min(Math.max(intensity, 0), 1) * (CODES - 1);
  const ii = Math.min(Math.floor(ip), CODES - 2);
  const iw = ip - ii;
  const gp = (latent - KNOT_START) / KNOT_STEP;
  const gi = Math.min(Math.max(Math.floor(gp), 0), KNOTS - 2);
  const gw = gp - gi;
  const at = (channel * CODES + ii) * KNOTS + gi;
  const lower = lookup[at] + gw * (lookup[at + 1] - lookup[at]);
  const upper = lookup[at + KNOTS] + gw * (lookup[at + KNOTS + 1] - lookup[at + KNOTS]);
  return lower + iw * (upper - lower);
}

/**
 * Add residuals to clean RGB (interleaved, [0, 1]) and deliver 8-bit RGBA by
 * clipping, scaling and rounding half up. Alpha comes from `alpha` (RGBA bytes).
 */
export function addNoise(clean, field, lookup, alpha, out) {
  for (let p = 0, q = 0; p < clean.length; p += 3, q += 4) {
    for (let c = 0; c < 3; c++) {
      const x = clean[p + c];
      const y = Math.min(Math.max(x + residual(lookup, c, x, field[p + c]), 0), 1);
      out[q + c] = Math.floor(y * 255 + 0.5);
    }
    out[q + 3] = alpha[q + 3];
  }
  return out;
}

// Clean image ---------------------------------------------------------------------

/** scipy.ndimage "reflect" (half-sample symmetric) index into [0, n). */
export function reflect(i, n) {
  const period = 2 * n;
  let m = i % period;
  if (m < 0) m += period;
  return m < n ? m : period - 1 - m;
}

/** Clean RGB in [0, 1] from RGBA bytes: x = code / 255. */
export function cleanFromBytes(rgba, w, h) {
  const clean = new Float32Array(w * h * 3);
  for (let p = 0, q = 0; q < w * h * 4; p += 3, q += 4) {
    clean[p] = rgba[q] / 255; clean[p + 1] = rgba[q + 1] / 255; clean[p + 2] = rgba[q + 2] / 255;
  }
  return clean;
}

/**
 * Film presets expect mu = clip(B ⋆ x, 0, 255) / 255 with the radius's 7×7
 * blur B (scipy correlate, reflected edges), per channel. `rgba` holds the
 * tile with a `halo`-pixel border already reflected.
 */
export function blurredClean(rgba, w, h, halo, blur) {
  const size = Math.round(Math.sqrt(blur.length)), r = size >> 1;
  if (r > halo) throw new RangeError("halo smaller than the blur radius");
  const stride = w + 2 * halo;
  const clean = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s0 = 0, s1 = 0, s2 = 0;
      for (let a = 0; a < size; a++) {
        let q = ((y + halo + a - r) * stride + (x + halo - r)) * 4;
        for (let b = 0; b < size; b++, q += 4) {
          const weight = blur[a * size + b];
          s0 += weight * rgba[q]; s1 += weight * rgba[q + 1]; s2 += weight * rgba[q + 2];
        }
      }
      const p = (y * w + x) * 3;
      clean[p] = Math.min(Math.max(s0, 0), 255) / 255;
      clean[p + 1] = Math.min(Math.max(s1, 0), 255) / 255;
      clean[p + 2] = Math.min(Math.max(s2, 0), 255) / 255;
    }
  }
  return clean;
}

/** Copy an RGBA window with a reflected border of `halo` pixels. */
export function extractTile(rgba, width, height, x0, y0, w, h, halo = 0) {
  const tw = w + 2 * halo, th = h + 2 * halo;
  const tile = new Uint8ClampedArray(tw * th * 4);
  for (let y = 0; y < th; y++) {
    const sy = reflect(y0 + y - halo, height);
    if (halo === 0) {
      const from = (sy * width + x0) * 4;
      tile.set(rgba.subarray(from, from + w * 4), y * tw * 4);
      continue;
    }
    for (let x = 0; x < tw; x++) {
      const from = (sy * width + reflect(x0 + x - halo, width)) * 4, to = (y * tw + x) * 4;
      tile[to] = rgba[from]; tile[to + 1] = rgba[from + 1];
      tile[to + 2] = rgba[from + 2]; tile[to + 3] = rgba[from + 3];
    }
  }
  return tile;
}

/** The tile's own pixels (without halo) from a haloed RGBA tile. */
export function stripHalo(rgba, w, h, halo) {
  if (halo === 0) return rgba;
  const stride = w + 2 * halo, out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const from = ((y + halo) * stride + halo) * 4;
    out.set(rgba.subarray(from, from + w * 4), y * w * 4);
  }
  return out;
}

// One tile, end to end --------------------------------------------------------------

/** Tile grid covering an image, nearest to (cx, cy) first. */
export function tiles(width, height, cx = width / 2, cy = height / 2) {
  const list = [];
  for (let y0 = 0; y0 < height; y0 += BLOCK) {
    for (let x0 = 0; x0 < width; x0 += BLOCK) {
      const w = Math.min(BLOCK, width - x0), h = Math.min(BLOCK, height - y0);
      list.push({ x0, y0, w, h, d: Math.hypot(x0 + w / 2 - cx, y0 + h / 2 - cy) });
    }
  }
  return list.sort((a, b) => a.d - b.d);
}

/** The seeded hash noise of a padded field for an image `width` wide. */
export function seededNoise(seed, width) {
  const fieldWidth = width + KERNEL - 1;
  return (x0, y0, w, h, planes, stride) => gaussianNoise(seed, fieldWidth, x0, y0, w, h, planes, stride);
}

/**
 * Render one tile: `job` = { seed, width, height, x0, y0, w, h, halo, pixels }
 * with `pixels` from extractTile; `preset` = { spectra, g, blur }.
 */
export function renderTile(builder, preset, job, noise = seededNoise(job.seed, job.width)) {
  const { x0, y0, w, h, halo, pixels } = job;
  const field = builder.build(preset.spectra, x0, y0, w, h, noise);
  const own = stripHalo(pixels, w, h, halo);
  const clean = preset.blur ? blurredClean(pixels, w, h, halo, preset.blur) : cleanFromBytes(own, w, h);
  return addNoise(clean, field, preset.g, own, new Uint8ClampedArray(w * h * 4));
}
