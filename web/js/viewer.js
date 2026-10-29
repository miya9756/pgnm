// Canvas viewer: clean on the left of the split, noisy on the right. `zoom` is
// in device pixels per image pixel; `scale` (what the buttons show) in screen
// (CSS) pixels per image pixel.

const MAX_ZOOM = 32;

export class Viewer {
  constructor(root, { onZoom } = {}) {
    this.root = root;
    this.canvas = root.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.handle = root.querySelector(".split");
    this.tags = root.querySelectorAll(".tag");
    this.onZoom = onZoom || (() => {});
    this.before = this.after = null;
    this.width = this.height = 0;
    this.zoom = 1; this.cx = 0; this.cy = 0;
    this.fitting = true;
    this.split = 0.5;
    this.pointers = new Map();
    this.frame = 0;
    new ResizeObserver(() => this.resize()).observe(root);
    this.bind();
  }

  get dpr() { return window.devicePixelRatio || 1; }
  get fitZoom() {
    return this.width ? Math.min(this.canvas.width / this.width, this.canvas.height / this.height) * 0.9 : 1;
  }

  /** Show two same-sized canvases; a new size refits the view. */
  show(before, after) {
    const resized = before.width !== this.width || before.height !== this.height;
    this.before = before; this.after = after;
    this.width = before.width; this.height = before.height;
    if (resized) this.fit();
    this.invalidate();
  }

  get scale() { return this.zoom / this.dpr; }

  center() { return { x: this.cx, y: this.cy }; }

  fit() {
    this.fitting = true;
    this.zoom = this.fitZoom;
    this.cx = this.width / 2; this.cy = this.height / 2;
    this.changed();
  }

  /** Zoom to `zoom`, keeping the image point under device pixel (px, py) still. */
  zoomTo(zoom, px = this.canvas.width / 2, py = this.canvas.height / 2) {
    const { width: cw, height: ch } = this.canvas;
    const ix = this.cx + (px - cw / 2) / this.zoom, iy = this.cy + (py - ch / 2) / this.zoom;
    this.zoom = Math.min(Math.max(zoom, Math.min(this.fitZoom, 1) / 2), MAX_ZOOM);
    this.cx = ix - (px - cw / 2) / this.zoom; this.cy = iy - (py - ch / 2) / this.zoom;
    this.fitting = false;
    this.changed();
  }

  panBy(dx, dy) {
    this.cx -= dx / this.zoom; this.cy -= dy / this.zoom;
    this.fitting = false;
    this.changed();
  }

  changed() {
    this.cx = Math.min(Math.max(this.cx, 0), this.width);
    this.cy = Math.min(Math.max(this.cy, 0), this.height);
    this.onZoom(this.fitting ? "fit" : this.scale);
    this.invalidate();
  }

  resize() {
    const { clientWidth, clientHeight } = this.root;
    this.canvas.width = Math.round(clientWidth * this.dpr);
    this.canvas.height = Math.round(clientHeight * this.dpr);
    if (this.fitting) this.zoom = this.fitZoom;
    this.draw();
  }

  invalidate() {
    if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.draw(); });
  }

  draw() {
    const { ctx, canvas, zoom } = this;
    const cw = canvas.width, ch = canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    this.handle.style.left = `${this.split * 100}%`;
    this.tags[0].classList.toggle("hidden", this.split < 0.12);
    this.tags[1].classList.toggle("hidden", this.split > 0.88);
    if (!this.before) return;
    let x = cw / 2 - this.cx * zoom, y = ch / 2 - this.cy * zoom;
    if (zoom >= 1) { x = Math.round(x); y = Math.round(y); }
    const w = this.width * zoom, h = this.height * zoom;
    ctx.save();
    ctx.shadowColor = "rgba(0, 0, 0, 0.55)";
    ctx.shadowBlur = 40 * this.dpr;
    ctx.shadowOffsetY = 8 * this.dpr;
    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);
    ctx.restore();
    ctx.imageSmoothingEnabled = zoom < 1;
    ctx.imageSmoothingQuality = "high";
    const split = Math.round(this.split * cw);
    for (const [image, left, right] of [[this.before, 0, split], [this.after, split, cw]]) {
      if (right <= left) continue;
      ctx.save();
      ctx.beginPath();
      ctx.rect(left, 0, right - left, ch);
      ctx.clip();
      ctx.drawImage(image, x, y, w, h);
      ctx.restore();
    }
  }

  // Input -------------------------------------------------------------------------

  local(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * this.dpr, y: (event.clientY - rect.top) * this.dpr };
  }

  bind() {
    const { canvas, handle } = this;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      this.pointers.set(event.pointerId, this.local(event));
      this.pinch = null;
      canvas.classList.add("grabbing");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.pointers.has(event.pointerId)) return;
      const previous = this.pointers.get(event.pointerId), point = this.local(event);
      this.pointers.set(event.pointerId, point);
      if (this.pointers.size === 1) return this.panBy(point.x - previous.x, point.y - previous.y);
      const [a, b] = [...this.pointers.values()];
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, distance = Math.hypot(a.x - b.x, a.y - b.y);
      if (this.pinch) {
        this.panBy(mid.x - this.pinch.mid.x, mid.y - this.pinch.mid.y);
        this.zoomTo(this.zoom * distance / this.pinch.distance, mid.x, mid.y);
      }
      this.pinch = { mid, distance };
    });
    const release = (event) => {
      this.pointers.delete(event.pointerId);
      this.pinch = null;
      if (!this.pointers.size) canvas.classList.remove("grabbing");
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1);
      const point = this.local(event);
      this.zoomTo(this.zoom * Math.exp(-delta * (event.ctrlKey ? 0.01 : 0.002)), point.x, point.y);
    }, { passive: false });
    canvas.addEventListener("dblclick", (event) => {
      const point = this.local(event);
      if (this.fitting || this.scale < 0.99) this.zoomTo(this.dpr, point.x, point.y);
      else this.fit();
    });

    const moveSplit = (event) => {
      const rect = this.root.getBoundingClientRect();
      this.split = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1);
      handle.setAttribute("aria-valuenow", Math.round(this.split * 100));
      this.invalidate();
    };
    handle.addEventListener("pointerdown", (event) => {
      handle.setPointerCapture(event.pointerId);
      handle.classList.add("dragging");
      moveSplit(event);
    });
    handle.addEventListener("pointermove", (event) => { if (handle.hasPointerCapture(event.pointerId)) moveSplit(event); });
    handle.addEventListener("pointerup", () => handle.classList.remove("dragging"));
    handle.addEventListener("keydown", (event) => {
      const step = { ArrowLeft: -0.05, ArrowRight: 0.05, Home: -1, End: 1 }[event.key];
      if (step === undefined) return;
      event.preventDefault();
      this.split = Math.min(Math.max(this.split + step, 0), 1);
      handle.setAttribute("aria-valuenow", Math.round(this.split * 100));
      this.invalidate();
    });
  }
}
