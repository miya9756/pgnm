"""Fig. calibration_dncnn.png: `python -m blind.figure_calibration` (needs runs/blind/results.json)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pgnm import load, fitting
from blind import estimate as E
from blind.run import source_image, render_batch, Cached, IMAGES
from blind.denoisers import dncnn

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = dict(naive="#2a78d6", envelope="#eb6834", calibrated="#1baf7a")
LAB = dict(naive="naive (x̂ as clean)", envelope="envelope", calibrated="calibrated")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

knots = fitting.GAUSSIAN_KNOTS; w = np.exp(-knots ** 2 / 2); w /= w.sum()
def std_curve(lookup, c=1):
    g = np.asarray(lookup)[c]; mu = (g * w).sum(-1)
    return 255 * np.sqrt(((g - mu[:, None]) ** 2 * w).sum(-1))

# ---- one case, reproducing the run (N6_800, coffee, seed 1001)
preset = load("presets/sidd/N6_800.json")
x = source_image("coffee"); y = render_batch(preset, [x], 1000 + IMAGES.index("coffee"))[0]
D = Cached(dncnn("real"))
curves = {"truth": std_curve(preset["marginal"]["g"]),
          "naive": std_curve(E.naive(y, D)["marginal"]["g"]),
          "envelope": std_curve(E.envelope(y, D)["marginal"]["g"])}
# calibration loop of estimate.calibrated, recording the lookup after every iteration
rng = np.random.default_rng(0)
clean, yf = E.to_float(D(y)), E.to_float(y)
mask = E._masks(clean, yf)
C_obs, hist = E.statistics(clean, yf, mask); s_obs = E.strength_curves(clean, yf, mask)
C, lookup = C_obs.copy(), fitting.build_residual_lookup(hist)
steps = [std_curve(lookup)]
for _ in range(4):
    params = E.parameters(C, lookup, "step"); C_sim, s_sim = 0, []
    for _ in range(2):
        sim = E.to_float(E.to_uint8(E.render_fresh(params, clean, rng)))
        sc = E.to_float(D(E.to_uint8(sim))); m = E._masks(sc, sim)
        c_, _ = E.statistics(sc, sim, m); C_sim = C_sim + c_ / 2; s_sim.append(E.strength_curves(sc, sim, m))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = s_obs / np.nanmean(s_sim, axis=0)
    C = C + (C_obs - C_sim); lookup = E.rescale_lookup(lookup, np.clip(ratio, .5, 2))
    steps.append(std_curve(lookup))
curves["calibrated"] = steps[-1]
present = np.bincount(np.clip(np.rint(clean[..., 1] * 255), 0, 255).astype(int).ravel(), minlength=256) >= 300
codes = np.arange(256)
def seg(ax, yv, **kw):
    v = np.where(present, yv, np.nan); return ax.plot(codes, v, **kw)

fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))
# A: strength curves
a = ax[0, 0]
seg(a, curves["truth"], color=INK, lw=2.4, label="true preset")
for k, ls in [("naive", ":"), ("envelope", "--"), ("calibrated", "-")]:
    seg(a, curves[k], color=COL[k], lw=2, ls=ls, label=LAB[k])
a.set_xlim(0, 255); a.set_xlabel("clean intensity (green code)"); a.set_ylabel("noise std (codes)")
a.set_title("A  One image: estimated noise strength vs. intensity\nN6_800 noise on 'coffee', DnCNN denoiser", loc="left", color=INK)
a.legend(frameon=False, fontsize=9, loc="upper left"); a.grid(color=GRID, lw=.8); a.set_axisbelow(True)

# B: iterations (sequential blue ramp, light -> dark)
b = ax[0, 1]
ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]
seg(b, curves["truth"], color=INK, lw=2.4, label="true preset")
for i, (s, col) in enumerate(zip(steps, ramp)):
    seg(b, s, color=col, lw=2, label="envelope (start)" if i == 0 else f"after iteration {i}")
b.set_xlim(0, 255); b.set_xlabel("clean intensity (green code)"); b.set_ylabel("noise std (codes)")
b.set_title("B  Calibration iterations on the same image\neach step: x̂ + PGNM noise → DnCNN → rescale by observed/simulated", loc="left", color=INK)
b.legend(frameon=False, fontsize=9, loc="upper left"); b.grid(color=GRID, lw=.8); b.set_axisbelow(True)

# C, D: grid summary for the DnCNN
r = json.load(open("runs/blind/results.json"))
presets = ["IP_1000", "N6_800", "S6_1600", "GP_10000"]
ests = ["naive", "envelope", "calibrated"]
for axis, key, title in [(ax[1, 0], "strength_bias", "C  Signed strength bias, DnCNN (0 = right strength)"),
                         (ax[1, 1], "strength_error", "D  Strength error, DnCNN (lower is better)")]:
    width = .24
    for j, e in enumerate(ests):
        vals = np.array([[r[f"{p}/{img}/dncnn/{e}"][key] for img in IMAGES] for p in presets])
        pos = np.arange(4) + (j - 1) * (width + .02)
        axis.bar(pos, vals.mean(1), width, color=COL[e], label=LAB[e], edgecolor=SURF, linewidth=2, zorder=2)
        for k in range(4):
            axis.scatter(np.full(6, pos[k]), vals[k], s=10, color=INK2, zorder=3, alpha=.7)
    if key == "strength_error":
        orc = [np.mean([r[f"{p}/{img}/none/oracle"][key] for img in IMAGES]) for p in presets]
        for k, o in enumerate(orc):
            axis.plot([k - .4, k + .4], [o, o], color=INK, lw=1.5, ls="--", zorder=4, label="oracle (true clean pair)" if k == 0 else None)
    axis.axhline(0, color=INK2, lw=1)
    axis.set_xticks(range(4)); axis.set_xticklabels([f"{p}\n({'weak' if i < 2 else 'strong'} noise)" for i, p in enumerate(presets)])
    axis.set_ylabel("%"); axis.set_title(title + "\nbars: mean of 6 images, dots: each image", loc="left", color=INK)
    axis.grid(axis="y", color=GRID, lw=.8); axis.set_axisbelow(True); axis.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
ax[1, 0].set_ylim(-45, 70); ax[1, 1].set_ylim(0, 60)
fig.tight_layout()
out = "/home/user/pgnm/blind/calibration_dncnn.png"
fig.savefig(out, dpi=160); print(out)

