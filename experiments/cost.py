"""Model size and speed (Tab. 4).

    python -m experiments.cost parameters --method naflow    # once per neural baseline, in its environment
    python -m experiments.cost time --method pgnm --device cpu --mode playback
    python -m experiments.cost time --method naflow --device cuda      # NAFlow environment
    python -m experiments.cost port-reference                 # PGNM on the CPU, base environment
    python -m experiments.cost port-check --device cuda       # PyTorch port, e.g. GMDCN environment
    python -m experiments.cost table                          # Tab. 4

Parameters are the learned values a model reads at inference. A PGNM preset
holds the 25x25x3x3 filter tensor and the 3x256x25 residual lookup; its
Gaussian knot grid is the same fixed grid in every preset and is not counted.
Film rows add the 49 values of the shared 7x7 blur B; Zhang23 reads a grain
radius and a filter width per radius. A neural baseline counts the learned
parameters of the network its adapter loads, which is the noise generator
only (no discriminator or denoiser); parameters that are not trained, such as
NAFlow's fixed identity entries, are not counted.

Time is for one 1920x1080 RGB frame of color ramps, batch 1, fp32. The timed
call starts from the 8-bit frame in host memory and returns the noisy float
frame in host memory, so type conversion and host-device copies are inside;
CUDA is synchronized before and after every call. Model loading, the first
call and the final 8-bit quantization are timed separately and excluded. GPU
rows report the median of 50 calls after 10 warm-up calls, CPU rows the median
of 10 calls after one. SIDD rows use the G4 ISO 800 condition and its PGNM
preset; film rows use radius 0.1, applied to each color channel.

PGNM is timed three ways. "new-field" draws and filters a new H x W latent
field for every frame, as the baselines draw new noise. "playback" is the
renderer of all reported results: one cyclic window of its bank of four
2H x 2W fields per frame. Its setup, the bank construction, is reported
separately; on the CPU it includes compiling the lookup kernel. The GPU rows
use the PyTorch port in experiments/torch_pgnm.py, checked against the CPU
renderer by port-reference and port-check.

For the paper every row ran alone in its own process with two threads,
pinned to four efficiency cores at the lowest priority with a 16 GB memory
cap, e.g. `taskset -c 28-31 nice -n 19 python -m experiments.cost time ...`;
GPU rows ran on one RTX A6000, capped at 12 GB.

Outputs: runs/cost/parameters/<method>.json, runs/cost/time/<row>.json,
runs/cost/port-check.json, runs/cost/table.json.
"""
from __future__ import annotations

import os

THREADS = 2
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, str(THREADS))  # before NumPy and PyTorch start their thread pools

import argparse
from pathlib import Path
import platform
import resource
import time

import numpy as np

from . import config
from .seeds import to_uint8

HEIGHT, WIDTH = 1080, 1920
SIDD_CLASS = "G4_800"   # covered by every SIDD method, including the 15-class ones
FILM_RADIUS = "r100"
GPU_MEMORY_GIB = 12.0
NEURAL = ("srgb_flow", "gmdcn", "naflow", "danet", "nmflowgan", "grainnet")
METHODS = NEURAL + ("zhang23", "pgnm")
LABELS = dict(srgb_flow="sRGB-Flow", gmdcn="GMDCN", naflow="NAFlow", danet="DANet",
              nmflowgan="NM-FlowGAN", grainnet="GrainNet", zhang23="Zhang23", pgnm="PGNM (ours)")
SIDD_BASELINES = ("srgb_flow", "gmdcn", "naflow", "danet", "nmflowgan")
FILM_BASELINES = ("zhang23", "grainnet")
PGNM_MODES = (("new-field", "new field per frame"), ("playback", "bank playback"))


def clean_frame():
    """Deterministic color ramps; the channels differ, so no gray shortcut applies."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    return np.stack([x * 255 // (WIDTH - 1), y * 255 // (HEIGHT - 1),
                     (x + y) * 255 // (HEIGHT + WIDTH - 2)], -1).astype(np.uint8)


def sidd_preset_path():
    return config.REPO / "presets/sidd" / (SIDD_CLASS + ".json")


# Parameters ----------------------------------------------------------------------

def pgnm_values(preset):
    """Filter tensor and residual lookup; the Gaussian knot grid is fixed."""
    return int(np.size(preset["kernels"]) + np.size(preset["marginal"]["g"]))


def gray_distinct_values(preset):
    """Values that differ between channels of a preset fitted on gray images.

    The nine 25x25 filter slices agree to fitting round-off (1e-8) and the
    three lookup channels are identical, so one slice and one channel remain.
    """
    kernels, lookup = np.asarray(preset["kernels"]), np.asarray(preset["marginal"]["g"])
    if not (np.allclose(kernels, kernels[:, :, :1, :1], rtol=0, atol=1e-8) and (lookup == lookup[:1]).all()):
        raise ValueError("preset channels differ")
    return int(kernels[:, :, 0, 0].size + lookup[0].size)


def presets_parameters():
    """PGNM (SIDD and film) and Zhang23 counts, read from presets/."""
    root = config.REPO / "presets"
    sidd = [config.read_json(p) for p in sorted((root / "sidd").glob("*.json"))]
    film = {p.stem: config.read_json(p) for p in sorted((root / "film").glob("r[0-9]*.json"))}
    blur = config.read_json(root / "film/blur.json")
    zhang = config.read_json(root / "film/zhang23.json")
    knots = {tuple(p["marginal"]["t"]) for p in sidd + list(film.values())}
    if len(knots) != 1:
        raise ValueError("presets use different Gaussian knot grids")

    def one_value(counts):
        if len(set(counts)) != 1:
            raise ValueError("models of one method differ in size")
        return counts[0]

    return dict(
        pgnm_sidd=dict(per_model=one_value([pgnm_values(p) for p in sidd]), models=len(sidd)),
        pgnm_film=dict(per_model=one_value([pgnm_values(p) + int(np.size(blur[k])) for k, p in film.items()]),
                       models=len(film),
                       distinct_per_model=one_value([gray_distinct_values(p) for p in film.values()])),
        # Zhang23 reads r and s of each layer (one gray layer per radius) and the blur.
        zhang23=dict(per_model=one_value([2 * len(zhang[k]["layers"]) + int(np.size(blur[k])) for k in film]),
                     models=len(film)))


def generator(method, device):
    """The adapter's gen(clean_u8, cond, seed) on the requested device."""
    from baselines.adapters import build
    if method == "srgb_flow" and device == "cpu":
        return _srgb_flow_on_cpu()
    return build(method, device)


def _srgb_flow_on_cpu():
    """sRGB-Flow's adapter and upstream flow layers fix CUDA; run the same source on the CPU."""
    import torch
    import baselines.adapters as adapters
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    path = Path(adapters.__file__).with_name("srgb_flow.py")
    source = path.read_text().replace('"cuda"', '"cpu"').replace(".cuda()", ".cpu()")
    namespace = dict(__name__="baselines.adapters.srgb_flow_cpu", __package__="baselines.adapters",
                     __file__=str(path))
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["build"]("cpu")


def condition(method):
    if method == "grainnet":
        return dict(radius=0.1)
    from baselines.render_sidd import conditioning
    camera, iso = SIDD_CLASS.split("_")
    return conditioning(method, camera, int(iso))


def count(method, device="cpu"):
    """Learned parameters of the network the adapter loads from its checkpoint.

    Every adapter loads exactly one network with load_state_dict; it is
    recorded there. Learned means requires_grad; buffers are not counted.
    """
    import torch
    loaded = []
    original = torch.nn.Module.load_state_dict

    def recording(module, *args, **kwargs):
        loaded.append(module)
        return original(module, *args, **kwargs)

    torch.nn.Module.load_state_dict = recording
    try:
        gen = generator(method, device)
        if method == "srgb_flow":  # builds its flow for the first image size it sees
            gen(np.zeros((64, 64, 3), np.uint8), condition(method), 0)
    finally:
        torch.nn.Module.load_state_dict = original
    if len(loaded) != 1:
        raise RuntimeError(f"{method}: expected one loaded network, got {len(loaded)}")
    network = loaded[0]
    result = dict(method=method, network=type(network).__name__,
                  learned=sum(p.numel() for p in network.parameters() if p.requires_grad),
                  fixed=sum(p.numel() for p in network.parameters() if not p.requires_grad),
                  buffers=sum(b.numel() for b in network.buffers()))
    config.write_json(config.runs("cost/parameters", method + ".json"), result)
    print(result)
    return result


# Timing --------------------------------------------------------------------------

def pgnm_renderer(device, mode):
    """(setup seconds, render(frame_u8) -> noisy float frame on the host)."""
    if device == "cpu":
        from pgnm import PGNM, load
        model = PGNM(load(sidd_preset_path()), workers=THREADS)
        if mode == "playback":
            start = time.perf_counter()
            model.warmup((HEIGHT, WIDTH, 3))  # bank construction and compilation of the lookup kernel
            setup = time.perf_counter() - start
            return setup, lambda frame: model.render(frame.astype(np.float32) / 255)
        # New field per frame: the renderer's own field and lookup steps, without a bank.
        tiny = np.zeros((1, 1, 3), np.float32)
        model._render_noise_window(tiny, tiny, 0, 0, np.empty_like(tiny))  # compile the lookup kernel
        rng = np.random.default_rng(1)

        def new_field(frame):
            clean = frame.astype(np.float32) / 255
            field = model._build_noise_field(clean.shape, rng).astype(np.float32)
            noisy = np.empty_like(clean)
            model._render_noise_window(field, clean, 0, 0, noisy)
            return noisy
        return 0.0, new_field

    import torch
    from .torch_pgnm import TorchPGNM
    port = TorchPGNM(config.read_json(sidd_preset_path()), device)
    noise_generator = torch.Generator(device=device).manual_seed(1)

    def render(frame):
        clean = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255
        if mode == "playback":
            noisy = port.render(clean)
        else:
            noisy = port.add_noise(port.new_field(HEIGHT, WIDTH, noise_generator), clean)
        return noisy.permute(1, 2, 0).cpu().numpy()

    setup = 0.0
    if mode == "playback":
        torch.cuda.synchronize()
        start = time.perf_counter()
        port.warmup(HEIGHT, WIDTH)
        torch.cuda.synchronize()
        setup = time.perf_counter() - start
    return setup, render


def neural_renderer(method, device):
    from baselines.adapters import seed_all
    gen = generator(method, device)
    cond = condition(method)
    calls = [0]

    def render(frame):
        calls[0] += 1
        seed_all(1202026 + calls[0])  # every call reseeds, as when rendering the evaluation images
        if method == "nmflowgan":  # its U-Net needs multiples of 16: reflect-pad, render, crop
            pad = (-frame.shape[0]) % 16, (-frame.shape[1]) % 16
            padded = np.pad(frame, ((0, pad[0]), (0, pad[1]), (0, 0)), mode="reflect")
            return gen(padded, cond, calls[0])[:frame.shape[0], :frame.shape[1]]
        return gen(frame, cond, calls[0])
    return 0.0, render


def zhang23_renderer():
    """Grain on the frame itself (the shared blur B is not timed); the gray fit per channel."""
    from baselines import zhang23
    params = dict(layers=config.read_json(config.REPO / "presets/film/zhang23.json")[FILM_RADIUS]["layers"] * 3)
    rng = np.random.default_rng(1)

    def render(frame):
        clean = frame.astype(np.float32) / 255
        return np.clip(clean + zhang23.synthesize(clean, params, rng), 0, 1).astype(np.float32)
    return 0.0, render


def row_name(method, device, mode=None):
    return "-".join([method] + ([mode] if method == "pgnm" else []) + [device])


def measure(method, device, mode="playback", warmup=None, repetitions=None, budget=900.0):
    """Time one row and write runs/cost/time/<row>.json."""
    if method == "zhang23" and device != "cpu":
        raise ValueError("Zhang23 has no GPU implementation")
    record = dict(method=method, mode=mode if method == "pgnm" else None, device=device,
                  frame=[HEIGHT, WIDTH, 3], batch=1, precision="fp32",
                  condition=FILM_RADIUS if method in FILM_BASELINES else SIDD_CLASS,
                  cpu=_cpu_name(), affinity=sorted(os.sched_getaffinity(0)),
                  threads={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    if method in NEURAL or device == "cuda":
        import torch
        torch.set_num_threads(THREADS)
        record.update(torch=torch.__version__, tf32=dict(
            cudnn=torch.backends.cudnn.allow_tf32, matmul=torch.backends.cuda.matmul.allow_tf32))
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_MEMORY_GIB * 2**30 / total), 0)
            record.update(gpu=torch.cuda.get_device_name(0), gpu_memory_cap_gib=GPU_MEMORY_GIB)

    frame = clean_frame()
    start = time.perf_counter()
    if method == "pgnm":
        setup, render = pgnm_renderer(device, mode)
    elif method == "zhang23":
        setup, render = zhang23_renderer()
    else:
        setup, render = neural_renderer(method, device)
    record.update(load_seconds=time.perf_counter() - start, setup_seconds=setup)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def timed():
        sync()
        begin = time.perf_counter()
        noisy = render(frame)
        sync()
        return time.perf_counter() - begin, noisy

    record["first_seconds"], noisy = timed()
    noisy = np.asarray(noisy)
    if noisy.shape != frame.shape or not np.isfinite(noisy).all():
        raise ValueError(f"invalid output of shape {noisy.shape}")
    begin = time.perf_counter()
    to_uint8(noisy)
    record["quantization_seconds"] = time.perf_counter() - begin

    default_warmup, default_repetitions = (10, 50) if device == "cuda" else (1, 10)
    warmup = default_warmup if warmup is None else warmup
    repetitions = default_repetitions if repetitions is None else repetitions
    for _ in range(warmup):
        timed()
    samples, deadline = [], time.perf_counter() + budget
    while len(samples) < repetitions and (len(samples) < 3 or time.perf_counter() < deadline):
        samples.append(timed()[0])
    record.update(warmup=warmup, seconds=samples, median_seconds=float(np.median(samples)))
    if device == "cuda":
        record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    record["cpu_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    config.write_json(config.runs("cost/time", row_name(method, device, mode) + ".json"), record)
    print({k: v for k, v in record.items() if k != "seconds"}, flush=True)
    return record


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# Check of the PyTorch port ---------------------------------------------------------

def port_reference(height=256, width=384):
    """CPU renderer on a crop of the timing frame: white noise, its field, the noisy image."""
    from pgnm import PGNM, load
    model = PGNM(load(sidd_preset_path()), workers=THREADS)
    margin = np.shape(model.params["kernels"])[0] - 1
    noise = np.random.default_rng(7).standard_normal((height + margin, width + margin, 3), dtype=np.float32)
    field = model._build_noise_field((height, width, 3), np.random.default_rng(7))  # filters exactly `noise`
    clean = clean_frame()[:height, :width].astype(np.float32) / 255
    noisy = np.empty_like(clean)
    model._render_noise_window(field.astype(np.float32), clean, 0, 0, noisy)
    target = config.runs("cost/port-reference.npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, noise=noise, field=field, clean=clean, noisy=noisy)
    print(target)


def port_check(device="cuda"):
    """The port against the CPU reference: same white noise, then the same field."""
    import torch
    from .torch_pgnm import TorchPGNM
    data = np.load(config.runs("cost/port-reference.npz"))
    port = TorchPGNM(config.read_json(sidd_preset_path()), device)

    def to_device(image):  # (H, W, 3) -> (3, H, W) float32
        return torch.from_numpy(np.ascontiguousarray(image, np.float32)).to(device).permute(2, 0, 1)

    def to_host(image):
        return image.permute(1, 2, 0).cpu().numpy()

    with torch.no_grad():
        field = port.filter(to_device(data["noise"])[None])
        clean = to_device(data["clean"])
        noisy = to_host(port.add_noise(to_device(data["field"]), clean))  # given the CPU field
        end_to_end = to_host(port.add_noise(field, clean))  # given the port's own field
    reference = to_uint8(data["noisy"])
    result = dict(crop=list(data["clean"].shape[:2]),
                  field_max_abs_error=float(np.abs(to_host(field) - data["field"]).max()),
                  field_std=float(data["field"].std()),
                  noisy_max_abs_error=float(np.abs(noisy - data["noisy"]).max()),
                  uint8_mismatch_fraction=float((to_uint8(noisy) != reference).mean()),
                  end_to_end_uint8_mismatch_fraction=float((to_uint8(end_to_end) != reference).mean()))
    config.write_json(config.runs("cost/port-check.json"), result)
    print(result)
    return result


# Table ---------------------------------------------------------------------------

def table(root=None):
    """Tab. 4 and the derived numbers quoted in the text (None where a file is missing)."""
    root = Path(root or config.runs("cost"))

    def seconds(method, device, mode=None, key="median_seconds"):
        path = root / "time" / (row_name(method, device, mode) + ".json")
        return config.read_json(path)[key] if path.exists() else None

    def learned(method):
        path = root / "parameters" / (method + ".json")
        return config.read_json(path)["learned"] if path.exists() else None

    counts = presets_parameters()
    rows = {method: dict(per_model=learned(method), models=1, gpu=seconds(method, "cuda"), cpu=seconds(method, "cpu"))
            for method in NEURAL}
    rows["zhang23"] = dict(counts["zhang23"], gpu=None, cpu=seconds("zhang23", "cpu"))
    rows["pgnm_sidd"] = dict(counts["pgnm_sidd"])
    rows["pgnm_film"] = dict(counts["pgnm_film"])
    for mode, _ in PGNM_MODES:
        rows["pgnm_" + mode] = dict(gpu=seconds("pgnm", "cuda", mode), cpu=seconds("pgnm", "cpu", mode))
    # Bank construction is the setup of the playback rows.
    rows["pgnm_bank"] = dict(gpu=seconds("pgnm", "cuda", "playback", "setup_seconds"),
                             cpu=seconds("pgnm", "cpu", "playback", "setup_seconds"))

    derived = {}
    pgnm = counts["pgnm_sidd"]["per_model"]
    larger = [m for m in NEURAL if rows[m]["per_model"] and rows[m]["per_model"] > pgnm]
    derived["size_ratio"] = {m: rows[m]["per_model"] / pgnm for m in larger}
    derived["pgnm_sidd_total"] = pgnm * counts["pgnm_sidd"]["models"]
    for device in ("gpu", "cpu"):
        new, play, bank = (rows["pgnm_" + k][device] for k in ("new-field", "playback", "bank"))
        if None not in (new, play, bank):
            derived[device + "_bank_amortized_frames"] = bank / (new - play)
        others = [rows[m][device] for m in NEURAL if rows[m][device] is not None]
        if new is not None and others:
            derived[device + "_speedup_over_neural"] = [min(others) / new, max(others) / new]
    result = dict(rows=rows, derived=derived)
    config.write_json(root / "table.json", result)
    return result


def print_table(result):
    rows, derived = result["rows"], result["derived"]

    def number(value):
        return "--" if value is None else f"{value:,}"

    def gpu_ms(seconds, digits=1):
        return "--" if seconds is None else "%.*f" % (digits, 1000 * seconds)

    def cpu_s(seconds):
        if seconds is None:
            return "--"
        return "%.2f" % seconds if seconds >= 0.1 else "%.3f" % seconds

    def line(label, parameters=None, models=None, gpu="", cpu=""):
        print("%-28s%12s%8s%10s%10s" % (label, parameters or "", models or "", gpu, cpu))

    line("Method", "Per model", "Models", "GPU ms", "CPU s")
    for title, methods, pgnm in (("SIDD (16 camera-ISO classes)", SIDD_BASELINES, "pgnm_sidd"),
                                 ("Boolean film grain (8 radii)", FILM_BASELINES, "pgnm_film")):
        print(title)
        for m in methods:
            line("  " + LABELS[m], number(rows[m]["per_model"]), str(rows[m]["models"]),
                 gpu_ms(rows[m]["gpu"]), cpu_s(rows[m]["cpu"]))
        line("  PGNM (ours)", number(rows[pgnm]["per_model"]), str(rows[pgnm]["models"]))
        if pgnm == "pgnm_sidd":
            for mode, label in PGNM_MODES:
                line("    " + label, gpu=gpu_ms(rows["pgnm_" + mode]["gpu"]), cpu=cpu_s(rows["pgnm_" + mode]["cpu"]))
            line("    bank construction", gpu=gpu_ms(rows["pgnm_bank"]["gpu"], 0), cpu=cpu_s(rows["pgnm_bank"]["cpu"]))
    print()
    print("PGNM on gray film: %s distinct values per radius" % number(rows["pgnm_film"]["distinct_per_model"]))
    print("PGNM total over the SIDD classes: %s" % number(derived["pgnm_sidd_total"]))
    print("larger neural models / PGNM: " + ", ".join(
        "%s %.1fx" % (LABELS[m], r) for m, r in derived["size_ratio"].items()))
    for device in ("gpu", "cpu"):
        if device + "_speedup_over_neural" in derived:
            low, high = derived[device + "_speedup_over_neural"]
            print("%s: new field per frame is %.1f-%.1fx faster than the neural baselines" % (
                device.upper(), low, high))
        if device + "_bank_amortized_frames" in derived:
            print("%s: the bank is amortized after %.1f frames" % (
                device.upper(), derived[device + "_bank_amortized_frames"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("parameters", "time", "port-reference", "port-check", "table"))
    parser.add_argument("--method", choices=METHODS, default="pgnm")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=[m for m, _ in PGNM_MODES], default="playback", help="PGNM rows only")
    parser.add_argument("--warmup", type=int, help="warm-up calls (default 10 GPU, 1 CPU)")
    parser.add_argument("--repetitions", type=int, help="timed calls (default 50 GPU, 10 CPU)")
    parser.add_argument("--budget", type=float, default=900.0,
                        help="seconds after which timing stops early, once at least 3 calls are timed")
    parser.add_argument("--crop", type=int, nargs=2, default=(256, 384), metavar=("H", "W"),
                        help="crop of the timing frame for port-reference")
    args = parser.parse_args()
    if args.stage == "parameters":
        if args.method not in NEURAL:
            raise SystemExit("PGNM and Zhang23 are counted from presets/ by the table stage")
        count(args.method, args.device)
    elif args.stage == "time":
        measure(args.method, args.device, args.mode, args.warmup, args.repetitions, args.budget)
    elif args.stage == "port-reference":
        port_reference(*args.crop)
    elif args.stage == "port-check":
        port_check(args.device)
    else:
        print_table(table())


if __name__ == "__main__":
    main()
