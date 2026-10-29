"""Output folder and Matplotlib setup shared by the figure scripts."""
from __future__ import annotations

from pathlib import Path

from experiments import config

# No software names or creation date in the PDFs, so a rerun gives the same file.
PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None}


def output(name, folder=None):
    """runs/figures/<name> (or <folder>/<name>), with its parent folder created."""
    path = Path(folder or config.runs("figures")) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pyplot():
    """Matplotlib's pyplot on the non-interactive Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def save(fig, path, dpi):
    """Write the figure as <path>.pdf and as a <path>.png preview at dpi; close it."""
    from matplotlib import pyplot as plt
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, metadata=PDF_METADATA)
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    plt.close(fig)
    print("wrote", path.with_suffix(".pdf"))
