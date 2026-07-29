"""Figures for one suite, one file per case.

Every case gets a 2x2 grid with one panel per framework. The panels share their axes so
the frameworks stay comparable, colour keys the framework and device, and the line style
keys the variant.
"""

from glob import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from xp_bench import core

matplotlib.use("Agg")

PALETTE = {
    "numpy cpu": "#052b59",
    "torch cpu": "#ff6e34",
    "torch gpu": "#a41900",
    "jax cpu": "#51B854",
    "jax gpu": "#065A09",
    "cupy gpu": "#9B28AF",
}
LINESTYLES = {"current": "-", "baseline": "--"}


def load_series(mirror, fn, variants):
    """Load every timing on disk for one case.

    Args:
        mirror: Scipy module path of the suite, e.g. "spatial/transform/rotation".
        fn: Case name.
        variants: Variant names to look for.

    Returns:
        {(framework, device): {variant: {size: [seconds per call]}}}, holding only the
        configs and variants that exist on disk.
    """
    series = {}
    for variant in variants:
        for path in glob(str(core.result_path(mirror, variant, "*", "*", fn))):
            xp, device = Path(path).parts[-3:-1]
            timings = core.load_result(mirror, variant, xp, device, fn)
            series.setdefault((xp, device), {})[variant] = timings
    return series


def figure_path(mirror, fn, fmt):
    """Path of one case figure, mirroring the scipy module tree."""
    return core.ROOT / "plots" / mirror / f"{fn}.{fmt}"


def _colors(configs):
    """Colour per "framework device", extending the palette for unknown configs."""
    extra = plt.get_cmap("tab10").colors
    unknown = [c for c in configs if c not in PALETTE]
    return PALETTE | {c: extra[i % len(extra)] for i, c in enumerate(unknown)}


def figure(mirror, fn, variants):
    """Draw the comparison figure for one case, or None when nothing is on disk.

    Args:
        mirror: Scipy module path of the suite, e.g. "spatial/transform/rotation".
        fn: Case name.
        variants: Variant names to draw.

    Returns:
        The matplotlib figure, or None if the case has no results.
    """
    series = load_series(mirror, fn, variants)
    if not series:
        return None
    frameworks = sorted({xp for xp, _ in series}, key=core.FRAMEWORKS.index)
    assert len(frameworks) <= 4, f"a 2x2 grid holds four frameworks, got {frameworks}"
    colors = _colors([f"{xp} {device}" for xp, device in series])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    for ax, framework in zip(axes.flat, frameworks):
        for (xp, device), by_variant in sorted(series.items()):
            if xp != framework:
                continue
            for variant, timings in sorted(by_variant.items()):
                sizes = sorted(timings, key=int)
                ax.errorbar(
                    [int(n) for n in sizes],
                    [np.mean(timings[n]) for n in sizes],
                    yerr=[np.std(timings[n]) for n in sizes],
                    label=f"{xp} {device} {variant}",
                    color=colors[f"{xp} {device}"],
                    linestyle=LINESTYLES.get(variant, ":"),
                    marker="o",
                    capsize=5,
                )
        ax.set_title(f"{fn} - {framework}")
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Time (seconds)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
    for ax in axes.flat[len(frameworks) :]:
        ax.set_visible(False)
    fig.suptitle(f"{mirror.replace('/', '.')}.{fn}")
    fig.tight_layout()
    return fig


def plot(mirror, fns, variants):
    """Write one figure per case as png and svg, mirroring the scipy module tree.

    Args:
        mirror: Scipy module path of the suite, e.g. "spatial/transform/rotation".
        fns: Case names to draw.
        variants: Variant names to compare, e.g. ["current", "baseline"].

    Returns:
        The figure paths written. Cases without any results are skipped.
    """
    paths = []
    for fn in fns:
        fig = figure(mirror, fn, variants)
        if fig is None:
            continue
        for fmt in ("png", "svg"):
            path = figure_path(mirror, fn, fmt)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, format=fmt)
            paths.append(path)
        plt.close(fig)
    return paths
