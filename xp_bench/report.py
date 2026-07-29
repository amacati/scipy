"""Markdown and PDF summaries of one suite's benchmark results.

The markdown compares the variants per case and per framework/device config, quoting
both the minimum and the mean over the repeats of every measurement, and stitches in the
figures that plots.py writes. The PDF holds the same figures in one file for sharing.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

from xp_bench import core, plots

COLUMNS = "|---:|---:|---:|---:|---:|---:|---:|---:|:--|"


def _comparison(config, by_variant, variant, reference):
    """Markdown block for one framework/device config, plus the sizes it compared."""
    label = " ".join(config)
    test, base = by_variant.get(variant, {}), by_variant.get(reference, {})
    shared = sorted(set(test) & set(base), key=int)
    if not shared:
        missing = f"- **{label}**: no size shared by `{variant}` and `{reference}`"
        return [missing], set()

    slowest = max(np.mean(base[n]) for n in shared)
    scale, unit = (1e6, "us") if slowest < 1e-3 else (1e3, "ms")
    lines = [
        "",
        f"### {label}",
        "",
        f"| size | {variant} min {unit} | {reference} min {unit} | min speedup "
        f"| {variant} mean {unit} | {reference} mean {unit} | mean speedup "
        "| p | significant |",
        COLUMNS,
    ]
    speedups = []
    for n in shared:
        t, b = np.array(test[n]) * scale, np.array(base[n]) * scale
        p = mannwhitneyu(t, b, alternative="less").pvalue
        speedups.append(b.mean() / t.mean())
        lines.append(
            f"| {n} | {t.min():.4g} | {b.min():.4g} | {b.min() / t.min():.2f}x "
            f"| {t.mean():.4g} | {b.mean():.4g} | {speedups[-1]:.2f}x "
            f"| {p:.1e} | {'yes' if p < 0.05 else 'no'} |"
        )
    med = float(np.median(speedups))
    verdict = (
        "REAL gain" if med > 1.05 else "REAL regression" if med < 0.95 else "no change"
    )
    lines += ["", f"Median speedup {med:.2f}x, **{verdict}**", ""]
    return lines, set(shared)


def _header(mirror, variants, cases, sizes):
    """Preamble stating what was compared over which sizes."""
    span = f"{min(sizes, key=int)} to {max(sizes, key=int)}" if sizes else "none"
    others = ", ".join(f"`{v}`" for v in variants[1:])
    return [
        f"# {mirror.replace('/', '.')} benchmark",
        "",
        f"`{variants[0]}` compared against {others} for "
        f"{', '.join(cases) or 'no case with results'} at sample sizes {span}. Sizes "
        "present in only one variant are excluded from a comparison.",
        "",
        "Every measurement is a list of repeats of seconds per call. Both the minimum "
        "and the mean over that list are reported, minimum first, in the unit named in "
        f"each table header. Speedup divides the reference by `{variants[0]}`, so "
        f"above 1 means `{variants[0]}` is faster. Significance is a one-sided "
        "Mann-Whitney U test over the repeats, `yes` when p < 0.05.",
        "",
    ]


def report(mirror, fns, variants):
    """Write the markdown summary and the stitched PDF for one suite.

    Args:
        mirror: Scipy module path of the suite, e.g. "spatial/transform/rotation".
        fns: Case names to summarize.
        variants: Variant names, the first compared against every other.

    Returns:
        Path of the written summary.md.
    """
    assert len(variants) > 1, f"nothing to compare {variants} against"
    directory = core.ROOT / "reports" / mirror
    directory.mkdir(parents=True, exist_ok=True)

    body, cases, sizes, figures = [], [], set(), []
    for fn in fns:
        series = plots.load_series(mirror, fn, variants)
        if not series:
            continue
        cases.append(fn)
        figures.append(plots.figure(mirror, fn, variants))
        png = plots.figure_path(mirror, fn, "png")
        body += [f"## {fn}", "", f"![{fn}]({os.path.relpath(png, directory)})", ""]
        for config in sorted(series):
            for reference in variants[1:]:
                lines, compared = _comparison(
                    config, series[config], variants[0], reference
                )
                body += lines
                sizes |= compared
        body.append("")

    summary = directory / "summary.md"
    summary.write_text("\n".join(_header(mirror, variants, cases, sizes) + body))
    if figures:
        with PdfPages(directory / "benchmark.pdf") as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
    return summary
