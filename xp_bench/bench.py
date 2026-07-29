#!/usr/bin/env python
r"""Single entry point for the scipy array API benchmarks.

The benchmarks time the scipy you built, so they run through spin:

    pixi run -e all-frameworks spin run --build-dir=build-all-frameworks \
        python xp_bench/bench.py run --module spatial/transform/rotation \
        --fn as_rotvec --xp jax --device gpu --low 0 --high 5 --variant baseline

Run `xp_bench/bench.py --help` for the subcommands. Every path this writes mirrors the
scipy module tree, so spatial/distance results land in results/spatial/distance.
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # must precede any scipy import

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Appended, not prepended: the repo root holds the scipy source tree, which would
# shadow the built scipy that spin puts on the path ahead of us.
sys.path.append(str(Path(__file__).parent.parent))

from xp_bench import core, registry  # noqa: E402


def _modules(module):
    """Registered mirrors, filtered by a path prefix such as spatial/transform."""
    cases = registry.discover()
    mirrors = [m for m in sorted(cases) if module is None or m.startswith(module)]
    assert mirrors, f"No suite matches {module}. Run `bench list` to see them."
    return {m: cases[m] for m in mirrors}


def _names(cases, fn):
    assert fn is None or fn in cases, f"Unknown case {fn}, have {sorted(cases)}"
    return [fn] if fn else sorted(cases)


def run(args):
    for mirror, cases in _modules(args.module).items():
        core.sweep(
            mirror,
            cases,
            _names(cases, args.fn),
            [args.xp] if args.xp else core.FRAMEWORKS,
            [args.device] if args.device else core.DEVICES,
            args.low,
            args.high,
            args.repeat,
            args.number,
            args.variant,
            args.append,
        )


def plot(args):
    from xp_bench import plots

    for mirror, cases in _modules(args.module).items():
        for path in plots.plot(mirror, _names(cases, args.fn), args.variant):
            print(path)


def report(args):
    from xp_bench import report as report_module

    for mirror, cases in _modules(args.module).items():
        print(report_module.report(mirror, _names(cases, args.fn), args.variant))


def show(args):
    for mirror, cases in _modules(args.module).items():
        print(f"{mirror} ({len(cases)} cases)")
        for name in sorted(cases):
            print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, helptext in [
        ("run", run, "time cases and store the results"),
        ("plot", plot, "draw one figure per case"),
        ("report", report, "write a markdown summary"),
        ("list", show, "show the registered suites and cases"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.set_defaults(handler=handler)
        p.add_argument("--module", help="mirror path prefix, e.g. spatial/distance")
        p.add_argument("--fn", help="single case name, default all")
        if name == "run":
            p.add_argument("--xp", choices=core.FRAMEWORKS, help="default all")
            p.add_argument("--device", choices=core.DEVICES, help="default both")
            p.add_argument("--low", type=int, default=0, help="log10 smallest size")
            p.add_argument("--high", type=int, default=7, help="log10 largest size")
            p.add_argument("--repeat", type=int, default=5, help="samples per size")
            p.add_argument("--number", type=int, default=100, help="calls per sample")
            p.add_argument(
                "--variant",
                default="current",
                help="name of the result tree to write into, so a reference run on main"
                " and a run on the branch can be compared later",
            )
            p.add_argument(
                "--append",
                action="store_true",
                help="add to the stored samples instead of replacing them, so"
                " repeated runs accumulate across processes",
            )
        if name in ("plot", "report"):
            p.add_argument(
                "--variant",
                nargs="+",
                default=["current", "baseline"],
                help="result trees to compare",
            )

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
