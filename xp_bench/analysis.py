"""Baseline-vs-optimized analysis of benchmark results.

Three views over `{results}/{framework}[_baseline]/{device}/{fn}.json`:

  python analysis.py table from_rotvec as_rotvec pow
  python analysis.py stability from_rotvec --results distance_results
  python analysis.py significance inv

`table` prints the speedup at the largest shared size; `stability` adds the coefficient of
variation (std/mean) as a noise gauge over the largest three sizes; `significance` runs a
one-sided Mann-Whitney U test per size and gives a per-config verdict.
"""
import json
from pathlib import Path

import fire
import numpy as np
from scipy.stats import mannwhitneyu

FRAMEWORKS = ["numpy", "torch", "jax", "cupy"]
DEVICES = ["cpu", "gpu"]
DEFAULT_FNS = ["from_rotvec", "as_rotvec", "pow"]


def _load(results, fw, device, fn):
    f = Path(__file__).parent / results / fw / device / f"{fn}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _pairs(results, fn):
    """Yield (label, baseline_dict, optimized_dict) for every framework/device with both."""
    for fw in FRAMEWORKS:
        for device in DEVICES:
            opt = _load(results, fw, device, fn)
            base = _load(results, f"{fw}_baseline", device, fn)
            if opt and base:
                yield f"{fw} {device}", base, opt


def table(*fns, results="rotation_results"):
    """Speedup (baseline mean / optimized mean) at the largest shared size."""
    for fn in fns or DEFAULT_FNS:
        print(f"\n### {fn}")
        print(f"{'config':<14} {'size':>10} {'base ms':>12} {'opt ms':>10} {'speedup':>9}")
        for label, base, opt in _pairs(results, fn):
            sizes = sorted(set(opt) & set(base), key=int)
            if not sizes:
                continue
            n = sizes[-1]
            tb, to = np.mean(base[n]) * 1e3, np.mean(opt[n]) * 1e3
            print(f"{label:<14} {n:>10} {tb:>12.3f} {to:>10.3f} {tb / to:>8.2f}x")


def stability(*fns, results="rotation_results"):
    """Mean, CV (std/mean), and speedup over the largest three shared sizes."""
    for fn in fns or DEFAULT_FNS:
        print(f"\n### {fn}")
        print(f"{'config':<14} {'size':>9} {'base ms':>9} {'baseCV':>7} "
              f"{'opt ms':>9} {'optCV':>7} {'speedup':>8}")
        for label, base, opt in _pairs(results, fn):
            for n in sorted(set(opt) & set(base), key=int)[-3:]:
                b, o = np.array(base[n]) * 1e3, np.array(opt[n]) * 1e3
                print(f"{label:<14} {n:>9} {b.mean():>9.4f} {b.std() / b.mean():>7.3f} "
                      f"{o.mean():>9.4f} {o.std() / o.mean():>7.3f} {b.mean() / o.mean():>7.2f}x")


def significance(*fns, results="rotation_results"):
    """One-sided Mann-Whitney U per size, with a per-config median-speedup verdict."""
    for fn in fns or DEFAULT_FNS:
        print(f"\n### {fn}")
        print(f"{'config':<12} {'size':>9} {'base ms':>9} {'opt ms':>9} "
              f"{'speedup':>8} {'p':>9} {'sig':>4}")
        for label, base, opt in _pairs(results, fn):
            speedups = []
            for n in sorted(set(opt) & set(base), key=int):
                b, o = np.array(base[n]) * 1e3, np.array(opt[n]) * 1e3
                p = mannwhitneyu(o, b, alternative="less").pvalue
                sp = b.mean() / o.mean()
                speedups.append(sp)
                sig = "yes" if p < 0.05 else "no"
                print(f"{label:<12} {n:>9} {b.mean():>9.4f} {o.mean():>9.4f} "
                      f"{sp:>7.2f}x {p:>9.1e} {sig:>4}")
            if speedups:
                med = np.median(speedups)
                verdict = ("REAL gain" if med > 1.05 else
                           "REAL regression" if med < 0.95 else "no change")
                print(f"  -> {label}: median {med:.2f}x  {verdict}")


if __name__ == "__main__":
    fire.Fire({"table": table, "stability": stability, "significance": significance})
