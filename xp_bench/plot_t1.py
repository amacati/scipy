"""Scaling plot for the T1 distance validators across frameworks."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "distance_results"
COLORS = {
    ("torch", "cpu"): "#ff6e34",
    ("torch", "gpu"): "#a41900",
    ("jax", "cpu"): "#51B854",
    ("jax", "gpu"): "#065A09",
    ("cupy", "gpu"): "#9B28AF",
    ("numpy", "cpu"): "#052b59",
}


def load(fw, dev, fn):
    f = R / fw / dev / f"{fn}.json"
    return json.loads(f.read_text()) if f.exists() else {}


fns = ["is_valid_dm", "num_obs_dm"]
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, fn in zip(axes, fns):
    for (fw, dev), color in COLORS.items():
        d = load(fw, dev, fn)
        if not d:
            continue
        ks = sorted(map(int, d))
        ts = [min(d[str(k)]) * 1e6 for k in ks]
        ax.plot(ks, ts, marker="o", color=color, label=f"{fw} {dev}")
    ax.set_title(fn)
    ax.set_xlabel("matrix dimension k")
    ax.set_ylabel("time (µs), min of 10")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

fig.suptitle("T1 distance validators: scaling across frameworks")
plt.tight_layout()
out = Path(__file__).parent / "reports" / "t1_validators"
out.parent.mkdir(exist_ok=True)
plt.savefig(f"{out}.png", format="png")
print(f"saved {out}.png")
