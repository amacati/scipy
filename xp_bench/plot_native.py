"""Plot the native-vs-bridge cdist prototype results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "torch cpu": "#ff6e34", "torch gpu": "#a41900",
    "jax cpu": "#51B854", "jax gpu": "#065A09",
    "cupy gpu": "#9B28AF", "numpy cpu": "#052b59",
}

data = json.loads((Path(__file__).parent / "native_prototype_results.json").read_text())
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
for ax, (cfg, series) in zip(axes.flatten(), data.items()):
    ms = sorted(int(m) for m in series)
    native = [series[str(m)]["native"] * 1e3 for m in ms]
    bridge = [series[str(m)]["bridge"] * 1e3 for m in ms]
    color = COLORS[cfg]
    ax.plot(ms, native, marker="o", color=color, label="native (array API)")
    ax.plot(ms, bridge, marker="s", ls="--", color=color, label="convert to numpy")
    ax.set_title(cfg)
    ax.set_xlabel("observations m (features=10)")
    ax.set_ylabel("cdist time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
fig.suptitle("Native array-API cdist vs convert-to-numpy (euclidean)")
plt.tight_layout()
out = Path(__file__).parent / "reports" / "native_cdist"
out.parent.mkdir(exist_ok=True)
plt.savefig(f"{out}.png", format="png")
print(f"saved {out}.png")
