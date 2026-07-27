from pathlib import Path
from multiprocessing import Pool
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fire

# Color per framework and device combination.
COLORS = {
    "torch cpu": "#ff6e34",
    "torch gpu": "#a41900",
    "jax cpu": "#51B854",
    "jax gpu": "#065A09",
    "cupy gpu": "#9B28AF",
    "numpy cpu": "#052b59",
}


def load_results(results_dir: Path | str = "rotation_results"):
    """Load all benchmark results from JSON files."""
    results_dir = Path(results_dir)
    all_results = {}

    # Iterate through all json files in results directory
    for result_file in results_dir.rglob("*.json"):
        # Parse path components
        xp = result_file.parent.parent.name
        device = result_file.parent.name
        fn_name = result_file.stem

        # Initialize dictionary structure
        if fn_name not in all_results:
            all_results[fn_name] = {}
        if xp not in all_results[fn_name]:
            all_results[fn_name][xp] = {}
        if device not in all_results[fn_name][xp]:
            all_results[fn_name][xp][device] = {}

        # Load and parse results
        with open(result_file, "r") as f:
            results = json.load(f)
        all_results[fn_name][xp][device] = results

    return all_results


def _plot_one(fn_name, fn_results, results_dir, save_dir):
    """Render and save the four-panel figure for a single function."""
    # The panels share axes so that the frameworks stay comparable without
    # drawing each other's series.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    frameworks = ["numpy", "torch", "jax", "cupy"]

    for ax, focus_framework in zip(axes, frameworks):
        for xp, xp_data in sorted(fn_results.items()):
            baseline = xp.endswith("_baseline")
            framework = xp.removesuffix("_baseline")
            native = framework.endswith("_native")
            framework = framework.removesuffix("_native")
            if framework != focus_framework:
                continue

            for device, timings in sorted(xp_data.items()):
                means = []
                std_devs = []
                for _, timing in sorted(timings.items()):
                    means.append(np.mean(timing))
                    std_devs.append(np.std(timing))
                sample_sizes = [int(s) for s in sorted(timings.keys())]

                if baseline:
                    label, linestyle = f"{framework} baseline {device}", "--"
                elif native:
                    label, linestyle = f"{framework} native {device}", ":"
                else:
                    label, linestyle = f"{framework} {device}", "-"

                ax.errorbar(
                    sample_sizes,
                    means,
                    yerr=std_devs,
                    label=label,
                    color=COLORS[f"{framework} {device}"],
                    linestyle=linestyle,
                    marker="o",
                    capsize=5,
                )

        ax.set_title(f"{fn_name} - {focus_framework.capitalize()}")
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Time (seconds)")
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    kind = "Rotation" if "rotation" in results_dir else "RigidTransform"
    fig.suptitle(f"{kind}.{fn_name}")
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{fn_name}.png", format="png")
    plt.savefig(save_dir / f"{fn_name}.svg", format="svg")
    plt.close(fig)


def plot_results(
    results_dir: Path | str = "rotation_results", save_path: str = "rotation_plots"
):
    """Plot benchmark results, creating a separate figure for each function."""
    all_results = load_results(Path(__file__).parent / results_dir)
    save_dir = Path(__file__).parent / save_path

    with Pool() as pool:
        pool.starmap(
            _plot_one,
            [(fn, res, str(results_dir), save_dir) for fn, res in all_results.items()],
        )


def main(rot: bool = True, tf: bool = True):
    if rot:
        plot_results()
    if tf:
        plot_results(results_dir="tf_results", save_path="tf_plots")


if __name__ == "__main__":
    fire.Fire(main)
