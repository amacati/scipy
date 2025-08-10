from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import fire


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


def plot_results(
    results_dir: Path | str = "rotation_results", save_path: str = "rotation_plots"
):
    """Plot benchmark results, creating a separate figure for each function."""
    all_results = load_results(Path(__file__).parent / results_dir)
    save_path = Path(__file__).parent / save_path

    # Define colors for each XP type and device combination
    colors = {
        "torch cpu": "#e67446",
        "torch gpu": "#eb5036",
        "jax cpu": "#469E49",
        "jax gpu": "#2F6B32",
        "cupy gpu": "#9B28AF",
        "numpy cpu": "#1152a3",
    }

    for fn_name, fn_results in all_results.items():
        # Create a figure with 4 subfigures
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        axes = {"numpy": ax1, "torch": ax2, "jax": ax3, "cupy": ax4}

        # First pass to find global min/max values
        global_xmin = float("inf")
        global_xmax = float("-inf")
        global_ymin = float("inf")
        global_ymax = float("-inf")

        # Plot data for each framework separately
        for framework, ax in axes.items():
            # Get both optimized and unoptimized results
            framework_data = {k: v for k, v in fn_results.items() if framework in k}

            for xp, xp_data in framework_data.items():
                for device, device_data in xp_data.items():
                    means = []
                    std_devs = []
                    for _, timing in sorted(device_data.items()):
                        means.append(np.mean(timing))
                        std_devs.append(np.std(timing))
                    sample_sizes = sorted(device_data.keys())
                    sample_sizes = [int(s) for s in sample_sizes]

                    # Update global min/max
                    global_xmin = min(global_xmin, min(sample_sizes))
                    global_xmax = max(global_xmax, max(sample_sizes))
                    global_ymin = min(global_ymin, min(means))
                    global_ymax = max(global_ymax, max(means))

                    color = colors.get(f"{framework} {device}")
                    label = f"{device} " + ("optimized" if "_no_opt" not in xp else "")
                    linestyle = "-" if "_no_opt" not in xp else "--"
                    if framework == "jax" and "_native" in xp:
                        label = f"{device} native"
                        linestyle = ":"

                    ax.errorbar(
                        sample_sizes,
                        means,
                        yerr=std_devs,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker="o",
                        capsize=5,
                    )

        # Set consistent limits for all axes
        for framework, ax in axes.items():
            ax.set_xlim(global_xmin * 0.9, global_xmax * 1.1)
            ax.set_ylim(global_ymin * 0.9, global_ymax * 1.1)

            ax.set_title(f"{framework} - {fn_name}")
            ax.set_xlabel("Number of samples")
            ax.set_ylabel("Time (seconds)")
            ax.grid(True)
            ax.legend()
            ax.set_xscale("log")
            ax.set_yscale("log")

        plt.suptitle(f"Rotation.{fn_name}", fontsize=20)
        plt.tight_layout()

        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"{fn_name}.png", format="png")
            plt.savefig(save_dir / f"{fn_name}.svg", format="svg")
            plt.close()
        else:
            plt.close()


def main(rot: bool = True, tf: bool = True):
    if rot:
        plot_results()
    if tf:
        plot_results(results_dir="tf_results", save_path="tf_plots")


if __name__ == "__main__":
    fire.Fire(main)
