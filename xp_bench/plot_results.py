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
        "scipy main": "#8dbff2",
        "XP enabled": "#082644",
        "XP disabled": "#1873cd",
        "torch cpu": "#e67446",
        "torch gpu": "#eb5036",
        "jax cpu": "#469E49",
        "jax gpu": "#2F6B32",
        "cupy gpu": "#9B28AF",
        "optimized XP disabled": "#e30202",
        "optimized XP enabled": "#4a0101",
    }

    for fn_name, fn_results in all_results.items():
        # Create a new figure for each function
        plt.figure(figsize=(10, 8))

        # Merge xp and device keys
        merged_results = {}
        for xp, xp_data in fn_results.items():
            for device, device_data in xp_data.items():
                merged_key = f"{xp} {device}"
                merged_results[merged_key] = device_data

        for xp_device, timings in merged_results.items():
            means = []
            std_devs = []
            for _, timing in sorted(timings.items()):
                means.append(np.mean(timing))
                std_devs.append(np.std(timing))
            sample_sizes = sorted(timings.keys())
            sample_sizes = [int(s) for s in sample_sizes]

            if xp_device == "numpy_scipy cpu":  # Replace with more accurate label
                xp_device = "scipy main"  # "scipy ENH #22777"
            if xp_device == "numpy_xp cpu":
                xp_device = "XP enabled"
            if xp_device == "numpy_no_xp cpu":
                xp_device = "XP disabled"
            if xp_device == "numpy_opt_no_xp cpu":
                xp_device = "optimized XP disabled"
            if xp_device == "numpy_opt_xp cpu":
                xp_device = "optimized XP enabled"
            color = colors.get(xp_device)

            plt.errorbar(
                sample_sizes,
                means,
                yerr=std_devs,
                label=xp_device,
                color=color,
                marker="o",
                capsize=5,
            )

        plt.title(fn_name)
        plt.xlabel("Number of samples")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")

        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            # Save each figure with the function name
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
