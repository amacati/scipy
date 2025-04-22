from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import fire
from typing import Optional


def load_results(results_dir: Path | str = "results"):
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


def plot_results(results_dir: Path | str = "results", save_path: str = ""):
    """Plot benchmark results, creating a separate figure for each function."""
    all_results = load_results(Path(__file__).parent / results_dir)
    save_path = Path(__file__).parent / "plots" / save_path

    # Define colors for each XP type and device combination
    colors = {
        "numpy|cpu": "#1f77b4",  # blue
        "torch|cpu": "#ff7f0e",  # orange
        "torch|gpu": "#d62728",  # red
        "jax|cpu": "#2ca02c",  # green
        "jax|gpu": "#9467bd",  # purple
    }

    for fn_name, fn_results in all_results.items():
        # Create a new figure for each function
        plt.figure(figsize=(10, 8))

        # Merge xp and device keys
        merged_results = {}
        for xp, xp_data in fn_results.items():
            for device, device_data in xp_data.items():
                merged_key = f"{xp}|{device}"
                merged_results[merged_key] = device_data

        for xp_device, timings in merged_results.items():
            means = []
            std_devs = []
            for n_samples, timing in sorted(timings.items()):
                means.append(np.mean(timing))
                std_devs.append(np.std(timing))
            sample_sizes = sorted(timings.keys())
            sample_sizes = [int(s) for s in sample_sizes]

            plt.errorbar(
                sample_sizes,
                means,
                yerr=std_devs,
                label=xp_device,
                color=colors.get(xp_device),
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
        plt.show()

        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            # Save each figure with the function name
            plt.savefig(save_dir / f"{fn_name}.png")
            plt.close()
        else:
            plt.close()


if __name__ == "__main__":
    fire.Fire(plot_results)
