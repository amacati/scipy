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
        "torch cpu": "#ff6e34",
        "torch gpu": "#a41900",
        "jax cpu": "#51B854",
        "jax gpu": "#065A09",
        "cupy gpu": "#9B28AF",
        "numpy cpu": "#052b59",
    }

    for fn_name, fn_results in all_results.items():
        # Create a new figure for each function
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Define framework order for subplots
        frameworks = ["numpy", "torch", "jax", "cupy"]

        # Merge xp and device keys
        merged_results = {}
        for xp, xp_data in fn_results.items():
            for device, device_data in xp_data.items():
                merged_key = f"{xp} {device}"
                merged_results[merged_key] = device_data

        for i, focus_framework in enumerate(frameworks):
            ax = axes[i]

            for xp_device, timings in merged_results.items():
                means = []
                std_devs = []
                for _, timing in sorted(timings.items()):
                    means.append(np.mean(timing))
                    std_devs.append(np.std(timing))
                sample_sizes = [int(s) for s in sorted(timings.keys())]

                # Determine if this is the focus framework
                is_focus = focus_framework in xp_device

                if is_focus:
                    color = colors.get(xp_device)
                    alpha = 1.0
                else:
                    color = "gray"
                    alpha = 0.3

                if "jax_native" in xp_device:
                    if focus_framework != "jax":
                        continue
                    linestyle = "--"
                    color = colors.get(xp_device.replace("jax_native", "jax"))
                else:
                    linestyle = "-"

                ax.errorbar(
                    sample_sizes,
                    means,
                    yerr=std_devs,
                    label=xp_device if is_focus else None,
                    color=color,
                    alpha=alpha,
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
                if any(focus_framework in key for key in merged_results.keys()):
                    ax.legend()

        if "rotation" in results_dir:
            fig.suptitle(f"Rotation.{fn_name}")
        else:
            fig.suptitle(f"RigidTransform.{fn_name}")
        plt.tight_layout()

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
