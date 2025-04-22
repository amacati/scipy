from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def combine_plots_to_pdf(
    plots_dir: str = "plots", output_name: str = "benchmark_results.pdf"
):
    """Combine all PNG plots into a single PDF file."""
    plots_dir = Path(__file__).parent / plots_dir
    output_path = Path(__file__).parent / output_name

    # Get all PNG files
    plot_files = sorted(plots_dir.glob("*.png"))

    if not plot_files:
        print("No plot files found in the specified directory.")
        return

    with PdfPages(output_path) as pdf:
        # Add a title page
        plt.figure(figsize=(11.69, 8.27))  # A4 size
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            "scipy.spatial.transform benchmark",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
        )
        pdf.savefig()
        plt.close()

        # Add each plot to the PDF
        for plot_file in plot_files:
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 size
            img = plt.imread(plot_file)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    combine_plots_to_pdf()
