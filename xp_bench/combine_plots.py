from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas


def combine_plots_to_pdf(
    plots_dir: str = "plots", output_name: str = "benchmark_results.pdf"
):
    """Combine all PNG plots into a single PDF file."""
    plots_dir = Path(__file__).parent / plots_dir
    output_path = Path(__file__).parent / output_name

    # Get all SVG files
    plot_files = sorted(plots_dir.glob("*.svg"))

    if not plot_files:
        print("No plot files found in the specified directory.")
        return

    # Create a PDF document
    page_width = 11.69 * 72  # A4 width in points
    page_height = 8.27 * 72  # A4 height in points
    c = canvas.Canvas(str(output_path))
    c.setPageSize((page_width, page_height))

    # Add a title page
    c.setFont("Helvetica", 20)
    c.drawCentredString(
        page_width / 2, page_height / 2, "scipy.spatial.transform benchmark"
    )
    c.showPage()

    # Add each plot to the PDF
    for plot_file in plot_files:
        drawing = svg2rlg(str(plot_file))
        if drawing:
            # Scale to fit page with margins
            margin = 36  # 0.5 inch margin
            scale = min(
                (page_width - 2 * margin) / drawing.width,
                (page_height - 2 * margin) / drawing.height,
            )
            drawing.scale(scale, scale)

            # Center on page
            x = (page_width - drawing.width * scale) / 2
            y = (page_height - drawing.height * scale) / 2

            renderPDF.draw(drawing, c, x, y)
            c.showPage()
        else:
            print(f"Failed to process {plot_file}")

    c.save()


if __name__ == "__main__":
    combine_plots_to_pdf()
