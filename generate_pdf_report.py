# Fugitive Emissions Estimation v1.0

# generate_pdf_report.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import numpy as np
import os
from PIL import Image
from datetime import datetime


def wrap_list(label, values, wrap=10):
    lines = [f"{label} ({len(values)} values):"]
    for i in range(0, len(values), wrap):
        chunk = values[i:i + wrap]
        lines.append(", ".join(f"{v:.4f}" for v in chunk))
    return lines


def add_branded_elements(fig, page_num, timestamp=None, footer_text=None):
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    import os

    fig_width, fig_height = fig.get_size_inches()
    dpi = fig.get_dpi()

    # --- Footer text (uses passed-in footer_text if provided; falls back to original v1.0 string) ---
    base_footer = "Fugitive Emissions Estimation v1.0 | Energy Emissions Modeling and Data Lab"
    footer = footer_text if isinstance(footer_text, str) and footer_text.strip() else base_footer
    if timestamp:
        footer += f" | Generated: {timestamp}"

    # Safe bottom margin
    fig.text(0.5, 0.03, footer, ha='center', fontsize=8, color='gray')
    fig.text(0.98, 0.03, f"Page {page_num}", ha='right', fontsize=8, color='gray')

    # --- Logo ---
    logo_path = os.path.join(os.path.dirname(__file__), "eemdl_logo.png")
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")

            # Resize logo to a fixed width (e.g., 1.8 inches)
            max_width_in = 1.8
            target_width_px = int(max_width_in * dpi)
            scale = target_width_px / logo.width
            target_height_px = int(logo.height * scale)

            logo_resized = logo.resize((target_width_px, target_height_px), Image.LANCZOS)
            logo_arr = np.array(logo_resized) / 255.0

            # Place logo top center with margin
            xo = int((fig_width * dpi - logo_resized.width) / 2)
            yo = int(fig_height * dpi - logo_resized.height - 10)

            fig.figimage(logo_arr, xo=xo, yo=yo, origin='upper', zorder=10)
        except Exception as e:
            print(f"Error rendering logo: {e}")


def generate_pdf_report(figs, stats, prop_rates, malf_rates):
    pdf_buffer = BytesIO()
    page_counter = 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- NEW: build footer text from app_version/version if provided ----
    app_name = stats.get("app_name", "Fugitive Emissions Estimation")
    app_version = stats.get("app_version") or stats.get("version") or "1.0"
    footer_text = f"{app_name} v{app_version} | Energy Emissions Modeling and Data Lab"

    with PdfPages(pdf_buffer) as pdf:
        # --- Render each passed-in figure ---
        for fig in figs:
            # Prevent logo overlap and improve spacing
            fig.subplots_adjust(top=0.80, bottom=0.15, left=0.1, right=0.95)
            add_branded_elements(fig, page_counter, timestamp, footer_text=footer_text)
            pdf.savefig(fig)
            plt.close(fig)
            page_counter += 1

        # --- Summary Page ---
        summary_fig = plt.figure(figsize=(8.27, 11.69))  # A4
        plt.axis('off')
        summary_lines = [
            "Simulation Summary:",
            f"Total Components Simulated: {stats.get('PC_count', 'N/A')}",
            f"Timesteps Simulated: {stats.get('timesteps', 'N/A')}",
            f"Average Emission Rate: {stats.get('avg_emission_rate', 'N/A'):.3f} kg/hr" if isinstance(stats.get('avg_emission_rate'), float) else f"avg_emission_rate: {stats.get('avg_emission_rate', 'N/A')}",
            f"Final Cumulative Emissions: {stats.get('final_cumulative_emission', 'N/A'):.2f} metric tons" if isinstance(stats.get('final_cumulative_emission'), float) else f"final_cumulative_emission: {stats.get('final_cumulative_emission', 'N/A')}",
            f"Report generated: {timestamp}"
        ]
        for i, line in enumerate(summary_lines):
            plt.text(0.05, 0.95 - i * 0.035, line, va='top', fontsize=10, family='monospace')
        add_branded_elements(summary_fig, page_counter, timestamp, footer_text=footer_text)
        pdf.savefig(summary_fig)
        plt.close(summary_fig)
        page_counter += 1

        # --- Emission Rate Listings ---
        for label, values in [("Component Emission Rates", prop_rates),
                              ("Malfunctioning Rates", malf_rates)]:
            if values is not None and len(values) > 0:
                lines = wrap_list(label, values)
                lines_per_page = 80
                for i in range(0, len(lines), lines_per_page):
                    chunk = lines[i:i + lines_per_page]
                    list_fig = plt.figure(figsize=(8.27, 11.69))  # A4
                    plt.axis('off')
                    for j, line in enumerate(chunk):
                        plt.text(0.05, 0.95 - j * 0.025, line, va='top', fontsize=9, family='monospace')
                    add_branded_elements(list_fig, page_counter, timestamp, footer_text=footer_text)
                    pdf.savefig(list_fig)
                    plt.close(list_fig)
                    page_counter += 1

    pdf_buffer.seek(0)
    return pdf_buffer
