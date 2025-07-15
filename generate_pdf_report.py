#Fugitive Emissions Estimation v1.0

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


def add_branded_elements(fig, page_num, timestamp=None):
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    import os

    fig_width, fig_height = fig.get_size_inches()
    dpi = fig.get_dpi()

    # --- Footer text positioning (safe above page edge) ---
    footer_text = "Fugitive Emissions Estimation v1.0 | Energy Emissions Modeling and Data Lab"
    if timestamp:
        footer_text += f" | Generated: {timestamp}"
    fig.text(0.5, 0.03, footer_text, ha='center', fontsize=8, color='gray')
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

    with PdfPages(pdf_buffer) as pdf:
        # --- Render each passed-in figure ---
        for fig in figs:
            # Prevent logo overlap and improve spacing
            fig.subplots_adjust(top=0.80, bottom=0.15, left=0.1, right=0.95)
            add_branded_elements(fig, page_counter)
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
            #f"Initial Properly Operating Fraction (S0): {stats.get('S0', 'N/A'):.2f}" if isinstance(stats.get('S0'), float) else f"S0: {stats.get('S0', 'N/A')}",
            f"Gas Conversion Factor (p_gas): {stats.get('p_gas', 'N/A'):.6f}" if isinstance(stats.get('p_gas'), float) else f"p_gas: {stats.get('p_gas', 'N/A')}",
            #f"Transition Probabilities: p = {stats.get('p', 'N/A'):.4f}, r = {stats.get('r', 'N/A'):.4f}" if all(isinstance(stats.get(k), float) for k in ['p', 'r']) else f"p: {stats.get('p')}, r: {stats.get('r')}",
            f"Average Emission Rate: {stats.get('avg_emission_rate', 'N/A'):.3f} scfh" if isinstance(stats.get('avg_emission_rate'), float) else f"avg_emission_rate: {stats.get('avg_emission_rate', 'N/A')}",
            f"Final Cumulative Emissions: {stats.get('final_cumulative_emission', 'N/A'):.2f} metric tons" if isinstance(stats.get('final_cumulative_emission'), float) else f"final_cumulative_emission: {stats.get('final_cumulative_emission', 'N/A')}",
            f"Report generated: {timestamp}"
        ]
        for i, line in enumerate(summary_lines):
            plt.text(0.05, 0.95 - i * 0.035, line, va='top', fontsize=10, family='monospace')
        add_branded_elements(summary_fig, page_counter, timestamp)
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
                    add_branded_elements(list_fig, page_counter, timestamp)
                    pdf.savefig(list_fig)
                    plt.close(list_fig)
                    page_counter += 1

    pdf_buffer.seek(0)
    return pdf_buffer
