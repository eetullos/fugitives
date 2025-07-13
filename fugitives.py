import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io

# Title
st.title("Fugitive Emissions Estimator")

# Instructions
st.markdown("""
Upload a CSV file with a column of fugitive emission rates (in kg/hr).  
The tool will simulate total population emissions using bootstrapping and generate a downloadable PDF report.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Constants
HOURS_MIN = 24
HOURS_MAX = 24 * 90
kg_to_mt = 1 / 1000  # kg to metric tons
n_iterations = 10000

if uploaded_file:
    try:
        # Read user-uploaded CSV
        df = pd.read_csv(uploaded_file)
        fugitives_data = df.iloc[:, 0].dropna().astype(float).values  # Use first column

        population_size = len(fugitives_data)

        # Run bootstrap
        bootstrap_population_totals_mt = []
        bootstrap_mean_durations = []

        for _ in range(n_iterations):
            sample_emissions = np.random.choice(fugitives_data, size=population_size, replace=True)
            sample_durations = np.random.uniform(HOURS_MIN, HOURS_MAX, size=population_size)
            kg = sample_emissions * sample_durations
            mt = kg * kg_to_mt
            sample_mean_mt = np.mean(mt)
            total_population_mt = sample_mean_mt * population_size
            bootstrap_population_totals_mt.append(total_population_mt)
            bootstrap_mean_durations.append(np.mean(sample_durations))

        bootstrap_population_totals_mt = np.array(bootstrap_population_totals_mt)
        bootstrap_mean_durations = np.array(bootstrap_mean_durations)

        # Summary stats
        mean_mt = np.mean(bootstrap_population_totals_mt)
        lower_mt = np.percentile(bootstrap_population_totals_mt, 2.5)
        upper_mt = np.percentile(bootstrap_population_totals_mt, 97.5)

        # Display summary
        st.subheader("Summary Statistics")
        st.write(f"**Population size:** {population_size}")
        st.write(f"**Estimated Total Emissions:** {mean_mt:.2f} metric tons")
        st.write(f"**95% Confidence Interval:** [{lower_mt:.2f}, {upper_mt:.2f}]")

        # Generate PDF in memory
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:

            # Histogram
            fig1, ax1 = plt.subplots()
            ax1.hist(bootstrap_population_totals_mt, bins=30, edgecolor='black')
            ax1.set_title("Bootstrapped Population Emissions (Metric Tons)")
            ax1.set_xlabel("Total Emissions (mt)")
            ax1.set_ylabel("Frequency")
            summary_text = (
                f"Mean: {mean_mt:.2f} mt\n"
                f"Population: {population_size}\n"
                f"95% CI: [{lower_mt:.2f}, {upper_mt:.2f}]"
            )
            ax1.annotate(summary_text, xy=(0.95, 0.95), xycoords='axes fraction',
                         fontsize=9, ha='right', va='top',
                         bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
            pdf.savefig(fig1)
            plt.close(fig1)

            # Boxplot
            fig2, ax2 = plt.subplots()
            ax2.boxplot(bootstrap_population_totals_mt, vert=False)
            ax2.set_title("Boxplot of Bootstrapped Total Emissions")
            ax2.set_xlabel("Total Emissions (mt)")
            pdf.savefig(fig2)
            plt.close(fig2)

            # Scatter
            fig3, ax3 = plt.subplots()
            ax3.scatter(bootstrap_population_totals_mt, bootstrap_mean_durations, alpha=0.3, s=10)
            ax3.set_title("Avg Duration vs. Bootstrapped Total Emissions")
            ax3.set_xlabel("Total Emissions (mt)")
            ax3.set_ylabel("Avg Duration (hours)")
            pdf.savefig(fig3)
            plt.close(fig3)

        st.success("Analysis complete. You can now download your report.")

        # Prepare download
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="Fugitives_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
