# Fugitive Emissions Estimation v1.0 — styled to match pneumatics.py UI
# - Base64 logo header + version
# - Sidebar "Estimation Input Options" with uploads/downloads/params
# - Main "Run Estimation" button
# - Sequential figures, summary text
# - Branded PDF download
# - Footer

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import base64

from generate_pdf_report import generate_pdf_report  # uses your existing helper

# ---------- Header with base64-embedded logo ----------
encoded_logo = None
try:
    with open("eemdl_logo.png", "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode()
except Exception:
    pass

title_html = """
    <div style='display: flex; align-items: center;'>
        {logo}
        <h3 style='margin: 0;'>Fugitive Emissions Estimator v1.0</h3>
    </div>
""".format(
    logo=f"<img src='data:image/png;base64,{encoded_logo}' style='height: 50px; margin-right: 15px;'>"
    if encoded_logo
    else ""
)

st.markdown(title_html, unsafe_allow_html=True)
st.markdown("<hr style='margin-top:0'>", unsafe_allow_html=True)

# ---------- Sidebar: Estimation Input Options ----------
st.sidebar.title("Estimation Input Options")

# Choose data source (to mirror your pneumatics UX)
data_source = st.sidebar.radio(
    "Choose Input:",
    options=["Upload CSV (first column = kg/hr)", "Use sample data (synthetic)"],
    index=0,
)

# CSV template / example download
st.sidebar.markdown("### 📅 Download CSV Template")
template_df = pd.DataFrame({"emission_kg_per_hr": [0.5, 1.2, 0.9, 3.4, 2.1]})
st.sidebar.download_button(
    "Download Template CSV",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="fugitives_template.csv",
    mime="text/csv",
)

uploaded_file = None
population_size_input = None

if data_source.startswith("Upload"):
    st.sidebar.markdown("### 📤 Upload Your CSV")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
else:
    st.sidebar.markdown("### 🔧 Sample Data Settings")
    population_size_input = st.sidebar.number_input(
        "Population size (records)", min_value=10, max_value=100000, value=500, step=10
    )
    sample_mean = st.sidebar.number_input(
        "Synthetic mean emission (kg/hr)", min_value=0.0, value=1.0, step=0.1
    )
    sample_std = st.sidebar.number_input(
        "Synthetic std dev (kg/hr)", min_value=0.0, value=0.8, step=0.1
    )

# Duration & Monte Carlo
st.sidebar.markdown("### ⏱️ Duration and Iterations")
HOURS_MIN = int(st.sidebar.number_input("Min duration (hours)", min_value=1, max_value=24*365, value=24, step=1))
HOURS_MAX = int(st.sidebar.number_input("Max duration (hours)", min_value=HOURS_MIN, max_value=24*365, value=24*90, step=1))
n_iterations = int(st.sidebar.number_input("Bootstrap iterations", min_value=100, max_value=200000, value=10000, step=500))

# Output file name (match pneumatics)
today = datetime.today().strftime("%Y-%m-%d")
default_filename = f"Fugitives_Report_{today}.pdf"
output_filename = st.sidebar.text_input("Output PDF Filename", value=default_filename)

# ---------- Main area instructions ----------
st.markdown("""
Upload a CSV file with a **single column** of fugitive emission rates (in kg/hr) **or** use synthetic sample data.  
Click **Run Estimation** to simulate total population emissions via bootstrapping and generate a PDF report.
""")

# ---------- Run button (main area, to mirror pneumatics) ----------
run_clicked = st.button("Run Estimation")

# ---------- Processing ----------
if run_clicked:
    try:
        # Load or synthesize data
        if data_source.startswith("Upload"):
            if not uploaded_file:
                st.error("Please upload a CSV or switch to sample data.")
                st.stop()
            df = pd.read_csv(uploaded_file)
            fugitives_data = df.iloc[:, 0].dropna().astype(float).values
        else:
            # Synthetic normal; clip at 0 to avoid negative emissions
            rng = np.random.default_rng(seed=42)
            fugitives_data = rng.normal(loc=sample_mean, scale=sample_std, size=population_size_input)
            fugitives_data = np.clip(fugitives_data, a_min=0, a_max=None)

        population_size = int(len(fugitives_data))
        if population_size == 0:
            st.error("No data rows found. Provide at least one emission value.")
            st.stop()

        # Constants
        kg_to_mt = 1.0 / 1000.0  # kg -> metric tons

        # Vectorized bootstrap for speed
        rng = np.random.default_rng()
        # (n_iterations, population_size)
        sample_emissions = rng.choice(fugitives_data, size=(n_iterations, population_size), replace=True)
        sample_durations = rng.uniform(HOURS_MIN, HOURS_MAX, size=(n_iterations, population_size))

        kg = sample_emissions * sample_durations
        mt = kg * kg_to_mt
        sample_mean_mt = mt.mean(axis=1)
        total_population_mt = sample_mean_mt * population_size

        # Summary stats
        mean_mt = float(np.mean(total_population_mt))
        lower_mt = float(np.percentile(total_population_mt, 2.5))
        upper_mt = float(np.percentile(total_population_mt, 97.5))
        mean_duration = float(sample_durations.mean())

        # Display summary (mirrors pneumatics "summary_text" approach)
        st.subheader("Summary Statistics")
        st.write(f"**Population size:** {population_size}")
        st.write(f"**Estimated Total Emissions:** {mean_mt:.2f} metric tons")
        st.write(f"**95% Confidence Interval:** [{lower_mt:.2f}, {upper_mt:.2f}]")
        st.write(f"**Average Duration (hours):** {mean_duration:.1f}")

        # ---------- Figures (three, like pneumatics) ----------
        figs = []

        # 1) Histogram of total population emissions
        fig1, ax1 = plt.subplots(figsize=(8.27, 5.0))
        ax1.hist(total_population_mt, bins=30, edgecolor='black')
        ax1.set_title("Bootstrapped Population Emissions (Metric Tons)")
        ax1.set_xlabel("Total Emissions (mt)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.2)
        figs.append(fig1)
        st.pyplot(fig1)

        # 2) Boxplot of total emissions
        fig2, ax2 = plt.subplots(figsize=(8.27, 4.5))
        ax2.boxplot(total_population_mt, vert=False)
        ax2.set_title("Boxplot of Bootstrapped Total Emissions")
        ax2.set_xlabel("Total Emissions (mt)")
        ax2.grid(True, axis='x', alpha=0.2)
        figs.append(fig2)
        st.pyplot(fig2)

        # 3) Scatter: avg duration vs total emissions (per iteration)
        fig3, ax3 = plt.subplots(figsize=(8.27, 5.0))
        ax3.scatter(total_population_mt, sample_durations.mean(axis=1), alpha=0.3, s=10)
        ax3.set_title("Avg Duration vs. Bootstrapped Total Emissions")
        ax3.set_xlabel("Total Emissions (mt)")
        ax3.set_ylabel("Avg Duration (hours)")
        ax3.grid(True, alpha=0.2)
        figs.append(fig3)
        st.pyplot(fig3)

        # ---------- Text summary (to mirror pneumatics "st.text" block) ----------
        summary_text = (
            "Fugitives Estimation Summary\n"
            "----------------------------\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Inputs\n"
            f"Population size: {population_size}\n"
            f"Bootstrap iterations: {n_iterations}\n"
            f"Duration range: {HOURS_MIN} to {HOURS_MAX} hours\n\n"
            "Results\n"
            f"Estimated total emissions (mt): {mean_mt:.2f}\n"
            f"95% CI (mt): [{lower_mt:.2f}, {upper_mt:.2f}]\n"
            f"Average duration (hours): {mean_duration:.1f}\n"
        )
        st.text(summary_text)

        # ---------- PDF generation (same helper you already use) ----------
        stats = {
            "PC_count": population_size,          # kept keys to align with your current helper
            "timesteps": n_iterations,
            "S0": None,
            "p_gas": None,
            "p": None,
            "r": None,
            "avg_emission_rate": mean_mt,
            "final_cumulative_emission": mean_mt,
            "ci_lower": lower_mt,
            "ci_upper": upper_mt
        }

        # Your existing signature: generate_pdf_report(figs, stats, fugitives_data, np.array([]))
        pdf_buffer = generate_pdf_report(figs, stats, fugitives_data, np.array([]))

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name=output_filename,
            mime="application/pdf",
        )

        # Footer (match the look of pneumatics)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center; color:gray;'>Fugitive Emissions Estimator v1.0 | LICENSE PLACEHOLDER</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
