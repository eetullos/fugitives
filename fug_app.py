# Fugitive Emissions Estimation v1.6
# - Removed "Preview of parsed data" section
# - avg_emission_rate passed to the PDF in kg/hr (mean of input rates)
# - Sidebar PDF button via placeholder (reliable)
# - Instructions .txt + two-column CSV ("label", "Emission Rate (kg/hr)")
# - SMALL figures for screen; ORIGINAL-SIZE for PDF
# - Gray sidebar only; no theme hacks (use .streamlit/config.toml)

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import base64
import io

from generate_pdf_report import generate_pdf_report  # your existing helper

# -------- Version helpers (single source of truth) --------
APP_NAME = "Fugitive Emissions Estimator"
DEFAULT_VERSION = "1.6"

def get_app_version(default: str = DEFAULT_VERSION) -> str:
    """Resolve app version from file, env var, or fallback default."""
    for candidate in ("VERSION", "version.txt", ".version"):
        try:
            if os.path.exists(candidate):
                v = open(candidate, "r", encoding="utf-8").read().strip()
                if v:
                    return v
        except Exception:
            pass
    return os.getenv("FUG_APP_VERSION", default)

APP_VERSION = get_app_version()

# -------- Page --------
st.set_page_config(page_title="Fugitive Emissions Estimator", page_icon="🧪", layout="wide")

# -------- Sidebar styling (scoped) --------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] > div:first-child { background-color:#f5f6f8; }
    [data-testid="stSidebar"] { border-right:1px solid #e2e4e7; }
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stDownloadButton,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stFileUploader { margin-bottom:0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------- Header --------
encoded_logo = None
try:
    with open("eemdl_logo.png", "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
except Exception:
    pass

st.markdown(
    """
    <div style='display:flex;align-items:center;gap:14px;'>
        %s
        <h3 style='margin:0;'>%s v%s</h3>
    </div>
    <hr style='margin-top:8px;'>
    """ % (
        f"<img src='data:image/png;base64,{encoded_logo}' style='height:48px;'>" if encoded_logo else "",
        APP_NAME,
        APP_VERSION,
    ),
    unsafe_allow_html=True,
)

# -------- Session --------
today = datetime.today().strftime("%Y-%m-%d")
default_filename = f"Fugitives_Report_v{APP_VERSION}_{today}.pdf"
st.session_state.setdefault("pdf_bytes", b"")
st.session_state.setdefault("last_summary", "")

# -------- Sidebar --------
st.sidebar.title("Estimation Input Options")

# Instructions (.txt)
instructions_txt = """FUGITIVE EMISSIONS INPUT INSTRUCTIONS
------------------------------------
Required column:
  - "Emission Rate (kg/hr)" : numeric emission rate in kilograms per hour

Optional column:
  - "label" : short name/ID for the row (string)

Notes:
  - One record per row.
  - Do not include units in cells (numbers only).
  - Lines beginning with '#' in your CSV will be ignored.
  - UTF-8 encoding is recommended.
"""
st.sidebar.markdown("### 📖 How to format your CSV")
st.sidebar.download_button(
    "Download Instructions (.txt)",
    data=instructions_txt.encode("utf-8"),
    file_name="fugitives_instructions.txt",
    mime="text/plain",
)

# Two-column template
template_csv = """label,Emission Rate (kg/hr)
Well-001,1.2
Well-002,0.9
Tank-Bay-A,3.4
Unit-17,2.1
"""
st.sidebar.markdown("### 📄 Download CSV Template")
st.sidebar.download_button(
    "Download Template CSV",
    data=template_csv.encode("utf-8"),
    file_name="fugitives_template.csv",
    mime="text/csv",
)

# Upload
st.sidebar.markdown("### 📤 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (label optional; 'Emission Rate (kg/hr)' required)",
    type=["csv"],
)

# Duration & iterations
st.sidebar.markdown("### ⏱️ Duration & Iterations")
HOURS_MIN = int(st.sidebar.number_input("Min duration (hours)", 1, 24 * 365, 24, step=1))
HOURS_MAX = int(st.sidebar.number_input("Max duration (hours)", HOURS_MIN, 24 * 365, 24 * 90, step=1))
n_iterations = int(st.sidebar.number_input("Bootstrap iterations", 100, 200_000, 10_000, step=500))

# Output filename
output_filename = st.sidebar.text_input("Output PDF Filename", value=default_filename)

# Sidebar PDF button SLOT (placeholder; replaced after run)
st.sidebar.markdown("### 📄 Report")
pdf_slot = st.sidebar.empty()
pdf_slot.download_button(
    label="📄 Download / Print PDF",
    data=b"",
    file_name=output_filename,
    mime="application/pdf",
    disabled=True,
    key="dl_sidebar_disabled",
)

# Last run summary (if any)
if st.session_state.get("last_summary"):
    with st.sidebar.expander("Last run summary", expanded=False):
        st.text(st.session_state["last_summary"])

# -------- Main instructions --------
st.markdown(
    """
Upload a CSV with **Emission Rate (kg/hr)** (and optionally **label**).  
Click **Run Estimation** to bootstrap population emissions and generate a PDF report.
"""
)

# -------- Helpers --------
def parse_upload(_uploaded):
    df = pd.read_csv(_uploaded, comment="#")
    df.columns = [c.strip() for c in df.columns]
    preferred = [
        "Emission Rate (kg/hr)",
        "emission_kg_per_hr", "emission_kg/hr", "kg/hr", "kg_per_hr", "emissions", "emission",
    ]
    emission_col = next((c for c in preferred if c in df.columns), None)
    if emission_col is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            raise ValueError("No numeric column found. Include 'Emission Rate (kg/hr)'.")
        emission_col = num_cols[0]

    emis = pd.to_numeric(df[emission_col], errors="coerce")
    valid = emis.notna()
    arr = emis[valid].astype(float).to_numpy()
    if arr.size == 0:
        raise ValueError(f"Column '{emission_col}' has no numeric data.")
    labels = df.loc[valid, "label"].fillna("").astype(str).to_numpy() if "label" in df.columns else None
    return labels, arr

def to_bytes(obj) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    for attr in ("getvalue", "getbuffer"):
        if hasattr(obj, attr):
            try:
                if hasattr(obj, "seek"): obj.seek(0)
            except Exception:
                pass
            try:
                return obj.getvalue()
            except Exception:
                try:
                    return bytes(obj.getbuffer())
                except Exception:
                    pass
    return b""

# -------- CTA --------
run_clicked = st.button("Run Estimation", type="primary")

# -------- Core --------
if run_clicked:
    try:
        st.session_state["pdf_bytes"] = b""

        if not uploaded_file:
            st.error("Please upload a CSV file (see the template in the sidebar).")
            st.stop()

        labels, fugitives_data = parse_upload(uploaded_file)
        N = len(fugitives_data)

        # --- Bootstrap ---
        kg_to_mt = 1.0 / 1000.0
        rng = np.random.default_rng()  # set a fixed seed here if you want reproducibility
        sample_emissions = rng.choice(fugitives_data, size=(n_iterations, N), replace=True)  # kg/hr
        sample_durations = rng.uniform(HOURS_MIN, HOURS_MAX, size=(n_iterations, N))        # hr
        mt = sample_emissions * sample_durations * kg_to_mt  # -> metric tons
        total_population_mt = mt.mean(axis=1) * N            # mean over sample, scaled to population size
        mean_durations = sample_durations.mean(axis=1)

        mean_mt = float(np.mean(total_population_mt))
        lower_mt = float(np.percentile(total_population_mt, 2.5))
        upper_mt = float(np.percentile(total_population_mt, 97.5))
        avg_duration = float(mean_durations.mean())

        # On-screen summary
        st.subheader("Summary Statistics")
        st.write(f"**Population size:** {N}")
        st.write(f"**Estimated Total Emissions:** {mean_mt:.2f} metric tons")
        st.write(f"**95% Confidence Interval:** [{lower_mt:.2f}, {upper_mt:.2f}]")
        st.write(f"**Average Duration (hours):** {avg_duration:.1f}")

        # ---- Figures: small for screen ----
        screen_figs = []

        fig1s, ax1s = plt.subplots(figsize=(6.0, 3.8))
        ax1s.hist(total_population_mt, bins=30, edgecolor="black")
        ax1s.set_title("Bootstrapped Population Emissions (Metric Tons)")
        ax1s.set_xlabel("Total Emissions (mt)")
        ax1s.set_ylabel("Frequency")
        ax1s.grid(True, alpha=0.2)
        screen_figs.append(fig1s)
        st.pyplot(fig1s, clear_figure=True)

        fig2s, ax2s = plt.subplots(figsize=(6.0, 3.4))
        ax2s.boxplot(total_population_mt, vert=False)
        ax2s.set_title("Boxplot of Bootstrapped Total Emissions")
        ax2s.set_xlabel("Total Emissions (mt)")
        ax2s.grid(True, axis="x", alpha=0.2)
        screen_figs.append(fig2s)
        st.pyplot(fig2s, clear_figure=True)

        fig3s, ax3s = plt.subplots(figsize=(6.0, 3.8))
        ax3s.scatter(total_population_mt, mean_durations, alpha=0.3, s=10)
        ax3s.set_title("Avg Duration vs. Bootstrapped Total Emissions")
        ax3s.set_xlabel("Total Emissions (mt)")
        ax3s.set_ylabel("Avg Duration (hours)")
        ax3s.grid(True, alpha=0.2)
        screen_figs.append(fig3s)
        st.pyplot(fig3s, clear_figure=True)

        # ---- Figures: ORIGINAL sizes for PDF ----
        pdf_figs = []

        fig1p, ax1p = plt.subplots(figsize=(8.27, 5.0), dpi=100)
        ax1p.hist(total_population_mt, bins=30, edgecolor="black")
        ax1p.set_title("Bootstrapped Population Emissions (Metric Tons)")
        ax1p.set_xlabel("Total Emissions (mt)")
        ax1p.set_ylabel("Frequency")
        ax1p.grid(True, alpha=0.2)
        pdf_figs.append(fig1p)

        fig2p, ax2p = plt.subplots(figsize=(8.27, 4.5), dpi=100)
        ax2p.boxplot(total_population_mt, vert=False)
        ax2p.set_title("Boxplot of Bootstrapped Total Emissions")
        ax2p.set_xlabel("Total Emissions (mt)")
        ax2p.grid(True, axis="x", alpha=0.2)
        pdf_figs.append(fig2p)

        fig3p, ax3p = plt.subplots(figsize=(8.27, 5.0), dpi=100)
        ax3p.scatter(total_population_mt, mean_durations, alpha=0.3, s=10)
        ax3p.set_title("Avg Duration vs. Bootstrapped Total Emissions")
        ax3p.set_xlabel("Total Emissions (mt)")
        ax3p.set_ylabel("Avg Duration (hours)")
        ax3p.grid(True, alpha=0.2)
        pdf_figs.append(fig3p)

        # ---- Summary text for sidebar and PDF metadata ----
        summary_text = (
            f"{APP_NAME} v{APP_VERSION} — Fugitives Estimation Summary\n"
            "----------------------------------------------\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Inputs\n"
            f"Population size: {N}\n"
            f"Bootstrap iterations: {n_iterations}\n"
            f"Duration range: {HOURS_MIN} to {HOURS_MAX} hours\n\n"
            "Results\n"
            f"Estimated total emissions (mt): {mean_mt:.2f}\n"
            f"95% CI (mt): [{lower_mt:.2f}, {upper_mt:.2f}]\n"
            f"Average duration (hours): {avg_duration:.1f}\n"
            f"Average emission rate (kg/hr): {float(np.mean(fugitives_data)):.3f}\n"
        )
        st.text(summary_text)
        st.session_state["last_summary"] = summary_text

        # ---- Generate PDF using your original helper ----
        stats = {
            "PC_count": N,
            "timesteps": n_iterations,
            "S0": None, "p_gas": None, "p": None, "r": None,
            # Ensure avg emission rate is kg/hr (mean of the input rates)
            "avg_emission_rate": float(np.mean(fugitives_data)),     # kg/hr
            "final_cumulative_emission": mean_mt,                    # mt
            "ci_lower": lower_mt, "ci_upper": upper_mt,
            "labels": (labels.tolist() if isinstance(labels, np.ndarray) else None),
            # >>> version keys so the PDF can read them <<<
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "version": APP_VERSION,  # alt key for compatibility
        }

        pdf_obj = generate_pdf_report(pdf_figs, stats, fugitives_data, np.array([]))
        pdf_bytes = to_bytes(pdf_obj)

        # Minimal fallback if helper returns empty
        if len(pdf_bytes) == 0:
            from matplotlib.backends.backend_pdf import PdfPages
            buf = io.BytesIO()
            with PdfPages(buf) as pdf:
                for f in pdf_figs:
                    pdf.savefig(f, bbox_inches="tight")
            pdf_bytes = buf.getvalue()

        st.session_state["pdf_bytes"] = pdf_bytes

        # Replace the sidebar button in-place (enabled now)
        pdf_slot.download_button(
            label="📄 Download / Print PDF",
            data=st.session_state["pdf_bytes"],
            file_name=output_filename,
            mime="application/pdf",
            disabled=False,
            key="dl_sidebar_enabled",
        )

        # Duplicate main-area button (optional convenience)
        st.success(f"PDF ready · {len(pdf_bytes)//1024} KB")
        st.download_button(
            "📄 Download / Print PDF",
            data=st.session_state["pdf_bytes"],
            file_name=output_filename,
            mime="application/pdf",
            key="dl_main",
        )

        # Footer
        st.markdown("<hr>", unsafe_allow_html=True)
        year = datetime.now().year
        st.markdown(
            f"<div style='text-align:center; color:gray;'>"
            f"© {year} The University of Texas at Austin · Licensed under Apache License 2.0"
            f"</div>",
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.session_state["pdf_bytes"] = b""
        st.error(f"An error occurred: {e}")
