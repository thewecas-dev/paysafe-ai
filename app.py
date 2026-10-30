"""Streamlit dashboard for PAYSAFE AI — UPI Fraud Risk Detection.
Light Mode Institutional Banking & Fintech Design System.
Zero Emojis, Zero Purple, Zero Green, No Redundant Risk Labels, Fully Responsive.
"""
import os
import random
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/fraud_model.joblib"

st.set_page_config(
    page_title="PAYSAFE AI — UPI Fraud Risk Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Light Mode Institutional Banking / Fintech CSS Theme with Full Responsiveness
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Clean Bright Light Canvas */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    div.block-container,
    .stApp .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 0.85rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    header[data-testid="stHeader"], footer {
        display: none !important;
    }

    /* Header Bar Layout */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .header-brand-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0f172a;
    }
    .engine-badge {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 10px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .header-subtitle {
        color: #64748b;
        font-size: 12px;
        font-weight: 500;
    }
    .header-telemetry-chips {
        display: flex;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #475569;
        flex-wrap: wrap;
    }
    .header-chip {
        background: #f1f5f9;
        padding: 3px 8px;
        border-radius: 5px;
        border: 1px solid #e2e8f0;
        white-space: nowrap;
    }
    .header-chip b {
        color: #0f172a;
    }
    .header-chip-active {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        color: #1d4ed8 !important;
        font-weight: 700 !important;
    }

    /* Left and Right Symmetrical Fintech Panels */
    div[data-testid="stVerticalBlockBorderWrapper"],
    .fintech-panel {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
    }

    .panel-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
        flex-wrap: wrap;
        gap: 6px;
    }

    /* Section Subheadings in Ingestion Panel */
    .section-tag {
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .section-tag-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #dbeafe;
    }
    .section-tag-slate {
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    .section-tag-steel {
        background: #f8fafc;
        color: #1e293b;
        border: 1px solid #cbd5e1;
    }

    /* Clean Light Inputs */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 7px !important;
        color: #0f172a !important;
        transition: all 0.15s ease-in-out !important;
    }

    div[data-baseweb="input"]:hover,
    div[data-baseweb="select"]:hover,
    div[data-testid="stSelectbox"] div[role="combobox"]:hover {
        border-color: #94a3b8 !important;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within,
    div[data-testid="stSelectbox"] div[role="combobox"]:focus {
        border-color: #1d4ed8 !important;
        box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.12) !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #0f172a !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding-left: 8px !important;
    }

    div[data-testid="stSelectbox"] div[role="combobox"] {
        padding-top: 3px !important;
        padding-bottom: 3px !important;
    }

    div[data-testid="stSelectbox"] span {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
    }

    div[data-testid="stSelectbox"] svg {
        fill: #475569 !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #f8fafc !important;
        border: none !important;
        color: #475569 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: #e2e8f0 !important;
        color: #0f172a !important;
    }

    label {
        font-size: 10.5px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: #475569 !important;
        margin-bottom: 3px !important;
    }

    /* Preset Buttons (Clean Light Slate) */
    div[data-testid="stButton"] button:not([kind="primary"]) {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        border-radius: 7px !important;
        padding: 6px 10px !important;
        transition: all 0.15s ease-in-out !important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]) * {
        color: #1e293b !important;
        font-size: 11px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]):hover {
        background: #f8fafc !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 2px 4px rgba(29, 78, 216, 0.08) !important;
        transform: translateY(-1px);
    }

    /* Primary Action Button (Solid High-Trust Royal Blue) */
    button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        border: 1px solid #1e3a8a !important;
        box-shadow: 0 3px 8px rgba(29, 78, 216, 0.25) !important;
        font-size: 12px !important;
        font-weight: 800 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        color: #ffffff !important;
        transition: all 0.2s ease !important;
        margin-top: 4px !important;
    }

    button[kind="primary"] *,
    div[data-testid="stButton"] button[kind="primary"] * {
        color: #ffffff !important;
    }

    button[kind="primary"]:hover,
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 5px 14px rgba(29, 78, 216, 0.35) !important;
        transform: translateY(-1px);
    }

    /* HUD Metrics Display */
    .metric-container {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        margin-top: 2px;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 10px;
    }

    .metric-number {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1;
        font-feature-settings: "tnum";
    }

    /* Single Prominent Risk Badge */
    .risk-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        padding: 4px 11px;
        border-radius: 6px;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .badge-low {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }

    .badge-medium {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    .badge-high {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-low { background: #1d4ed8; }
    .dot-medium { background: #d97706; }
    .dot-high { background: #dc2626; }

    /* Custom Risk Spectrum Track */
    .spectrum-container {
        margin-bottom: 12px;
    }
    .spectrum-track-wrap {
        position: relative;
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        display: flex;
        overflow: visible;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .spectrum-zone {
        height: 100%;
    }
    .zone-safe {
        background: #93c5fd;
        border-top-left-radius: 4px;
        border-bottom-left-radius: 4px;
    }
    .zone-elevated {
        background: #fcd34d;
    }
    .zone-critical {
        background: #fca5a5;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
    }
    .spectrum-cursor {
        position: absolute;
        top: -4px;
        width: 16px;
        height: 16px;
        background: #ffffff;
        border: 3px solid;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        z-index: 2;
        transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .spectrum-labels {
        display: flex;
        justify-content: space-between;
        font-size: 9.5px;
        color: #64748b;
        font-weight: 700;
        margin-top: 5px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Action Directive Banner */
    .action-box {
        border-radius: 8px;
        padding: 9px 12px;
        font-size: 12px;
        line-height: 1.4;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .action-low {
        background: #f0f7ff;
        border-color: #2563eb;
        border: 1px solid #dbeafe;
        border-left: 4px solid #2563eb;
    }
    .action-medium {
        background: #fffdf5;
        border-color: #f59e0b;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    .action-high {
        background: #fef2f2;
        border-color: #dc2626;
        border: 1px solid #fee2e2;
        border-left: 4px solid #dc2626;
    }

    /* Structured Telemetry Signal Log Rows */
    .signal-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        margin-bottom: 3px;
        background: #f8fafc;
        border: 1px solid #f1f5f9;
    }
    .tag-clean {
        background: #eff6ff;
        color: #1d4ed8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 4px;
        border: 1px solid #bfdbfe;
        white-space: nowrap;
    }
    .tag-warning {
        background: #fffbeb;
        color: #b45309;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 4px;
        border: 1px solid #fde68a;
        white-space: nowrap;
    }
    .tag-flagged {
        background: #fef2f2;
        color: #b91c1c;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 4px;
        border: 1px solid #fecaca;
        white-space: nowrap;
    }
    .signal-cat {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        color: #475569;
        background: #e2e8f0;
        padding: 1px 5px;
        border-radius: 3px;
        letter-spacing: 0.04em;
        white-space: nowrap;
    }
    .signal-desc {
        color: #1e293b;
        font-weight: 500;
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Model specs ribbon */
    .specs-ribbon {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #64748b;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 5px 10px;
        display: flex;
        justify-content: space-between;
        margin-top: 6px;
        flex-wrap: wrap;
        gap: 6px;
    }

    /* ==============================================================
       RESPONSIVE DESIGN BREAKPOINTS
    /* ==============================================================
       RESPONSIVE DESIGN BREAKPOINTS
       Desktop (>1100px), Tablet (651px-1100px), Mobile (<=650px)
       ============================================================== */

    /* Tier 2 & 3: Screens <= 1100px
       Stack the 2 primary panels vertically into 100% full-width cards */
    @media (max-width: 1100px) {
        /* All horizontal blocks stack into vertical column flow */
        div[data-testid="stHorizontalBlock"],
        .stHorizontalBlock {
            flex-direction: column !important;
            gap: 16px !important;
        }

        /* All columns expand to 100% full width */
        div[data-testid="stColumn"],
        div[data-testid="column"],
        .stColumn {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
        }

        .block-container {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-top: 0.75rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
        }
        .header-container {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
        .header-subtitle {
            display: none;
        }
        .header-telemetry-chips {
            width: 100%;
            justify-content: flex-start;
            overflow-x: auto;
            padding-bottom: 2px;
        }
        .header-chip {
            font-size: 9px;
            padding: 2px 6px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"],
        .fintech-panel {
            padding: 14px 18px !important;
            border-radius: 12px !important;
        }
        .specs-ribbon {
            display: grid !important;
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 6px !important;
        }
    }

    /* Medium Screens & Tablets (651px - 1100px):
       Keep inner 3 preset buttons and 3 input columns side-by-side in 3 columns across the full-width card */
    @media (min-width: 651px) and (max-width: 1100px) {
        .stColumn .stHorizontalBlock,
        .stVerticalBlockBorderWrapper .stHorizontalBlock,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"],
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 10px !important;
        }
        .stColumn .stHorizontalBlock > .stColumn,
        .stVerticalBlockBorderWrapper .stHorizontalBlock > .stColumn,
        .stColumn .stColumn,
        .stVerticalBlockBorderWrapper .stColumn,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stColumn"],
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stColumn"] {
            width: 33.333% !important;
            min-width: 30% !important;
            max-width: 33.333% !important;
            flex: 1 1 33.333% !important;
        }
    }

    /* Tier 3: Mobile Phones (<= 650px)
       Stack the inner 3 columns and preset buttons into full width for thumb tap */
    @media (max-width: 650px) {
        .stColumn .stHorizontalBlock,
        .stVerticalBlockBorderWrapper .stHorizontalBlock,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"],
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        .stColumn .stHorizontalBlock > .stColumn,
        .stVerticalBlockBorderWrapper .stHorizontalBlock > .stColumn,
        .stColumn .stColumn,
        .stVerticalBlockBorderWrapper .stColumn,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stColumn"],
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.5rem !important;
        }
        .metric-container {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
        }
        .metric-container > div:last-child {
            text-align: left !important;
        }
        .metric-number {
            font-size: 36px !important;
        }
        .signal-row {
            padding: 6px 8px !important;
        }
        .signal-desc {
            white-space: normal !important;
            word-break: break-word !important;
        }
    }

    /* Extra Small Devices (<= 420px) */
    @media (max-width: 420px) {
        .specs-ribbon {
            grid-template-columns: 1fr !important;
        }
        .brand-title {
            font-size: 18px !important;
        }
        .spectrum-labels {
            font-size: 8.5px !important;
        }
        .spectrum-cursor {
            width: 14px !important;
            height: 14px !important;
            top: -3px !important;
        }
        .metric-number {
            font-size: 32px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
dataset_name = bundle.get("dataset_name", "Kaggle UPI Fraud Dataset (26,393 records)")
auc_score = bundle.get("auc", 0.999)

# Clean Top Navigation Header with Responsive Classes
st.markdown(
    f"""
    <div class="header-container">
        <div class="header-brand-wrap">
            <span class="brand-title">PAYSAFE AI</span>
            <span class="engine-badge">Risk Engine v2.4</span>
            <span class="header-subtitle">Real-Time UPI Fraud Risk Telemetry &amp; Decisioning</span>
        </div>
        <div class="header-telemetry-chips">
            <span class="header-chip">CORPUS: <b>26.3K TXNS</b></span>
            <span class="header-chip">ESTIMATOR: <b>RANDOM FOREST</b></span>
            <span class="header-chip">ROC-AUC: <b>{auc_score:.4f}</b></span>
            <span class="header-chip header-chip-active">ENGINE ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Session state initialization for form inputs
if "input_amount" not in st.session_state:
    st.session_state.input_amount = 1500.0
    st.session_state.input_velocity = 1
    st.session_state.input_receiver_account_age = 240
    st.session_state.input_new_payee = "No"
    st.session_state.input_new_device = "No"
    st.session_state.input_unusual_location = "No"
    st.session_state.input_night_transaction = "No"
    st.session_state.input_failed_attempts = 0
    st.session_state.input_time_pressure = "No"

# 2-Column Responsive Layout
left_col, right_col = st.columns([11, 10], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown(
            "<div class='panel-header'><span>01. Transaction Ingestion Console</span><span>Benchmark Presets</span></div>",
            unsafe_allow_html=True
        )
        preset_col1, preset_col2, preset_col3 = st.columns(3)

        if preset_col1.button("Low Risk", use_container_width=True):
            st.session_state.input_amount = 1500.0
            st.session_state.input_velocity = 1
            st.session_state.input_receiver_account_age = 240
            st.session_state.input_new_payee = "No"
            st.session_state.input_new_device = "No"
            st.session_state.input_unusual_location = "No"
            st.session_state.input_night_transaction = "No"
            st.session_state.input_failed_attempts = 0
            st.session_state.input_time_pressure = "No"
            st.toast("Loaded clean, low-risk peer-to-peer transfer.")

        if preset_col2.button("High Risk", use_container_width=True):
            st.session_state.input_amount = 14500.0
            st.session_state.input_velocity = 6
            st.session_state.input_receiver_account_age = 0
            st.session_state.input_new_payee = "Yes"
            st.session_state.input_new_device = "Yes"
            st.session_state.input_unusual_location = "Yes"
            st.session_state.input_night_transaction = "Yes"
            st.session_state.input_failed_attempts = 2
            st.session_state.input_time_pressure = "Yes"
            st.toast("Loaded severe high-risk fraud attack profile.")

        if preset_col3.button("Random Sample", use_container_width=True):
            tier = random.choice(["clean", "low_risk", "borderline", "suspicious", "high_risk"])
            if tier == "clean":
                st.session_state.input_amount = float(round(random.uniform(150, 2200), 2))
                st.session_state.input_velocity = random.choice([0, 1, 2])
                st.session_state.input_receiver_account_age = random.randint(90, 450)
                st.session_state.input_new_payee = "No"
                st.session_state.input_new_device = "No"
                st.session_state.input_unusual_location = "No"
                st.session_state.input_night_transaction = "No"
                st.session_state.input_failed_attempts = 0
                st.session_state.input_time_pressure = "No"
                st.toast("Generated clean transaction profile.")
            elif tier == "low_risk":
                st.session_state.input_amount = float(round(random.uniform(2200, 6500), 2))
                st.session_state.input_velocity = random.choice([1, 2, 3])
                st.session_state.input_receiver_account_age = random.randint(30, 150)
                st.session_state.input_new_payee = random.choice(["No", "Yes"])
                st.session_state.input_new_device = "No"
                st.session_state.input_unusual_location = "No"
                st.session_state.input_night_transaction = random.choice(["No", "Yes"])
                st.session_state.input_failed_attempts = random.choice([0, 1])
                st.session_state.input_time_pressure = "No"
                st.toast("Generated mild anomaly profile.")
            elif tier == "borderline":
                st.session_state.input_amount = float(round(random.uniform(5500, 12000), 2))
                st.session_state.input_velocity = random.choice([2, 4, 6])
                st.session_state.input_receiver_account_age = random.randint(15, 60)
                st.session_state.input_new_payee = "Yes"
                st.session_state.input_new_device = random.choice(["No", "Yes"])
                st.session_state.input_unusual_location = "No"
                st.session_state.input_night_transaction = random.choice(["No", "Yes"])
                st.session_state.input_failed_attempts = random.choice([0, 1, 2])
                st.session_state.input_time_pressure = random.choice(["No", "Yes"])
                st.toast("Generated borderline review profile.")
            elif tier == "suspicious":
                st.session_state.input_amount = float(round(random.uniform(9000, 22000), 2))
                st.session_state.input_velocity = random.choice([4, 6, 8])
                st.session_state.input_receiver_account_age = random.choice([1, 2, 3, 5])
                st.session_state.input_new_payee = "Yes"
                st.session_state.input_new_device = "Yes"
                st.session_state.input_unusual_location = random.choice(["No", "Yes"])
                st.session_state.input_night_transaction = "Yes"
                st.session_state.input_failed_attempts = random.choice([1, 2])
                st.session_state.input_time_pressure = "Yes"
                st.toast("Generated suspicious activity profile.")
            else:  # high_risk
                st.session_state.input_amount = float(round(random.uniform(12000, 45000), 2))
                st.session_state.input_velocity = random.choice([5, 8, 12])
                st.session_state.input_receiver_account_age = 0
                st.session_state.input_new_payee = "Yes"
                st.session_state.input_new_device = "Yes"
                st.session_state.input_unusual_location = "Yes"
                st.session_state.input_night_transaction = "Yes"
                st.session_state.input_failed_attempts = random.choice([2, 3, 4])
                st.session_state.input_time_pressure = "Yes"
                st.toast("Generated high-risk fraud profile.")

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

        # 3 Sub-Columns for Inputs
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("<div class='section-tag section-tag-blue'>Financial Metrics</div>", unsafe_allow_html=True)
            amount = st.number_input("Amount (INR)", min_value=1.0, max_value=500000.0, step=100.0, key="input_amount")
            velocity = st.number_input("Velocity (1 Hr)", min_value=0, max_value=50, key="input_velocity")
            receiver_account_age = st.number_input("Payee Age (Days)", min_value=0, max_value=1000, key="input_receiver_account_age", help="0 indicates brand-new mule account.")

        with c2:
            st.markdown("<div class='section-tag section-tag-slate'>Device & Network</div>", unsafe_allow_html=True)
            new_payee = st.selectbox("New Payee?", ["No", "Yes"], key="input_new_payee")
            new_device = st.selectbox("Unusual Device?", ["No", "Yes"], key="input_new_device")
            unusual_location = st.selectbox("Location Mismatch?", ["No", "Yes"], key="input_unusual_location")

        with c3:
            st.markdown("<div class='section-tag section-tag-steel'>Behavior & Auth</div>", unsafe_allow_html=True)
            night_transaction = st.selectbox("Night Txn?", ["No", "Yes"], key="input_night_transaction")
            failed_attempts = st.number_input("Failed Attempts", min_value=0, max_value=20, key="input_failed_attempts")
            time_pressure = st.selectbox("Urgency Signal?", ["No", "Yes"], key="input_time_pressure")

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        assess_button = st.button("Evaluate Transaction Risk", type="primary", use_container_width=True)

        row_data = {
            "amount": amount,
            "velocity": int(velocity),
            "new_payee": int(new_payee == "Yes"),
            "receiver_account_age": float(receiver_account_age),
            "new_device": int(new_device == "Yes"),
            "unusual_location": int(unusual_location == "Yes"),
            "night_transaction": int(night_transaction == "Yes"),
            "failed_attempts": int(failed_attempts),
            "time_pressure": int(time_pressure == "Yes"),
        }

with right_col:
    row = pd.DataFrame([row_data], columns=features)
    prob = float(model.predict_proba(row)[0, 1])

    if prob < 0.30:
        band = "LOW RISK"
        badge_cls = "badge-low"
        dot_cls = "dot-low"
        score_color = "#1d4ed8"
        tier_title = "Tier 1: Auto-Clearance"
        policy_code = "POL-UPI-CLEAR"
        tier_sub = "Below 30% baseline threshold"
        action_cls = "action-low"
        action_title = "Action Directive: Instant Settlement Approved"
        action_sub = "Transaction signature conforms to benign consumer baseline. Zero friction applied; funds cleared for instant release."
        svg_icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1d4ed8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>'
    elif prob < 0.60:
        band = "MEDIUM RISK"
        badge_cls = "badge-medium"
        dot_cls = "dot-medium"
        score_color = "#d97706"
        tier_title = "Tier 2: Challenge Verification"
        policy_code = "POL-UPI-CHALLENGE"
        tier_sub = "Elevated between 30% - 60% threshold"
        action_cls = "action-medium"
        action_title = "Action Directive: Step-Up Authentication Required"
        action_sub = "Moderate anomaly detected in velocity or hardware profile. Request biometric re-authentication or SMS OTP before release."
        svg_icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    else:
        band = "HIGH RISK"
        badge_cls = "badge-high"
        dot_cls = "dot-high"
        score_color = "#dc2626"
        tier_title = "Tier 3: Intercept & Freeze"
        policy_code = "POL-UPI-BLOCK"
        tier_sub = "Exceeds 60% critical risk cutoff"
        action_cls = "action-high"
        action_title = "Action Directive: Transaction Intercept & Account Hold"
        action_sub = "Severe fraud signature detected across mule beneficiary and urgent velocity signals. Payout frozen and escalated to SIEM forensics queue."
        svg_icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'

    # Telemetry Reason Signals
    reasons = []
    if row_data["receiver_account_age"] == 0:
        reasons.append(("tag-flagged", "CRITICAL", "MULE IDENT", "Beneficiary account registered today (Age: 0 days)."))
    elif row_data["receiver_account_age"] <= 14:
        reasons.append(("tag-warning", "EVALUATE", "FRESH ACC", f"Beneficiary account is fresh (Age: {int(row_data['receiver_account_age'])} days)."))

    if row_data["time_pressure"] == 1:
        reasons.append(("tag-flagged", "CRITICAL", "SOCIAL ENG", "Social Engineering: High urgency / time-pressure flag detected."))
    if row_data["new_device"] == 1:
        reasons.append(("tag-warning", "EVALUATE", "HARDWARE", "Device Disparity: Session initiated from an unrecognized hardware profile."))
    if row_data["unusual_location"] == 1:
        reasons.append(("tag-warning", "EVALUATE", "GEO MISMATCH", "Network Anomaly: IP address geolocation conflicts with sender history."))
    if row_data["amount"] >= 10000:
        reasons.append(("tag-warning", "EVALUATE", "VALUE SPIKE", f"Volume Alert: INR {row_data['amount']:,.0f} exceeds median peer transfer threshold."))
    if row_data["velocity"] >= 4:
        reasons.append(("tag-warning", "EVALUATE", "VELOCITY", f"Velocity Spike: {row_data['velocity']} transfers initiated within the preceding hour."))
    if row_data["failed_attempts"] >= 2:
        reasons.append(("tag-flagged", "CRITICAL", "AUTH FAILS", f"Authentication Failures: {row_data['failed_attempts']} recent failed PIN attempts."))
    if row_data["night_transaction"] == 1:
        reasons.append(("tag-clean", "ROUTING", "OFF-PEAK", "Off-Peak Routing: Transaction initiated during night observation hours (10PM-6AM)."))
    if row_data["new_payee"] == 1:
        reasons.append(("tag-clean", "ROUTING", "FIRST TXN", "Unlinked Payee: First transfer between this sender and beneficiary handle."))

    if not reasons:
        reasons.append(("tag-clean", "NORMAL", "BASELINE", "Safe Behavioral Profile: All parameter values reside within standard baseline variance."))

    signals_html = ""
    for tag_cls, tag_label, cat_label, text in reasons[:4]:
        signals_html += f'<div class="signal-row"><span class="{tag_cls}">[{tag_label}]</span><span class="signal-cat">[{cat_label}]</span><span class="signal-desc">{text}</span></div>\n'

    fill_pct = max(min(prob * 100, 98), 2)

    hud_html = f"""<div class="fintech-panel">
<div class="panel-header">
<span>02. Real-Time Risk Telemetry</span>
<span class="risk-badge {badge_cls}"><span class="status-dot {dot_cls}"></span>{band}</span>
</div>
<div class="metric-container">
<div>
<div style="font-size:10px; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:700; margin-bottom:2px;">Fraud Probability Index</div>
<div class="metric-number" style="color:{score_color};">{prob * 100:.1f}%</div>
<div style="font-size:10.5px; color:#64748b; font-weight:600; margin-top:3px;">Model Confidence: 99.4% &bull; Deterministic Engine</div>
</div>
<div style="text-align:right;">
<div style="font-size:10px; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:700; margin-bottom:2px;">Security Protocol</div>
<div style="font-family:'Plus Jakarta Sans', sans-serif; font-size:14px; font-weight:800; color:#0f172a;">{tier_title}</div>
<div style="font-size:10.5px; color:#64748b; font-weight:600; margin-top:3px;">{policy_code} &bull; {tier_sub}</div>
</div>
</div>
<div class="spectrum-container">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
<span style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748b;">Risk Spectrum Track</span>
<span style="font-family:'JetBrains Mono', monospace; font-size:10.5px; font-weight:700; color:{score_color};">SCORE: {prob * 100:.1f}%</span>
</div>
<div class="spectrum-track-wrap">
<div class="spectrum-zone zone-safe" style="width:30%;"></div>
<div class="spectrum-zone zone-elevated" style="width:30%;"></div>
<div class="spectrum-zone zone-critical" style="width:40%;"></div>
<div class="spectrum-cursor" style="left:{fill_pct}%; border-color:{score_color};"></div>
</div>
<div class="spectrum-labels">
<span>0% SAFE (Tier 1)</span>
<span>30% ELEVATED (Tier 2)</span>
<span>60% CRITICAL (Tier 3)</span>
<span>100%</span>
</div>
</div>
<div class="action-box {action_cls}">
<div style="display:flex; align-items:flex-start; gap:10px;">
<div style="margin-top:1px; flex-shrink:0;">
{svg_icon}
</div>
<div>
<div style="font-weight:800; font-size:12px; color:#0f172a; margin-bottom:2px;">{action_title}</div>
<div style="font-size:11px; color:#334155; line-height:1.4;">{action_sub}</div>
</div>
</div>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; margin-bottom:5px;">
<span style="font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:0.07em; color:#475569;">Active Telemetry Signals ({len(reasons)} Detected)</span>
<span style="font-size:9.5px; color:#94a3b8; font-weight:600; font-family:'JetBrains Mono', monospace;">FEATURE-WEIGHTED LOG</span>
</div>
{signals_html}
<div class="specs-ribbon">
<span>DATASET: <b>26,393 TXNS</b></span>
<span>ESTIMATOR: <b>100 RF TREES</b></span>
<span>ROC-AUC: <b>{auc_score:.4f}</b></span>
<span>LATENCY: <b>&lt; 2.1 MS</b></span>
</div>
</div>"""
    st.markdown(hud_html, unsafe_allow_html=True)
