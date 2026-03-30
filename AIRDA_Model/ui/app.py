"""
ui/app.py — AIRDA Research Lab Control Panel
=============================================
Swiss International + Minimalist Dark hybrid.
High-density research-grade dashboard.
"""

import sys
import os
import time
import io

# ── Fix imports ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd

# ── Page Config ──
st.set_page_config(
    page_title="AIRDA // CONTROL",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# MASSIVE CSS INJECTION — Research Lab Aesthetic
# ============================================================================
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════
       GOOGLE FONTS
       ═══════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

    /* ═══════════════════════════════════════════════════════════════
       GLOBAL RESET — Kill all Streamlit defaults
       ═══════════════════════════════════════════════════════════════ */
    .main, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0A0A0A !important;
        color: #E0E0E0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Grid paper background */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    .main .block-container {
        padding: 1rem 1.5rem !important;
        max-width: 100% !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       SIDEBAR — Control Panel
       ═══════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background-color: #0A0A0A !important;
        border-right: 2px solid #00FFAA !important;
        width: 280px !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #0A0A0A !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #E0E0E0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00FFAA !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        font-size: 0.85rem !important;
        border-bottom: 1px solid #2D2D2D !important;
        padding-bottom: 6px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       KILL ALL BORDER RADIUS
       ═══════════════════════════════════════════════════════════════ */
    *, *::before, *::after {
        border-radius: 0px !important;
    }
    div[data-testid="stMetric"],
    div[data-testid="stMetricValue"],
    .stButton > button,
    .stNumberInput input,
    .stSelectbox > div,
    .stDataFrame,
    div[data-testid="stExpander"],
    .stAlert,
    .stProgress > div > div,
    div[data-testid="stImage"] > img,
    .stTabs [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab-panel"] {
        border-radius: 0px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       BUTTONS — Flat, sharp, neon
       ═══════════════════════════════════════════════════════════════ */
    .stButton > button {
        background-color: #0A0A0A !important;
        color: #00FFAA !important;
        border: 2px solid #00FFAA !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        padding: 8px 16px !important;
        transition: none !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background-color: #00FFAA !important;
        color: #0A0A0A !important;
        border-color: #00FFAA !important;
    }
    .stButton > button:active {
        background-color: #00CC88 !important;
        color: #0A0A0A !important;
    }
    /* Primary buttons — filled */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #00FFAA !important;
        color: #0A0A0A !important;
        border: 2px solid #00FFAA !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #0A0A0A !important;
        color: #00FFAA !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       INPUTS
       ═══════════════════════════════════════════════════════════════ */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background-color: #0A0A0A !important;
        color: #00FFAA !important;
        border: 2px solid #2D2D2D !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        box-shadow: none !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #00FFAA !important;
        box-shadow: none !important;
    }
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #888 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-size: 0.7rem !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       DATAFRAMES / TABLES
       ═══════════════════════════════════════════════════════════════ */
    .stDataFrame {
        border: 2px solid #2D2D2D !important;
    }
    [data-testid="stDataFrame"] th {
        background-color: #1A1A1A !important;
        color: #00FFAA !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.08em !important;
        border-bottom: 2px solid #00FFAA !important;
    }
    [data-testid="stDataFrame"] td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        color: #E0E0E0 !important;
        background-color: #0A0A0A !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       METRICS
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stMetric"] {
        background-color: #0A0A0A !important;
        border: 2px solid #2D2D2D !important;
        padding: 12px 14px !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00FFAA !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00FFAA !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #888 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        font-size: 0.65rem !important;
    }
    div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       TABS
       ═══════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0A0A0A !important;
        border-bottom: 2px solid #2D2D2D !important;
        gap: 0px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0A0A0A !important;
        color: #888 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-size: 0.72rem !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 20px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00FFAA !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00FFAA !important;
        border-bottom: 2px solid #00FFAA !important;
        background-color: #0A0A0A !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0A0A0A !important;
        padding-top: 16px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       EXPANDERS
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stExpander"] {
        border: 2px solid #2D2D2D !important;
        background-color: #0A0A0A !important;
    }
    div[data-testid="stExpander"] summary {
        color: #E0E0E0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: #00FFAA !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       ALERTS — Terminal feel
       ═══════════════════════════════════════════════════════════════ */
    .stAlert {
        border: 2px solid !important;
        background-color: #0A0A0A !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }
    div[data-testid="stAlert"][data-baseweb*="positive"],
    .stSuccess, .element-container .stAlert:has([data-testid="stMarkdownContainer"]) {
        border-color: #00FFAA !important;
        color: #00FFAA !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PROGRESS BAR
       ═══════════════════════════════════════════════════════════════ */
    .stProgress > div > div > div {
        background-color: #00FFAA !important;
    }
    .stProgress > div {
        background-color: #2D2D2D !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       SPINNER
       ═══════════════════════════════════════════════════════════════ */
    .stSpinner > div {
        border-top-color: #00FFAA !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       SCROLLBAR
       ═══════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0A0A0A; }
    ::-webkit-scrollbar-thumb { background: #2D2D2D; }
    ::-webkit-scrollbar-thumb:hover { background: #00FFAA; }

    /* ═══════════════════════════════════════════════════════════════
       CUSTOM CLASSES
       ═══════════════════════════════════════════════════════════════ */
    .headline-serif {
        font-family: 'DM Serif Display', serif !important;
        font-size: 48px !important;
        color: #FFFFFF !important;
        line-height: 1.05 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .headline-serif-lg {
        font-family: 'DM Serif Display', serif !important;
        font-size: 64px !important;
        color: #FFFFFF !important;
        line-height: 1.0 !important;
        margin: 0 0 4px 0 !important;
    }
    .label-mono {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        color: #888 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        margin: 0 !important;
    }
    .data-mono {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        color: #E0E0E0 !important;
    }
    .neon-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #00FFAA !important;
    }
    .alert-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #FF3B3B !important;
    }
    .kpi-block {
        background-color: #0A0A0A;
        border: 2px solid #2D2D2D;
        padding: 16px 14px;
        transition: border-color 0.0s;
    }
    .kpi-block:hover {
        border-color: #00FFAA;
    }
    .divider-neon {
        border: none;
        border-top: 2px solid #00FFAA;
        margin: 16px 0;
    }
    .divider-gray {
        border: none;
        border-top: 1px solid #2D2D2D;
        margin: 10px 0;
    }
    .status-live {
        display: inline-block;
        width: 8px; height: 8px;
        background: #00FFAA;
        margin-right: 6px;
        animation: blink 1.2s infinite;
    }
    .status-off {
        display: inline-block;
        width: 8px; height: 8px;
        background: #FF3B3B;
        margin-right: 6px;
    }
    .status-idle {
        display: inline-block;
        width: 8px; height: 8px;
        background: #2D2D2D;
        border: 1px solid #555;
        margin-right: 6px;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .terminal-log {
        background: #0A0A0A;
        border: 2px solid #2D2D2D;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #00FFAA;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        border-bottom: 1px solid #2D2D2D;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }
    .garf-highlight {
        background-color: rgba(0, 255, 170, 0.08) !important;
        border-left: 3px solid #00FFAA !important;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background-color: #0A0A0A !important; }

    /* Image borders */
    div[data-testid="stImage"] > img {
        border: 2px solid #2D2D2D !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
def init_state():
    defaults = {
        'data': None,
        'training_results': None,
        'table_iv': None,
        'table_v': None,
        'table_v_raw': None,
        'table_vi': None,
        'table_vi_raw': None,
        'table_vii': None,
        'plots_generated': False,
        'plot_paths': [],
        'log_buffer': [],
        'train_svm': True,
        'train_lstm': True,
        'train_rf': True,
        'train_garf': True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def log(msg):
    """Append a terminal log line."""
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_buffer.append(f"[{ts}] {msg}")
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer = st.session_state.log_buffer[-50:]


def get_status_html(ready, label_on="ONLINE", label_off="OFFLINE"):
    if ready:
        return f'<span class="status-live"></span><span class="data-mono" style="color:#00FFAA;font-size:0.72rem;">{label_on}</span>'
    else:
        return f'<span class="status-off"></span><span class="data-mono" style="color:#FF3B3B;font-size:0.72rem;">{label_off}</span>'


# ============================================================================
# SIDEBAR — CONTROL PANEL
# ============================================================================
with st.sidebar:
    # System title
    st.markdown("""
    <div style="border-bottom:2px solid #00FFAA; padding-bottom:12px; margin-bottom:16px;">
        <p class="label-mono" style="color:#00FFAA; font-size:0.6rem; margin-bottom:2px;">SYSTEM //</p>
        <p style="font-family:'DM Serif Display',serif; font-size:28px; color:#FFF; margin:0; line-height:1.1;">AIRDA</p>
        <p class="label-mono" style="font-size:0.55rem; margin-top:4px;">AI-Enabled Resource Detection<br>& Allocation Framework</p>
    </div>
    """, unsafe_allow_html=True)

    # ── STATUS ──
    st.markdown('<div class="section-label">SYSTEM STATUS</div>', unsafe_allow_html=True)

    data_ready = st.session_state.data is not None
    models_ready = st.session_state.training_results is not None
    plots_ready = st.session_state.plots_generated

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; line-height:2.2;">
        {get_status_html(data_ready, "DATA READY", "NO DATA")}<br>
        {get_status_html(models_ready, "MODELS TRAINED", "NOT TRAINED")}<br>
        {get_status_html(plots_ready, "PLOTS READY", "NOT GENERATED")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

    # ── DATA GENERATION ──
    st.markdown('<div class="section-label">DATA GENERATION</div>', unsafe_allow_html=True)

    n_samples = st.number_input("SAMPLE COUNT", min_value=1000, max_value=100000,
                                value=20000, step=1000, label_visibility="visible")

    gen_btn = st.button("► GENERATE DATA", type="primary", use_container_width=True)

    if gen_btn:
        log("Initiating data generation pipeline...")
        with st.spinner(""):
            from pipeline.trainer import run_data_generation
            t0 = time.time()
            data = run_data_generation(n_samples=n_samples)
            elapsed = time.time() - t0
            st.session_state.data = data
            st.session_state.training_results = None
            st.session_state.table_iv = None
            st.session_state.table_v = None
            st.session_state.table_v_raw = None
            st.session_state.table_vi = None
            st.session_state.table_vii = None
            st.session_state.plots_generated = False
            log(f"Data generated: {n_samples} samples in {elapsed:.1f}s")
            log(f"Train/Test split: {len(data['y_train'])}/{len(data['y_test'])}")
            log(f"Augmented features: {data['X_train_aug'].shape[1]}")
        st.rerun()

    st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

    # ── MODEL TRAINING ──
    st.markdown('<div class="section-label">MODEL SELECTION</div>', unsafe_allow_html=True)

    st.session_state.train_svm = st.toggle("SVM", value=st.session_state.train_svm)
    st.session_state.train_lstm = st.toggle("LSTM", value=st.session_state.train_lstm)
    st.session_state.train_rf = st.toggle("RANDOM FOREST", value=st.session_state.train_rf)
    st.session_state.train_garf = st.toggle("GA-RF ★", value=st.session_state.train_garf)

    train_btn = st.button("► TRAIN SELECTED", use_container_width=True)

    if train_btn:
        if not data_ready:
            st.error("ERR: Generate data first")
        else:
            data = st.session_state.data
            selected = []
            if st.session_state.train_svm: selected.append('SVM')
            if st.session_state.train_lstm: selected.append('LSTM')
            if st.session_state.train_rf: selected.append('Random Forest')
            if st.session_state.train_garf: selected.append('GA-RF')

            if not selected:
                st.error("ERR: Select at least one model")
            else:
                log(f"Training pipeline: {', '.join(selected)}")
                from pipeline.trainer import train_single_model

                if st.session_state.training_results is None:
                    st.session_state.training_results = {
                        'models': {}, 'metrics': [], 'predictions': {}, 'ga_fitness': None
                    }

                progress = st.progress(0)
                for i, name in enumerate(selected):
                    log(f"Training {name}...")
                    progress.progress((i) / len(selected), text=f"TRAINING {name.upper()}")
                    model, metrics, preds, ga_fit = train_single_model(
                        name, data['X_train_aug'], data['y_train'],
                        data['X_test_aug'], data['y_test']
                    )
                    st.session_state.training_results['models'][name] = model
                    existing = [m for m in st.session_state.training_results['metrics']
                               if m['model'] != metrics['model']]
                    existing.append(metrics)
                    st.session_state.training_results['metrics'] = existing
                    st.session_state.training_results['predictions'][name] = preds
                    if ga_fit:
                        st.session_state.training_results['ga_fitness'] = ga_fit
                    log(f"{name} complete: Acc={metrics['accuracy']:.1f}% F1={metrics['f1']:.1f}%")

                progress.progress(1.0, text="COMPLETE")
                from pipeline.evaluator import get_table_iv
                st.session_state.table_iv = get_table_iv(
                    st.session_state.training_results['metrics']
                )
                log("Training pipeline complete.")
                st.rerun()

    st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

    # ── SIMULATION ──
    st.markdown('<div class="section-label">SIMULATION</div>', unsafe_allow_html=True)

    sim_btn = st.button("► RUN ALLOCATION SIM", use_container_width=True)
    if sim_btn:
        if not models_ready:
            st.error("ERR: Train models first")
        else:
            results = st.session_state.training_results
            required = ['SVM', 'LSTM', 'Random Forest', 'GA-RF']
            missing = [m for m in required if m not in results['predictions']]
            if missing:
                st.error(f"ERR: Missing {', '.join(missing)}")
            else:
                log("Running allocation simulation...")
                from pipeline.evaluator import get_table_v
                data = st.session_state.data
                df, alloc_raw = get_table_v(
                    data['X_test'], data['y_test'], results['predictions']
                )
                st.session_state.table_v = df
                st.session_state.table_v_raw = alloc_raw
                log("Simulation complete. Table V generated.")
                st.rerun()

    plots_btn = st.button("► GENERATE PLOTS", use_container_width=True)
    if plots_btn:
        if not models_ready:
            st.error("ERR: Train models first")
        else:
            log("Generating visualizations...")
            from pipeline.evaluator import (generate_all_plots, get_table_v,
                                           get_table_vi, get_table_vii)
            data = st.session_state.data
            results = st.session_state.training_results

            # Ensure Table V
            if st.session_state.table_v_raw is None:
                req = ['SVM', 'LSTM', 'Random Forest', 'GA-RF']
                mis = [m for m in req if m not in results['predictions']]
                if not mis:
                    df_v, alloc_raw = get_table_v(data['X_test'], data['y_test'],
                                                  results['predictions'])
                    st.session_state.table_v = df_v
                    st.session_state.table_v_raw = alloc_raw

            # Ensure Table VI
            if st.session_state.table_vi is None and 'GA-RF' in results['models']:
                df_vi, imp, names = get_table_vi(results['models']['GA-RF'],
                                                 data['feature_names'])
                st.session_state.table_vi = df_vi
                st.session_state.table_vi_raw = (imp, names)

            # Ensure Table VII
            if st.session_state.table_vii is None and 'GA-RF' in results['models']:
                st.session_state.table_vii = get_table_vii(
                    results['models']['GA-RF'], data['X'], data['y'],
                    data['feature_names'], data['scaler']
                )

            garf_pred = results['predictions'].get('GA-RF')
            alloc_raw = st.session_state.table_v_raw or []

            paths = generate_all_plots(
                results['metrics'], alloc_raw,
                results['models'].get('GA-RF', results['models'].get('Random Forest')),
                data['feature_names'], results['ga_fitness'],
                garf_pred=garf_pred, y_test=data['y_test'],
                cross_domain_df=st.session_state.table_vii
            )
            st.session_state.plot_paths = paths
            st.session_state.plots_generated = True
            log(f"{len(paths)} plots generated → /outputs")
            st.rerun()

    st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

    # ── TERMINAL LOG ──
    st.markdown('<div class="section-label">SYSTEM LOG</div>', unsafe_allow_html=True)
    log_text = "\n".join(st.session_state.log_buffer[-12:]) if st.session_state.log_buffer else "Awaiting input..."
    st.markdown(f'<div class="terminal-log">{log_text}<span style="animation:blink 1s infinite;">█</span></div>',
                unsafe_allow_html=True)


# ============================================================================
# MAIN CANVAS
# ============================================================================

# ── HEADER ──
st.markdown("""
<div style="border-bottom:2px solid #2D2D2D; padding-bottom:14px; margin-bottom:20px;">
    <p class="label-mono" style="font-size:0.55rem; margin-bottom:0;">RESEARCH MODULE // DASHBOARD v2.0</p>
    <p class="headline-serif-lg">Resource Allocation<br>Control System</p>
    <p class="label-mono" style="color:#555; font-size:0.55rem;">GA-OPTIMIZED RANDOM FOREST · K-MEANS PROFILING · MULTI-STRATEGY ALLOCATION</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# PRIMARY PANEL — TOP 70%
# ============================================================================

if st.session_state.data is not None:
    data = st.session_state.data

    # ── DATASET KPIs ──
    st.markdown('<div class="section-label">DATASET OVERVIEW</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("SAMPLES", f"{len(data['y']):,}")
    k2.metric("FEATURES", f"{len(data['feature_names'])}")
    k3.metric("TRAIN SET", f"{len(data['y_train']):,}")
    k4.metric("TEST SET", f"{len(data['y_test']):,}")
    k5.metric("AUGMENTED DIM", f"{data['X_train_aug'].shape[1]}")

    st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

# ── TABS: Classification | Allocation | Advanced ──
if st.session_state.training_results is not None:
    results = st.session_state.training_results
    data = st.session_state.data

    tab_class, tab_alloc, tab_advanced, tab_plots = st.tabs([
        "CLASSIFICATION", "ALLOCATION", "ADVANCED", "VISUALIZATIONS"
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1: CLASSIFICATION PERFORMANCE
    # ══════════════════════════════════════════════════════════════
    with tab_class:
        st.markdown("""
        <div style="margin-bottom:14px;">
            <p class="label-mono" style="font-size:0.5rem; margin-bottom:0;">TABLE IV //</p>
            <p class="headline-serif" style="font-size:36px;">Classification Performance</p>
        </div>
        """, unsafe_allow_html=True)

        metrics = results['metrics']

        # ── Key metrics row ──
        garf_m = next((m for m in metrics if 'GA-RF' in m['model']), None)
        rf_m = next((m for m in metrics if 'Vanilla' in m['model']), None)
        lstm_m = next((m for m in metrics if 'LSTM' in m['model']), None)
        svm_m = next((m for m in metrics if 'SVM' in m['model']), None)

        if garf_m:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GA-RF ACCURACY", f"{garf_m['accuracy']:.2f}%")
            c2.metric("GA-RF F1-SCORE", f"{garf_m['f1']:.2f}%")
            if rf_m:
                delta_f1 = garf_m['f1'] - rf_m['f1']
                c3.metric("VS VANILLA RF", f"+{delta_f1:.2f}%", "F1 GAIN")
            if lstm_m:
                delta_acc = garf_m['accuracy'] - lstm_m['accuracy']
                c4.metric("VS LSTM", f"+{delta_acc:.2f}%", "ACC GAIN")

            st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

        # ── Table IV DataFrame ──
        if st.session_state.table_iv is not None:
            df_iv = st.session_state.table_iv

            def style_table_iv(row):
                if 'GA-RF' in str(row['Model']):
                    return ['background-color: rgba(0,255,170,0.06); color: #00FFAA; font-weight:700;'] * len(row)
                return ['color: #E0E0E0;'] * len(row)

            st.dataframe(
                df_iv.style.apply(style_table_iv, axis=1).format({
                    'Accuracy (%)': '{:.2f}', 'Precision (%)': '{:.2f}',
                    'Recall (%)': '{:.2f}', 'F1-Score (%)': '{:.2f}',
                    'Train Time (s)': '{:.2f}',
                }),
                use_container_width=True, hide_index=True, height=210
            )

        # ── Per-model detail ──
        st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">MODEL DETAIL</div>', unsafe_allow_html=True)

        for m in metrics:
            tag = "★ " if 'GA-RF' in m['model'] else ""
            with st.expander(f"{tag}{m['model']}  //  ACC {m['accuracy']:.1f}%  F1 {m['f1']:.1f}%"):
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("ACCURACY", f"{m['accuracy']:.2f}%")
                mc2.metric("PRECISION", f"{m['precision']:.2f}%")
                mc3.metric("RECALL", f"{m['recall']:.2f}%")
                mc4.metric("F1-SCORE", f"{m['f1']:.2f}%")
                mc5.metric("TRAIN TIME", f"{m['train_time']:.2f}s")

    # ══════════════════════════════════════════════════════════════
    # TAB 2: ALLOCATION STRATEGIES
    # ══════════════════════════════════════════════════════════════
    with tab_alloc:
        st.markdown("""
        <div style="margin-bottom:14px;">
            <p class="label-mono" style="font-size:0.5rem; margin-bottom:0;">TABLE V //</p>
            <p class="headline-serif" style="font-size:36px;">Allocation Efficiency</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.table_v is not None:
            df_v = st.session_state.table_v
            alloc_raw = st.session_state.table_v_raw

            # ── KPI strip ──
            garf_r = next((r for r in alloc_raw if 'GA-RF' in r['strategy']), None)
            tb_r = next((r for r in alloc_raw if 'Threshold' in r['strategy']), None)
            rr_r = next((r for r in alloc_raw if 'Round' in r['strategy']), None)

            if garf_r:
                st.markdown('<div class="section-label">GA-RF ALLOCATION METRICS</div>',
                           unsafe_allow_html=True)
                a1, a2, a3, a4 = st.columns(4)
                a1.markdown(f"""
                <div class="kpi-block">
                    <p class="label-mono">AVG LATENCY</p>
                    <p class="neon-value">{garf_r['avg_latency_ms']:.0f}<span style="font-size:0.8rem;">ms</span></p>
                </div>
                """, unsafe_allow_html=True)
                a2.markdown(f"""
                <div class="kpi-block">
                    <p class="label-mono">UTILIZATION</p>
                    <p class="neon-value">{garf_r['utilization_pct']:.1f}<span style="font-size:0.8rem;">%</span></p>
                </div>
                """, unsafe_allow_html=True)
                a3.markdown(f"""
                <div class="kpi-block">
                    <p class="label-mono">ENERGY</p>
                    <p class="neon-value">{garf_r['energy_kwh']:.1f}<span style="font-size:0.8rem;">kWh</span></p>
                </div>
                """, unsafe_allow_html=True)
                a4.markdown(f"""
                <div class="kpi-block">
                    <p class="label-mono">SLA VIOLATIONS</p>
                    <p class="{'neon-value' if garf_r['sla_violations_pct'] < 5 else 'alert-value'}">{garf_r['sla_violations_pct']:.1f}<span style="font-size:0.8rem;">%</span></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

                # ── Comparison deltas ──
                if tb_r:
                    lat_imp = (1 - garf_r['avg_latency_ms'] / tb_r['avg_latency_ms']) * 100
                    energy_imp = (1 - garf_r['energy_kwh'] / tb_r['energy_kwh']) * 100
                    d1, d2, d3 = st.columns(3)
                    d1.metric("LATENCY vs THRESHOLD", f"-{abs(lat_imp):.1f}%", "REDUCTION")
                    d2.metric("ENERGY vs THRESHOLD", f"-{abs(energy_imp):.1f}%", "SAVINGS")
                    if rr_r:
                        sla_imp = rr_r['sla_violations_pct'] - garf_r['sla_violations_pct']
                        d3.metric("SLA vs ROUND ROBIN", f"-{sla_imp:.1f}pp", "IMPROVEMENT")

            st.markdown('<hr class="divider-gray">', unsafe_allow_html=True)

            # ── Full Table V ──
            st.markdown('<div class="section-label">ALL STRATEGIES</div>', unsafe_allow_html=True)

            def style_table_v(row):
                if 'GA-RF' in str(row['Strategy']):
                    return ['background-color: rgba(0,255,170,0.06); color: #00FFAA; font-weight:700;'] * len(row)
                return ['color: #E0E0E0;'] * len(row)

            st.dataframe(
                df_v.style.apply(style_table_v, axis=1),
                use_container_width=True, hide_index=True, height=280
            )
        else:
            st.markdown("""
            <div class="terminal-log">
            > Allocation simulation not run.
            > Use control panel: ► RUN ALLOCATION SIM
            <span style="animation:blink 1s infinite;">█</span>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3: ADVANCED — Feature Importance, Cross-Domain, Confusion
    # ══════════════════════════════════════════════════════════════
    with tab_advanced:
        st.markdown("""
        <div style="margin-bottom:14px;">
            <p class="label-mono" style="font-size:0.5rem; margin-bottom:0;">TABLES VI + VII //</p>
            <p class="headline-serif" style="font-size:36px;">Advanced Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        adv1, adv2 = st.columns(2)

        # ── Feature Importance ──
        with adv1:
            st.markdown('<div class="section-label">FEATURE IMPORTANCE · TABLE VI</div>',
                       unsafe_allow_html=True)

            if st.session_state.table_vi is not None:
                st.dataframe(st.session_state.table_vi, use_container_width=True,
                            hide_index=True, height=350)
            elif 'GA-RF' in results['models']:
                calc_vi = st.button("► COMPUTE FEATURE IMPORTANCE", key="calc_vi")
                if calc_vi:
                    from pipeline.evaluator import get_table_vi
                    df_vi, imp, names = get_table_vi(
                        results['models']['GA-RF'], data['feature_names']
                    )
                    st.session_state.table_vi = df_vi
                    st.session_state.table_vi_raw = (imp, names)
                    log("Feature importance computed.")
                    st.rerun()
            else:
                st.markdown('<p class="data-mono" style="color:#FF3B3B;">GA-RF model required.</p>',
                           unsafe_allow_html=True)

        # ── Cross-Domain ──
        with adv2:
            st.markdown('<div class="section-label">CROSS-DOMAIN VALIDATION · TABLE VII</div>',
                       unsafe_allow_html=True)

            if st.session_state.table_vii is not None:
                df_vii = st.session_state.table_vii

                def style_cd(row):
                    acc = row['Accuracy (%)']
                    if acc >= 95:
                        return ['color: #00FFAA;'] * len(row)
                    elif acc >= 90:
                        return ['color: #E0E0E0;'] * len(row)
                    return ['color: #FF3B3B;'] * len(row)

                st.dataframe(
                    df_vii.style.apply(style_cd, axis=1),
                    use_container_width=True, hide_index=True, height=220
                )

                min_acc = df_vii['Accuracy (%)'].min()
                max_acc = df_vii['Accuracy (%)'].max()
                st.markdown(f"""
                <div style="border:2px solid #2D2D2D; padding:12px; margin-top:8px;">
                    <p class="label-mono" style="margin-bottom:6px;">VALIDATION SUMMARY</p>
                    <p class="data-mono">Range: <span style="color:#00FFAA;">{min_acc:.1f}%</span> – <span style="color:#00FFAA;">{max_acc:.1f}%</span></p>
                    <p class="data-mono">All domains >{min_acc:.0f}% · Cross-domain validated</p>
                </div>
                """, unsafe_allow_html=True)
            elif 'GA-RF' in results['models']:
                calc_vii = st.button("► RUN CROSS-DOMAIN", key="calc_vii")
                if calc_vii:
                    from pipeline.evaluator import get_table_vii
                    log("Running cross-domain validation...")
                    st.session_state.table_vii = get_table_vii(
                        results['models']['GA-RF'], data['X'], data['y'],
                        data['feature_names'], data['scaler']
                    )
                    log("Cross-domain validation complete.")
                    st.rerun()
            else:
                st.markdown('<p class="data-mono" style="color:#FF3B3B;">GA-RF model required.</p>',
                           unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4: VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════
    with tab_plots:
        st.markdown("""
        <div style="margin-bottom:14px;">
            <p class="label-mono" style="font-size:0.5rem; margin-bottom:0;">FIGURE OUTPUT //</p>
            <p class="headline-serif" style="font-size:36px;">Visualizations</p>
        </div>
        """, unsafe_allow_html=True)

        outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
        if os.path.isdir(outputs_dir):
            pngs = sorted([f for f in os.listdir(outputs_dir) if f.endswith('.png')])
            if pngs:
                plot_labels = {
                    'fig_classification_performance.png': 'CLASSIFICATION PERFORMANCE',
                    'fig_allocation_latency.png': 'ALLOCATION LATENCY',
                    'fig_feature_importance.png': 'FEATURE IMPORTANCE',
                    'fig_ga_convergence.png': 'GA CONVERGENCE',
                    'fig_confusion_matrix.png': 'CONFUSION MATRIX',
                    'fig_allocation_multimetric.png': 'MULTI-METRIC COMPARISON',
                    'fig_training_time.png': 'TRAINING TIME',
                    'fig_cross_domain.png': 'CROSS-DOMAIN VALIDATION',
                }

                # Grid layout: 2 columns
                for i in range(0, len(pngs), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(pngs):
                            with col:
                                fname = pngs[idx]
                                label = plot_labels.get(fname, fname.upper())
                                st.markdown(f'<div class="section-label">{label}</div>',
                                           unsafe_allow_html=True)
                                st.image(os.path.join(outputs_dir, fname),
                                        use_container_width=True)
            else:
                st.markdown("""
                <div class="terminal-log">
                > No plots in /outputs.
                > Use control panel: ► GENERATE PLOTS
                <span style="animation:blink 1s infinite;">█</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="terminal-log">
            > Output directory not found.
            > Generate plots via control panel.
            <span style="animation:blink 1s infinite;">█</span>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── NO DATA STATE ──
    if st.session_state.data is None:
        st.markdown("""
        <div style="border:2px solid #2D2D2D; padding:40px; text-align:center; margin-top:40px;">
            <p class="label-mono" style="font-size:0.6rem; margin-bottom:8px;">SYSTEM IDLE</p>
            <p class="headline-serif" style="font-size:42px; margin-bottom:12px;">Awaiting Data</p>
            <p class="data-mono" style="color:#888;">
                Use the control panel to generate synthetic workload data.<br>
                Set sample count → Click ► GENERATE DATA
            </p>
            <hr class="divider-gray">
            <p class="label-mono" style="color:#555; font-size:0.5rem;">
                PIPELINE: DATA GENERATION → K-MEANS PROFILING → MODEL TRAINING → ALLOCATION SIM → ANALYSIS
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.training_results is None:
        st.markdown("""
        <div style="border:2px solid #2D2D2D; padding:40px; text-align:center; margin-top:40px;">
            <p class="label-mono" style="font-size:0.6rem; margin-bottom:8px; color:#00FFAA;">DATA LOADED</p>
            <p class="headline-serif" style="font-size:42px; margin-bottom:12px;">Ready to Train</p>
            <p class="data-mono" style="color:#888;">
                Select models in the control panel → Click ► TRAIN SELECTED
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# SECONDARY PANEL — BOTTOM 30% (KPI strip always visible when data exists)
# ============================================================================
if st.session_state.table_v_raw is not None:
    st.markdown('<hr class="divider-neon">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:10px;">
        <p class="label-mono" style="font-size:0.5rem; margin-bottom:0;">SECONDARY METRICS //</p>
        <p class="headline-serif" style="font-size:28px;">System KPIs</p>
    </div>
    """, unsafe_allow_html=True)

    alloc_raw = st.session_state.table_v_raw

    # Build comparison data
    strategies = [r['strategy'] for r in alloc_raw]
    s1, s2, s3, s4 = st.columns(4)

    # SLA Violations comparison
    with s1:
        st.markdown('<div class="section-label">SLA VIOLATIONS (%)</div>', unsafe_allow_html=True)
        for r in alloc_raw:
            is_garf = 'GA-RF' in r['strategy']
            color = '#00FFAA' if is_garf else '#E0E0E0'
            bar_w = max(5, r['sla_violations_pct'] * 2)
            name_short = r['strategy'].replace('-Based', '').replace(' (Proposed)', '★')
            st.markdown(f"""
            <div style="margin-bottom:4px;">
                <span class="label-mono" style="display:inline-block; width:90px; font-size:0.55rem;">{name_short}</span>
                <span style="display:inline-block; width:{bar_w}px; height:8px; background:{color};"></span>
                <span class="data-mono" style="font-size:0.7rem; color:{color}; margin-left:6px;">{r['sla_violations_pct']:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    # Energy comparison
    with s2:
        st.markdown('<div class="section-label">ENERGY (kWh)</div>', unsafe_allow_html=True)
        max_energy = max(r['energy_kwh'] for r in alloc_raw)
        for r in alloc_raw:
            is_garf = 'GA-RF' in r['strategy']
            color = '#00FFAA' if is_garf else '#E0E0E0'
            bar_w = max(5, (r['energy_kwh'] / max_energy) * 120)
            name_short = r['strategy'].replace('-Based', '').replace(' (Proposed)', '★')
            st.markdown(f"""
            <div style="margin-bottom:4px;">
                <span class="label-mono" style="display:inline-block; width:90px; font-size:0.55rem;">{name_short}</span>
                <span style="display:inline-block; width:{bar_w}px; height:8px; background:{color};"></span>
                <span class="data-mono" style="font-size:0.7rem; color:{color}; margin-left:6px;">{r['energy_kwh']:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    # Utilization comparison
    with s3:
        st.markdown('<div class="section-label">UTILIZATION (%)</div>', unsafe_allow_html=True)
        for r in alloc_raw:
            is_garf = 'GA-RF' in r['strategy']
            color = '#00FFAA' if is_garf else '#E0E0E0'
            bar_w = max(5, r['utilization_pct'] * 1.2)
            name_short = r['strategy'].replace('-Based', '').replace(' (Proposed)', '★')
            st.markdown(f"""
            <div style="margin-bottom:4px;">
                <span class="label-mono" style="display:inline-block; width:90px; font-size:0.55rem;">{name_short}</span>
                <span style="display:inline-block; width:{bar_w}px; height:8px; background:{color};"></span>
                <span class="data-mono" style="font-size:0.7rem; color:{color}; margin-left:6px;">{r['utilization_pct']:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    # Latency comparison
    with s4:
        st.markdown('<div class="section-label">LATENCY (ms)</div>', unsafe_allow_html=True)
        max_lat = max(r['avg_latency_ms'] for r in alloc_raw)
        for r in alloc_raw:
            is_garf = 'GA-RF' in r['strategy']
            color = '#00FFAA' if is_garf else '#E0E0E0'
            bar_w = max(5, (r['avg_latency_ms'] / max_lat) * 120)
            name_short = r['strategy'].replace('-Based', '').replace(' (Proposed)', '★')
            st.markdown(f"""
            <div style="margin-bottom:4px;">
                <span class="label-mono" style="display:inline-block; width:90px; font-size:0.55rem;">{name_short}</span>
                <span style="display:inline-block; width:{bar_w}px; height:8px; background:{color};"></span>
                <span class="data-mono" style="font-size:0.7rem; color:{color}; margin-left:6px;">{r['avg_latency_ms']:.0f}</span>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<div style="border-top:1px solid #2D2D2D; margin-top:30px; padding-top:10px;">
    <p class="label-mono" style="color:#333; font-size:0.5rem; text-align:center;">
        AIRDA FRAMEWORK · DAA SCE PROJECT · SONALI · TEJAS · NIRANT · RIDDHI · ROHAN · DURVESH
    </p>
</div>
""", unsafe_allow_html=True)
