"""
Vectory - Precision LLM Evaluation Platform
Main Entry Point with animated UI and motion graphics
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from components import __version__, __product_name__, __tagline__, __author__, __website__
from components.models import init_session_state
from components.ui import (
    inject_custom_css,
    feature_card,
    animated_metric,
    gradient_banner,
    animated_logo,
    app_footer,
    get_current_theme,
    sidebar_logo,
)

st.set_page_config(
    page_title=f"{__product_name__} - {__tagline__}",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize
init_session_state(st)
inject_custom_css()

# Get current theme
theme = get_current_theme()

# Sidebar navigation is handled by Streamlit's built-in page navigation
# The "app" link is restyled to show "⚡ Vectory" via CSS

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero Section with Animated Logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    animated_logo(size="large", show_text=True)

# Subtitle
st.markdown(f"""
<div style="text-align: center; margin-top: -10px; margin-bottom: 40px;">
    <p style="
        color: {theme.text_muted};
        font-family: var(--theme-font-mono);
        font-size: 1rem;
    ">
        <span style="color: {theme.terminal};">$</span> Comprehensive toolkit for evaluating LLM outputs
    </p>
</div>
""", unsafe_allow_html=True)

# Feature Cards - Row 1
st.markdown(f"""
<div style="
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
">
    <div style="
        height: 1px;
        flex: 1;
        background: linear-gradient(90deg, transparent, {theme.border});
    "></div>
    <span style="
        color: {theme.text_muted};
        font-family: var(--theme-font-mono);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    ">Capabilities</span>
    <div style="
        height: 1px;
        flex: 1;
        background: linear-gradient(90deg, {theme.border}, transparent);
    "></div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    feature_card(
        "📊",
        "Load Datasets",
        "Upload JSON, CSV, or PDF files for evaluation.",
        delay=1
    )

with col2:
    feature_card(
        "🤖",
        "LLM-as-Judge",
        "Use current OpenAI or Anthropic models to evaluate output quality.",
        delay=2
    )

with col3:
    feature_card(
        "📏",
        "Rule-Based Metrics",
        "BLEU, ROUGE, exact match, and similarity scores.",
        delay=3
    )

with col4:
    feature_card(
        "👤",
        "Human Evaluation",
        "Rate outputs with customizable criteria.",
        delay=4
    )

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    feature_card(
        "🏆",
        "Model Leaderboard",
        "Compare embedding models on benchmarks.",
        delay=1
    )

with col2:
    feature_card(
        "🧪",
        "Custom Benchmarks",
        "Run your own evaluation frameworks.",
        delay=2
    )

with col3:
    feature_card(
        "📈",
        "Visualizations",
        "Interactive charts and analytics.",
        delay=3
    )

with col4:
    feature_card(
        "💾",
        "Export Results",
        "Download as CSV or JSON.",
        delay=4
    )

# Quick Stats Section
st.markdown("<br><br>", unsafe_allow_html=True)

if st.session_state.dataset is not None or st.session_state.evaluation_results or st.session_state.human_annotations:
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
    ">
        <div style="
            height: 1px;
            flex: 1;
            background: linear-gradient(90deg, transparent, {theme.border});
        "></div>
        <span style="
            color: {theme.text_muted};
            font-family: var(--theme-font-mono);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        ">Your Progress</span>
        <div style="
            height: 1px;
            flex: 1;
            background: linear-gradient(90deg, {theme.border}, transparent);
        "></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rows = len(st.session_state.dataset) if st.session_state.dataset is not None else 0
        animated_metric("Dataset Rows", str(rows), "📊", delay=1)

    with col2:
        evals = len(st.session_state.evaluation_results)
        animated_metric("Evaluations", str(evals), "✅", delay=2)

    with col3:
        annotations = len(st.session_state.human_annotations)
        animated_metric("Annotations", str(annotations), "👤", delay=3)

    with col4:
        status = "Ready" if st.session_state.dataset is not None else "No Data"
        animated_metric("Status", status, "🚀", delay=4)

# Getting Started Banner
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.dataset is None:
    gradient_banner("""
    <div style="text-align: center;">
        <h2 style="margin: 0; font-size: 1.8rem; color: white;">🚀 Get Started</h2>
        <p style="margin: 16px 0 0 0; opacity: 0.9; font-size: 1.1rem; color: white;">
            Navigate to <strong>📊 Dataset</strong> in the sidebar to upload your evaluation data
        </p>
    </div>
    """)

# Footer
app_footer(__product_name__, __tagline__, __author__, __website__, __version__)
