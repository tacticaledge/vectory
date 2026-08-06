"""Dataset Management Page - With Animations"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state, ExtractionMode, ColumnMapping, DataSourceType
from components.document_loader import load_file, infer_columns, SUPPORTED_EXTENSIONS
from components.ui import (
    inject_custom_css,
    animated_metric,
    display_lottie,
    score_bar,
    feature_card,
    section_header,
    get_current_theme,
    sidebar_logo,
)

st.set_page_config(page_title="Dataset | Vectory", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)
inject_custom_css()

# Header with developer gradient
theme = get_current_theme()
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    📊 Dataset Studio
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {theme.text_muted}; font-size: 1rem; margin-top: -10px;">
    <span style="color: {theme.terminal};">$</span> upload, preview, and configure your evaluation data
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="glass-card" style="background: {theme.bg_card}; padding: 24px; border-radius: 12px; border: 1px solid {theme.border};">
        <h3 style="margin-top: 0; color: {theme.text_primary}; font-family: 'DM Sans', sans-serif;">📁 Upload Your Data</h3>
        <p style="color: {theme.text_muted}; margin-bottom: 0; font-family: 'DM Sans', sans-serif; font-size: 0.9rem;">Supports JSON, CSV, PDF, and images</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=list(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS.keys()),
        help="Drag and drop or click to browse",
        label_visibility="collapsed",
    )

with col2:
    if not display_lottie("data", height=150):
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <span style="font-size: 4rem;">📂</span>
        </div>
        """, unsafe_allow_html=True)

# PDF extraction mode
extraction_mode = ExtractionMode.MARKDOWN
if uploaded_file and uploaded_file.name.lower().endswith(".pdf"):
    st.markdown("""
    <div class="feature-card" style="margin-top: 16px;">
        <h4 style="margin-top: 0;">🔧 PDF Extraction Settings</h4>
    </div>
    """, unsafe_allow_html=True)

    extraction_mode = ExtractionMode(
        st.selectbox(
            "Mode",
            [e.value for e in ExtractionMode],
            format_func=lambda x: {
                "markdown": "📝 Markdown (best for LLMs)",
                "text": "📄 Plain text",
                "chunks": "📑 Page chunks (for RAG)",
            }.get(x, x),
            label_visibility="collapsed",
        )
    )

# Process uploaded file - load new data if a file was uploaded
if uploaded_file:
    try:
        with st.spinner("Loading..."):
            df = load_file(uploaded_file, extraction_mode)

        # Detect data source type based on file extension
        ext = uploaded_file.name.lower().split('.')[-1]
        if ext in ['csv', 'json', 'jsonl']:
            st.session_state.data_source_type = DataSourceType.TABULAR
        elif ext == 'pdf':
            st.session_state.data_source_type = DataSourceType.DOCUMENT
        elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
            st.session_state.data_source_type = DataSourceType.IMAGE
            # Store raw file content for image evaluation
            st.session_state.raw_file_content = uploaded_file.getvalue()
        else:
            st.session_state.data_source_type = DataSourceType.TABULAR

        st.session_state.dataset = df
        st.session_state.dataset_filename = uploaded_file.name
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

# Show dataset view if data is loaded (whether from new upload or session state)
if st.session_state.dataset is not None:
    df = st.session_state.dataset
    filename = st.session_state.get("dataset_filename", "unknown")

    # Success banner
    st.markdown(f"""
    <div style="background: linear-gradient(-45deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}, {theme.primary_dark}); background-size: 400% 400%; animation: gradient 8s ease infinite; border-radius: 12px; padding: 30px; color: white; text-align: center;">
        <span style="font-size: 2rem;">✨</span>
        <h3 style="margin: 10px 0 5px 0; color: white;">Dataset Loaded</h3>
        <p style="margin: 0; opacity: 0.9; color: white;">{len(df)} rows × {len(df.columns)} columns from {filename}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Animated metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        animated_metric("Rows", str(len(df)), "📊", delay=1)
    with col2:
        animated_metric("Columns", str(len(df.columns)), "📋", delay=2)
    with col3:
        memory_kb = df.memory_usage(deep=True).sum() / 1024
        animated_metric("Memory", f"{memory_kb:.1f} KB", "💾", delay=3)
    with col4:
        file_type = filename.split(".")[-1].upper() if "." in filename else "DATA"
        animated_metric("Type", file_type, "📁", delay=4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Preview", "🎯 Column Mapping", "📊 Statistics"])

    with tab1:
        st.dataframe(df.head(15), use_container_width=True, height=400)
        if st.button("🗑️ Clear Dataset", type="secondary"):
            st.session_state.dataset = None
            st.session_state.dataset_filename = None
            st.session_state.column_mapping = ColumnMapping()
            st.session_state.evaluation_results = {}
            st.rerun()

    with tab2:
        section_header("🎯 Map Your Columns", style="accent", subtitle="Tell us which columns contain your input, output, and reference data")

        inferred = infer_columns(df)
        cols = ["(Select column)"] + list(df.columns)

        # Get existing mapping values
        existing_mapping = st.session_state.column_mapping
        if isinstance(existing_mapping, dict):
            existing_mapping = ColumnMapping(**existing_mapping)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**📝 Input/Prompt**")
            default_input = existing_mapping.input_col if existing_mapping.input_col in cols else (inferred.input_col if inferred.input_col in cols else "(Select column)")
            input_col = st.selectbox(
                "Input column",
                cols,
                index=cols.index(default_input) if default_input in cols else 0,
                label_visibility="collapsed",
            )

        with col2:
            st.markdown(f"**🤖 LLM Output**")
            default_output = existing_mapping.output_col if existing_mapping.output_col in cols else (inferred.output_col if inferred.output_col in cols else "(Select column)")
            output_col = st.selectbox(
                "Output column",
                cols,
                index=cols.index(default_output) if default_output in cols else 0,
                label_visibility="collapsed",
            )

        with col3:
            st.markdown(f"**✅ Expected/Reference**")
            default_expected = existing_mapping.expected_col if existing_mapping.expected_col in cols else (inferred.expected_col if inferred.expected_col in cols else "(Select column)")
            expected_col = st.selectbox(
                "Expected column",
                cols,
                index=cols.index(default_expected) if default_expected in cols else 0,
                label_visibility="collapsed",
            )

        mapping = ColumnMapping(
            input_col=input_col if input_col != "(Select column)" else None,
            output_col=output_col if output_col != "(Select column)" else None,
            expected_col=expected_col if expected_col != "(Select column)" else None,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if mapping.is_valid():
            st.session_state.column_mapping = mapping
            st.success("✨ Column mapping saved! Ready for evaluation.")
        else:
            st.warning("⚠️ Please select at least the **LLM Output** column to continue")

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**📋 Column Types**")
            for i, col in enumerate(df.columns[:8]):
                dtype = str(df[col].dtype)
                st.markdown(f"""
                <div class="animated-list-item delay-{i % 5}" style="display: flex; justify-content: space-between; padding: 12px 16px;">
                    <span style="font-weight: 500; color: {theme.text_primary};">{col[:25]}</span>
                    <span style="color: {theme.accent}; font-family: 'DM Sans', sans-serif;">{dtype}</span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"**📊 Data Completeness**")
            for col in df.columns[:8]:
                completeness = 1 - (df[col].isnull().sum() / len(df))
                score_bar(completeness, label=col[:25])

else:
    # Empty state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not display_lottie("search", height=180):
            st.markdown("""
            <div style="text-align: center; padding: 40px;">
                <span style="font-size: 5rem; opacity: 0.5;">📂</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h3 style="color: {theme.text_primary}; margin-bottom: 8px;">No Data Yet</h3>
            <p style="color: {theme.text_muted};">Upload a file above to get started</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Format cards
    col1, col2, col3, col4 = st.columns(4)
    formats = [
        ("📄", "JSON/JSONL", "Arrays or newline-delimited objects"),
        ("📊", "CSV", "Tabular data with headers"),
        ("📕", "PDF", "Extracted via PyMuPDF4LLM"),
        ("🖼️", "Images", "Metadata extraction"),
    ]

    for i, (col, (icon, title, desc)) in enumerate(zip([col1, col2, col3, col4], formats)):
        with col:
            feature_card(icon, title, desc, delay=i+1)
