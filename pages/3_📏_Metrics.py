"""Rule-Based Metrics Evaluation Page - With Animations"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state, ColumnMapping, DataSourceType
from components.evaluators.metrics import (
    ExactMatchEvaluator,
    ContainsEvaluator,
    LevenshteinEvaluator,
    BLEUEvaluator,
    ROUGEEvaluator,
    LengthEvaluator,
    RegexEvaluator,
)
from components.ui import (
    inject_custom_css,
    animated_metric,
    score_bar,
    feature_card,
    display_lottie,
    section_header,
    get_current_theme,
    sidebar_logo,
)

st.set_page_config(page_title="Metrics | Vectory", page_icon="📏", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)
inject_custom_css()

# Animated header with developer colors
theme = get_current_theme()
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    📏 Rule-Based Metrics
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {theme.text_muted}; font-size: 1rem; margin-top: -10px;">
    <span style="color: {theme.terminal};">$</span> automated evaluation with BLEU, ROUGE, exact match, and more
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Check prerequisites
if st.session_state.dataset is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not display_lottie("search", height=180):
            st.markdown("""
            <div style="text-align: center; padding: 40px;">
                <span style="font-size: 5rem; opacity: 0.5;">📊</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
            <h3 style="margin: 0; color: white;">📁 No Dataset Loaded</h3>
            <p style="margin: 10px 0 0 0; opacity: 0.9; color: white;">
                Please upload a dataset on the <strong>Dataset</strong> page first
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# Get data source type
data_source_type = st.session_state.get("data_source_type", DataSourceType.TABULAR)
df = st.session_state.dataset

# Rule-based metrics work best with tabular data that has output/expected columns
if data_source_type != DataSourceType.TABULAR:
    st.info(f"📊 **Rule-based metrics** work best with tabular data (CSV/JSON) that has output and expected/reference columns.")
    st.markdown("For documents and images, consider using **LLM-as-Judge** evaluation instead.")

    # Still allow basic text analysis
    text_col = "text" if "text" in df.columns else df.columns[0]
    mapping_dict = {
        "input": None,
        "output": text_col,
        "expected": None,
    }
else:
    mapping = st.session_state.column_mapping
    if isinstance(mapping, dict):
        mapping = ColumnMapping(**{k.replace("output", "output_col").replace("input", "input_col").replace("expected", "expected_col"): v for k, v in mapping.items() if v})
    if not mapping.output_col:
        st.warning("⚠️ Please configure column mapping on the Dataset page.")
        st.stop()
    mapping_dict = {
        "input": mapping.input_col,
        "output": mapping.output_col,
        "expected": mapping.expected_col,
    }

# Metric selection with animated cards
section_header("🎯 Select Metrics", style="success", subtitle="Choose which metrics to compute on your data")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="feature-card" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">🔄 Comparison Metrics</h4>
        <p style="color: {theme.text_muted}; font-size: 0.85rem; margin-bottom: 16px;">Require a reference/expected column</p>
    </div>
    """, unsafe_allow_html=True)

    use_exact_match = st.checkbox("✓ Exact Match", value=True, help="Check if output exactly matches reference")
    use_contains = st.checkbox("✓ Contains Reference", help="Check if output contains the reference text")
    use_levenshtein = st.checkbox("✓ Levenshtein Similarity", help="Edit distance-based similarity")
    use_bleu = st.checkbox("✓ BLEU Score", help="N-gram overlap metric for translation/generation")
    use_rouge = st.checkbox("✓ ROUGE Scores", help="Recall-oriented metric for summarization")

with col2:
    st.markdown(f"""
    <div class="feature-card" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">📊 Standalone Metrics</h4>
        <p style="color: {theme.text_muted}; font-size: 0.85rem; margin-bottom: 16px;">Do not require a reference</p>
    </div>
    """, unsafe_allow_html=True)

    use_length = st.checkbox("✓ Length Analysis", value=True, help="Character, word, and sentence counts")
    use_regex = st.checkbox("✓ Custom Regex Pattern", help="Match outputs against a custom pattern")

    if use_regex:
        regex_pattern = st.text_input(
            "Regex Pattern",
            placeholder=r"\d+",
            help="Enter a valid regex pattern to match against outputs"
        )

# Check if reference is needed but not available
needs_reference = use_exact_match or use_contains or use_levenshtein or use_bleu or use_rouge
has_reference = mapping_dict.get("expected") is not None

if needs_reference and not has_reference:
    st.markdown(f"""
    <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; border-left: 4px solid {theme.error};">
        <p style="margin: 0; color: {theme.error};">
            <strong>⚠️ Missing Reference Column</strong><br>
            <span style="color: {theme.text_secondary};">Some selected metrics require a reference column. Please configure one on the Dataset page.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Run evaluation section
st.markdown("<br>", unsafe_allow_html=True)
section_header("⚡ Run Evaluation", style="info")

# Dataset info cards
col1, col2, col3 = st.columns(3)
with col1:
    animated_metric("Rows", str(len(df)), "📊", delay=1)
with col2:
    selected_count = sum([use_exact_match, use_contains, use_levenshtein, use_bleu, use_rouge, use_length, use_regex])
    animated_metric("Metrics", str(selected_count), "📏", delay=2)
with col3:
    animated_metric("Output Column", mapping_dict["output"][:15] if mapping_dict["output"] else "N/A", "📝", delay=3)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Run Selected Metrics", type="primary", use_container_width=True):
    outputs = df[mapping_dict["output"]].astype(str).tolist()
    references = df[mapping_dict["expected"]].astype(str).tolist() if has_reference else None
    inputs = df[mapping_dict["input"]].astype(str).tolist() if mapping_dict.get("input") else None

    all_results = []

    # Animated progress
    st.markdown('<div class="animated-progress" style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_metrics = sum([
        use_exact_match, use_contains, use_levenshtein,
        use_bleu, use_rouge, use_length, use_regex
    ])
    metric_count = [0]

    def update_progress():
        metric_count[0] += 1
        progress_bar.progress(metric_count[0] / total_metrics)

    # Run each selected metric
    if use_exact_match:
        status_text.markdown("**Computing Exact Match...**")
        evaluator = ExactMatchEvaluator()
        results = evaluator.evaluate_batch(outputs, references, inputs)
        all_results.append(results[["exact_match"]])
        update_progress()

    if use_contains:
        status_text.markdown("**Computing Contains Reference...**")
        evaluator = ContainsEvaluator()
        results = evaluator.evaluate_batch(outputs, references, inputs)
        all_results.append(results[["contains"]])
        update_progress()

    if use_levenshtein:
        status_text.markdown("**Computing Levenshtein Similarity...**")
        evaluator = LevenshteinEvaluator()
        results = evaluator.evaluate_batch(outputs, references, inputs)
        all_results.append(results[["levenshtein_similarity", "levenshtein_distance"]])
        update_progress()

    if use_bleu:
        status_text.markdown("**Computing BLEU Score...**")
        evaluator = BLEUEvaluator()
        results = evaluator.evaluate_batch(outputs, references, inputs)
        all_results.append(results[["bleu"]])
        update_progress()

    if use_rouge:
        status_text.markdown("**Computing ROUGE Scores...**")
        evaluator = ROUGEEvaluator()
        results = evaluator.evaluate_batch(outputs, references, inputs)
        all_results.append(results[["rouge1", "rouge2", "rougeL"]])
        update_progress()

    if use_length:
        status_text.markdown("**Computing Length Analysis...**")
        evaluator = LengthEvaluator()
        results = evaluator.evaluate_batch(outputs, references, inputs)
        cols = ["char_count", "word_count", "sentence_count"]
        if "length_ratio" in results.columns:
            cols.append("length_ratio")
        all_results.append(results[cols])
        update_progress()

    if use_regex and regex_pattern:
        status_text.markdown("**Computing Regex Match...**")
        try:
            evaluator = RegexEvaluator(regex_pattern)
            results = evaluator.evaluate_batch(outputs, references, inputs)
            all_results.append(results[["regex_match"]])
        except Exception as e:
            st.error(f"Invalid regex pattern: {e}")
        update_progress()

    status_text.empty()
    progress_bar.progress(1.0)

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, axis=1)
        st.session_state.evaluation_results["metrics"] = combined

        # Success banner
        st.markdown("""
        <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
            <span style="font-size: 2rem;">✨</span>
            <h3 style="margin: 10px 0 5px 0; color: white;">Evaluation Complete!</h3>
            <p style="margin: 0; opacity: 0.9; color: white;">All selected metrics have been computed</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary statistics with animated cards
        section_header("📊 Summary Statistics", style="primary")

        numeric_cols = combined.select_dtypes(include=["float64", "int64"]).columns

        # Score bars for key metrics
        score_cols = [c for c in numeric_cols if any(
            m in c for m in ["match", "bleu", "rouge", "similarity", "contains"]
        )]

        if score_cols:
            cols = st.columns(min(4, len(score_cols)))
            for i, col_name in enumerate(score_cols[:4]):
                with cols[i]:
                    mean_val = combined[col_name].mean()
                    animated_metric(col_name.replace("_", " ").title(), f"{mean_val:.3f}", "📈", delay=i+1)

        st.markdown("<br>", unsafe_allow_html=True)

        summary = combined[numeric_cols].describe()
        st.dataframe(summary, use_container_width=True)

        # Visualizations
        section_header("📈 Score Distributions", style="warning")

        if score_cols:
            fig = go.Figure()
            colors = ["#6366f1", "#a855f7", "#22d3ee", "#4ade80", "#fbbf24"]
            for i, col in enumerate(score_cols):
                fig.add_trace(go.Histogram(
                    x=combined[col].dropna(),
                    name=col.replace("_", " ").title(),
                    opacity=0.7,
                    marker_color=colors[i % len(colors)]
                ))
            fig.update_layout(
                xaxis_title="Score",
                yaxis_title="Count",
                barmode="overlay",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Per-sample results
        section_header("📋 Per-Sample Results", style="success")

        # Get original columns, avoiding duplicates with results
        orig_cols = [c for c in [mapping_dict["input"], mapping_dict["expected"], mapping_dict["output"]] if c]
        display_df = df[orig_cols].copy().reset_index(drop=True)

        # Rename duplicate columns in combined results
        results_df = combined.reset_index(drop=True)
        for col in results_df.columns:
            if col in display_df.columns:
                results_df = results_df.rename(columns={col: f"{col}_result"})

        display_df = pd.concat([display_df, results_df], axis=1)
        st.dataframe(display_df, use_container_width=True)

        # Export
        section_header("💾 Export Results", style="info")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download CSV",
                display_df.to_csv(index=False),
                "evaluation_results.csv",
                "text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "📋 Download JSON",
                display_df.to_json(orient="records", indent=2),
                "evaluation_results.json",
                "application/json",
                use_container_width=True,
            )

# Show previous results if available
elif "metrics" in st.session_state.get("evaluation_results", {}):
    st.markdown(f"""
    <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid {theme.info};">
        <p style="margin: 0; color: {theme.info};">
            📊 Previous evaluation results available. Run new evaluation to update.
        </p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.evaluation_results["metrics"]

    st.markdown(f"""
    <div class="feature-card">
        <h4 style="margin: 0 0 16px 0; color: {theme.text_primary};">📊 Previous Results Summary</h4>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = results.select_dtypes(include=["float64", "int64"]).columns
    summary = results[numeric_cols].describe()
    st.dataframe(summary, use_container_width=True)
