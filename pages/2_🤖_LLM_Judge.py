"""LLM-as-Judge Evaluation Page - With Animations"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_api_key(provider: str) -> str:
    """Get API key from session state or environment. Returns empty string if not found."""
    # Map provider to session state key and env var name
    key_map = {
        "openai": ("openai_api_key", "OPENAI_API_KEY"),
        "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    }

    session_key, env_key = key_map.get(provider, (None, None))
    if not session_key:
        return ""

    # Check session state first (set via Settings page)
    if session_key in st.session_state and st.session_state[session_key]:
        return st.session_state[session_key]

    # Check environment variables
    env_value = os.environ.get(env_key, "")
    if env_value:
        return env_value

    return ""

from components.models import init_session_state, ColumnMapping, DataSourceType
from components.model_catalog import DEFAULT_MODEL_BY_PROVIDER, get_model_ids, get_model_label
from components.evaluators.llm_judge import LLMJudgeEvaluator, estimate_cost, CRITERIA_TEMPLATES
from components.ui import (
    inject_custom_css,
    animated_metric,
    feature_card,
    display_lottie,
    score_bar,
    section_header,
    get_current_theme,
    sidebar_logo,
)

st.set_page_config(page_title="LLM Judge | Vectory", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)
inject_custom_css()

# Animated header with developer colors
theme = get_current_theme()
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    🤖 LLM-as-Judge Evaluation
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {theme.text_muted}; font-size: 1rem; margin-top: -10px;">
    <span style="color: {theme.terminal};">$</span> use current OpenAI or Anthropic models to evaluate your outputs
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Check prerequisites
if st.session_state.dataset is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not display_lottie("robot", height=180):
            st.markdown("""
            <div style="text-align: center; padding: 40px;">
                <span style="font-size: 5rem; opacity: 0.5;">🤖</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
            <h3 style="margin: 0; color: white; font-family: 'DM Sans', sans-serif;">📁 No Dataset Loaded</h3>
            <p style="margin: 10px 0 0 0; opacity: 0.9; color: white; font-family: 'DM Sans', sans-serif;">
                Please upload a dataset on the <strong>Dataset</strong> page first
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# Get data source type
data_source_type = st.session_state.get("data_source_type", DataSourceType.TABULAR)
df = st.session_state.dataset

# For tabular data, check column mapping
mapping = st.session_state.column_mapping
if isinstance(mapping, dict):
    mapping = ColumnMapping(**{k.replace("output", "output_col").replace("input", "input_col").replace("expected", "expected_col"): v for k, v in mapping.items() if v})

# Different handling based on data source type
if data_source_type == DataSourceType.TABULAR:
    if not mapping.output_col:
        st.warning("⚠️ Please configure column mapping on the Dataset page.")
        st.stop()
    mapping_dict = {
        "input": mapping.input_col,
        "output": mapping.output_col,
        "expected": mapping.expected_col,
    }
else:
    # For DOCUMENT and IMAGE types, we'll use the 'text' column or content directly
    mapping_dict = {
        "input": None,
        "output": "text" if "text" in df.columns else df.columns[0],
        "expected": None,
    }

# Show data source type indicator
source_type_labels = {
    DataSourceType.TABULAR: ("📊", "Tabular Data", "CSV/JSON with columns"),
    DataSourceType.DOCUMENT: ("📄", "Document", "PDF/Text content"),
    DataSourceType.IMAGE: ("🖼️", "Image", "Visual content"),
}
icon, label, desc = source_type_labels.get(data_source_type, ("📊", "Data", ""))
st.info(f"{icon} **{label}** - {desc}")

# Configuration section
section_header("⚙️ Configuration", style="primary")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="feature-card" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">🔑 API Settings</h4>
    </div>
    """, unsafe_allow_html=True)

    provider = st.selectbox(
        "Provider",
        ["openai", "anthropic"],
        help="Select the LLM provider"
    )

    # Get API key from session state/env or allow manual input
    default_key = get_api_key(provider)
    if default_key:
        st.success(f"✓ {provider.title()} API key loaded")
        api_key = default_key
    else:
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder=f"Enter your {provider.title()} API key",
            help="Enter your API key here or configure it in the Settings page"
        )
        # Save to session state for persistence
        if api_key:
            session_key = f"{provider}_api_key"
            st.session_state[session_key] = api_key

    model_options = get_model_ids(provider)
    if not model_options:
        st.warning(f"No public model catalog entries are configured for {provider.title()}.")
        st.stop()

    default_model = DEFAULT_MODEL_BY_PROVIDER.get(provider, model_options[0])

    model = st.selectbox(
        "Model",
        model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
        format_func=lambda model_id: get_model_label(provider, model_id),
        help="Select the model for evaluation"
    )

with col2:
    st.markdown(f"""
    <div class="feature-card" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">🎯 Evaluation Settings</h4>
    </div>
    """, unsafe_allow_html=True)

    criteria_type = st.selectbox(
        "Evaluation Criteria",
        list(CRITERIA_TEMPLATES.keys()),
        help="What aspect to evaluate"
    )

    custom_criteria = None
    if criteria_type == "custom":
        custom_criteria = st.text_area(
            "Custom Criteria",
            placeholder="Describe what the LLM should evaluate...",
            help="Enter your custom evaluation criteria"
        )

    include_reference = st.checkbox(
        "Include reference in prompt",
        value=True,
        help="Whether to show the reference/expected output to the judge"
    )

    rate_limit = st.slider(
        "Rate limit (seconds between calls)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Delay between API calls to avoid rate limits"
    )

# Document/Image specific settings
if data_source_type in [DataSourceType.DOCUMENT, DataSourceType.IMAGE]:
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("📝 Evaluation Prompt", style="accent", subtitle="Define what the LLM should evaluate about your content")

    if data_source_type == DataSourceType.DOCUMENT:
        eval_prompt = st.text_area(
            "Evaluation Prompt",
            value="Analyze this document and evaluate it based on the following criteria:\n1. Clarity and organization\n2. Completeness of information\n3. Quality of writing\n\nProvide a score from 1-5 and explain your reasoning.",
            height=150,
            help="Describe what the LLM should evaluate about the document"
        )
    else:  # IMAGE
        eval_prompt = st.text_area(
            "Evaluation Prompt",
            value="Analyze this image and evaluate it based on:\n1. Visual quality\n2. Content relevance\n3. Information conveyed\n\nProvide a score from 1-5 and explain your reasoning.",
            height=150,
            help="Describe what the LLM should evaluate about the image"
        )
else:
    eval_prompt = None

# Cost estimation
st.markdown("<br>", unsafe_allow_html=True)
section_header("💰 Cost Estimate", style="warning")

num_samples = len(df)
cost_info = estimate_cost(num_samples, provider, model)

col1, col2, col3 = st.columns(3)
with col1:
    animated_metric("Samples", str(num_samples), "📊", delay=1)
with col2:
    animated_metric("Est. Tokens", f"{cost_info.get('total_tokens', 'N/A'):,}", "📝", delay=2)
with col3:
    animated_metric("Est. Cost", cost_info.get("estimated_cost", "Unknown"), "💵", delay=3)

if "warning" in cost_info:
    st.caption(cost_info["warning"])

# Advanced settings
with st.expander("🔧 Advanced Settings"):
    use_custom_prompt = st.checkbox("Use custom prompt template")

    custom_prompt = None
    if use_custom_prompt:
        custom_prompt = st.text_area(
            "Custom Prompt Template",
            value="""Evaluate the following response:

Input: {input}
Response: {output}
Reference: {reference}

Rate the response on a scale of 1-5 and provide reasoning.

Format:
Score: [1-5]
Reasoning: [Your explanation]""",
            height=200,
            help="Use {input}, {output}, and {reference} as placeholders"
        )

    sample_subset = st.checkbox("Evaluate a subset of samples")
    if sample_subset:
        max_samples = st.number_input(
            "Maximum samples to evaluate",
            min_value=1,
            max_value=len(df),
            value=min(10, len(df))
        )
    else:
        max_samples = len(df)

# Run evaluation
st.markdown("<br>", unsafe_allow_html=True)
section_header("⚡ Run Evaluation", style="info")

if not api_key:
    st.markdown(f"""
    <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; border-left: 4px solid {theme.error};">
        <p style="margin: 0; color: {theme.error};">
            <strong>🔑 API Key Required</strong><br>
            <span style="color: {theme.text_secondary};">Please enter your API key above to run evaluation.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    if st.button("🚀 Run LLM Evaluation", type="primary", use_container_width=True):
        # Prepare data based on data source type
        eval_df = df.head(max_samples) if sample_subset else df

        if data_source_type == DataSourceType.TABULAR:
            # Standard tabular evaluation with column mapping
            outputs = eval_df[mapping_dict["output"]].astype(str).tolist()
            references = eval_df[mapping_dict["expected"]].astype(str).tolist() if mapping_dict.get("expected") else None
            inputs = eval_df[mapping_dict["input"]].astype(str).tolist() if mapping_dict.get("input") else None
            effective_prompt = custom_prompt if use_custom_prompt else None

        elif data_source_type == DataSourceType.DOCUMENT:
            # Document evaluation - use text column with custom eval prompt
            text_col = "text" if "text" in eval_df.columns else eval_df.columns[0]
            outputs = eval_df[text_col].astype(str).tolist()
            references = None
            inputs = [eval_prompt] * len(outputs)  # Use eval_prompt as the instruction
            effective_prompt = f"""You are evaluating a document based on the following criteria:

{eval_prompt}

Document Content:
{{output}}

Provide your evaluation in this format:
Score: [1-5]
Reasoning: [Your detailed explanation]"""

        elif data_source_type == DataSourceType.IMAGE:
            # Image evaluation - currently limited to metadata
            # Full image evaluation would require vision API
            text_col = "text" if "text" in eval_df.columns else eval_df.columns[0]
            outputs = eval_df[text_col].astype(str).tolist()
            references = None
            inputs = [eval_prompt] * len(outputs)
            effective_prompt = f"""You are evaluating image metadata based on the following criteria:

{eval_prompt}

Image Information:
{{output}}

Note: This evaluation is based on image metadata. For full visual analysis, vision-capable models are required.

Provide your evaluation in this format:
Score: [1-5]
Reasoning: [Your explanation based on available information]"""
            st.info("📝 Note: Image evaluation is currently based on metadata. Full visual analysis requires vision API integration.")

        # Initialize evaluator
        try:
            evaluator = LLMJudgeEvaluator(
                provider=provider,
                api_key=api_key,
                model=model,
                criteria=criteria_type if data_source_type == DataSourceType.TABULAR else "custom",
                custom_criteria=custom_criteria if data_source_type == DataSourceType.TABULAR else eval_prompt,
                custom_prompt=effective_prompt,
                include_reference=include_reference if data_source_type == DataSourceType.TABULAR else False,
                rate_limit_delay=rate_limit,
            )
        except Exception as e:
            st.error(f"Error initializing evaluator: {e}")
            st.stop()

        # Run evaluation with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("**Starting evaluation...**")

        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.markdown(f"**Evaluating... {int(progress * 100)}% complete**")

        try:
            results = evaluator.evaluate_batch(outputs, references, inputs, progress_callback=update_progress)

            status_text.empty()
            progress_bar.progress(1.0)

            # Store results
            st.session_state.evaluation_results["llm_judge"] = results

            # Success banner
            st.markdown("""
            <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
                <span style="font-size: 2rem;">✨</span>
                <h3 style="margin: 10px 0 5px 0; color: white;">Evaluation Complete!</h3>
                <p style="margin: 0; opacity: 0.9; color: white;">All samples have been evaluated by the LLM</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Summary metrics
            section_header("📊 Score Summary", style="success")

            valid_scores = results["score"].dropna()

            if len(valid_scores) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    animated_metric("Average", f"{valid_scores.mean():.2f}", "📈", delay=1)
                with col2:
                    animated_metric("Median", f"{valid_scores.median():.1f}", "📊", delay=2)
                with col3:
                    animated_metric("Min", f"{valid_scores.min():.0f}", "📉", delay=3)
                with col4:
                    animated_metric("Max", f"{valid_scores.max():.0f}", "📈", delay=4)

                st.markdown("<br>", unsafe_allow_html=True)

                # Score distribution
                section_header("📈 Score Distribution", style="warning")

                fig = px.histogram(
                    results,
                    x="score",
                    nbins=5,
                    color_discrete_sequence=["#667eea"],
                    labels={"score": "Score (1-5)", "count": "Count"}
                )
                fig.update_xaxes(dtick=1)
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed results
            section_header("📋 Detailed Results", style="primary")

            display_cols = [c for c in [mapping_dict["input"], mapping_dict["expected"], mapping_dict["output"]] if c]
            display_df = eval_df[display_cols].reset_index(drop=True).copy()
            display_df["Score"] = results["score"].values
            display_df["Reasoning"] = results["reasoning"].values

            if "error" in results.columns:
                errors = results["error"].dropna()
                if len(errors) > 0:
                    st.warning(f"⚠️ {len(errors)} samples had errors during evaluation")
                    display_df["Error"] = results["error"].values

            st.dataframe(display_df, use_container_width=True)

            # Export
            section_header("💾 Export Results", style="info")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download CSV",
                    display_df.to_csv(index=False),
                    "llm_judge_results.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "📋 Download JSON",
                    display_df.to_json(orient="records", indent=2),
                    "llm_judge_results.json",
                    "application/json",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Error during evaluation: {e}")

# Show previous results if available
if "llm_judge" in st.session_state.get("evaluation_results", {}):
    st.markdown(f"""
    <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid {theme.info};">
        <p style="margin: 0; color: {theme.info};">
            📊 Previous LLM Judge results available. Configure settings and run to update.
        </p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.evaluation_results["llm_judge"]
    valid_scores = results["score"].dropna()

    if len(valid_scores) > 0:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin: 0 0 16px 0;">📊 Previous Results Summary</h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            score_bar(valid_scores.mean() / 5, label="Average Score")
        with col2:
            st.markdown(f"**Samples Evaluated:** {len(results)}")
