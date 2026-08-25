"""
Error Analysis Page - Trace Viewer and Failure Mode Taxonomy.

Based on "Application-Centric AI Evals" by Shankar & Husain.
Implements grounded theory approach: Open Coding → Axial Coding → Failure Taxonomy
"""

import streamlit as st
import pandas as pd
import hashlib
import json
import os
from datetime import datetime
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state, ColumnMapping, DataSourceType
from components.model_catalog import DEFAULT_MODEL_BY_PROVIDER, get_model_ids, get_model_label
from components.error_discovery import generate_taxonomy_suggestions
from components.eval_workflow import (
    build_evaluator_definition,
    build_review_bundle,
    dataframe_to_records,
    find_related_records,
    normalize_trace_segments,
    select_diverse_samples,
    summarize_coverage,
)
from components.ui import (
    inject_custom_css,
    animated_metric,
    section_header,
    get_current_theme,
    hex_to_rgb,
    display_lottie,
)


def get_api_key(provider: str) -> str:
    """Get provider API key from session state or environment."""
    key_map = {
        "openai": ("openai_api_key", "OPENAI_API_KEY"),
        "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    }
    session_key, env_key = key_map.get(provider, (None, None))
    if not session_key:
        return ""
    if st.session_state.get(session_key):
        return st.session_state[session_key]
    return os.environ.get(env_key, "")

st.set_page_config(
    page_title="Error Analysis | Vectory",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_session_state(st)
inject_custom_css()

# Initialize error analysis state
if "failure_modes" not in st.session_state:
    st.session_state.failure_modes = {}  # {name: {description, examples}}

if "trace_annotations" not in st.session_state:
    st.session_state.trace_annotations = {}  # {idx: {open_code, failure_modes, pass_fail}}

if "open_codes" not in st.session_state:
    st.session_state.open_codes = []  # List of raw annotations from open coding

if "taxonomy_suggestions" not in st.session_state:
    st.session_state.taxonomy_suggestions = []

if "dismissed_taxonomy_suggestions" not in st.session_state:
    st.session_state.dismissed_taxonomy_suggestions = []

if "related_trace_suggestions" not in st.session_state:
    st.session_state.related_trace_suggestions = {}

if "promoted_evaluators" not in st.session_state:
    st.session_state.promoted_evaluators = {}

if "principal_reviewer_name" not in st.session_state:
    st.session_state.principal_reviewer_name = ""

if "principal_reviewer_role" not in st.session_state:
    st.session_state.principal_reviewer_role = ""

theme = get_current_theme()

# Header
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    🔍 Error Analysis
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {theme.text_muted}; font-size: 1rem; margin-top: -10px;">
    <span style="color: {theme.terminal};">$</span> systematic failure mode discovery using grounded theory
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
                <span style="font-size: 5rem; opacity: 0.5;">🔍</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
            <h3 style="margin: 0; color: white; font-family: 'DM Sans', sans-serif;">📁 No Dataset Loaded</h3>
            <p style="margin: 10px 0 0 0; opacity: 0.9; color: white;">
                Please upload a dataset on the <strong>Dataset</strong> page first
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.dataset
data_source_type = st.session_state.get("data_source_type", DataSourceType.TABULAR)

if data_source_type == DataSourceType.TABULAR:
    mapping = st.session_state.column_mapping
    if not isinstance(mapping, ColumnMapping):
        mapping = ColumnMapping(**mapping) if isinstance(mapping, dict) else ColumnMapping()
else:
    text_col = "text" if "text" in df.columns else df.columns[0]
    mapping = ColumnMapping(output_col=text_col)

total = len(df)
if total == 0:
    st.warning("⚠️ The loaded dataset has no rows to review.")
    st.stop()

records = dataframe_to_records(df)
default_dimension_fields = [
    column
    for column in df.columns
    if str(column).casefold() in {"feature", "scenario", "persona"}
]
dataset_digest = hashlib.sha256(
    json.dumps(records, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
).hexdigest()
dataset_signature = f"{st.session_state.get('dataset_filename')}:{dataset_digest}"
previous_dataset_signature = st.session_state.get("error_discovery_dataset_signature")
if previous_dataset_signature != dataset_signature:
    if previous_dataset_signature is not None:
        st.session_state.failure_modes = {}
        st.session_state.trace_annotations = {}
        st.session_state.open_codes = []
        st.session_state.taxonomy_suggestions = []
        st.session_state.dismissed_taxonomy_suggestions = []
    st.session_state.error_discovery_dataset_signature = dataset_signature
    st.session_state.discovery_samples = select_diverse_samples(
        records,
        sample_size=min(30, total),
        seed=42,
        dimension_fields=default_dimension_fields,
    )
    st.session_state.discovery_sample_position = 0
    st.session_state.related_trace_suggestions = {}
    st.session_state.promoted_evaluators = {}

annotations = st.session_state.trace_annotations
failure_modes = st.session_state.failure_modes
discovery_samples = st.session_state.discovery_samples

# Sidebar - Workflow phases
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">🔬</span>
        <h3 style="margin: 8px 0; color: {theme.text_primary};">Analysis Phase</h3>
    </div>
    """, unsafe_allow_html=True)

    phase = st.radio(
        "Select Phase",
        ["📖 Open Coding", "🔗 Axial Coding", "📊 Taxonomy Dashboard"],
        label_visibility="collapsed"
    )

    with st.expander("🧭 Discovery Sample", expanded=False):
        st.text_input(
            "Principal domain expert",
            key="principal_reviewer_name",
            help="The person whose judgment defines acceptable product behavior.",
        )
        st.text_input(
            "Expert role",
            key="principal_reviewer_role",
            placeholder="e.g., Support Director",
        )
        selected_dimension_fields = st.multiselect(
            "Coverage dimensions",
            list(df.columns),
            default=default_dimension_fields,
            help="Product dimensions such as feature, scenario, or persona.",
        )
        requested_sample_size = st.number_input(
            "Sample size",
            min_value=1,
            max_value=total,
            value=min(max(1, len(discovery_samples)), total),
            help="Mixes cluster representatives with random exploration records.",
        )
        discovery_seed = st.number_input("Sampling seed", min_value=0, value=42)
        if st.button("Regenerate diverse sample", use_container_width=True):
            st.session_state.discovery_samples = select_diverse_samples(
                records,
                sample_size=int(requested_sample_size),
                seed=int(discovery_seed),
                dimension_fields=selected_dimension_fields,
            )
            st.session_state.discovery_sample_position = 0
            discovery_samples = st.session_state.discovery_samples
            st.rerun()
        representative_count = sum(
            sample.get("strategy") in {"cluster_representative", "dimension_coverage"}
            for sample in discovery_samples
        )
        st.caption(
            f"{len(discovery_samples)} selected · "
            f"{representative_count} coverage selections"
        )
        st.warning("Use this sample to discover failure modes, not to estimate prevalence.")

    st.markdown("---")

    # Progress
    annotated = len(annotations)
    progress_pct = annotated / total if total > 0 else 0

    st.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 1.5rem;">📊</span>
        <h4 style="margin: 8px 0; color: {theme.text_primary};">Progress</h4>
    </div>
    """, unsafe_allow_html=True)

    st.progress(progress_pct)
    st.markdown(f"""
    <div style="text-align: center;">
        <span style="font-size: 1.2rem; font-weight: 600;">{annotated}</span>
        <span style="color: {theme.text_muted};"> / {total} traces</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick stats
    if annotations:
        fail_count = sum(1 for a in annotations.values() if not a.get("pass_fail", True))
        st.markdown(f"""
        <div style="background: rgba({hex_to_rgb(theme.error)}, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 8px;">
            <div style="text-align: center;">
                <span style="font-size: 1.5rem;">❌</span>
                <p style="margin: 4px 0 0 0; color: {theme.error}; font-weight: 600;">{fail_count} Failures</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pass_count = annotated - fail_count
        st.markdown(f"""
        <div style="background: rgba({hex_to_rgb(theme.success)}, 0.1); padding: 12px; border-radius: 8px;">
            <div style="text-align: center;">
                <span style="font-size: 1.5rem;">✅</span>
                <p style="margin: 4px 0 0 0; color: {theme.success}; font-weight: 600;">{pass_count} Passes</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PHASE 1: Open Coding - Read and label traces
# =============================================================================
if phase == "📖 Open Coding":
    section_header("📖 Open Coding", style="primary")

    st.markdown(f"""
    <div style="background: rgba({hex_to_rgb(theme.info)}, 0.1); padding: 16px; border-radius: 8px; border-left: 4px solid {theme.info}; margin-bottom: 20px;">
        <strong>Open Coding</strong> is the first step of error analysis. Read each trace carefully and write
        free-form notes about what you observe: where outputs are incorrect, surprising, or feel wrong.
        Focus on the <strong>first (most upstream) failure</strong> to avoid cataloging cascading errors.
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    nav_mode = st.radio(
        "Navigation",
        ["Diverse Sample", "Sequential", "Failures Only", "Unannotated"],
        horizontal=True,
    )

    if nav_mode == "Diverse Sample":
        sample_indices = [int(item["index"]) for item in discovery_samples]
        sample_position = min(
            int(st.session_state.get("discovery_sample_position", 0)),
            len(sample_indices) - 1,
        )
        st.session_state.discovery_sample_position = sample_position
        idx = sample_indices[sample_position]
        selected_sample = discovery_samples[sample_position]
        st.caption(
            f"Discovery sample {sample_position + 1} of {len(sample_indices)} · "
            f"{selected_sample.get('reason', 'Coverage sample')}"
        )
    elif nav_mode == "Sequential":
        if "error_idx" not in st.session_state:
            st.session_state.error_idx = 0
        idx = st.session_state.error_idx
    elif nav_mode == "Failures Only":
        failures = [i for i, a in annotations.items() if not a.get("pass_fail", True)]
        if failures:
            idx = st.selectbox("Select failure", failures, format_func=lambda x: f"Trace {x + 1}")
        else:
            st.info("No failures annotated yet.")
            idx = 0
    else:  # Unannotated
        unannotated = [i for i in range(total) if i not in annotations]
        if unannotated:
            idx = unannotated[0]
            st.info(f"📋 {len(unannotated)} traces remaining")
        else:
            st.success("✅ All traces annotated!")
            idx = 0

    # Navigation buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        previous_disabled = sample_position == 0 if nav_mode == "Diverse Sample" else idx == 0
        if st.button("← Prev", use_container_width=True, disabled=previous_disabled):
            if nav_mode == "Diverse Sample":
                st.session_state.discovery_sample_position = max(0, sample_position - 1)
            else:
                st.session_state.error_idx = max(0, idx - 1)
            st.rerun()
    with col2:
        next_disabled = (
            sample_position >= len(sample_indices) - 1
            if nav_mode == "Diverse Sample"
            else idx >= total - 1
        )
        if st.button("Next →", use_container_width=True, disabled=next_disabled):
            if nav_mode == "Diverse Sample":
                st.session_state.discovery_sample_position = min(
                    len(sample_indices) - 1,
                    sample_position + 1,
                )
            else:
                st.session_state.error_idx = min(total - 1, idx + 1)
            st.rerun()
    with col3:
        if nav_mode == "Diverse Sample":
            jump = st.number_input(
                "Jump to sample #",
                1,
                len(sample_indices),
                sample_position + 1,
                label_visibility="collapsed",
            )
        else:
            jump = st.number_input("Jump to trace #", 1, total, idx + 1, label_visibility="collapsed")
    with col4:
        if st.button("Go", use_container_width=True):
            if nav_mode == "Diverse Sample":
                st.session_state.discovery_sample_position = jump - 1
            else:
                st.session_state.error_idx = jump - 1
            st.rerun()
    with col5:
        status = "✅" if idx in annotations and annotations[idx].get("pass_fail", True) else "❌" if idx in annotations else "○"
        st.markdown(f"""
        <div style="text-align: center; padding: 8px;">
            <span style="font-size: 1.5rem;">{status}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Trace display header
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <h3 style="margin: 0; color: {theme.text_primary};">Trace {idx + 1} of {total}</h3>
    </div>
    """, unsafe_allow_html=True)

    row = df.iloc[idx]
    existing = annotations.get(idx, {})

    trace_segments = normalize_trace_segments(records[idx])
    if trace_segments:
        st.markdown(f"<h4 style=\"color: {theme.text_primary};\">🧩 Agent Trace</h4>", unsafe_allow_html=True)
        for segment in trace_segments:
            label = (
                f"{segment['index'] + 1}. {segment['title']} "
                f"· {segment['kind'].replace('_', ' ')}"
            )
            with st.expander(label, expanded=segment["index"] == 0):
                if segment["content"]:
                    st.code(segment["content"], language=None)
                if segment["details"]:
                    st.json(segment["details"])
        st.markdown("<br>", unsafe_allow_html=True)

    # Display trace
    col1, col2 = st.columns(2)

    with col1:
        if mapping.input_col and mapping.input_col in df.columns:
            st.markdown(f"""
            <div style="background: rgba({hex_to_rgb(theme.info)}, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
                <h4 style="margin: 0 0 8px 0; color: {theme.info};">📝 Input / Query</h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area("Input", str(row[mapping.input_col]), height=150, disabled=True, label_visibility="collapsed")

        if mapping.expected_col and mapping.expected_col in df.columns:
            st.markdown(f"""
            <div style="background: rgba({hex_to_rgb(theme.success)}, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
                <h4 style="margin: 0 0 8px 0; color: {theme.success};">✓ Expected / Reference</h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area("Expected", str(row[mapping.expected_col]), height=150, disabled=True, label_visibility="collapsed")

    with col2:
        if mapping.output_col and mapping.output_col in df.columns:
            st.markdown(f"""
            <div style="background: rgba({hex_to_rgb(theme.warning)}, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
                <h4 style="margin: 0 0 8px 0; color: {theme.warning};">🤖 LLM Output</h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area("Output", str(row[mapping.output_col]), height=350, disabled=True, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # Annotation form
    section_header("✏️ Annotate This Trace", style="info")

    col1, col2 = st.columns([2, 1])

    with col1:
        open_code = st.text_area(
            "Expert Critique / Open Code",
            value=existing.get("open_code", ""),
            placeholder=(
                "Explain why this passes or fails, with enough detail for a new teammate to understand. "
                "For failures, focus on the first observable problem rather than guessing at its technical cause."
            ),
            height=120,
            help="Detailed critiques later become grounded examples for evaluator development."
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        pass_fail = st.radio(
            "Overall Verdict",
            ["✅ Pass", "❌ Fail"],
            index=0 if existing.get("pass_fail", True) else 1,
            help="Is this trace acceptable or unacceptable?"
        )
        is_pass = pass_fail == "✅ Pass"

        # If failure modes exist, allow tagging
        if failure_modes and not is_pass:
            st.markdown("<br>", unsafe_allow_html=True)
            tagged_modes = st.multiselect(
                "Failure Modes",
                list(failure_modes.keys()),
                default=existing.get("failure_modes", []),
                help="Tag with structured failure modes from axial coding"
            )
        else:
            tagged_modes = []

    # Save button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💾 Save & Next", type="primary", use_container_width=True):
            if not st.session_state.principal_reviewer_name.strip():
                st.error("Identify the principal domain expert in the Discovery Sample sidebar first.")
            elif not open_code.strip():
                st.error("Add a detailed critique before saving the verdict.")
            else:
                annotations[idx] = {
                    "open_code": open_code.strip(),
                    "critique": open_code.strip(),
                    "pass_fail": is_pass,
                    "failure_modes": tagged_modes,
                    "reviewer": st.session_state.principal_reviewer_name,
                    "reviewer_role": st.session_state.principal_reviewer_role,
                    "timestamp": datetime.now().isoformat(),
                }
                st.session_state.trace_annotations = annotations

                # Add to open codes list if it's a new annotation
                if open_code not in st.session_state.open_codes:
                    st.session_state.open_codes.append(open_code.strip())

                st.success("✅ Saved!")

                # Auto-advance
                if nav_mode == "Diverse Sample" and sample_position < len(sample_indices) - 1:
                    st.session_state.discovery_sample_position = sample_position + 1
                    st.rerun()
                elif nav_mode != "Diverse Sample" and idx < total - 1:
                    st.session_state.error_idx = idx + 1
                    st.rerun()

    with col2:
        if idx in annotations and st.button("🗑️ Clear", use_container_width=True):
            del annotations[idx]
            st.session_state.trace_annotations = annotations
            st.rerun()


# =============================================================================
# PHASE 2: Axial Coding - Structure failure modes
# =============================================================================
elif phase == "🔗 Axial Coding":
    section_header("🔗 Axial Coding", style="primary")

    st.markdown(f"""
    <div style="background: rgba({hex_to_rgb(theme.info)}, 0.1); padding: 16px; border-radius: 8px; border-left: 4px solid {theme.info}; margin-bottom: 20px;">
        <strong>Axial Coding</strong> organizes your open codes into structured failure categories.
        Group similar annotations together to create a coherent, non-overlapping taxonomy of failure modes.
    </div>
    """, unsafe_allow_html=True)

    # Display existing open codes
    open_codes = [a.get("open_code", "") for a in annotations.values() if a.get("open_code") and not a.get("pass_fail", True)]

    if not open_codes:
        st.warning("⚠️ No failure annotations yet. Complete Open Coding first to identify failures.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <h4 style="color: {theme.text_primary};">📝 Open Codes (Failures)</h4>
        """, unsafe_allow_html=True)

        for i, code in enumerate(open_codes[:20], 1):  # Show first 20
            st.markdown(f"""
            <div style="background: rgba({hex_to_rgb(theme.error)}, 0.1); padding: 10px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid {theme.error};">
                <small style="color: {theme.text_muted};">#{i}</small><br>
                {code[:200]}{'...' if len(code) > 200 else ''}
            </div>
            """, unsafe_allow_html=True)

        if len(open_codes) > 20:
            st.info(f"Showing 20 of {len(open_codes)} open codes")

    with col2:
        st.markdown(f"""
        <h4 style="color: {theme.text_primary};">📂 Failure Mode Taxonomy</h4>
        """, unsafe_allow_html=True)

        # Display existing failure modes
        for name, details in failure_modes.items():
            with st.expander(f"🏷️ {name}", expanded=False):
                st.write(details.get("description", ""))
                examples = details.get("examples", [])
                if examples:
                    st.markdown("**Examples:**")
                    for ex in examples[:3]:
                        st.markdown(f"- {ex[:100]}...")

                if st.button("Find related traces", key=f"find_related_{name}"):
                    tagged_notes = [
                        annotation.get("open_code", "")
                        for annotation in annotations.values()
                        if name in annotation.get("failure_modes", [])
                    ]
                    st.session_state.related_trace_suggestions[name] = find_related_records(
                        records,
                        [details.get("description", ""), *examples, *tagged_notes],
                        exclude_indices=annotations.keys(),
                        limit=10,
                    )

                related = st.session_state.related_trace_suggestions.get(name, [])
                if related:
                    st.caption("Candidate traces for human confirmation")
                    for candidate in related:
                        candidate_idx = int(candidate["index"])
                        preview_field = mapping.output_col if mapping.output_col in df.columns else df.columns[0]
                        st.code(str(df.iloc[candidate_idx][preview_field])[:500], language=None)
                        action_col, score_col = st.columns([2, 1])
                        with action_col:
                            already_selected = candidate_idx in {
                                int(item["index"]) for item in st.session_state.discovery_samples
                            }
                            if st.button(
                                "Added to review queue" if already_selected else "Add to review queue",
                                key=f"queue_related_{name}_{candidate_idx}",
                                disabled=already_selected,
                                use_container_width=True,
                            ):
                                st.session_state.discovery_samples.append(
                                    {
                                        "index": candidate_idx,
                                        "cluster": -1,
                                        "strategy": "targeted_related",
                                        "reason": f"Related candidate for {name}",
                                        "similarity": candidate["score"],
                                    }
                                )
                                st.rerun()
                        with score_col:
                            st.caption(f"Similarity {candidate['score']:.2f}")

                if st.button(f"🗑️ Delete", key=f"del_{name}"):
                    del failure_modes[name]
                    st.session_state.failure_modes = failure_modes
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Add new failure mode
        st.markdown(f"""
        <h5 style="color: {theme.text_primary};">➕ Add Failure Mode</h5>
        """, unsafe_allow_html=True)

        new_name = st.text_input("Name", placeholder="e.g., Missing Constraint, Hallucinated Facts")
        new_desc = st.text_area("Description", placeholder="Clear definition of when this failure occurs", height=80)

        if st.button("Add Failure Mode", type="primary"):
            if new_name:
                failure_modes[new_name] = {
                    "description": new_desc,
                    "examples": [],
                    "created": datetime.now().isoformat(),
                }
                st.session_state.failure_modes = failure_modes
                st.success(f"✅ Added: {new_name}")
                st.rerun()
            else:
                st.error("Please provide a name")

    st.markdown("<br>", unsafe_allow_html=True)

    # AI-assisted clustering
    section_header("🤖 AI Taxonomy Suggestions", style="info")

    st.markdown(f"""
    <div style="background: rgba({hex_to_rgb(theme.info)}, 0.08); padding: 16px; border-radius: 8px; border-left: 4px solid {theme.info}; margin-bottom: 20px;">
        Use an LLM to propose failure modes from your failure notes. Suggestions stay pending until you accept them.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        suggestion_provider = st.selectbox(
            "Provider",
            ["openai", "anthropic"],
            key="taxonomy_provider",
            help="Select the LLM provider for taxonomy suggestions",
        )
    with c2:
        model_options = get_model_ids(suggestion_provider)
        if not model_options:
            st.warning(f"No public model catalog entries are configured for {suggestion_provider.title()}.")
            st.stop()

        default_model = DEFAULT_MODEL_BY_PROVIDER.get(suggestion_provider, model_options[0])
        suggestion_model = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            format_func=lambda model_id: get_model_label(suggestion_provider, model_id),
            key=f"taxonomy_model_{suggestion_provider}",
            help="Select the model for clustering open-code notes",
        )
    with c3:
        default_key = get_api_key(suggestion_provider)
        if default_key:
            st.success(f"✓ {suggestion_provider.title()} API key loaded")
            suggestion_api_key = default_key
        else:
            suggestion_api_key = st.text_input(
                "API Key",
                type="password",
                placeholder=f"Enter your {suggestion_provider.title()} API key",
                key=f"taxonomy_{suggestion_provider}_api_key",
            )
            if suggestion_api_key:
                st.session_state[f"{suggestion_provider}_api_key"] = suggestion_api_key

    button_col, info_col = st.columns([1, 2])
    with button_col:
        generate_disabled = not suggestion_api_key or len(open_codes) < 2
        if st.button(
            "Generate Suggestions",
            type="primary",
            use_container_width=True,
            disabled=generate_disabled,
            help="Requires at least two failure notes and an API key",
        ):
            try:
                with st.spinner("Clustering failure notes..."):
                    st.session_state.taxonomy_suggestions = generate_taxonomy_suggestions(
                        open_codes=open_codes,
                        existing_failure_modes=failure_modes,
                        provider=suggestion_provider,
                        api_key=suggestion_api_key,
                        model=suggestion_model,
                    )
                st.success(f"Generated {len(st.session_state.taxonomy_suggestions)} suggestions")
            except Exception as e:
                st.error(f"Could not generate suggestions: {e}")

    with info_col:
        st.caption(f"Using {len(open_codes)} failure notes. Existing failure modes are included so the model can avoid duplicates.")

    suggestions = st.session_state.taxonomy_suggestions
    if suggestions:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <h4 style="color: {theme.text_primary};">Pending Suggestions</h4>
        """, unsafe_allow_html=True)

        for i, suggestion in enumerate(suggestions):
            name = suggestion.get("name", "")
            with st.expander(f"💡 {name}", expanded=i == 0):
                st.markdown(f"**Definition:** {suggestion.get('description', '')}")
                if suggestion.get("rationale"):
                    st.markdown(f"**Rationale:** {suggestion.get('rationale')}")
                examples = suggestion.get("examples", [])
                if examples:
                    st.markdown("**Supporting notes:**")
                    for example in examples[:3]:
                        st.markdown(f"- {example}")

                accept_col, dismiss_col = st.columns(2)
                with accept_col:
                    if st.button("Accept", key=f"accept_taxonomy_{i}", use_container_width=True):
                        failure_modes[name] = {
                            "description": suggestion.get("description", ""),
                            "examples": examples,
                            "created": datetime.now().isoformat(),
                            "source": "ai_suggestion",
                        }
                        st.session_state.failure_modes = failure_modes
                        st.session_state.taxonomy_suggestions = [
                            item for j, item in enumerate(suggestions) if j != i
                        ]
                        st.success(f"Added: {name}")
                        st.rerun()
                with dismiss_col:
                    if st.button("Dismiss", key=f"dismiss_taxonomy_{i}", use_container_width=True):
                        st.session_state.dismissed_taxonomy_suggestions.append(
                            {**suggestion, "dismissed_at": datetime.now().isoformat()}
                        )
                        st.session_state.taxonomy_suggestions = [
                            item for j, item in enumerate(suggestions) if j != i
                        ]
                        st.rerun()
    elif len(open_codes) < 2:
        st.info("Add at least two failed trace notes before generating taxonomy suggestions.")


# =============================================================================
# PHASE 3: Taxonomy Dashboard
# =============================================================================
else:  # Taxonomy Dashboard
    section_header("📊 Failure Mode Dashboard", style="primary")

    if not annotations:
        st.warning("⚠️ No traces annotated yet. Complete Open Coding first.")
        st.stop()

    # Compute statistics
    total_annotated = len(annotations)
    failures = [a for a in annotations.values() if not a.get("pass_fail", True)]
    passes = total_annotated - len(failures)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        animated_metric("Annotated", f"{total_annotated}/{total}", "📊", delay=1)
    with col2:
        pass_rate = (passes / total_annotated * 100) if total_annotated > 0 else 0
        animated_metric("Pass Rate", f"{pass_rate:.1f}%", "✅", delay=2)
    with col3:
        animated_metric("Failures", str(len(failures)), "❌", delay=3)
    with col4:
        animated_metric("Failure Modes", str(len(failure_modes)), "🏷️", delay=4)

    coverage = summarize_coverage(discovery_samples, annotations.keys())
    st.info(
        f"Discovery coverage: {coverage['reviewed_selected_count']}/{coverage['selected_count']} "
        f"selected traces reviewed across {coverage['clusters_reviewed']}/{coverage['clusters_selected']} "
        "selected clusters. " + coverage["prevalence_warning"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Failure mode distribution
    if failure_modes and failures:
        section_header("📈 Failure Mode Distribution", style="info")

        # Count failure modes across annotations
        mode_counts = Counter()
        for a in failures:
            for mode in a.get("failure_modes", []):
                mode_counts[mode] += 1

        if mode_counts:
            mode_df = pd.DataFrame([
                {"Failure Mode": mode, "Count": count, "Rate": f"{count / len(failures) * 100:.1f}%"}
                for mode, count in mode_counts.most_common()
            ])

            st.dataframe(mode_df, use_container_width=True, hide_index=True)

            # Bar chart
            import plotly.express as px
            fig = px.bar(
                mode_df,
                x="Count",
                y="Failure Mode",
                orientation="h",
                title="Failure Mode Frequency",
                color="Count",
                color_continuous_scale="Reds"
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color=theme.text_primary,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tag failures with failure modes in Open Coding to see distribution.")

    if selected_dimension_fields and annotations:
        section_header("🧩 Reviewed Outcomes by Product Dimension", style="info")
        dimension_rows = []
        for annotation_idx, annotation in annotations.items():
            dimension_rows.append(
                {
                    **{
                        field: records[annotation_idx].get(field, "(missing)")
                        for field in selected_dimension_fields
                    },
                    "Failed": 0 if annotation.get("pass_fail", True) else 1,
                }
            )
        dimension_df = pd.DataFrame(dimension_rows)
        grouped_dimensions = (
            dimension_df.groupby(selected_dimension_fields, dropna=False)["Failed"]
            .agg([("Reviewed", "size"), ("Failures", "sum"), ("Failure Rate", "mean")])
            .reset_index()
        )
        grouped_dimensions["Failure Rate"] = grouped_dimensions["Failure Rate"].map(
            lambda value: f"{value * 100:.1f}%"
        )
        st.dataframe(grouped_dimensions, use_container_width=True, hide_index=True)
        st.caption(
            "Descriptive results for reviewed records only. If these records came from diverse or targeted "
            "sampling, the rates are not production prevalence estimates."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Evaluator promotion
    section_header("🧪 Promote a Failure Mode", style="info")

    if failure_modes:
        st.markdown(
            "Turn one human-accepted failure mode into a focused binary evaluator. "
            "New evaluators remain drafts until they pass held-out label validation."
        )
        promote_col1, promote_col2 = st.columns(2)
        with promote_col1:
            promoted_mode = st.selectbox(
                "Failure mode",
                sorted(failure_modes),
                key="promote_failure_mode",
            )
            evaluator_kind = st.selectbox(
                "Evaluator type",
                ["llm_judge", "regex", "contains"],
                format_func=lambda value: {
                    "llm_judge": "LLM judge",
                    "regex": "Regex rule",
                    "contains": "Contains rule",
                }[value],
                key="promote_evaluator_kind",
            )
        with promote_col2:
            default_fields = [
                value
                for value in (mapping.input_col, mapping.output_col, mapping.expected_col)
                if value in df.columns
            ]
            promoted_fields = st.multiselect(
                "Evaluator input fields",
                list(df.columns),
                default=default_fields or list(df.columns[: min(3, len(df.columns))]),
                key="promote_input_fields",
            )
            pass_definition = st.text_area(
                "Pass definition (optional)",
                placeholder="Describe what acceptable behavior looks like for this failure mode.",
                height=80,
                key="promote_pass_definition",
            )

        rule_pattern = None
        if evaluator_kind in {"regex", "contains"}:
            rule_pattern = st.text_input(
                "Failure pattern",
                help="A match means Fail. Regex syntax is used only for regex evaluators.",
                key="promote_rule_pattern",
            )

        errors_reviewed = st.checkbox(
            "Obvious product errors were fixed or reviewed, and a specialized evaluator is still useful",
            help="Pervasive errors should be fixed and re-reviewed before investing in judge automation.",
        )

        if st.button("Create draft evaluator", type="primary", disabled=not errors_reviewed):
            try:
                failure_examples = []
                pass_examples = []
                for annotation_idx, annotation in annotations.items():
                    critique = str(annotation.get("critique") or annotation.get("open_code") or "").strip()
                    if not critique:
                        continue
                    expert_example = {
                        "input": {
                            field: records[annotation_idx].get(field)
                            for field in promoted_fields
                            if field in records[annotation_idx]
                        },
                        "critique": critique,
                        "result": (
                            "Fail"
                            if promoted_mode in annotation.get("failure_modes", [])
                            else "Pass"
                        ),
                        "annotation_id": str(annotation_idx),
                    }
                    if expert_example["result"] == "Fail":
                        failure_examples.append(expert_example)
                    elif annotation.get("pass_fail", True):
                        pass_examples.append(expert_example)
                expert_examples = [*failure_examples[:4], *pass_examples[:4]]
                source_annotation_ids = [
                    example["annotation_id"] for example in expert_examples
                ]
                evaluator = build_evaluator_definition(
                    promoted_mode,
                    failure_modes[promoted_mode],
                    kind=evaluator_kind,
                    input_fields=promoted_fields,
                    pass_definition=pass_definition or None,
                    rule_pattern=rule_pattern,
                    source_annotation_ids=source_annotation_ids,
                    expert_examples=expert_examples,
                    errors_reviewed=errors_reviewed,
                )
                st.session_state.promoted_evaluators[evaluator["evaluator_id"]] = evaluator
                st.success("Draft evaluator created. Validate it before using it for release decisions.")
            except ValueError as error:
                st.error(str(error))

        promoted_for_mode = [
            evaluator
            for evaluator in st.session_state.promoted_evaluators.values()
            if evaluator.get("name") == promoted_mode
        ]
        if promoted_for_mode:
            evaluator = promoted_for_mode[-1]
            st.warning(
                "Status: draft_unvalidated. Export human labels, create train/dev/test splits, "
                "and validate true-pass and true-fail rates before gating CI."
            )
            if evaluator.get("prompt"):
                with st.expander("Inspect judge prompt"):
                    st.code(evaluator["prompt"], language=None)
            st.download_button(
                "Download evaluator definition",
                json.dumps(evaluator, indent=2, default=str),
                f"{evaluator['evaluator_id']}.json",
            )
    else:
        st.info("Create and accept at least one failure mode before promoting an evaluator.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Export section
    section_header("💾 Export Analysis", style="success")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        export_annotations = [
            {"trace_index": idx, **data}
            for idx, data in sorted(annotations.items())
        ]
        st.download_button(
            "📄 Download Annotations (JSON)",
            json.dumps(export_annotations, indent=2, default=str),
            "trace_annotations.json",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "📂 Download Taxonomy (JSON)",
            json.dumps(failure_modes, indent=2, default=str),
            "failure_taxonomy.json",
            use_container_width=True,
        )

    with col3:
        # Create summary report
        report = {
            "summary": {
                "total_traces": total,
                "annotated": total_annotated,
                "pass_count": passes,
                "fail_count": len(failures),
                "pass_rate": f"{pass_rate:.1f}%",
            },
            "failure_modes": failure_modes,
            "mode_distribution": dict(mode_counts) if failure_modes and failures else {},
            "generated_at": datetime.now().isoformat(),
        }
        st.download_button(
            "📊 Download Report (JSON)",
            json.dumps(report, indent=2, default=str),
            "error_analysis_report.json",
            use_container_width=True,
        )

    with col4:
        review_bundle = build_review_bundle(
            records,
            discovery_samples,
            annotations=annotations,
            taxonomy=failure_modes,
            suggestions={
                "pending_taxonomy": st.session_state.taxonomy_suggestions,
                "dismissed_taxonomy": st.session_state.dismissed_taxonomy_suggestions,
                "related_traces": st.session_state.related_trace_suggestions,
            },
            evaluators=st.session_state.promoted_evaluators,
            source=st.session_state.get("dataset_filename"),
            reviewer={
                "name": st.session_state.principal_reviewer_name,
                "role": st.session_state.principal_reviewer_role,
            },
            dimension_fields=selected_dimension_fields,
        )
        st.download_button(
            "📦 Download Review Bundle",
            json.dumps(review_bundle, indent=2, default=str),
            "vectory_review_bundle.json",
            use_container_width=True,
            help="Portable dataset, sample rationale, annotations, taxonomy, suggestions, and draft evaluators.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Failure examples table
    if failures:
        section_header("📋 Failure Examples", style="warning")

        failure_data = []
        for idx, a in annotations.items():
            if not a.get("pass_fail", True):
                row = df.iloc[idx]
                failure_data.append({
                    "Trace #": idx + 1,
                    "Open Code": a.get("open_code", "")[:100] + "...",
                    "Failure Modes": ", ".join(a.get("failure_modes", [])) or "Untagged",
                    "Output Preview": str(row[mapping.output_col])[:100] + "..." if mapping.output_col else "N/A",
                })

        failure_df = pd.DataFrame(failure_data)
        st.dataframe(failure_df, use_container_width=True, hide_index=True)


# Sidebar - Import/Reset
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<h4 style="color: {theme.text_primary};">🔧 Tools</h4>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Import Taxonomy", type=["json"])
if uploaded:
    imported = json.load(uploaded)
    st.session_state.failure_modes = imported
    st.sidebar.success("✅ Imported!")
    st.rerun()

if st.sidebar.button("🗑️ Reset All Analysis"):
    st.session_state.failure_modes = {}
    st.session_state.trace_annotations = {}
    st.session_state.open_codes = []
    st.session_state.related_trace_suggestions = {}
    st.session_state.promoted_evaluators = {}
    st.session_state.discovery_samples = select_diverse_samples(
        records,
        sample_size=min(30, total),
        seed=42,
        dimension_fields=default_dimension_fields,
    )
    st.session_state.discovery_sample_position = 0
    st.rerun()
