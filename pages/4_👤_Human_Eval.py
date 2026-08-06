"""Human Evaluation Page - With Animations"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state, RatingType, HumanAnnotation, ColumnMapping, DataSourceType
from components.ui import (
    inject_custom_css,
    animated_metric,
    feature_card,
    display_lottie,
    score_bar,
    section_header,
    get_current_theme,
    hex_to_rgb,
    sidebar_logo,
)

st.set_page_config(page_title="Human Evaluation | Vectory", page_icon="👤", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)
inject_custom_css()

# Animated header with developer colors
theme = get_current_theme()
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    👤 Human Evaluation
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {theme.text_muted}; font-size: 1rem; margin-top: -10px;">
    <span style="color: {theme.terminal};">$</span> rate LLM outputs with customizable criteria and feedback
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
                <span style="font-size: 5rem; opacity: 0.5;">👤</span>
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

# Get data source type and set up column mapping accordingly
data_source_type = st.session_state.get("data_source_type", DataSourceType.TABULAR)
df = st.session_state.dataset

if data_source_type == DataSourceType.TABULAR:
    mapping = st.session_state.column_mapping
    if not isinstance(mapping, ColumnMapping):
        mapping = ColumnMapping(**mapping) if isinstance(mapping, dict) else ColumnMapping()
    if not mapping.output_col:
        st.warning("⚠️ Please configure column mapping on the Dataset page.")
        st.stop()
else:
    # For document/image, use the text column
    text_col = "text" if "text" in df.columns else df.columns[0]
    mapping = ColumnMapping(output_col=text_col)
annotations = st.session_state.human_annotations
total = len(df)

# Sidebar - Progress with animation
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">📊</span>
        <h3 style="margin: 8px 0; color: {theme.text_primary};">Progress</h3>
    </div>
    """, unsafe_allow_html=True)

    annotated = len(annotations)
    progress_pct = annotated / total if total > 0 else 0
    st.progress(progress_pct)

    st.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <span class="metric-value" style="font-size: 1.5rem;">{annotated}</span>
        <span style="color: {theme.text_muted};"> / {total}</span>
        <p style="color: {theme.text_muted}; margin: 4px 0 0 0; font-size: 0.85rem;">samples annotated</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"""
    <div style="text-align: center; padding: 5px;">
        <span style="font-size: 1.5rem;">🧭</span>
        <h4 style="margin: 8px 0; color: {theme.text_primary};">Navigation</h4>
    </div>
    """, unsafe_allow_html=True)

    nav_mode = st.radio("Mode", ["Sequential", "Jump", "Unannotated"], label_visibility="collapsed")

# Get current index
if nav_mode == "Sequential":
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("← Prev"):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
    with c2:
        st.markdown(f"""
        <div style="text-align: center; padding: 5px;">
            <strong>{st.session_state.current_idx + 1}</strong>/{total}
        </div>
        """, unsafe_allow_html=True)
    with c3:
        if st.button("Next →"):
            st.session_state.current_idx = min(total - 1, st.session_state.current_idx + 1)
    idx = st.session_state.current_idx

elif nav_mode == "Jump":
    idx = st.sidebar.number_input("Sample #", 1, total, 1) - 1

else:  # Unannotated
    unannotated = [i for i in range(total) if i not in annotations]
    if unannotated:
        if "unannotated_pos" not in st.session_state:
            st.session_state.unannotated_pos = 0
        pos = min(st.session_state.unannotated_pos, len(unannotated) - 1)
        idx = unannotated[pos]
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #fef3c7, #fde68a); border-radius: 8px;">
            <strong>{pos + 1}</strong> / {len(unannotated)} remaining
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #d1fae5, #a7f3d0); border-radius: 8px;">
            ✅ All done!
        </div>
        """, unsafe_allow_html=True)
        idx = 0

# Sample header with status
annotated_badge = f"<span style='background: rgba({hex_to_rgb(theme.success)}, 0.2); padding: 8px 16px; border-radius: 20px; color: {theme.success}; font-weight: 500; border: 1px solid {theme.success};'>✓ Annotated</span>" if idx in annotations else f"<span style='background: rgba({hex_to_rgb(theme.warning)}, 0.2); padding: 8px 16px; border-radius: 20px; color: {theme.warning}; font-weight: 500; border: 1px solid {theme.warning};'>○ Needs annotation</span>"
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div>
        <h2 style="margin: 0; color: {theme.text_primary};">Sample {idx + 1} / {total}</h2>
    </div>
    <div>
        {annotated_badge}
    </div>
</div>
""", unsafe_allow_html=True)

row = df.iloc[idx]

# Display sample in animated cards
c1, c2 = st.columns(2)

with c1:
    if mapping.input_col:
        st.markdown(f"""
        <div class="feature-card delay-1" style="margin-bottom: 16px;">
            <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">📝 Input</h4>
        </div>
        """, unsafe_allow_html=True)
        st.text_area("Input text", str(row[mapping.input_col]), height=150, disabled=True, label_visibility="collapsed")

    if mapping.expected_col:
        st.markdown(f"""
        <div class="feature-card delay-2" style="margin-bottom: 16px;">
            <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">✅ Expected</h4>
        </div>
        """, unsafe_allow_html=True)
        st.text_area("Expected text", str(row[mapping.expected_col]), height=150, disabled=True, label_visibility="collapsed")

with c2:
    st.markdown(f"""
    <div class="feature-card delay-3" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">🤖 LLM Output</h4>
    </div>
    """, unsafe_allow_html=True)
    st.text_area("Output text", str(row[mapping.output_col]), height=320, disabled=True, label_visibility="collapsed")

# Annotation form
st.markdown("<br>", unsafe_allow_html=True)
section_header("✏️ Your Evaluation", style="primary")

existing = annotations.get(idx, {})

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"""
    <div class="feature-card" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">⭐ Rating</h4>
    </div>
    """, unsafe_allow_html=True)

    rating_type = st.radio(
        "Rating Type",
        [rt.value for rt in RatingType],
        horizontal=True,
        key=f"rt_{idx}",
        label_visibility="collapsed",
    )

    if rating_type == RatingType.SCALE_1_5.value:
        rating = st.slider("Score", 1, 5, existing.get("rating", 3))
        labels = {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}
        colors = {1: "#ef4444", 2: "#f97316", 3: "#eab308", 4: "#84cc16", 5: "#22c55e"}
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: {colors[rating]}22; border-radius: 8px; margin-top: 8px;">
            <span style="color: {colors[rating]}; font-weight: 600;">{labels[rating]}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        choice = st.radio(
            "Verdict",
            ["Good", "Bad"] if "Thumbs" in rating_type else ["Pass", "Fail"],
            horizontal=True,
        )
        rating = 1 if choice in ["Good", "Pass"] else 0

with c2:
    st.markdown(f"""
    <div class="feature-card" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">🏷️ Criteria</h4>
    </div>
    """, unsafe_allow_html=True)

    criteria = st.multiselect(
        "This response is:",
        ["Accurate", "Relevant", "Coherent", "Complete", "Helpful"],
        default=existing.get("criteria", []),
        label_visibility="collapsed",
    )

st.markdown(f"""
<div class="feature-card" style="margin: 16px 0;">
    <h4 style="margin: 0 0 12px 0; color: {theme.text_primary};">📝 Notes</h4>
</div>
""", unsafe_allow_html=True)

feedback = st.text_area(
    "Notes (optional)",
    existing.get("feedback", ""),
    placeholder="Any additional feedback...",
    label_visibility="collapsed",
)

# Navigation and Save buttons
st.markdown("<br>", unsafe_allow_html=True)

# Navigation row
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])

with nav_col1:
    if st.button("← Previous", use_container_width=True, disabled=(idx == 0)):
        st.session_state.current_idx = max(0, idx - 1)
        st.rerun()

with nav_col2:
    if st.button("Skip / Next →", use_container_width=True, disabled=(idx >= total - 1)):
        st.session_state.current_idx = min(total - 1, idx + 1)
        st.rerun()

with nav_col3:
    jump_to = st.number_input("Go to #", min_value=1, max_value=total, value=idx + 1, label_visibility="collapsed")

with nav_col4:
    if st.button("Go", use_container_width=True):
        st.session_state.current_idx = jump_to - 1
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Action buttons
c1, c2, _ = st.columns([1, 1, 2])

with c1:
    if st.button("💾 Save Annotation", type="primary", use_container_width=True):
        annotations[idx] = {
            "rating": rating,
            "rating_type": rating_type,
            "criteria": criteria,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        }
        st.session_state.human_annotations = annotations

        st.markdown("""
        <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 20px; color: white; text-align: center;">
            <span style="font-size: 1.5rem;">✅</span>
            <p style="margin: 8px 0 0 0; color: white;">Annotation saved!</p>
        </div>
        """, unsafe_allow_html=True)

        # Auto-advance
        if nav_mode == "Sequential" and idx < total - 1:
            st.session_state.current_idx = idx + 1
            st.rerun()

with c2:
    if idx in annotations and st.button("🗑️ Clear", use_container_width=True):
        del annotations[idx]
        st.session_state.human_annotations = annotations
        st.rerun()

# Summary section
if annotations:
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("📊 Summary", style="success")

    c1, c2, c3 = st.columns(3)

    with c1:
        scale_ratings = [a["rating"] for a in annotations.values()
                        if a.get("rating_type") == RatingType.SCALE_1_5.value]
        if scale_ratings:
            avg = sum(scale_ratings)/len(scale_ratings)
            animated_metric("Avg Rating", f"{avg:.2f}", "⭐", delay=1)
            score_bar(avg / 5, label="Rating")

    with c2:
        binary = [a["rating"] for a in annotations.values()
                 if a.get("rating_type") != RatingType.SCALE_1_5.value]
        if binary:
            approval = 100*sum(binary)/len(binary)
            animated_metric("Approval", f"{approval:.0f}%", "👍", delay=2)

    with c3:
        animated_metric("Annotated", f"{len(annotations)}/{total}", "📝", delay=3)

    # Export section
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("💾 Export Annotations", style="info")

    export_data = [
        {"index": i, **annotations[i]} for i in sorted(annotations.keys())
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📄 Download CSV",
            pd.DataFrame(export_data).to_csv(index=False),
            "annotations.csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📋 Download JSON",
            json.dumps(export_data, indent=2, default=str),
            "annotations.json",
            use_container_width=True,
        )

# Clear all in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Annotations"):
    st.session_state.human_annotations = {}
    st.rerun()
