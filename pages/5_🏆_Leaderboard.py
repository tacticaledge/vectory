"""MTEB Leaderboard Page."""

import streamlit as st
import streamlit.components.v1 as st_components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state
from components.mteb_leaderboard import (
    fetch_mteb_leaderboard,
    get_leaderboard_info,
    refresh_leaderboard_cache,
    get_available_models,
    AVAILABLE_MODELS_INFO,
)

from components.ui import (
    inject_custom_css,
    animated_metric,
    rank_badge,
    score_bar,
    display_lottie,
    section_header,
    get_current_theme,
    sidebar_logo,
)

st.set_page_config(page_title="MTEB Leaderboard | Vectory", page_icon="🏆", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)
inject_custom_css()

# Animated header with developer colors
theme = get_current_theme()
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    🏆 MTEB Embedding Leaderboard
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {theme.text_muted}; font-size: 1rem; margin-top: -10px;">
    <span style="color: {theme.terminal};">$</span> real-time benchmarks from the Massive Text Embedding Benchmark
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

AUTO_REFRESH_SECONDS = 1800
st_components.html(
    f"""
    <script>
      window.setTimeout(function() {{
        window.parent.location.reload();
      }}, {AUTO_REFRESH_SECONDS * 1000});
    </script>
    """,
    height=0,
)

refresh_col, status_col = st.columns([0.22, 0.78])
with refresh_col:
    force_refresh = st.button("Refresh now", use_container_width=True)
with status_col:
    st.caption("Leaderboard data refreshes in the background every 30 minutes. This page also reloads every 30 minutes while open.")

if force_refresh:
    with st.spinner("Refreshing leaderboard cache..."):
        refreshed = refresh_leaderboard_cache(top_n=100)
    if refreshed:
        st.success("Leaderboard cache refreshed.")
    else:
        st.warning("Live refresh did not return new rows. Showing the best available cached data.")

# Load leaderboard data (fetches from HuggingFace or uses cache)
with st.spinner("Loading leaderboard data..."):
    df = fetch_mteb_leaderboard(top_n=20, force_refresh=False)
    leaderboard_info = get_leaderboard_info()

# Show data source info
if str(leaderboard_info.get("source", "")).startswith("live"):
    age_minutes = int(leaderboard_info.get("age_seconds", 0) // 60)
    st.caption(
        f"Live leaderboard · updated {age_minutes} min ago · "
        f"auto-refresh every {int(leaderboard_info.get('refresh_interval_seconds', AUTO_REFRESH_SECONDS) // 60)} min"
    )
elif leaderboard_info.get("source") == "disk":
    age_minutes = int(leaderboard_info.get("age_seconds", 0) // 60)
    st.caption(f"Cached leaderboard · updated {age_minutes} min ago · auto-refresh active")
else:
    st.caption("Bundled fallback leaderboard · auto-refresh will replace it when live data is available")

# Sidebar filters
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">🎯</span>
        <h3 style="margin: 8px 0; color: {theme.text_primary};">Filters</h3>
    </div>
    """, unsafe_allow_html=True)

    providers = ["All"] + sorted(df["provider"].unique().tolist())
    selected_provider = st.selectbox("Provider", providers, label_visibility="collapsed")

    if selected_provider != "All":
        df = df[df["provider"] == selected_provider].reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

    st.markdown("---")
    st.markdown(f"**{len(df)}** models displayed")

# Task columns present in data
task_cols = [c for c in ["retrieval", "sts", "classification"] if c in df.columns]

# Tabs with custom styling
tab1, tab2, tab3 = st.tabs(["📊 Leaderboard", "📈 Visualizations", "🔧 Available Models"])

# ==================== TAB 1: LEADERBOARD ====================
with tab1:
    # Top 3 podium
    if len(df) >= 3:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: {theme.text_primary};">🥇 Top Performers</h3>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns([1, 1.2, 1])
        podium_order = [1, 0, 2]  # Silver, Gold, Bronze positions

        for idx, col_idx in enumerate(podium_order):
            if col_idx < len(df):
                row = df.iloc[col_idx]
                with cols[idx]:
                    medal = ["🥇", "🥈", "🥉"][col_idx]
                    bg_colors = [
                        "linear-gradient(135deg, #ffd700, #ffed4a)",
                        "linear-gradient(135deg, #c0c0c0, #e5e7eb)",
                        "linear-gradient(135deg, #cd7f32, #d97706)"
                    ][col_idx]

                    st.markdown(f"""
                    <div class="feature-card delay-{idx+1}" style="text-align: center; padding: 20px;">
                        <div style="font-size: 2.5rem; margin-bottom: 10px;">{medal}</div>
                        <div style="font-weight: 700; font-size: 1.1rem; color: {theme.text_primary}; margin-bottom: 4px;">
                            {row['model'][:20]}
                        </div>
                        <div style="font-size: 0.85rem; color: {theme.text_muted}; margin-bottom: 8px;">
                            {row['provider']}
                        </div>
                        <div class="metric-value" style="font-size: 1.8rem;">
                            {row['mean_score']:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full rankings table
    st.markdown(f"""
    <div style="background: {theme.bg_card}; padding: 16px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid {theme.primary};">
        <h4 style="margin: 0; color: {theme.primary};">📋 Full Rankings</h4>
    </div>
    """, unsafe_allow_html=True)

    display_df = df.copy()
    st.dataframe(
        display_df.style.format({
            "mean_score": "{:.3f}",
            **{c: "{:.3f}" for c in task_cols}
        }).background_gradient(subset=["mean_score"], cmap="RdYlGn"),
        use_container_width=True,
        height=400,
    )

    # Export section
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("💾 Export Results", style="success")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📄 Download CSV",
            df.to_csv(index=False),
            "mteb_leaderboard.csv",
            "text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📋 Download JSON",
            df.to_json(orient="records", indent=2),
            "mteb_leaderboard.json",
            "application/json",
            use_container_width=True,
        )

# ==================== TAB 2: VISUALIZATIONS ====================
with tab2:
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3 style="color: {theme.text_primary};">📈 Performance Analytics</h3>
    </div>
    """, unsafe_allow_html=True)

    # Bar chart with custom styling
    fig = px.bar(
        df.sort_values("mean_score"),
        x="mean_score",
        y="model",
        orientation="h",
        color="provider",
        title="Mean Score by Model",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Mean Score",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    if task_cols and len(df) >= 3:
        section_header("🎯 Task Performance Comparison", style="info", subtitle="Top 5 models across different task types")

        fig = go.Figure()
        colors = ["#6366f1", "#a855f7", "#22d3ee", "#4ade80", "#fbbf24"]

        for i, (_, row) in enumerate(df.head(5).iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[row.get(c, 0) for c in task_cols],
                theta=[c.title() for c in task_cols],
                fill="toself",
                name=row["model"][:20],
                line=dict(color=colors[i % len(colors)]),
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Provider comparison
    if len(df["provider"].unique()) > 1:
        section_header("🏢 Provider Performance", style="warning")

        provider_stats = df.groupby("provider")["mean_score"].agg(["mean", "count"]).round(3)
        provider_stats.columns = ["Mean Score", "Model Count"]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(provider_stats, use_container_width=True)
        with col2:
            for provider, row in provider_stats.iterrows():
                score_bar(row["Mean Score"], label=provider[:15])

# ==================== TAB 3: AVAILABLE MODELS ====================
with tab3:
    col1, col2 = st.columns([2, 1])

    with col1:
        section_header("🔧 Available Models", style="primary", subtitle="Open-source models ready for custom evaluation")

    with col2:
        if not display_lottie("robot", height=100):
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <span style="font-size: 3rem;">🤖</span>
            </div>
            """, unsafe_allow_html=True)

    models_list = get_available_models()
    models_df = pd.DataFrame([
        {
            "Model": m["name"],
            "Provider": m["provider"],
            "Dimensions": m["dimensions"],
            "Max Tokens": m["max_tokens"],
        }
        for m in models_list
    ])

    # Display as animated list
    for i, (_, row) in enumerate(models_df.iterrows()):
        st.markdown(f"""
        <div class="animated-list-item delay-{i % 5}" style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-weight: 600; color: {theme.text_primary};">{row['Model']}</span>
                <span style="color: {theme.text_muted}; margin-left: 8px; font-size: 0.85rem;">{row['Provider']}</span>
            </div>
            <div style="text-align: right; font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: {theme.accent};">
                {row['Dimensions']}d · {row['Max Tokens']} tokens
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
        <h4 style="margin: 0; color: white;">💡 Pro Tip</h4>
        <p style="margin: 10px 0 0 0; opacity: 0.9; color: white;">
            Use these models in the <strong>Custom Benchmark</strong> page for your own evaluations.
            API-based models (OpenAI, etc.) require an API key.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer with animation
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; padding: 20px; background: {theme.bg_card}; border-radius: 16px; border: 1px solid {theme.border};">
    <p style="margin: 0; color: {theme.text_muted};">
        📊 <a href="https://huggingface.co/spaces/mteb/leaderboard" target="_blank" style="color: {theme.primary};">Full MTEB Leaderboard</a> ·
        📄 <a href="https://arxiv.org/abs/2210.07316" target="_blank" style="color: {theme.primary};">MTEB Paper</a> ·
        💻 <a href="https://github.com/embeddings-benchmark/mteb" target="_blank" style="color: {theme.primary};">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
