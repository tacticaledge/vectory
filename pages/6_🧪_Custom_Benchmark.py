"""Custom Benchmark Page."""

import html
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state, EvalTaskType
from components.document_loader import load_file, load_files, infer_columns
from components.ui import (
    inject_custom_css,
    score_bar,
    get_current_theme,
)
from components.mteb_leaderboard import get_available_models

# Check if MTEB benchmarking is available (requires PyTorch >= 2.1)
MTEB_BENCHMARK_AVAILABLE = False
MTEB_ERROR = None
run_benchmark = None
get_mteb_tasks = None

def _check_torch_version():
    """Check if torch version is compatible (>= 2.1.0)."""
    try:
        import torch
        version = torch.__version__.split('+')[0]
        parts = version.split('.')
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor) >= (2, 1)
    except ImportError:
        return False
    except Exception:
        return False

# Try to import MTEB benchmark functions if PyTorch is available
if _check_torch_version():
    try:
        from components.mteb_evaluator import (
            run_benchmark,
            get_mteb_tasks,
        )
        MTEB_BENCHMARK_AVAILABLE = True
    except Exception as e:
        MTEB_ERROR = str(e)
else:
    MTEB_ERROR = "PyTorch >= 2.1.0 is required for running benchmarks"

# Get available models (doesn't require PyTorch)
AVAILABLE_MODELS = get_available_models()

st.set_page_config(page_title="Custom Benchmark | Vectory", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)
inject_custom_css()

theme = get_current_theme()
st.markdown(f"""
<style>
.cb-shell {{
    display: grid;
    gap: 18px;
}}
.cb-hero {{
    border: 1px solid {theme.border};
    border-radius: 8px;
    padding: 24px 26px;
    background: {theme.bg_card};
}}
.cb-kicker {{
    color: {theme.primary};
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.cb-title-row {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 18px;
    flex-wrap: wrap;
}}
.cb-title {{
    color: {theme.text_primary};
    font-family: var(--theme-font-sans);
    font-size: clamp(2rem, 4vw, 3.1rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: 0;
    margin: 0;
}}
.cb-subtitle {{
    color: {theme.text_secondary};
    font-size: 1rem;
    line-height: 1.55;
    max-width: 920px;
    margin: 14px 0 0 0;
}}
.cb-status {{
    border: 1px solid {theme.border};
    border-radius: 999px;
    color: {theme.text_secondary};
    background: {theme.bg_secondary};
    padding: 6px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
}}
.cb-flow {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}}
.cb-step {{
    min-height: 148px;
    height: 100%;
    border: 1px solid {theme.border};
    border-radius: 8px;
    background: {theme.bg_card};
    padding: 16px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}}
.cb-step-number {{
    color: {theme.primary};
    font-weight: 800;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.cb-step-title {{
    color: {theme.text_primary};
    font-weight: 800;
    font-size: 1rem;
    line-height: 1.25;
    margin-top: 12px;
}}
.cb-step-text {{
    color: {theme.text_muted};
    font-size: 0.88rem;
    line-height: 1.45;
    margin-top: 8px;
}}
.cb-panel {{
    border: 1px solid {theme.border};
    border-radius: 8px;
    background: {theme.bg_card};
    padding: 18px;
    margin-bottom: 16px;
}}
.cb-section {{
    margin: 18px 0 14px;
}}
.cb-section-title {{
    color: {theme.text_primary};
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0;
}}
.cb-section-subtitle {{
    color: {theme.text_muted};
    font-size: 0.92rem;
    line-height: 1.5;
    margin: 6px 0 0;
}}
.cb-panel-title {{
    color: {theme.text_primary};
    font-weight: 800;
    font-size: 1rem;
    margin-bottom: 6px;
}}
.cb-muted {{
    color: {theme.text_muted};
    font-size: 0.9rem;
    line-height: 1.5;
}}
.cb-control-panel {{
    border: 1px solid {theme.border};
    border-radius: 8px;
    background: {theme.bg_card};
    padding: 14px;
    margin-bottom: 14px;
}}
.cb-upload-zone {{
    border: 1px dashed {theme.border_hover};
    border-radius: 8px;
    background: {theme.bg_card};
    padding: 16px;
    margin-bottom: 14px;
}}
.cb-empty {{
    border: 1px dashed {theme.border_hover};
    border-radius: 8px;
    padding: 34px 24px;
    text-align: center;
    background: {theme.bg_card};
    color: {theme.text_muted};
}}
.cb-empty-title {{
    color: {theme.text_primary};
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 6px;
}}
.cb-callout {{
    border: 1px solid {theme.border};
    border-left: 3px solid {theme.primary};
    border-radius: 8px;
    padding: 13px 15px;
    background: {theme.bg_card};
    margin: 14px 0;
}}
.cb-callout-warning {{
    border-left-color: {theme.warning};
    background: rgba(251, 191, 36, 0.08);
}}
.cb-callout-error {{
    border-left-color: {theme.error};
    background: rgba(248, 113, 113, 0.08);
}}
.cb-pill {{
    display: inline-flex;
    align-items: center;
    border: 1px solid {theme.border};
    border-radius: 999px;
    padding: 5px 10px;
    margin: 4px 4px 4px 0;
    color: {theme.text_secondary};
    background: {theme.bg_secondary};
    font-size: 0.78rem;
    font-weight: 700;
}}
.cb-label {{
    color: {theme.text_muted};
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.cb-stat-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 14px 0;
}}
.cb-stat {{
    border: 1px solid {theme.border};
    border-radius: 8px;
    background: {theme.bg_card};
    padding: 14px;
}}
.cb-stat-value {{
    color: {theme.text_primary};
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1;
}}
.cb-stat-label {{
    color: {theme.text_muted};
    font-size: 0.78rem;
    margin-top: 8px;
    font-weight: 700;
}}
@media (max-width: 1100px) {{
    .cb-flow, .cb-stat-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}
@media (max-width: 720px) {{
    .cb-flow, .cb-stat-grid {{
        grid-template-columns: 1fr;
    }}
    .cb-hero {{
        padding: 20px;
    }}
}}
</style>
""", unsafe_allow_html=True)


def _step_card_html(number: str, title: str, text: str) -> str:
    return f"""
<div class="cb-step">
  <div class="cb-step-number">{html.escape(number)}</div>
  <div class="cb-step-title">{html.escape(title)}</div>
  <div class="cb-step-text">{html.escape(text)}</div>
</div>
    """


def _section(title: str, subtitle: str = ""):
    subtitle_html = f'<p class="cb-section-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f"""
<div class="cb-section">
  <h2 class="cb-section-title">{html.escape(title)}</h2>
  {subtitle_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def _stat_grid(items):
    cards = "\n".join(
        f"""
<div class="cb-stat">
  <div class="cb-stat-value">{html.escape(str(value))}</div>
  <div class="cb-stat-label">{html.escape(label)}</div>
</div>
        """
        for label, value in items
    )
    st.markdown(f'<div class="cb-stat-grid">{cards}</div>', unsafe_allow_html=True)


runner_status = "Local runner ready" if MTEB_BENCHMARK_AVAILABLE else "Local runner optional"

steps_html = "\n".join(
    [
        _step_card_html("Step 1", "Upload data", "Bring in CSV, JSON, JSONL, or PDF files from support, sales, policy, product, or research workflows."),
        _step_card_html("Step 2", "Map columns", "Identify the text, label, reference, or expected-answer fields Vectory should treat as signal."),
        _step_card_html("Step 3", "Choose scope", "Select task families and candidate embedding models when local model execution is enabled."),
        _step_card_html("Step 4", "Export report", "Download CSV or JSON artifacts for internal review, baselines, and follow-up analysis."),
    ]
)

st.markdown(f"""
<div class="cb-shell">
  <div class="cb-hero">
    <div class="cb-title-row">
      <div>
        <div class="cb-kicker">Organization evaluation workspace</div>
        <h1 class="cb-title">Custom Benchmark</h1>
      </div>
      <div class="cb-status">{runner_status}</div>
    </div>
    <p class="cb-subtitle">
      Build a repeatable evaluation packet from your own business data. Upload files, normalize columns,
      choose the task surface, and export artifacts your team can review.
    </p>
  </div>
  <div class="cb-flow">
    {steps_html}
  </div>
</div>
""", unsafe_allow_html=True)

if not MTEB_BENCHMARK_AVAILABLE:
    st.markdown(
        f"""
<div class="cb-callout cb-callout-warning" style="margin-top: 16px;">
  <div class="cb-panel-title">Local model runner unavailable</div>
  <div class="cb-muted">Data upload, preview, and column mapping still work. Install the optional PyTorch runtime only when this page needs to download and run local embedding models.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar configuration
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 8px 0 14px 0;">
        <div class="cb-kicker">Run setup</div>
        <h3 style="margin: 0; color: {theme.text_primary};">Benchmark Scope</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cb-control-panel">
        <div class="cb-label">Evaluation shape</div>
        <div class="cb-muted">Choose the task families this dataset should exercise.</div>
    </div>
    """, unsafe_allow_html=True)

    task_types = st.multiselect(
        "Select tasks",
        [t.value for t in EvalTaskType],
        default=["Classification", "STS"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div class="cb-control-panel">
        <div class="cb-label">Candidate models</div>
        <div class="cb-muted">Pick the models to compare when the local runner is enabled.</div>
    </div>
    """, unsafe_allow_html=True)

    model_names = [m["name"] for m in AVAILABLE_MODELS]
    selected_models = st.multiselect(
        "Select models",
        model_names,
        default=model_names[:2] if model_names else [],
        label_visibility="collapsed",
    )

    # Show model details
    if selected_models:
        for model_name in selected_models:
            model_info = next((m for m in AVAILABLE_MODELS if m["name"] == model_name), None)
            if model_info:
                st.caption(f"  {model_info['provider']} · {model_info['dimensions']}d")

# Tabs with custom styling
tab1, tab2, tab3 = st.tabs(["Prepare data", "Configure & run", "Results"])

# ==================== TAB 1: DATASET ====================
with tab1:
    _section("Upload organization data", "Bring business data in JSON, JSONL, CSV, or PDF format.")

    st.markdown(
        f"""
<div class="cb-upload-zone">
  <div class="cb-panel-title">Source files</div>
  <div class="cb-muted">Use representative rows, preserve the fields reviewers care about, and avoid uploading secrets unless this app is running in your approved private environment.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload evaluation file",
        type=["json", "jsonl", "csv", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        try:
            st.markdown('<div class="animated-progress" style="margin: 20px 0;"></div>', unsafe_allow_html=True)

            if len(uploaded) == 1:
                df = load_file(uploaded[0])
            else:
                df = load_files(uploaded)

            st.session_state.benchmark_dataset = df

            st.markdown(f"""
            <div class="cb-callout">
                <div class="cb-panel-title">Dataset loaded</div>
                <div class="cb-muted">{len(df)} rows are ready for preview and mapping.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Preview
            st.markdown(f"""
            <div class="cb-panel">
                <div class="cb-panel-title">Preview</div>
                <div class="cb-muted">First rows from the uploaded data. Confirm the shape before mapping columns.</div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

            _section("Column mapping", "Confirm the fields Vectory should use for inputs, labels, and expected answers.")

            inferred = infer_columns(df)
            cols = [""] + list(df.columns)

            c1, c2, c3 = st.columns(3)
            with c1:
                text_col = st.selectbox(
                    "Text Column",
                    cols,
                    index=cols.index(inferred.input_col) if inferred.input_col in cols else 0,
                )
            with c2:
                label_col = st.selectbox(
                    "Label Column (for classification)",
                    cols,
                    index=cols.index(inferred.expected_col) if inferred.expected_col in cols else 0,
                )
            with c3:
                reference_col = st.selectbox(
                    "Reference / Expected Column",
                    cols,
                    index=cols.index(inferred.expected_col) if inferred.expected_col in cols else 0,
                )

            st.session_state.benchmark_column_mapping = {
                "text": text_col or None,
                "label": label_col or None,
                "reference": reference_col or None,
            }
            st.markdown(
                f"""
<div class="cb-panel">
  <div class="cb-panel-title">Dataset Profile</div>
  <span class="cb-pill">{len(df)} rows</span>
  <span class="cb-pill">{len(df.columns)} columns</span>
  <span class="cb-pill">{df.isna().sum().sum()} missing cells</span>
  <span class="cb-pill">{df.duplicated().sum()} duplicate rows</span>
</div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            error_str = str(e)
            # Check if this is a torch-related error
            if "torch" in error_str.lower() or "pytorch" in error_str.lower():
                st.error(f"""
                ❌ **PyTorch Compatibility Error**

                Your local embedding runtime is incompatible. Add the optional embedding runtime to the environment that runs Vectory:
                ```
                # pipx install
                pipx inject vectoryai "torch>=2.1.0" mteb sentence-transformers

                # source checkout or activated virtualenv
                pip install -e ".[embedding]"
                ```

                This error occurred during file loading due to a transitive dependency.
                The file upload itself does not require PyTorch. Restart the app after installing the optional runtime.

                Technical details: `{error_str}`
                """)
            else:
                st.error(f"❌ Error loading file: {e}")

    elif st.session_state.get("benchmark_dataset") is not None:
        st.markdown(f"""
        <div class="cb-callout">
            <div class="cb-panel-title">Dataset loaded</div>
            <div class="cb-muted">{len(st.session_state.benchmark_dataset)} rows are available in this session.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Clear dataset"):
            st.session_state.benchmark_dataset = None
            st.session_state.benchmark_results = None
            st.rerun()

    else:
        st.markdown(f"""
        <div class="cb-empty">
            <div class="cb-empty-title">No dataset loaded</div>
            <div>Upload organization data to prepare a benchmark workspace.</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== TAB 2: RUN ====================
with tab2:
    _section("Configure and run", "Review the run shape before executing local model comparisons.")

    if st.session_state.get("benchmark_dataset") is None:
        st.markdown(f"""
        <div class="cb-callout cb-callout-error">
            <div class="cb-panel-title">Dataset required</div>
            <div class="cb-muted">Upload a dataset in the Prepare data tab before starting a benchmark run.</div>
        </div>
        """, unsafe_allow_html=True)
    elif not selected_models:
        st.markdown(f"""
        <div class="cb-callout cb-callout-warning">
            <div class="cb-panel-title">Model required</div>
            <div class="cb-muted">Select at least one candidate model in the sidebar.</div>
        </div>
        """, unsafe_allow_html=True)
    elif not task_types:
        st.markdown(f"""
        <div class="cb-callout cb-callout-warning">
            <div class="cb-panel-title">Task required</div>
            <div class="cb-muted">Select at least one task type in the sidebar.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.benchmark_dataset
        mapping = st.session_state.get("benchmark_column_mapping", {})

        st.markdown(
            f"""
<div class="cb-panel">
  <div class="cb-panel-title">Run Summary</div>
  <div class="cb-muted">This run compares selected embedding models against the chosen evaluation task types. Your uploaded data is kept in-session for preview and mapping; the local embedding runner also needs a compatible model runtime.</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        _stat_grid(
            [
                ("Rows", len(df)),
                ("Models", len(selected_models)),
                ("Tasks", len(task_types)),
            ]
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cb-callout cb-callout-warning">
            <div class="cb-panel-title">Runtime note</div>
            <div class="cb-muted">Local embedding evaluation downloads and runs candidate models. This can take several minutes and uses disk, CPU, or GPU resources.</div>
        </div>
        """, unsafe_allow_html=True)

        if not MTEB_BENCHMARK_AVAILABLE:
            st.markdown(f"""
            <div class="cb-callout">
                <div class="cb-panel-title">Local runner disabled</div>
                <div class="cb-muted">PyTorch is only required for executing local embedding models. Uploading data and preparing the benchmark do not require it.</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Enable local embedding execution"):
                st.code(
                    '# pipx install\n'
                    'pipx inject vectoryai "torch>=2.1.0" mteb sentence-transformers\n\n'
                    '# source checkout or activated virtualenv\n'
                    'pip install -e ".[embedding]"',
                    language="bash",
                )
            st.button("Run benchmark", type="primary", use_container_width=True, disabled=True)
        elif st.button("Run benchmark", type="primary", use_container_width=True):
            # Import EmbeddingModel for creating model objects
            from components.models import EmbeddingModel

            # Convert selected model dicts to EmbeddingModel objects
            models = []
            for model_name in selected_models:
                model_info = next((m for m in AVAILABLE_MODELS if m["name"] == model_name), None)
                if model_info:
                    # Get the model ID (HuggingFace format)
                    provider_prefix = {
                        "Sentence Transformers": "sentence-transformers",
                        "BAAI": "BAAI",
                        "Microsoft": "intfloat",
                        "Jina AI": "jinaai",
                        "Nomic AI": "nomic-ai",
                    }.get(model_info["provider"], model_info["provider"].lower().replace(" ", "-"))

                    model_id = f"{provider_prefix}/{model_info['name']}"

                    models.append(EmbeddingModel(
                        name=model_info["name"],
                        provider=model_info["provider"],
                        model_id=model_id,
                        dimensions=model_info["dimensions"],
                        max_tokens=model_info["max_tokens"],
                    ))

            tasks = [EvalTaskType(t) for t in task_types]

            st.markdown('<div class="animated-progress" style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            progress = st.progress(0)
            status = st.empty()

            def update(p, msg):
                progress.progress(p)
                status.markdown(f"**{msg}**")

            try:
                results = run_benchmark(models, tasks, progress_callback=update)
                st.session_state.benchmark_results = results
                status.empty()

                st.success("Benchmark complete. Review the Results tab.")
            except Exception as e:
                error_str = str(e)
                if "torch" in error_str.lower() or "pytorch" in error_str.lower():
                    st.error(f"""
                    ❌ **PyTorch Compatibility Error**

                    Local embedding benchmarking requires the optional embedding runtime in the environment that runs Vectory:
                    ```
                    # pipx install
                    pipx inject vectoryai "torch>=2.1.0" mteb sentence-transformers

                    # source checkout or activated virtualenv
                    pip install -e ".[embedding]"
                    ```

                    Technical details: `{error_str}`
                    """)
                else:
                    st.error(f"❌ Error running benchmark: {e}")

# ==================== TAB 3: RESULTS ====================
with tab3:
    _section("Results", "Review rankings, compare scores, and export the finished benchmark packet.")

    results = st.session_state.get("benchmark_results")

    if results is None:
        st.markdown(f"""
        <div class="cb-empty">
            <div class="cb-empty-title">No results yet</div>
            <div>Run a benchmark to populate rankings, charts, and export files.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Rankings
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom: 16px;">
            <h4 style="margin: 0; color: {theme.text_primary};">Model rankings</h4>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            results.style.format({"mean_score": "{:.3f}"}).background_gradient(
                subset=["mean_score"], cmap="RdYlGn"
            ),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        _section("Visualization")

        fig = px.bar(
            results.sort_values("mean_score"),
            x="mean_score",
            y="model",
            orientation="h",
            color="provider",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Mean Score",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Score bars for each model
        st.markdown(f"""
        <div class="feature-card" style="margin: 20px 0;">
            <h4 style="margin: 0 0 16px 0; color: {theme.text_primary};">Score comparison</h4>
        </div>
        """, unsafe_allow_html=True)

        for _, row in results.iterrows():
            score_bar(row["mean_score"], label=f"{row['model'][:20]} ({row['provider']})")

        # Export
        st.markdown("<br>", unsafe_allow_html=True)
        _section("Export")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download CSV",
                results.to_csv(index=False),
                "benchmark_results.csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download JSON",
                results.to_json(orient="records", indent=2),
                "benchmark_results.json",
                use_container_width=True,
            )
