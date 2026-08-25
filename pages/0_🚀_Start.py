"""Route users to the right Vectory evaluation workflow."""

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state
from components.ui import inject_custom_css, section_header


st.set_page_config(
    page_title="Start | Vectory",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_session_state(st)
inject_custom_css()

st.title("🚀 Start an Evaluation")
st.caption("Choose the evidence you have now; Vectory will route you to the next useful step.")

goal = st.radio(
    "What are you trying to do?",
    [
        "Find failures in outputs or traces",
        "Build or validate an evaluator",
        "Audit an existing evaluation pipeline",
        "Gate an agent release",
    ],
)

dataset_loaded = st.session_state.dataset is not None
annotations = st.session_state.get("trace_annotations", {})
failure_modes = st.session_state.get("failure_modes", {})
promoted = st.session_state.get("promoted_evaluators", {})

section_header("Recommended next step", style="primary")

if goal == "Find failures in outputs or traces":
    if not dataset_loaded:
        st.info("Upload a JSON, JSONL, or CSV export on **📊 Dataset**, then open **🔍 Error Analysis**.")
        target_page = "pages/1_📊_Dataset.py"
        button_label = "Open Dataset"
    elif not annotations:
        st.success(
            "Your dataset is ready. Open **🔍 Error Analysis** and begin with the diverse discovery sample."
        )
        target_page = "pages/8_🔍_Error_Analysis.py"
        button_label = "Start Error Analysis"
    elif not failure_modes:
        st.success(
            f"You have {len(annotations)} annotations. Continue to **Axial Coding** to group notes into failure modes."
        )
        target_page = "pages/8_🔍_Error_Analysis.py"
        button_label = "Continue Error Analysis"
    else:
        st.success(
            f"You have {len(failure_modes)} failure modes. Find related examples, refine the taxonomy, "
            "then export the portable review bundle."
        )
        target_page = "pages/8_🔍_Error_Analysis.py"
        button_label = "Open Taxonomy"

elif goal == "Build or validate an evaluator":
    if not failure_modes:
        st.info(
            "Evaluator definitions should come from observed failures. Complete open and axial coding in "
            "**🔍 Error Analysis** first."
        )
        target_page = "pages/8_🔍_Error_Analysis.py" if dataset_loaded else "pages/1_📊_Dataset.py"
        button_label = "Open Error Analysis" if dataset_loaded else "Open Dataset"
    elif not promoted:
        st.success(
            "Promote one accepted failure mode into a binary draft evaluator from the Taxonomy Dashboard."
        )
        target_page = "pages/8_🔍_Error_Analysis.py"
        button_label = "Promote an Evaluator"
    else:
        st.warning(
            "Your promoted evaluator is still untrusted until it is measured on a held-out test set. "
            "Use the CLI workflow below and require both true-pass and true-fail rates to clear the gate."
        )
        target_page = "pages/8_🔍_Error_Analysis.py"
        button_label = "Inspect Evaluators"
        st.code(
            "vectory split-labels labels.jsonl --label-column human_label --out eval-splits\n"
            "vectory validate-judge eval-splits/test.jsonl --human-column human_label "
            "--judge-column judge_label --split-manifest eval-splits/split_manifest.json "
            "--out validation.json",
            language="bash",
        )

elif goal == "Audit an existing evaluation pipeline":
    st.markdown(
        "Check the pipeline in this order: representative production traces, human-written failure modes, "
        "binary evaluator definitions, disjoint train/dev/test labels, held-out TPR/TNR, and a CI threshold."
    )
    st.warning(
        "Do not accept a single aggregate score as evidence. A judge can look accurate while missing most "
        "real failures, so inspect class balance and both sides of the confusion matrix."
    )
    target_page = "pages/8_🔍_Error_Analysis.py"
    button_label = "Inspect Error Analysis"

else:
    st.success(
        "Use **🧭 Vectory Benchmark** for proof-grounded trace scoring and `vectory gate` for CI. "
        "Use validated failure-mode judges alongside it for product-specific regressions."
    )
    st.code(
        "vectory gate submission.jsonl --min-score 0.90 --block-severity critical "
        "--report-out vectory-report",
        language="bash",
    )
    target_page = "pages/9_🧭_VectoryBenchmark.py"
    button_label = "Open Vectory Benchmark"

if hasattr(st, "switch_page"):
    if st.button(button_label, type="primary"):
        st.switch_page(target_page)
else:
    st.caption(f"Select **{button_label}** from the sidebar.")

st.divider()
st.caption(
    "Discovery samples maximize coverage and are intentionally biased. Use an independent random sample "
    "when estimating production prevalence."
)
