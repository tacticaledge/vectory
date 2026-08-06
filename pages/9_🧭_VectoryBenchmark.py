"""Vectory Benchmark page for agent reliability scoring."""

import html
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state
from components.ui import (
    animated_metric,
    get_current_theme,
    hex_to_rgb,
    inject_custom_css,
    section_header,
)
from components.vectory_benchmark import build_leaderboard, load_suite, score_submission
from components.vectory_benchmark.reports import checkpoint_rows, claim_evidence_rows
from components.vectory_benchmark.trace_parser import load_submission_payload


st.set_page_config(
    page_title="Vectory Benchmark | Vectory",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_session_state(st)
inject_custom_css()
theme = get_current_theme()


def _score_rows(scores):
    rows = []
    for score in scores:
        row = {
            "agent": score.agent,
            "model": score.model,
            "task_id": score.task_id,
            "domain": score.domain.value,
            "run_id": score.run_id,
            "vectory_score": score.vectory_score,
            "band": score.band.value,
            "passed": score.passed,
            "pathology_count": len(score.pathologies),
            "pathology_risk": round(min(1.0, sum(f.score_penalty for f in score.pathologies)), 4),
        }
        for name, dimension in score.dimensions.items():
            row[name] = dimension.score
        rows.append(row)
    return pd.DataFrame(rows)


def _inject_benchmark_css():
    st.markdown(
        f"""
<style>
.vb-hero {{
    border: 1px solid {theme.border};
    border-radius: 8px;
    padding: 26px 28px;
    margin: 4px 0 22px 0;
    background: linear-gradient(135deg, rgba({hex_to_rgb(theme.bg_card)}, 0.95), rgba({hex_to_rgb(theme.bg_secondary)}, 0.72));
}}
.vb-eyebrow {{
    color: {theme.text_muted};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.vb-title {{
    color: {theme.text_primary};
    font-family: 'DM Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: 0;
    line-height: 1.1;
    margin: 0;
}}
.vb-subtitle {{
    color: {theme.text_secondary};
    font-size: 1rem;
    line-height: 1.55;
    max-width: 920px;
    margin: 10px 0 0 0;
}}
.vb-stat {{
    min-height: 118px;
    border: 1px solid {theme.border};
    border-radius: 8px;
    padding: 16px;
    background: {theme.bg_card};
}}
.vb-stat-label {{
    color: {theme.text_muted};
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.vb-stat-value {{
    color: {theme.text_primary};
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 8px;
}}
.vb-stat-note {{
    color: {theme.text_muted};
    font-size: 0.82rem;
    margin-top: 8px;
    line-height: 1.35;
}}
.vb-panel {{
    border: 1px solid {theme.border};
    border-radius: 8px;
    padding: 18px 18px 16px 18px;
    background: {theme.bg_card};
    margin-bottom: 16px;
}}
.vb-panel-title {{
    color: {theme.text_primary};
    font-size: 1.05rem;
    font-weight: 750;
    margin-bottom: 8px;
}}
.vb-muted {{
    color: {theme.text_muted};
    font-size: 0.88rem;
    line-height: 1.5;
}}
.vb-pill-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}}
.vb-pill {{
    border: 1px solid {theme.border};
    border-radius: 999px;
    padding: 5px 10px;
    color: {theme.text_secondary};
    background: rgba({hex_to_rgb(theme.bg_secondary)}, 0.55);
    font-size: 0.78rem;
    white-space: nowrap;
}}
.vb-list {{
    margin: 10px 0 0 0;
    padding-left: 18px;
}}
.vb-list li {{
    margin-bottom: 7px;
    color: {theme.text_secondary};
}}
.vb-table-note {{
    color: {theme.text_muted};
    font-size: 0.82rem;
    margin: 4px 0 10px 0;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid {theme.border};
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px 8px 0 0;
    padding: 10px 14px;
}}
@media (max-width: 760px) {{
    .vb-hero {{
        padding: 20px;
    }}
    .vb-title {{
        font-size: 1.8rem;
    }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _stat_card(label: str, value: str, note: str):
    st.markdown(
        f"""
<div class="vb-stat">
  <div class="vb-stat-label">{html.escape(label)}</div>
  <div class="vb-stat-value">{html.escape(str(value))}</div>
  <div class="vb-stat-note">{html.escape(note)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _proof_rows(scores):
    rows = []
    for score in scores:
        claims = score.facts.get("claims", 0)
        obligations = score.facts.get("proof_obligations", 0)
        checkers = score.facts.get("checker_results", 0)
        if claims or obligations or checkers or "proof_grounding" in score.dimensions:
            rows.append(
                {
                    "agent": score.agent,
                    "model": score.model,
                    "task_id": score.task_id,
                    "run_id": score.run_id,
                    "proof_grounding": score.dimensions.get("proof_grounding").score if "proof_grounding" in score.dimensions else None,
                    "claims": claims,
                    "proof_obligations": obligations,
                    "checker_results": checkers,
                }
            )
    return pd.DataFrame(rows)


def _pathology_rows(scores):
    rows = []
    for score in scores:
        for finding in score.pathologies:
            rows.append(
                {
                    "agent": score.agent,
                    "model": score.model,
                    "task_id": score.task_id,
                    "run_id": score.run_id,
                    "code": finding.code,
                    "name": finding.name,
                    "severity": finding.severity.value,
                    "penalty": finding.score_penalty,
                    "evidence": "; ".join(finding.evidence),
                    "recommendation": finding.recommendation,
                }
            )
    return pd.DataFrame(rows)


def _task_panel(task):
    criteria = "".join(
        f"<li>{html.escape(criterion)}</li>" for criterion in task.success_criteria[:4]
    )
    pressures = "".join(
        f"<span class='vb-pill'>{html.escape(pressure)}</span>"
        for pressure in task.adversarial_pressures[:4]
    )
    tags = "".join(f"<span class='vb-pill'>{html.escape(tag)}</span>" for tag in task.tags)
    st.markdown(
        f"""
<div class="vb-panel">
  <div class="vb-panel-title">{html.escape(task.title)}</div>
  <div class="vb-muted">{html.escape(task.intent)}</div>
  <ul class="vb-list">{criteria}</ul>
  <div class="vb-pill-row">{pressures}</div>
  <div class="vb-pill-row">{tags}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _download_json(label, payload, filename):
    st.download_button(
        label,
        json.dumps(payload, indent=2),
        filename,
        "application/json",
        use_container_width=True,
    )


suite = load_suite()
_inject_benchmark_css()

st.markdown(
    f"""
<div class="vb-hero">
  <div class="vb-eyebrow">Agent reliability evaluation</div>
  <h1 class="vb-title">🧭 Vectory Benchmark</h1>
  <p class="vb-subtitle">
    Score submitted agent traces across task outcome, grounding, productive work, retrieval discipline,
    recovery behavior, turn efficiency, evidence quality, proof grounding, and control.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px;">
            <span style="font-size: 2rem;">🧭</span>
            <h3 style="margin: 8px 0; color: {theme.text_primary};">Suite</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{suite.title} · v{suite.version}")
    domains = ["All"] + sorted({task.domain.value for task in suite.tasks})
    selected_domain = st.selectbox("Domain", domains, label_visibility="collapsed")
    st.markdown("---")
    st.metric("Tasks", len(suite.tasks))
    st.metric("Domains", len({task.domain for task in suite.tasks}))


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Suite", "Score", "Report", "Applications", "Format"])


with tab1:
    section_header("Suite Snapshot", style="primary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _stat_card("Tasks", len(suite.tasks), "Versioned scenarios across agent workflows.")
    with c2:
        _stat_card("Domains", len({task.domain for task in suite.tasks}), "Coverage across coding, research, data, and operations.")
    with c3:
        _stat_card("Dimensions", "9", "Outcome, behavior, evidence, and proof signals from the trace.")
    with c4:
        _stat_card("Pathology Checks", "23", "Looping, drift, proof, weak sampling, and control failures.")

    with st.expander("Current boundary", expanded=False):
        st.markdown(
            "Vectory Benchmark scores uploaded JSON/JSONL traces against a built-in task manifest. "
            "It does not run agents, execute tool calls, or provision task environments."
        )

    filtered_tasks = [
        task for task in suite.tasks if selected_domain == "All" or task.domain.value == selected_domain
    ]
    task_rows = [
        {
            "Task": task.title,
            "Domain": task.domain.value,
            "Difficulty": task.difficulty,
            "Turn cliff": task.limits.turn_cliff_events,
            "Max tools": task.limits.max_tool_calls,
        }
        for task in filtered_tasks
    ]
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Task Browser", style="info")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            '<div class="vb-table-note">Filtered manifest view. Select a task on the right for criteria and pressure details.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(task_rows), use_container_width=True, height=330, hide_index=True)

    with right:
        selected_task_id = st.selectbox(
            "Inspect task",
            [task.task_id for task in filtered_tasks],
            format_func=lambda task_id: next(task.title for task in filtered_tasks if task.task_id == task_id),
        )
        selected_task = next(task for task in suite.tasks if task.task_id == selected_task_id)
        _task_panel(selected_task)
        with st.expander("Operational limits"):
            st.json(selected_task.limits.model_dump())


with tab2:
    section_header("Score Agent Runs", style="success")
    upload_col, example_col = st.columns([1.05, 0.95], gap="large")
    with upload_col:
        st.markdown(
            f"""
<div class="vb-panel">
  <div class="vb-panel-title">Upload trace file</div>
  <div class="vb-muted">JSON or JSONL submissions are scored immediately and saved in this browser session.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload Vectory Benchmark JSON or JSONL",
            type=["json", "jsonl"],
            label_visibility="collapsed",
        )

        if uploaded:
            try:
                payload = load_submission_payload(uploaded.getvalue())
                scores = score_submission(suite.tasks, payload)
                st.session_state.vectory_benchmark_scores = scores
                st.session_state.vectory_benchmark_payload = payload
                st.success(f"Scored {len(scores)} run(s).")
            except Exception as exc:
                st.error(f"Could not score submission: {exc}")

    with example_col:
        st.markdown(
            f"""
<div class="vb-panel">
  <div class="vb-panel-title">Expected event shape</div>
  <div class="vb-muted">Use ordered events with action type, tool name, path or input, output summary, success flag, final answer, and optional token or wall-time metrics.</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    example = {
        "agent": "ExampleAgent",
        "model": "example-model",
        "task_id": suite.tasks[0].task_id,
        "run_id": f"{suite.tasks[0].task_id}.run_0",
        "status": "completed",
        "final_answer": "Implemented the fix in src/parser.py and verified the test.",
        "events": [
            {"type": "tool_call", "name": "read_file", "path": "tests/test_parser.py", "success": True},
            {"type": "file_edit", "path": "src/parser.py", "success": True},
            {"type": "verification", "name": "pytest", "success": True},
            {"type": "final", "content": "Completed with verification."},
        ],
        "metrics": {"tokens": 8200},
    }
    policy_proof_example = {
        "agent": "PolicyProofAgent",
        "model": "example-model",
        "task_id": "proof_grounding.policy_guardrail.001",
        "run_id": "proof_grounding.policy_guardrail.001.run_0",
        "status": "completed",
        "final_answer": "VALID policy answer with supportingRules, source-reference, and NO_TRANSLATIONS warning.",
        "claims": [
            {
                "claim_id": "claim.policy.valid",
                "text": "The translated policy claim is valid.",
                "evidence_ids": ["evidence.policy.check"],
                "obligation_ids": ["obligation.policy_translation", "obligation.supporting_rules"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence.policy.check",
                "source_type": "policy_checker_output",
                "output_span": "VALID with supportingRules and source-reference.",
            }
        ],
        "proof_obligations": [
            {
                "obligation_id": "obligation.policy_translation",
                "description": "Policy translation is represented by typed variables.",
                "checker_type": "policy",
                "status": "closed",
            },
            {
                "obligation_id": "obligation.supporting_rules",
                "description": "Supporting policy rules and source references are attached.",
                "checker_type": "policy",
                "status": "closed",
            },
            {
                "obligation_id": "obligation.untranslated_claims",
                "description": "NO_TRANSLATIONS is preserved as a warning, not a proved fact.",
                "checker_type": "formal",
                "status": "closed",
            },
        ],
        "checker_results": [
            {
                "checker_id": "checker.policy.proof",
                "name": "policy-proof-check",
                "checker_type": "policy",
                "status": "passed",
                "obligation_ids": [
                    "obligation.policy_translation",
                    "obligation.supporting_rules",
                    "obligation.untranslated_claims",
                ],
                "output_summary": "VALID with supportingRules; NO_TRANSLATIONS warning retained.",
            }
        ],
        "events": [
            {"type": "evidence", "content": "policy-rule and source-reference evidence loaded", "success": True},
            {"type": "proof_obligation", "content": "required obligations closed", "success": True},
            {"type": "checker_result", "content": "policy checker passed", "success": True},
            {"type": "policy_check", "content": "VALID with NO_TRANSLATIONS warning", "success": True},
            {"type": "verification", "content": "claim, evidence, obligation, and checker chain reviewed", "success": True},
            {"type": "checkpoint", "name": "proof-grounding-checkpoint", "content": "all obligations closed", "output": {"wall_time_seconds": 420, "score": 0.98, "open_obligations": 0, "failed_checkers": 0}, "success": True},
            {"type": "final", "content": "done"},
        ],
    }
    policy_proof_failure_example = {
        "agent": "PolicyProofBadAgent",
        "model": "example-model",
        "task_id": "proof_grounding.policy_guardrail.001",
        "run_id": "proof_grounding.policy_guardrail.001.bad_run_0",
        "status": "completed",
        "declared_success": True,
        "final_answer": "The policy answer is complete and verified despite failed checker output.",
        "claims": [
            {
                "claim_id": "claim.policy.valid",
                "text": "The translated policy claim is VALID.",
                "evidence_ids": ["evidence.missing"],
                "obligation_ids": ["obligation.policy_translation", "obligation.supporting_rules"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence.no_translations",
                "source_type": "policy_checker_output",
                "output_span": "NO_TRANSLATIONS for side claim outside policy variables.",
            }
        ],
        "proof_obligations": [
            {
                "obligation_id": "obligation.policy_translation",
                "description": "Translated claim includes an unsupported policy variable.",
                "checker_type": "policy",
                "status": "failed",
            },
            {
                "obligation_id": "obligation.supporting_rules",
                "description": "supportingRules and source-reference evidence are missing.",
                "checker_type": "policy",
                "status": "open",
            },
            {
                "obligation_id": "obligation.untranslated_claims",
                "description": "NO_TRANSLATIONS was incorrectly treated as a proved policy claim.",
                "checker_type": "formal",
                "status": "open",
            },
        ],
        "checker_results": [
            {
                "checker_id": "checker.policy.proof",
                "name": "policy-proof-check",
                "checker_type": "policy",
                "status": "failed",
                "obligation_ids": ["obligation.policy_translation", "obligation.supporting_rules"],
                "output_summary": "INVALID: no supportingRules prove the requested conclusion.",
            }
        ],
        "events": [
            {"type": "claim", "content": "The policy answer is complete and verified.", "success": True},
            {"type": "proof_obligation", "content": "required obligations remained open", "success": False},
            {"type": "checker_result", "content": "policy checker failed", "success": False},
            {"type": "policy_check", "content": "INVALID with NO_TRANSLATIONS warning", "success": False},
            {"type": "final", "content": "done"},
        ],
    }
    with example_col:
        with st.expander("Example run payload", expanded=True):
            st.json(example)
            _download_json("Download Example JSON", [example], "vectory_benchmark_example_submission.json")
        with st.expander("Policy proof payload", expanded=False):
            st.caption(
                "Synthetic policy-proof trace: claims must connect to source references, closed obligations, and passing policy checker results."
            )
            st.json(policy_proof_example)
            _download_json("Download Policy Proof JSON", [policy_proof_example], "vectory_benchmark_policy_proof_submission.json")
        with st.expander("Policy proof failure payload", expanded=False):
            st.caption(
                "Negative trace: claims success despite failed checkers, open obligations, and unsupported evidence."
            )
            st.json(policy_proof_failure_example)
            _download_json("Download Policy Proof Failure JSON", [policy_proof_failure_example], "vectory_benchmark_policy_proof_failure_submission.json")


with tab3:
    scores = st.session_state.get("vectory_benchmark_scores", [])
    if not scores:
        st.info("Upload a submission on the Score tab to see results.")
    else:
        score_df = _score_rows(scores)
        leaderboard_df = build_leaderboard(scores)
        pathology_df = _pathology_rows(scores)
        proof_df = _proof_rows(scores)
        payload = st.session_state.get("vectory_benchmark_payload", [])
        checkpoint_df = pd.DataFrame(checkpoint_rows(payload, scores))
        claim_evidence_df = pd.DataFrame(claim_evidence_rows(payload, scores))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            _stat_card("Mean Score", f"{score_df['vectory_score'].mean():.3f}", "Overall weighted reliability.")
        with col2:
            _stat_card("Pass@1", f"{score_df['passed'].mean():.1%}", "Completed runs meeting pass threshold.")
        with col3:
            _stat_card("Pathology Risk", f"{score_df['pathology_risk'].mean():.3f}", "Aggregate behavior penalty.")
        with col4:
            _stat_card("Control Index", f"{score_df['agent_control'].mean():.3f}", "Scope discipline and boundedness.")

        dimension_cols = [
            "task_success",
            "reality_sampling",
            "trace_productivity",
            "tool_retrieval_discipline",
            "recovery",
            "agent_control",
            "turn_efficiency",
            "evidence_quality",
            "proof_grounding",
        ]
        report_left, report_right = st.columns([0.95, 1.05], gap="large")
        with report_left:
            st.markdown("### Leaderboard")
            st.dataframe(
                leaderboard_df.style.format(
                    {
                        "vectory_score": "{:.3f}",
                        "pass_at_1": "{:.1%}",
                        "robust_pass_at_5": "{:.1%}",
                        "productive_work_ratio": "{:.3f}",
                        "pathology_risk": "{:.3f}",
                        "agent_control_index": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with report_right:
            melted = score_df.melt(
                id_vars=["agent", "model", "task_id", "run_id"],
                value_vars=[col for col in dimension_cols if col in score_df.columns],
                var_name="dimension",
                value_name="score",
            )
            fig = px.bar(
                melted,
                x="dimension",
                y="score",
                color="agent",
                barmode="group",
                title="Dimension Scores",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                yaxis_range=[0, 1],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=48, b=70),
                legend_title_text="Agent",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Run scores", expanded=False):
            st.dataframe(
                score_df.style.format({"vectory_score": "{:.3f}", "pathology_risk": "{:.3f}"}),
                use_container_width=True,
                height=360,
                hide_index=True,
            )

        with st.expander("Proof grounding", expanded=not proof_df.empty):
            if proof_df.empty:
                st.info("No proof artifacts were submitted. Non-proof tasks receive full proof-grounding credit by default.")
            else:
                st.dataframe(
                    proof_df.style.format({"proof_grounding": "{:.3f}"}),
                    use_container_width=True,
                    height=260,
                    hide_index=True,
                )

        with st.expander("Checkpoint timeline", expanded=not checkpoint_df.empty):
            if checkpoint_df.empty:
                st.info("No checkpoint events were submitted. Add `checkpoint`, `snapshot`, or `progress` events to show score movement over time.")
            else:
                st.dataframe(checkpoint_df, use_container_width=True, height=260, hide_index=True)
                if "wall_time_seconds" in checkpoint_df.columns and "checkpoint_score" in checkpoint_df.columns:
                    chart_df = checkpoint_df.dropna(subset=["wall_time_seconds", "checkpoint_score"])
                    if not chart_df.empty:
                        fig = px.line(chart_df, x="wall_time_seconds", y="checkpoint_score", color="run_id", markers=True, title="Checkpoint Score Timeline")
                        fig.update_layout(yaxis_range=[0, 1], plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Claim-to-evidence table", expanded=not claim_evidence_df.empty):
            if claim_evidence_df.empty:
                st.info("No run-level claims were submitted.")
            else:
                st.dataframe(claim_evidence_df, use_container_width=True, height=280, hide_index=True)

        with st.expander("Report bundle artifacts", expanded=False):
            st.markdown("The open-source CLI can write an inspectable bundle with `index.html`, `scores.json`, `leaderboard.json`, `pathologies.json`, `claim_evidence_table.json`, `checkpoints.json`, and `benchmark_card.json`.")
            st.code("vectory benchmark submission.json --report-out /tmp/vectory-report", language="bash")
            st.code("vectory gate submission.json --min-score 0.90 --block-severity critical --report-out /tmp/vectory-gate-report", language="bash")

        with st.expander("Pathology breakdown", expanded=not pathology_df.empty):
            if pathology_df.empty:
                st.success("No pathologies detected in the uploaded runs.")
            else:
                severity_counts = Counter(pathology_df["severity"])
                chips = " ".join(
                    f"<span class='vb-pill'>{html.escape(severity)}: {count}</span>"
                    for severity, count in severity_counts.items()
                )
                st.markdown(f"<div class='vb-pill-row'>{chips}</div>", unsafe_allow_html=True)
                st.dataframe(pathology_df, use_container_width=True, height=320, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Run Scores CSV",
                score_df.to_csv(index=False),
                "vectorybenchmark_run_scores.csv",
                "text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download Leaderboard CSV",
                leaderboard_df.to_csv(index=False),
                "vectorybenchmark_leaderboard.csv",
                "text/csv",
                use_container_width=True,
            )


with tab4:
    section_header("Where Vectory Benchmark Fits", style="warning")
    st.markdown(
        """
Vectory Benchmark is useful when a team already has agent traces and needs to understand behavior, not just final answers.
It is designed to identify whether an agent observed before deciding, used tools productively, converged during retrieval,
changed hypothesis after failure, respected scope, and stopped before extra turns became low-value work.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Use it for**")
        st.markdown(
            """
- Comparing agent versions before model, prompt, tool, or harness changes
- Reviewing private traces before broader rollout
- Turning production incidents into regression checks
- Auditing workflow boundaries and approval rules
- Measuring search convergence and retrieval discipline
- Building a baseline before a larger executable eval harness
            """
        )
    with c2:
        st.markdown("**Do not use it as**")
        st.markdown(
            """
- A live agent runner
- A replacement for private ground-truth checks
- A hosted public leaderboard
- A substitute for human expert calibration
- A source verifier for facts not represented in the trace
- A sandbox for executing submitted tool calls
            """
        )

    st.markdown("### Private Evaluations")
    st.markdown(
        """
Vectory can help turn internal workflows, agent traces, and production incidents into confidential
Vectory Benchmark suites with custom task taxonomies, pathology rules, baselines, and remediation reports.
        """
    )
    st.markdown("### Proof-Grounded Policy Validation")
    st.markdown(
        """
The policy-proof sample demonstrates a proof-based validation path for policy-sensitive AI answers. A normal LLM judge can decide that an answer sounds right, a keyword check can find `VALID`, and a RAG metric can measure retrieval quality. Vectory proof grounding checks the stronger chain: final claim -> evidence reference -> source document or checker output -> closed proof obligation -> passing checker result.

That chain matters when a result includes partial validation. If a checker returns `NO_TRANSLATIONS`, Vectory keeps it visible as an unproved span instead of letting the agent silently count it as verified.
        """
    )
    st.link_button("Contact Vectory", "https://vectoryai.com/#contact", use_container_width=False)


with tab5:
    section_header("Submission Contract", style="info")
    st.markdown(
        """
Vectory Benchmark accepts JSON or JSONL. Submit one object per agent run with:

- `agent`, `model`, `task_id`, `run_id`
- `status`: `completed`, `failed`, `timeout`, `aborted`, or `unknown`
- `final_answer`
- `events`: ordered trace events
- optional `metrics`: tokens, wall time, cost, or run metadata

Trace event types: `message`, `tool_call`, `tool_result`, `file_edit`, `test_run`, `verification`, `decision`, `claim`, `evidence`, `proof_obligation`, `checker_result`, `policy_check`, `checkpoint`, `error`, `final`.

Proof-sensitive runs may also include run-level `claims`, `evidence`, `proof_obligations`, and `checker_results`. Local formal checker execution is disabled by default and only runs trusted suite-defined commands when explicitly enabled from the CLI.

Current boundary: Vectory Benchmark scores the trace you submit. It does not execute uploaded tool calls or independently verify external source truth.
        """
    )
    st.code(
        json.dumps(
            {
                "agent": "ExampleAgent",
                "model": "example-model",
                "task_id": "research.search_converges.001",
                "run_id": "research.search_converges.001.run_0",
                "status": "completed",
                "final_answer": "Answer with source and as-of evidence.",
                "events": [
                    {"type": "tool_call", "name": "search", "input": {"query": "specific query"}},
                    {"type": "verification", "content": "Cross-checked source recency."},
                    {"type": "final", "content": "Final answer prepared."},
                ],
                "metrics": {"tokens": 12000},
            },
            indent=2,
        ),
        language="json",
    )
