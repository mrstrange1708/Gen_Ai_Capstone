"""
Agent UI — 7-Section Streamlit Display
========================================
Renders the complete ReAct agentic pipeline results:
  §1 Risk Assessment (gauge + SHAP waterfall)
  §2 Clinical Analysis (Llama 3.3 70B reasoning)
  §3 Evidence Base (RAG retrieved guidelines)
  §4 Intervention Plan (3-step cards)
  §5 Quality Evaluation (traffic lights + verdict)
  §6 Full Report Download (JSON)
  §7 Agent Trace (tool call sequence — proves dynamic reasoning)
"""

import streamlit as st
import plotly.graph_objects as go
import json


def render_agent_results(pipeline_result):
    """Render the complete agentic pipeline results in Streamlit."""

    if not pipeline_result:
        st.warning("No results to display.")
        return

    risk = pipeline_result.get("risk_assessment", {})
    risk_score = risk.get("score", 0) * 100
    risk_level = risk.get("level", "UNKNOWN")
    top_factors = risk.get("top_factors", [])
    clinical_analysis = pipeline_result.get("clinical_analysis", "")
    evidence = pipeline_result.get("evidence_base", [])
    plan = pipeline_result.get("intervention_plan", {})
    evaluation = pipeline_result.get("quality_evaluation", {})
    trace = pipeline_result.get("agent_trace", [])

    # ═══════════════════════════════════════════════════════════
    # §1 — RISK ASSESSMENT CARD
    # ═══════════════════════════════════════════════════════════

    st.markdown(
        '<div class="section-header">§1 — Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    # Determine colors by risk level
    if risk_level == "HIGH":
        bar_color = "#ef4444"
        badge_bg = "rgba(239,68,68,0.15)"
    elif risk_level == "MEDIUM":
        bar_color = "#f59e0b"
        badge_bg = "rgba(245,158,11,0.15)"
    else:
        bar_color = "#10b981"
        badge_bg = "rgba(16,185,129,0.15)"

    col_gauge, col_info = st.columns([2, 1])

    with col_gauge:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={"suffix": "%", "font": {"size": 48, "color": "white"}},
                title={
                    "text": "No-Show Risk Score",
                    "font": {"size": 16, "color": "#94a3b8"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 2,
                        "tickcolor": "#334155",
                        "tickfont": {"color": "#94a3b8"},
                    },
                    "bar": {"color": bar_color, "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16,185,129,0.12)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": risk_score,
                    },
                },
            )
        )
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=280,
            margin=dict(l=30, r=30, t=50, b=20),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with col_info:
        st.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 12px; border: 1px solid {bar_color};
                 background: {badge_bg}; text-align: center; margin-top: 1rem;">
                <p style="font-size: 2.2rem; font-weight: 800; color: {bar_color}; margin: 0;">
                    {risk_level}</p>
                <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">
                    Risk Classification</p>
                <p style="font-size: 1.8rem; font-weight: 700; color: white; margin-top: 0.5rem;">
                    {risk_score:.1f}%</p>
                <p style="color: #94a3b8; font-size: 0.8rem;">Probability of No-Show</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── SHAP Waterfall Chart ──
    if top_factors:
        st.markdown(
            '<div class="section-header" style="font-size:1.1rem;">SHAP Feature Impact</div>',
            unsafe_allow_html=True,
        )

        features = [
            f.get("feature", "").replace("_", " ").title() for f in top_factors
        ]
        impacts = [f.get("shap_impact", 0) for f in top_factors]
        values = [f.get("value", 0) for f in top_factors]
        colors = ["#ef4444" if v > 0 else "#10b981" for v in impacts]

        fig_shap = go.Figure(
            go.Bar(
                x=impacts[::-1],
                y=[f"{feat} (={val})" for feat, val in zip(features, values)][::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{v:+.4f}" for v in impacts[::-1]],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=11),
            )
        )
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(
                title="SHAP Impact on No-Show Risk",
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            yaxis=dict(title=""),
            height=250,
            margin=dict(l=10, r=80, t=10, b=30),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(
            """
            <p style="color:#64748b; font-size:0.75rem; text-align:center;">
            🔴 Red = increases no-show risk &nbsp;&nbsp;|&nbsp;&nbsp; 🟢 Green = decreases risk
            </p>
            """,
            unsafe_allow_html=True,
        )

    # ── LOW risk: brief summary and stop ──
    if risk_level == "LOW":
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center; padding:1.5rem;">
                <p style="color:#10b981; font-size:1.2rem; font-weight:600;">
                    Low Risk — Standard Protocol</p>
                <p style="color:#e2e8f0;">
                    {clinical_analysis or "Patient is likely to attend. Standard SMS reminder is sufficient."}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_download(pipeline_result)
        _render_trace(trace, pipeline_result)
        return

    # ═══════════════════════════════════════════════════════════
    # §2 — CLINICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════

    st.markdown(
        '<div class="section-header">§2 — Clinical Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="glass-card">
            <p style="color:#38bdf8; font-size:0.75rem; font-weight:600;
               text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">
                AI-Generated Clinical Reasoning — Llama 3.3 70B</p>
            <p style="color:#e2e8f0; line-height:1.7;">{clinical_analysis}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════
    # §3 — EVIDENCE BASE
    # ═══════════════════════════════════════════════════════════

    if evidence:
        st.markdown(
            '<div class="section-header">§3 — Evidence Base</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style="color:#64748b; font-size:0.8rem; margin-bottom:0.5rem;">
            Retrieved from RAG Knowledge Base (Chroma) —
            sentence-transformers/all-MiniLM-L6-v2</p>
            """,
            unsafe_allow_html=True,
        )
        for i, item in enumerate(evidence):
            source = item.get("source", "Unknown")
            text = item.get("text", "")
            with st.expander(f"{source}", expanded=(i == 0)):
                st.markdown(
                    f"<p style='color:#e2e8f0; line-height:1.6;'>{text}</p>",
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════════════════════════════════════
    # §4 — INTERVENTION PLAN
    # ═══════════════════════════════════════════════════════════

    if plan:
        st.markdown(
            '<div class="section-header">§4 — Intervention Plan</div>',
            unsafe_allow_html=True,
        )

        steps = []
        for key in ["step_1", "step_2", "step_3"]:
            if key in plan:
                steps.append(plan[key])

        if steps:
            cols = st.columns(len(steps))
            step_colors = ["#38bdf8", "#10b981", "#f59e0b"]
            step_icons = ["📱", "📞", "🔄"]

            for i, (step, col) in enumerate(zip(steps, cols)):
                with col:
                    color = step_colors[i % len(step_colors)]
                    icon = step_icons[i % len(step_icons)]
                    action = step.get("action", "N/A")
                    timing = step.get("timing", "N/A")
                    rationale = step.get("rationale", "N/A")
                    responsible = step.get("responsible", "N/A")

                    st.markdown(
                        f"""
                        <div style="padding:1.5rem; border-radius:12px;
                             border:1px solid {color};
                             background:rgba(255,255,255,0.03); min-height:300px;
                             box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <p style="color:{color}; font-size:1.1rem; font-weight:700; margin:0; text-transform:uppercase; letter-spacing:0.5px;">
                                Step {i + 1}</p>
                            <p style="color:white; font-weight:600; margin:0.5rem 0 0.3rem;">
                                {action}</p>
                            <p style="color:#94a3b8; font-size:0.85rem; margin:0.3rem 0;">
                                <strong style="color:#64748b;">⏱ Timing:</strong> {timing}</p>
                            <p style="color:#94a3b8; font-size:0.85rem; margin:0.3rem 0;">
                                <strong style="color:#64748b;">📋 Rationale:</strong> {rationale}</p>
                            <p style="color:#94a3b8; font-size:0.85rem; margin:0.3rem 0;">
                                <strong style="color:#64748b;">👤 Responsible:</strong>
                                {responsible}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ═══════════════════════════════════════════════════════════
    # §5 — QUALITY EVALUATION PANEL
    # ═══════════════════════════════════════════════════════════

    st.markdown(
        '<div class="section-header">§5 — Quality Evaluation</div>',
        unsafe_allow_html=True,
    )

    # Traffic light indicators
    eval_cols = st.columns(3)
    criteria = [
        ("Clinical Accuracy", evaluation.get("clinical_accuracy", "N/A")),
        ("Guideline Alignment", evaluation.get("guideline_alignment", "N/A")),
        ("Ethical Soundness", evaluation.get("ethical_soundness", "N/A")),
    ]

    for col, (name, result) in zip(eval_cols, criteria):
        with col:
            if result == "PASS":
                light_color = "#10b981"
            elif result == "FAIL":
                light_color = "#ef4444"
            else:
                light_color = "#64748b"

            st.markdown(
                f"""
                <div style="text-align:center; padding:1.5rem; border-radius:12px;
                     border:1px solid {light_color};
                     background:rgba(255,255,255,0.03);
                     box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <p style="color:{light_color}; font-weight:700; font-size:1.2rem;
                       margin:0 0 0.5rem 0;">{result}</p>
                    <p style="color:#94a3b8; font-size:0.85rem; font-weight:500; margin:0;">{name}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Verdict badge
    verdict = evaluation.get("verdict", "N/A")
    retry_count = evaluation.get("retry_count", 0)
    verdict_color = "#10b981" if verdict == "APPROVED" else "#f59e0b"

    retry_html = ""
    if retry_count > 0:
        retry_html = (
            f'<p style="color:#f59e0b; font-size:0.85rem; margin:0.3rem 0;">'
            f"Plan revised after {retry_count} evaluation cycle(s)</p>"
        )

    st.markdown(
        f"""
        <div style="text-align:center; padding:1rem; margin-top:0.8rem;
             border-radius:12px; border:1px solid {verdict_color};
             background:rgba(255,255,255,0.03);">
            <p style="color:{verdict_color}; font-size:1.3rem; font-weight:700; margin:0; letter-spacing:1px; text-transform:uppercase;">
                {verdict}</p>
            <p style="color:#64748b; font-size:0.8rem; margin:0.3rem 0;">
                Reviewed by Qwen QwQ 32B</p>
            {retry_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pipeline_result.get("human_review_required"):
        st.warning(
            "Maximum retries reached. Human review is required before "
            "acting on this plan."
        )

    # Justification expander
    justification = evaluation.get("justification", {})
    if justification:
        with st.expander("Detailed Evaluation Justification"):
            for criterion, note in justification.items():
                label = criterion.replace("_", " ").title()
                st.markdown(f"**{label}**: {note}")

    critique = evaluation.get("critique", "")
    if critique:
        with st.expander("Critic Feedback"):
            st.markdown(f"<p style='color:#f59e0b;'>{critique}</p>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # §6 — FULL REPORT DOWNLOAD
    # ═══════════════════════════════════════════════════════════

    _render_download(pipeline_result)

    # ═══════════════════════════════════════════════════════════
    # §7 — AGENT TRACE
    # ═══════════════════════════════════════════════════════════

    _render_trace(trace, pipeline_result)


# ───────────────────────────────────────────────────────────────
# HELPER: Download Button
# ───────────────────────────────────────────────────────────────

def _render_download(pipeline_result):
    st.markdown(
        '<div class="section-header">§6 — Full Report</div>',
        unsafe_allow_html=True,
    )

    report_json = json.dumps(pipeline_result, indent=2, default=str)
    patient_id = pipeline_result.get("patient_id", "unknown")

    st.download_button(
        "Download Report as JSON",
        data=report_json,
        file_name=f"care_coordination_report_{patient_id}.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("View Full Structured Data"):
        def dict_to_html(d):
            if isinstance(d, dict):
                html = '<div style="margin-left: 1rem; border-left: 2px solid rgba(255,255,255,0.1); padding-left: 1rem; margin-bottom: 0.5rem;">'
                for k, v in d.items():
                    key_fmt = str(k).replace('_', ' ').title()
                    html += f'<div style="margin-bottom: 0.4rem;"><span style="color: #38bdf8; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px;">{key_fmt}:</span><br>'
                    html += dict_to_html(v)
                    html += "</div>"
                html += '</div>'
                return html
            elif isinstance(d, list):
                if not d:
                    return '<span style="color: #64748b; font-size: 0.85rem;">[Empty]</span>'
                html = '<div style="margin-left: 1rem; margin-bottom: 0.4rem;">'
                for item in d:
                    html += f'<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 0.6rem; border-radius: 8px; margin-bottom: 0.4rem;">'
                    html += dict_to_html(item)
                    html += '</div>'
                html += '</div>'
                return html
            else:
                return f'<span style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.5;">{str(d)}</span>'

        # Remove agent_trace from the recursive html view to display it natively
        display_dict = {k: v for k, v in pipeline_result.items() if k != "agent_trace"}

        # Wrap in a neat card
        st.markdown(
            f'<div style="background: #0f172a; padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); overflow-x: auto;">'
            f'{dict_to_html(display_dict)}'
            f'</div>',
            unsafe_allow_html=True
        )

        if "agent_trace" in pipeline_result:
            import pandas as pd
            st.markdown('<p style="color: #38bdf8; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px; margin-top: 1rem;">Agent Trace Details:</p>', unsafe_allow_html=True)
            df_trace = pd.DataFrame(pipeline_result["agent_trace"])
            rename_map = {
                "step": "Step",
                "tool": "Action/Tool",
                "agent": "Agent Model",
                "result": "Result",
                "retry": "Retry Loop",
                "thought": "Internal Thought"
            }
            df_trace = df_trace.rename(columns={k: v for k, v in rename_map.items() if k in df_trace.columns})
            df_trace = df_trace.fillna("")
            st.dataframe(df_trace, use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────
# HELPER: Agent Trace — proves dynamic tool selection
# ───────────────────────────────────────────────────────────────

def _render_trace(trace, pipeline_result=None):
    pipeline_result = pipeline_result or {}
    if not trace:
        return

    st.markdown(
        '<div class="section-header">§7 — Agent Trace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style="color:#64748b; font-size:0.8rem;">
        Shows which tools the agent <strong>dynamically chose</strong> to call
        and in what order. This is NOT a hardcoded pipeline — the LLM decides
        the execution path based on reasoning.</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @keyframes slideInNode {
            0% { opacity: 0; transform: translateY(-10px) scale(0.95); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        .trace-node {
            position: relative;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            opacity: 0;
            animation: slideInNode 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        }
        .trace-node:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.4) !important;
        }
        .trace-node .hover-dialog {
            visibility: hidden;
            background-color: #0f172a;
            color: #f8fafc;
            text-align: left;
            border-radius: 8px;
            padding: 1rem;
            position: absolute;
            z-index: 100;
            top: 50%;
            left: 105%; /* Position to the right of the node */
            transform: translateY(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            transition-delay: 0.5s; /* Only show after hovering a bit */
            width: 250px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(56,189,248,0.3);
            font-size: 0.8rem;
            font-weight: 400;
            pointer-events: none;
        }
        .trace-node:hover .hover-dialog {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    trace_html = (
        '<div style="display:flex; flex-direction:column; gap:0.6rem; '
        'align-items:center; padding:1rem 0;">'
    )

    for i, step in enumerate(trace):
        tool_name = step.get("tool", "unknown")
        agent_type = step.get("agent", "main")
        retry = step.get("retry", 0)
        result = step.get("result", "")

        # Color by agent type
        if agent_type == "critic":
            bg = "rgba(168,85,247,0.15)"
            border = "#a855f7"
        else:
            bg = "rgba(56,189,248,0.15)"
            border = "#38bdf8"

        # Result badge (for critic checks)
        result_badge = ""
        if result:
            r_color = "#10b981" if result == "PASS" else "#ef4444"
            result_badge = (
                f'<span style="color:{r_color}; font-size:0.7rem;"> [{result}]</span>'
            )

        # Retry badge
        retry_badge = ""
        if retry > 0:
            retry_badge = (
                f'<span style="color:#f59e0b; font-size:0.65rem;"> R{retry}</span>'
            )

        # Generate hovering dialog detailed info based on pipeline results
        hover_info = ""
        description = "Agent reasoning execution"
        
        if "predict_noshow" in tool_name:
            description = "Calculates risk score using XGBoost & SHAP"
            score = pipeline_result.get("risk_assessment", {}).get("score", "N/A")
            hover_info = f"<strong style='color:#38bdf8'>XGBoost Output:</strong><br>Calculated Risk Score: {score}"
        elif "calculate_risk_flags" in tool_name:
            description = "Analyzes demographics for risk flags"
            top_flags = len(pipeline_result.get("risk_assessment", {}).get("top_factors", []))
            hover_info = f"<strong style='color:#38bdf8'>Demographics Output:</strong><br>Found {top_flags} active clinical risk flags."
        elif "retrieve_guidelines" in tool_name:
            description = "Queries RAG database for guidelines"
            evidence = len(pipeline_result.get("evidence_base", []))
            hover_info = f"<strong style='color:#38bdf8'>RAG Output:</strong><br>Vector search returned {evidence} matched guidelines."
        elif "analyze_risk_factors" in tool_name:
            description = "Synthesizes risk factors & summaries"
            hover_info = "<strong style='color:#38bdf8'>LLM Output:</strong><br>Generated clinical reasoning summary."
        elif "generate_intervention_plan" in tool_name:
            description = "Creates 3-step intervention plan"
            hover_info = "<strong style='color:#38bdf8'>LLM Output:</strong><br>3-step intervention plan requested and built."
        elif "critic" in agent_type:
            description = f"Qwen QwQ 32B evaluation ({result})"
            hover_info = f"<strong style='color:#a855f7'>Critic Output:</strong><br>Plan evaluated. Verdict: {result}"
        else:
            thought = step.get("thought", "Executed reasoning module")
            hover_info = f"<strong style='color:#cbd5e1'>Output:</strong><br>{thought}"

        # Add staggered animation delay so they appear one by one sequentially down the Y-axis
        anim_delay = i * 0.2
        
        trace_html += (
            f'<div class="trace-node" style="padding:0.8rem 1.5rem; border-radius:16px; '
            f'border:1px solid {border}; background:{bg}; '
            f'width:300px; text-align:center; '
            f'box-shadow: 0 4px 6px rgba(0,0,0,0.2); '
            f'animation-delay: {anim_delay}s;">'
            f'<div style="font-size:0.9rem; font-weight:700; color:white; margin-bottom:0.3rem;">{tool_name}{result_badge}{retry_badge}</div>'
            f'<div style="font-size:0.75rem; color:#cbd5e1; font-weight:400;">{description}</div>'
            f'<div class="hover-dialog">{hover_info}</div>'
            f"</div>"
        )

        if i < len(trace) - 1:
            trace_html += '<div style="color:#64748b; font-size:1.5rem; line-height:1; font-weight:700;">↓</div>'

    trace_html += "</div>"

    # Legend
    trace_html += (
        '<div style="margin-top:0.5rem; display:flex; gap:1rem; '
        'font-size:0.75rem; color:#64748b;">'
        '<span>Main Agent (Llama 70B)</span>'
        '<span>Critic (Qwen QwQ 32B)</span>'
        "</div>"
    )

    st.markdown(trace_html, unsafe_allow_html=True)


