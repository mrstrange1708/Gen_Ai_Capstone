"""
ReAct Agentic Care Coordination Pipeline
=========================================
Architecture:
  - Main Agent: Llama 3.3 70B (Groq) — ReAct loop with 5 tools
    The LLM DECIDES which tools to call and in what order.
  - Critic: Qwen QwQ 32B (Groq) — Evaluates plan on 3 criteria
    Runs 3 evaluation checks, then reasons about the verdict.
  - Retry Loop: Max 2 retries if critic returns NEEDS_REVISION
    Critique is injected back into the agent's messages.
"""

import os
import json
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

MAIN_AGENT_SYSTEM = """You are a clinical care coordination AI agent for a hospital appointment no-show prediction system.

You have access to tools to predict risk, retrieve guidelines, analyze factors, and build intervention plans. The LLM (you) DECIDES which tools to call and in what order based on reasoning.

MANDATORY FIRST STEP:
You MUST call predict_noshow as your VERY FIRST action. Do NOT skip this tool.
The risk_score in your final JSON MUST equal the EXACT value returned by the predict_noshow tool.
NEVER write a risk_score that was not returned by predict_noshow.
If predict_noshow has not been called, you do not know the score — do NOT guess or infer one.

WORKFLOW — follow this order strictly:
1. Call predict_noshow FIRST to get the patient's risk score and SHAP explanation (MANDATORY)
2. Call calculate_risk_flags to identify demographic/social risk factors
3. If risk level is LOW → give a brief summary as your final answer immediately
4. If MEDIUM or HIGH:
   a. Call retrieve_guidelines with a search query based on the identified risk factors
   b. Call analyze_risk_factors to compile a structured clinical summary
   c. Call generate_intervention_plan to get guideline-matched intervention templates
   d. Synthesize everything into your final comprehensive answer

Your FINAL ANSWER must be ONLY valid JSON (no markdown, no ```json blocks, no extra text) in this format:
{
  "risk_assessment": {
    "score": 0.87,
    "level": "HIGH",
    "top_factors": [
      {"feature": "previous_no_shows", "value": 3, "shap_impact": 0.23}
    ]
  },
  "risk_flags": ["elderly_patient", "uninsured"],
  "clinical_analysis": "3-4 sentence clinical reasoning paragraph explaining the combination of factors driving the risk...",
  "evidence_base": [
    {"text": "guideline excerpt...", "source": "document_name.txt"}
  ],
  "intervention_plan": {
    "step_1": {"action": "...", "timing": "...", "rationale": "...", "responsible": "..."},
    "step_2": {"action": "...", "timing": "...", "rationale": "...", "responsible": "..."},
    "step_3": {"action": "...", "timing": "...", "rationale": "...", "responsible": "..."}
  }
}

For LOW risk patients: include only risk_assessment, risk_flags, and a brief clinical_analysis. Omit evidence_base and intervention_plan.

CRITICAL RULES:
- The risk_assessment.score MUST be the exact value from predict_noshow — never invent a score
- Be PRECISE, CLINICAL, and EVIDENCE-BASED
- Cite source document names in your rationale fields
- Each intervention step MUST have: action, timing, rationale, responsible
- Use positive patient-choice language (offer, consider) — never assume or mandate
- Output ONLY the JSON object — no markdown, no code fences, no explanatory text before or after"""


# ═══════════════════════════════════════════════════════════════════════════════
# JSON EXTRACTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks and mixed text."""
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first complete JSON object by brace matching
    brace_depth = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                try:
                    return json.loads(text[start_idx:i + 1])
                except json.JSONDecodeError:
                    start_idx = None

    raise ValueError(f"Could not parse JSON from response: {text[:300]}")


# ═══════════════════════════════════════════════════════════════════════════════
# CRITIC EVALUATION CHECKS (deterministic Python functions)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_clinical_accuracy(intervention_plan: dict, shap_features: list) -> dict:
    """Check if interventions address the identified SHAP risk factors."""
    plan_text = json.dumps(intervention_plan).lower()
    top_features = shap_features if isinstance(shap_features, list) else []

    addressed = 0
    for feat in top_features:
        name = feat.get("feature", "").lower().replace("_", " ")
        keywords = [kw for kw in name.split() if len(kw) > 3]
        if any(kw in plan_text for kw in keywords):
            addressed += 1

    total = max(len(top_features), 1)
    return {
        "criterion": "clinical_accuracy",
        "result": "PASS" if addressed / total >= 0.4 else "FAIL",
        "detail": f"Plan addresses {addressed}/{total} top SHAP risk factors",
    }


def _check_guideline_alignment(intervention_plan: dict, guidelines: list) -> dict:
    """Check if interventions reference the retrieved guidelines."""
    plan_text = json.dumps(intervention_plan).lower()
    guideline_list = guidelines if isinstance(guidelines, list) else []

    matched = 0
    for g in guideline_list:
        text = g.get("text", "").lower()
        words = set(text.split())
        significant = {w for w in words if len(w) > 5}
        overlap = sum(1 for w in significant if w in plan_text)
        if overlap >= 3:
            matched += 1

    total = max(len(guideline_list), 1)
    return {
        "criterion": "guideline_alignment",
        "result": "PASS" if matched / total >= 0.3 else "FAIL",
        "detail": f"{matched}/{total} guidelines reflected in intervention plan",
    }


def _check_ethical_soundness(intervention_plan: dict, patient_data: dict) -> dict:
    """Check for discriminatory assumptions or coercive language."""
    plan_text = json.dumps(intervention_plan).lower()

    issues = []
    bad_phrases = [
        "patient cannot", "patient is unable", "force", "mandate",
        "due to their poverty", "because they are unemployed",
        "require the patient to",
    ]
    for phrase in bad_phrases:
        if phrase in plan_text:
            issues.append(f"Contains assumption: '{phrase}'")

    positive = any(w in plan_text for w in [
        "offer", "option", "consider", "if needed", "evaluate",
        "provide", "assist", "support",
    ])
    if not positive:
        issues.append("Plan lacks patient-choice language (offer, consider, option)")

    return {
        "criterion": "ethical_soundness",
        "result": "PASS" if not issues else "FAIL",
        "detail": "; ".join(issues) if issues else "Plan is ethically sound",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent_pipeline(
    patient_data: dict,
    user_query: str = "What interventions should we take for this patient?",
):
    """
    Runs the full agentic pipeline:
      1. Main ReAct Agent (Llama 70B) — dynamically calls tools
      2. Evaluation checks + Critic reasoning (Qwen QwQ 32B)
      3. Retry loop (max 2) if critic says NEEDS_REVISION
    Returns the complete structured report dict.
    """

    # ── Shared caches for closure-based tools ──
    _shap_cache = {}
    _guidelines_cache = {}
    _risk_flags_cache = {}

    # ═══════════════════════════════════════════════════════════
    # TOOL DEFINITIONS — closures capturing patient_data
    # The LLM decides WHICH tools to call and WHEN.
    # ═══════════════════════════════════════════════════════════

    @tool
    def predict_noshow(reason: str) -> str:
        """Predict no-show probability for the current patient using XGBoost and explain with SHAP.
        Input: brief reason for calling (e.g. 'Initial risk assessment').
        Returns: JSON with risk_score (0-1), risk_level (LOW/MEDIUM/HIGH), top 5 SHAP features.
        ALWAYS call this tool FIRST before any other tools."""
        try:
            import shap

            model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
            feature_columns = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns.pkl"))

            input_df = pd.DataFrame([patient_data])
            input_df = input_df.reindex(columns=list(feature_columns), fill_value=0)

            prob = float(model.predict_proba(input_df)[0][1])

            if prob < 0.4:
                risk_level = "LOW"
            elif prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # Handle all three possible SHAP output shapes
            if isinstance(shap_values, list):
                # Old SHAP: list of arrays per class
                sv = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                # 3D: (samples, features, classes)
                sv = shap_values[0, :, 1]
            else:
                # 2D: (samples, features) — modern SHAP with XGBoost
                sv = shap_values[0]

            feature_names = list(feature_columns)
            shap_vals = np.array(sv).flatten()

            # Top 5 by absolute SHAP value
            top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_names[idx],
                    "value": float(input_df.iloc[0, idx]),
                    "shap_impact": round(float(shap_vals[idx]), 4),
                })

            result = {
                "risk_score": round(prob, 4),
                "risk_level": risk_level,
                "top_shap_features": top_features,
            }
            _shap_cache.update(result)

            print(f"[PREDICT_NOSHOW DEBUG] Output: {json.dumps(result)}")
            return json.dumps(result)

        except Exception as e:
            # DO NOT return a fake score — raise so the real error is visible
            print(f"[PREDICT_NOSHOW ERROR] {type(e).__name__}: {str(e)}")
            raise RuntimeError(
                f"predict_noshow tool failed: {str(e)}. "
                f"Check SHAP version compatibility with XGBoost."
            )

    @tool
    def calculate_risk_flags(reason: str) -> str:
        """Calculate demographic and social risk flags for the current patient.
        Input: brief reason (e.g. 'Identify risk flags').
        Returns: JSON list of active flags like 'elderly_patient', 'uninsured', 'long_distance', etc."""

        flags = []
        age = patient_data.get("age", 0)

        if age >= 65:
            flags.append("elderly_patient")
        if 18 <= age <= 30:
            flags.append("young_adult")
        if patient_data.get("is_uninsured", 0) == 1:
            flags.append("uninsured")
        if patient_data.get("is_unemployed", 0) == 1:
            flags.append("unemployed")
        if patient_data.get("distance_km", 0) > 15:
            flags.append("long_distance")
        if patient_data.get("travel_time_min", 0) > 45:
            flags.append("high_travel_time")
        if patient_data.get("lead_time", 0) > 21:
            flags.append("long_lead_time")
        if patient_data.get("lead_time", 0) <= 3:
            flags.append("short_lead_time")
        if patient_data.get("previous_no_shows", 0) >= 3:
            flags.append("high_risk_prior_noshow_3plus")
        elif patient_data.get("previous_no_shows", 0) >= 1:
            flags.append("prior_noshow")
        if patient_data.get("has_chronic_condition", 0) == 1:
            flags.append("chronic_condition")
        if patient_data.get("multiple_chronic", 0) == 1:
            flags.append("multiple_chronic_conditions")
        if patient_data.get("got_reminder", 0) == 0:
            flags.append("no_reminder_sent")
        if patient_data.get("is_new_patient", 0) == 1:
            flags.append("new_patient")
        if patient_data.get("rainy_day", 0) == 1:
            flags.append("rainy_day")

        result = {"risk_flags": flags, "total_flags": len(flags)}
        _risk_flags_cache.update(result)
        return json.dumps(result)

    @tool
    def retrieve_guidelines(query: str) -> str:
        """Retrieve evidence-based clinical guidelines from the RAG knowledge base.
        Input: natural language query describing risk factors to search for.
        Example: 'interventions for uninsured patient with previous no-shows and long lead time'
        You decide the query based on what you learned from predict_noshow and calculate_risk_flags.
        Returns: top relevant guideline excerpts with source document names."""
        try:
            from rag.retriever import retrieve

            # Build a deterministic query from risk flags and SHAP features
            # to ensure consistent RAG retrieval across runs
            flags = _risk_flags_cache.get("risk_flags", [])
            shap_features = _shap_cache.get("top_shap_features", [])
            risk_level = _shap_cache.get("risk_level", "UNKNOWN")

            parts = []
            if risk_level != "UNKNOWN":
                parts.append(f"{risk_level} risk patient")
            # Add top SHAP feature names for targeted retrieval
            for feat in shap_features[:3]:
                parts.append(feat.get("feature", "").replace("_", " "))
            # Add risk flags
            for flag in sorted(flags):  # sort for determinism
                parts.append(flag.replace("_", " "))

            final_query = ("interventions for " + ", ".join(parts)) if parts else query

            results = retrieve(final_query, k=5)
            _guidelines_cache["retrieved_guidelines"] = results
            return json.dumps({"retrieved_guidelines": results})

        except Exception as e:
            return json.dumps({"error": str(e), "retrieved_guidelines": []})

    @tool
    def analyze_risk_factors(reason: str) -> str:
        """Compile a structured clinical risk analysis summary from all gathered data.
        Input: brief reason (e.g. 'Compile risk analysis from SHAP and flags').
        Uses prediction results, SHAP features, risk flags, and patient demographics.
        Returns: formatted clinical risk summary that you can use for your final analysis."""

        shap_features = _shap_cache.get("top_shap_features", [])
        risk_score = _shap_cache.get("risk_score", 0)
        risk_level = _shap_cache.get("risk_level", "UNKNOWN")
        flags = _risk_flags_cache.get("risk_flags", [])

        parts = [
            f"PATIENT RISK PROFILE: {risk_score * 100:.1f}% probability ({risk_level} risk)",
            f"\nACTIVE RISK FLAGS: {', '.join(flags) if flags else 'None identified'}",
            "\nTOP SHAP CONTRIBUTING FACTORS:",
        ]
        for f in shap_features:
            direction = "INCREASES" if f["shap_impact"] > 0 else "DECREASES"
            parts.append(
                f"  • {f['feature']}: value={f['value']}, "
                f"impact={f['shap_impact']:+.4f} ({direction} risk)"
            )
        parts.extend([
            "\nPATIENT CONTEXT:",
            f"  • Age: {patient_data.get('age', 'N/A')}",
            f"  • Previous No-Shows: {patient_data.get('previous_no_shows', 0)}",
            f"  • Previous Appointments: {patient_data.get('previous_appointments', 0)}",
            f"  • Lead Time: {patient_data.get('lead_time', 0)} days",
            f"  • Distance: {patient_data.get('distance_km', 0)} km",
            f"  • Travel Time: {patient_data.get('travel_time_min', 0)} min",
            f"  • Insurance: {'Uninsured' if patient_data.get('is_uninsured', 0) == 1 else 'Insured'}",
            f"  • Reminders Sent: {patient_data.get('num_reminders', 0)}",
            f"  • SMS Reminder: {'Yes' if patient_data.get('sms_reminder', 0) == 1 else 'No'}",
            f"  • Chronic Conditions: {'Yes' if patient_data.get('has_chronic_condition', 0) == 1 else 'No'}",
        ])

        return "\n".join(parts)

    @tool
    def generate_intervention_plan(reason: str) -> str:
        """Generate intervention plan templates matched to retrieved guidelines and risk factors.
        Input: brief reason (e.g. 'Build intervention plan from guidelines').
        Uses the retrieved guidelines and risk flags to suggest structured interventions.
        Returns: guideline-matched intervention templates for each area.
        You should refine these templates into your final 3-step plan."""

        guidelines = _guidelines_cache.get("retrieved_guidelines", [])
        flags = _risk_flags_cache.get("risk_flags", [])

        # Categorize guidelines by intervention area
        reminder_g, outreach_g, barrier_g = [], [], []
        for g in guidelines:
            t = g["text"].lower()
            if any(kw in t for kw in ["reminder", "sms", "confirm", "message", "notification"]):
                reminder_g.append(g)
            if any(kw in t for kw in ["coordinator", "outreach", "pre-visit", "prep", "call", "phone"]):
                outreach_g.append(g)
            if any(kw in t for kw in ["transport", "insurance", "financial", "barrier",
                                       "distance", "telecon", "cost"]):
                barrier_g.append(g)

        template = {
            "step_1_template": {
                "focus": "Communication & Reminders",
                "matched_guidelines": [g["text"][:300] for g in reminder_g[:2]],
                "sources": [g["source"] for g in reminder_g[:2]],
                "active_flags": [f for f in flags if f in [
                    "no_reminder_sent", "young_adult", "elderly_patient", "long_lead_time"
                ]],
            },
            "step_2_template": {
                "focus": "Personal Outreach & Engagement",
                "matched_guidelines": [g["text"][:300] for g in outreach_g[:2]],
                "sources": [g["source"] for g in outreach_g[:2]],
                "active_flags": [f for f in flags if f in [
                    "high_risk_prior_noshow_3plus", "prior_noshow", "new_patient"
                ]],
            },
            "step_3_template": {
                "focus": "Barrier Removal & Alternatives",
                "matched_guidelines": [g["text"][:300] for g in barrier_g[:2]],
                "sources": [g["source"] for g in barrier_g[:2]],
                "active_flags": [f for f in flags if f in [
                    "uninsured", "long_distance", "high_travel_time", "unemployed"
                ]],
            },
            "note": (
                "Use these templates and matched guidelines to formulate your final "
                "3-step plan. Each step must have: action, timing, rationale "
                "(cite source documents), and responsible party."
            ),
        }
        return json.dumps(template)

    # ═══════════════════════════════════════════════════════════
    # BUILD MAIN AGENT — Llama 3.3 70B ReAct loop
    # ═══════════════════════════════════════════════════════════

    llm_main = ChatGroq(
        temperature=0.0,
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY_llama"),
    )

    main_tools = [
        predict_noshow,
        calculate_risk_flags,
        retrieve_guidelines,
        analyze_risk_factors,
        generate_intervention_plan,
    ]

    # create_react_agent builds a 2-node graph:
    #   Agent Node (LLM decides) ↔ Tool Node (executes)
    # The LLM loops until it gives a final answer.
    try:
        main_agent = create_react_agent(
            llm_main, main_tools,
            state_modifier=SystemMessage(content=MAIN_AGENT_SYSTEM),
        )
    except TypeError:
        # Fallback for older LangGraph versions
        main_agent = create_react_agent(
            llm_main, main_tools,
            prompt=SystemMessage(content=MAIN_AGENT_SYSTEM),
        )

    # ═══════════════════════════════════════════════════════════
    # BUILD CRITIC — Qwen QwQ 32B
    # ═══════════════════════════════════════════════════════════

    llm_critic = ChatGroq(
        temperature=0.0,
        model_name="qwen-qwq-32b",
        api_key=os.getenv("GROQ_API_KEY_Qwen"),
    )

    # ═══════════════════════════════════════════════════════════
    # ORCHESTRATOR LOOP
    # ═══════════════════════════════════════════════════════════

    max_retries = 2
    retry_count = 0
    critique = ""
    agent_trace = []
    best_output = None
    best_evaluation = None

    # Build concise patient summary for the initial message
    patient_summary = json.dumps({
        k: v for k, v in patient_data.items()
        if k in [
            "age", "distance_km", "travel_time_min", "lead_time",
            "previous_appointments", "previous_no_shows", "diabetes",
            "hypertension", "chronic_disease", "sms_reminder",
            "email_reminder", "num_reminders", "education_level",
            "rainy_day", "public_holiday", "is_uninsured", "is_unemployed",
            "got_reminder", "is_new_patient", "high_risk_patient",
            "has_chronic_condition", "multiple_chronic", "long_distance",
            "no_show_rate",
        ]
    }, indent=2)

    while retry_count <= max_retries:
        # ── Build message for the agent ──
        msg_content = (
            f"Patient data:\n{patient_summary}\n\n"
            f"Query: {user_query}\n\n"
            f"STEP 1 (MANDATORY): Call predict_noshow NOW to get the exact risk score. "
            f"Do NOT skip this step. Do NOT estimate the score yourself. "
            f"Then continue with the remaining tools to build the care coordination plan."
        )

        if critique:
            msg_content += (
                f"\n\n⚠️ PREVIOUS PLAN WAS REJECTED by the clinical quality reviewer.\n"
                f"Critique:\n{critique}\n\n"
                f"You MUST address this feedback. Call the relevant tools again "
                f"and revise your analysis and plan accordingly."
            )

        messages = [HumanMessage(content=msg_content)]

        # ── Run Main Agent (ReAct loop) ──
        try:
            result = main_agent.invoke({"messages": messages})
        except Exception as e:
            agent_trace.append({
                "tool": "main_agent_error",
                "error": str(e),
                "retry": retry_count,
                "agent": "main",
            })
            break

        # ── Extract tool trace (proves agent chose tools dynamically) ──
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    agent_trace.append({
                        "tool": tc["name"],
                        "retry": retry_count,
                        "agent": "main",
                    })

        # ── Get final answer from agent ──
        final_msg = ""
        if result.get("messages"):
            final_msg = result["messages"][-1].content

        try:
            agent_output = _extract_json(final_msg)
        except Exception:
            # If JSON parsing fails, construct from caches
            agent_output = {
                "risk_assessment": {
                    "score": _shap_cache.get("risk_score", 0),
                    "level": _shap_cache.get("risk_level", "UNKNOWN"),
                    "top_factors": _shap_cache.get("top_shap_features", []),
                },
                "risk_flags": _risk_flags_cache.get("risk_flags", []),
                "clinical_analysis": final_msg[:500] if final_msg else "Analysis unavailable",
                "evidence_base": _guidelines_cache.get("retrieved_guidelines", []),
                "intervention_plan": {},
                "parse_fallback": True,
            }

        best_output = agent_output

        # ── Check risk level — use tool output, NEVER trust LLM-generated risk level for routing ──
        risk_level = _shap_cache.get("risk_level", "UNKNOWN")

        if risk_level == "LOW":
            return _format_final_result(
                agent_output,
                {"verdict": "APPROVED", "note": "Low risk — evaluation skipped"},
                0, False, agent_trace,
            )

        # ── Run 3 Evaluation Checks (deterministic) ──
        intervention_plan = agent_output.get("intervention_plan", {})
        shap_features = (
            agent_output.get("risk_assessment", {}).get("top_factors")
            or _shap_cache.get("top_shap_features", [])
        )
        evidence = (
            agent_output.get("evidence_base")
            or _guidelines_cache.get("retrieved_guidelines", [])
        )

        check_1 = _check_clinical_accuracy(intervention_plan, shap_features)
        check_2 = _check_guideline_alignment(intervention_plan, evidence)
        check_3 = _check_ethical_soundness(intervention_plan, patient_data)

        agent_trace.extend([
            {"tool": "critic:check_clinical_accuracy", "result": check_1["result"],
             "retry": retry_count, "agent": "critic"},
            {"tool": "critic:check_guideline_alignment", "result": check_2["result"],
             "retry": retry_count, "agent": "critic"},
            {"tool": "critic:check_ethical_soundness", "result": check_3["result"],
             "retry": retry_count, "agent": "critic"},
        ])

        # ── Send check results to Qwen QwQ 32B for reasoning + verdict ──
        critic_prompt = (
            "You are a senior clinical quality reviewer. You have received the results of "
            "three evaluation checks on a care coordination plan.\n\n"
            f"PLAN UNDER REVIEW:\n{json.dumps(agent_output, indent=2, default=str)}\n\n"
            f"EVALUATION CHECK RESULTS:\n"
            f"1. Clinical Accuracy: {check_1['result']} — {check_1['detail']}\n"
            f"2. Guideline Alignment: {check_2['result']} — {check_2['detail']}\n"
            f"3. Ethical Soundness: {check_3['result']} — {check_3['detail']}\n\n"
            "Based on these results and your own expert judgment, provide your final verdict.\n"
            "Return ONLY valid JSON (no markdown, no code fences):\n"
            '{\n'
            f'  "clinical_accuracy": "{check_1["result"]}",\n'
            f'  "guideline_alignment": "{check_2["result"]}",\n'
            f'  "ethical_soundness": "{check_3["result"]}",\n'
            '  "justification": {\n'
            '    "clinical_accuracy": "one sentence justification",\n'
            '    "guideline_alignment": "one sentence justification",\n'
            '    "ethical_soundness": "one sentence justification"\n'
            '  },\n'
            '  "verdict": "APPROVED or NEEDS_REVISION",\n'
            '  "critique_for_retry": "specific critique if NEEDS_REVISION, empty if APPROVED"\n'
            '}'
        )

        try:
            critic_response = llm_critic.invoke([HumanMessage(content=critic_prompt)])
            evaluation = _extract_json(critic_response.content)
        except Exception as e:
            # Fallback: derive verdict from check results
            all_pass = all(c["result"] == "PASS" for c in [check_1, check_2, check_3])
            evaluation = {
                "clinical_accuracy": check_1["result"],
                "guideline_alignment": check_2["result"],
                "ethical_soundness": check_3["result"],
                "justification": {
                    "clinical_accuracy": check_1["detail"],
                    "guideline_alignment": check_2["detail"],
                    "ethical_soundness": check_3["detail"],
                },
                "verdict": "APPROVED" if all_pass else "NEEDS_REVISION",
                "critique_for_retry": (
                    "" if all_pass else
                    "Failed: " + ", ".join(
                        c["criterion"] for c in [check_1, check_2, check_3]
                        if c["result"] == "FAIL"
                    )
                ),
            }

        best_evaluation = evaluation
        verdict = evaluation.get("verdict", "APPROVED")

        if verdict == "APPROVED" or retry_count >= max_retries:
            return _format_final_result(
                agent_output, evaluation,
                retry_count,
                retry_count >= max_retries and verdict != "APPROVED",
                agent_trace,
            )

        # ── NEEDS_REVISION — inject critique and retry ──
        # Do NOT clear caches: the LLM already has previous tool results in
        # its message history and will likely not re-call predict_noshow.
        # Clearing would cause analyze_risk_factors to return 0% / UNKNOWN.
        critique = evaluation.get("critique_for_retry", "Please improve the plan.")
        retry_count += 1

    # Safety fallback — should not reach here
    return _format_final_result(
        best_output or {},
        best_evaluation or {"verdict": "FORCE_APPROVED"},
        retry_count, True, agent_trace,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT FORMATTER (Node 7 equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

def _format_final_result(
    agent_output: dict,
    evaluation: dict,
    retry_count: int,
    human_review: bool,
    agent_trace: list,
) -> dict:
    """Assemble the final structured JSON report — Node 7 (FORMATTER)."""

    risk_assessment = agent_output.get("risk_assessment", {})

    return {
        "patient_id": f"PT-{abs(hash(json.dumps(agent_output, default=str))) % 100000:05d}",
        "timestamp": datetime.now().isoformat(),
        "risk_assessment": {
            "score": risk_assessment.get("score", risk_assessment.get("risk_score", 0)),
            "level": risk_assessment.get("level", risk_assessment.get("risk_level", "UNKNOWN")),
            "top_factors": risk_assessment.get(
                "top_factors", risk_assessment.get("top_shap_features", [])
            ),
        },
        "risk_flags": agent_output.get("risk_flags", []),
        "clinical_analysis": agent_output.get("clinical_analysis", ""),
        "evidence_base": agent_output.get("evidence_base", []),
        "intervention_plan": agent_output.get("intervention_plan", {}),
        "quality_evaluation": {
            "verdict": evaluation.get("verdict", "N/A"),
            "clinical_accuracy": evaluation.get("clinical_accuracy", "N/A"),
            "guideline_alignment": evaluation.get("guideline_alignment", "N/A"),
            "ethical_soundness": evaluation.get("ethical_soundness", "N/A"),
            "justification": evaluation.get("justification", {}),
            "critique": evaluation.get("critique_for_retry", ""),
            "retry_count": retry_count,
        },
        "models_used": {
            "predictor": "XGBoost (Tuned)",
            "main_agent": "llama-3.3-70b-versatile (Groq)",
            "evaluator": "qwen-qwq-32b (Groq)",
        },
        "human_review_required": human_review,
        "agent_trace": agent_trace,
    }
