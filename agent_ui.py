import streamlit as st

def render_agent_pipeline(input_data, prob):
    """
    Executes the LangGraph Agentic workflow and renders its outputs using Streamlit.
    """
    # ---- AGENTIC CARE COORDINATION ----
    with st.spinner("Agentic AI is analyzing risk and retrieving best-practice interventions..."):
        from agent.graph import build_graph
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        app_graph = build_graph()
        agent_input = {
            "patient_data": input_data,
            "risk_score": float(prob)
        }
        agent_result = app_graph.invoke(agent_input)
        final_output = agent_result.get("final_output", {})

    # ---- RENDERING UI ----
    st.markdown('<div class="section-header">Agentic Care Coordination (AI Assistant)</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #94a3b8; font-size: 0.9rem;">
    The AI assistant has autonomously analyzed the patient risk profile against hospital best practices (RAG) to recommend the following interventions.
    </p>
    """, unsafe_allow_html=True)
    
    # Display Evaluator metadata
    is_valid = final_output.get("is_valid", True)
    eval_color = "#10b981" if is_valid else "#f59e0b"
    val_status = "Approved & Verified" if is_valid else "Critiqued & Refined"
    
    st.markdown(f"""
    <div style="padding:0.75rem; border-radius:8px; border: 1px solid {eval_color}; margin-bottom: 1rem; background: rgba(255,255,255,0.03);">
        <p style="color:{eval_color}; margin:0; font-weight:600;">Qwen Evaluator Status: {val_status}</p>
        <p style="color:#d1d5db; font-size:0.9rem; margin-top:0.3rem;">{final_output.get("evaluator_notes", "No notes provided.")}</p>
    </div>
    """, unsafe_allow_html=True)

    improved = final_output.get("improved_recommendation", final_output) # fallback to old struct if missing

    agent_col1, agent_col2 = st.columns(2)
    with agent_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#38bdf8;'>Key Risk Factors Identified</h4>", unsafe_allow_html=True)
        for factor in improved.get("Key_Factors", []):
            st.markdown(f"- {factor}")
            
        conf = final_output.get('confidence_score', improved.get('Confidence_Score', 'N/A'))
        st.markdown(f"<p style='margin-top:1rem; color:#94a3b8;'><strong>AI Confidence:</strong> {conf}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with agent_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#10b981;'>Recommended Actions</h4>", unsafe_allow_html=True)
        for i, action in enumerate(improved.get("Recommended_Actions", []), 1):
            st.markdown(f"**{i}.** {action}")
            
        issues = final_output.get("issues", [])
        if issues and isinstance(issues, list) and len(issues) > 0 and issues[0] and issues[0] != "list any issues found, or empty list if none":
            st.markdown("<h5 style='color:#f43f5e; margin-top:1rem;'>Remediated Issues:</h5>", unsafe_allow_html=True)
            for iss in issues:
                st.markdown(f"<span style='color:#fb7185; font-size:0.85rem;'>- {iss}</span>", unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
