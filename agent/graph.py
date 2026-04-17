import os
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class AgentState(TypedDict):
    patient_data: dict
    risk_score: float
    risk_level: str
    retrieved_docs: List[str]
    final_output: dict

def analyze_risk(state: AgentState):
    risk_score = state.get("risk_score", 0)
    risk_pct = risk_score * 100
    if risk_pct < 40:
        risk_level = "Low Risk"
    elif risk_pct < 70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    return {"risk_level": risk_level}

def retrieve_guidelines(state: AgentState):
    risk_level = state.get("risk_level", "Low Risk")
    patient_data = state.get("patient_data", {})
    
    if risk_level == "Low Risk":
        return {"retrieved_docs": []}
    
    try:
        # Init chroma and embedder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_dir = os.path.join(base_dir, "rag", "chroma_db")
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
        
        # Formulate query
        query = f"no-show intervention guidelines for {risk_level} patient"
        if patient_data.get("previous_no_shows", 0) > 0:
            query += " previous no shows"
        if patient_data.get("lead_time", 0) > 15:
            query += " long lead time"
            
        docs = db.similarity_search(query, k=3)
        retrieved_docs = [doc.page_content for doc in docs]
    except Exception as e:
        print(f"Retrieval error: {e}")
        retrieved_docs = []
        
    return {"retrieved_docs": retrieved_docs}

def generate_recommendation(state: AgentState):
    risk_score = state.get("risk_score", 0)
    risk_level = state.get("risk_level", "Low Risk")
    patient_data = state.get("patient_data", {})
    retrieved_docs = state.get("retrieved_docs", [])
    
    docs_text = "\n".join(f"- {doc}" for doc in retrieved_docs)
    
    prompt = f"""
You are an expert healthcare care coordination assistant. 
Analyze the patient data and risk of no-show, review the provided best practice guidelines, and recommend specific interventions.

Patient Data summary:
- Age: {patient_data.get('age', 'Unknown')}
- Previous No Shows: {patient_data.get('previous_no_shows', 'Unknown')}
- Lead time to appointment: {patient_data.get('lead_time', 'Unknown')} days
- Travel Distance: {patient_data.get('distance_km', 'Unknown')} km
- Total Reminders Given: {patient_data.get('num_reminders', 'Unknown')}
- Risk Score: {risk_score*100:.1f}% ({risk_level})

Retrieved Hospital Guidelines:
{docs_text if docs_text else "None required (Low Risk). Standard SMS reminder is sufficient."}

Return ONLY a valid JSON object in the exact format shown below, with no markdown formatting or extra text. Only JSON. Do not include ```json tags.
{{
  "Risk_Level": "{risk_level}",
  "Key_Factors": ["factor 1", "factor 2"],
  "Recommended_Actions": ["Action 1", "Action 2"],
  "Confidence_Score": "XX%"
}}
"""
    # use groq
    llm = ChatGroq(temperature=0.0, model_name="llama-3.1-8b-instant")
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        
        # Clean markdown if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        parsed = json.loads(text.strip())
        return {"final_output": parsed}
    except Exception as e:
        print(f"Error calling LLM or parsing JSON: {e}")
        print(f"Raw Output: {text if 'text' in locals() else 'None'}")
        
        # Fallback output
        return {"final_output": {
            "Risk_Level": risk_level,
            "Key_Factors": ["High lead time or previous history" if risk_score > 0.5 else "Standard risk profile"],
            "Recommended_Actions": ["Send standard appointment reminder via SMS.", "Monitor appointment attendance."],
            "Confidence_Score": "None (LLM Failed)"
        }}

def evaluate_recommendation(state: AgentState):
    risk_level = state.get("risk_level", "Low Risk")
    patient_data = state.get("patient_data", {})
    retrieved_docs = state.get("retrieved_docs", [])
    initial_output = state.get("final_output", {})
    
    docs_text = "\n".join(f"- {doc}" for doc in retrieved_docs)
    initial_output_str = json.dumps(initial_output, indent=2)
    
    prompt = f"""
You are an expert Clinical Safety and Workflow Validator. 
Your job is to strictly evaluate the AI-generated recommendation below.

Patient Data summary:
- Age: {patient_data.get('age', 'Unknown')}
- Previous No Shows: {patient_data.get('previous_no_shows', 'Unknown')}
- Lead time to appointment: {patient_data.get('lead_time', 'Unknown')} days
- Travel Distance: {patient_data.get('distance_km', 'Unknown')} km

Retrieved Hospital Guidelines:
{docs_text if docs_text else "None."}

Initial AI Recommendation:
{initial_output_str}

You MUST check for:
1. Logical consistency with patient data.
2. Clinical safety (no harmful suggestions).
3. Alignment with retrieved guidelines (no hallucinations).
4. Missing critical actions.
5. Overconfidence or vague advice.

Return ONLY a valid JSON object in the exact format shown below, with no markdown formatting or extra text. Do not include ```json tags.
{{
  "is_valid": true,
  "issues": ["list any issues found, or empty list if none"],
  "improved_recommendation": {{
     "Risk_Level": "...",
     "Key_Factors": ["..."],
     "Recommended_Actions": ["..."]
  }},
  "confidence_score": 0.95,
  "evaluator_notes": "Detailed critical feedback."
}}
"""
    try:
        # User requested Qwen 32B. Using Groq's qwen integration.
        llm = ChatGroq(temperature=0.0, model_name="qwen-2.5-32b")
        try:
            response = llm.invoke(prompt)
        except Exception:
            # Fallback if qwen-2.5-32b is offline or name is slightly different
            llm = ChatGroq(temperature=0.0, model_name="llama-3.1-8b-instant")
            response = llm.invoke(prompt)
            
        text = response.content.strip()
        
        # Clean markdown if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        parsed = json.loads(text.strip())
        return {"final_output": parsed} 
    except Exception as e:
        print(f"Error in Evaluator: {e}")
        return {"final_output": initial_output}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_risk", analyze_risk)
    workflow.add_node("retrieve_guidelines", retrieve_guidelines)
    workflow.add_node("generate_recommendation", generate_recommendation)
    workflow.add_node("evaluate_recommendation", evaluate_recommendation)
    
    workflow.set_entry_point("analyze_risk")
    workflow.add_edge("analyze_risk", "retrieve_guidelines")
    workflow.add_edge("retrieve_guidelines", "generate_recommendation")
    workflow.add_edge("generate_recommendation", "evaluate_recommendation")
    workflow.add_edge("evaluate_recommendation", END)
    
    return workflow.compile()
