# Clinical Appointment No-Show Prediction & Care Coordination Agent

A comprehensive healthcare operations system that predicts the likelihood of patient appointment no-shows and proactively generates evidence-based intervention plans. Built with **Scikit-learn**, **XGBoost**, **Streamlit**, and an advanced **ReAct Agentic Workflow** powered by **Groq**, **Llama 3.3 70B**, **Qwen QwQ 32B**, and **ChromaDB**.

---

## 🚀 Live Demo

Deployed on **Streamlit Community Cloud**: [Open App](https://gen-ai-capstone.streamlit.app)

---

## 📊 Features & UI Dashboard

| Page | What It Does |
|------|-------------|
| **Prediction** | Enter patient demographics, medical history, and appointment details → get a real-time no-show risk percentage (via an interactive gauge chart), a risk category determination, and see the top contributing factors affecting the prediction. |
| **Model Performance** | Compare classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost) using metrics tables, bar charts, confusion matrices, and radar charts. |
| **Insights** | Visualizations for feature importance, no-show rates by department/day/time, age distributions, SMS & insurance impact, and correlation heatmaps. |
| **Care Coordination Agent** | An AI agentic pipeline designed to synthesize XGBoost output with real guidelines. Produces risk analysis, RAG-retrieved evidence, structured 3-step intervention plans, and a quality evaluation. Uses a dynamic execution trace! |

---

## 🧠 System Architecture

The project represents a full integration of Traditional Machine Learning and Generative AI (Agentic Workflow). 

### 1. Predictive ML Layer
Trained on 50,000 hospital appointment records, the system generates over 20 engineered features.
* **XGBoost Classifier**: Serves as the primary predictive model, outputting a precise risk score alongside SHAP explanations.
* **Exploratory Data Analysis**: Missing value imputation, one-hot encoding, and derived insights.

### 2. Generative AI "Care Coordination Agent"
We implement a highly customized **ReAct (Reasoning and Acting) Agent Flow** built with LangChain and LangGraph context:
* **The Main Agent (Llama 3.3 70B via Groq)**: Decides dynamically which custom tools to call. Tools include predicting no-shows, calculating risk flags, querying the RAG database, analyzing risk factors, and formulating an intervention plan.
* **Knowledge Retrieval (ChromaDB)**: A dedicated RAG module uses `sentence-transformers/all-MiniLM-L6-v2` to retrieve relevant healthcare protocols.
* **The Critic (Qwen QwQ 32B via Groq)**: Automatically evaluates the generated plan based on clinical accuracy, guideline alignment, and ethical soundness. Returns an `APPROVED` or `NEEDS_REVISION` verdict.

---

## 📁 Project Structure

```
Gen_Ai_Capstone/
│
├── app.py                      # Main Streamlit dashboard
├── agent_ui.py                 # Renders the 7-section ReAct agent UI in Streamlit
├── agent/
│   └── graph.py                # ReAct Agent pipeline (Llama + Qwen Critic Workflow)
│
├── rag/                        # RAG framework built with Chroma
│   ├── retriever.py            # Sentence-transformer document retrieval logic
│   └── create_db.py            # Utility to build the vector database
│
├── Data/
│   └── hospital_appointment_no_show_50000.csv  # Base dataset (50k records)
│
├── models/                     # Pickled serializations for all ML models
│   ├── xgboost_model.pkl       # Highly tuned XGBoost primary model
│   ├── random_forest_model.pkl
│   ├── decision_tree_model.pkl
│   ├── logistic_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl     # Used to match feature vector shapes natively
│   └── metrics.pkl
│
├── noteBooks/
│   └── CLINICAL_APPOINTMENT_NO_SHOW_PREDICTION.ipynb # Feature engineering and training setup
│
├── requirements.txt            # Python dependencies
└── runtime.txt                 # Specifies Python 3.11 for Streamlit Cloud
```

---

## 🛠 Tech Stack

- **Python 3.11**
- **Scikit-learn** — Logistic Regression, Decision Tree, Random Forest
- **XGBoost** — Primary classification model
- **Streamlit & Plotly** — Interactive frontend, charts, & custom agent dashboards
- **LangChain / LangGraph** — Orchestrates the Agent workflow
- **Groq API** — Super-fast inference for `llama-3.3-70b-versatile` & `qwen-qwq-32b`
- **ChromaDB & Sentence-Transformers** — Lightweight vector database for RAG

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mrstrange1708/Gen_Ai_Capstone.git
cd Gen_Ai_Capstone
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Environment Variables

To run the Agent logic, you'll need a Groq API key. Create a `.env` file in the project root:
```ini
GROQ_API_KEY_llama=your_groq_api_key_here
GROQ_API_KEY_Qwen=your_groq_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 67.5% | 89.0% | 66.0% | 75.8% | 76.4% |
| Decision Tree | 65.1% | 87.3% | 64.2% | 74.0% | 71.5% |
| Random Forest | 71.8% | 86.8% | 74.9% | 80.4% | 75.5% |
| **XGBoost (Tuned)** | **78.4%** | **80.2%** | **95.8%** | **87.3%** | **76.2%** |

> XGBoost (Tuned) maintains the strongest performance when balancing precision and recall.

---

## 👥 Team

- **Shaik Junaid Sami**
- **Siraparapu ManiKanta**
- **Srikar Balanagu**

---

## 📄 License

This project was developed as part of an academic capstone.
