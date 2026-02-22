# Clinical Appointment No-Show Prediction

A machine learning–based healthcare operations system that predicts the likelihood of patient appointment no-shows using historical scheduling data. Built with **Scikit-learn**, **XGBoost**, and **Streamlit**.

> **Milestone 1** — Traditional ML system (no GenAI/LLMs).

---

## Live Demo

Deployed on **Streamlit Community Cloud**:  
🔗 [Open App](https://gen-ai-capstone.streamlit.app)

---

## Features

| Page | What It Does |
|------|-------------|
| **Prediction** | Enter patient details → get no-show risk percentage via gauge chart + risk category (Low / Medium / High) + top contributing factors |
| **Model Performance** | Compare 4 models (Logistic Regression, Decision Tree, Random Forest, XGBoost) with metrics table, bar charts, confusion matrices, radar chart |
| **Insights** | Feature importance, no-show rates by department/day/time, age distributions, SMS & insurance impact, correlation heatmap |

---

## Project Structure

```
Gen_Ai_Capstone/
│
├── app.py                      # Streamlit dashboard (main application)
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version pin for Streamlit Cloud
│
├── Data/
│   └── hospital_appointment_no_show_50000.csv   # Dataset (50,000 records)
│
├── models/
│   ├── xgboost_model.pkl       # Final tuned XGBoost model (primary)
│   ├── random_forest_model.pkl # Random Forest model (comparison)
│   ├── decision_tree_model.pkl # Decision Tree model (comparison)
│   ├── logistic_model.pkl      # Logistic Regression model (comparison)
│   ├── scaler.pkl              # StandardScaler (used for Logistic Regression)
│   ├── feature_columns.pkl     # Feature column names after encoding (55 features)
│   └── metrics.pkl             # Evaluation metrics for all models
│
└── noteBooks/
    └── CLINICAL_APPOINTMENT_NO_SHOW_PREDICTION.ipynb  # Model training notebook
```

### File Details

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app with 3 pages — Prediction, Model Performance, Insights. Loads trained models, processes patient input through feature engineering (55 features), and renders interactive Plotly visualizations. |
| `requirements.txt` | All Python dependencies: pandas, numpy, scikit-learn, xgboost, streamlit, plotly, matplotlib, seaborn, joblib. |
| `runtime.txt` | Pins Python 3.11 for Streamlit Community Cloud deployment. |
| `Data/hospital_appointment_no_show_50000.csv` | Historical hospital appointment dataset with 23 raw columns — age, gender, city type, distance, travel time, department, waiting days, previous appointments/no-shows, medical conditions, reminders, employment, education, insurance, weather, and holiday flags. Target: `no_show` (0=Show, 1=No-Show). |
| `noteBooks/CLINICAL_APPOINTMENT_NO_SHOW_PREDICTION.ipynb` | Full ML pipeline — EDA, preprocessing, feature engineering (22 derived features), model training (Logistic Regression, Decision Tree, Random Forest, XGBoost), hyperparameter tuning via GridSearchCV, threshold optimization, and model export. |
| `models/*.pkl` | Serialized models, scaler, feature columns, and evaluation metrics saved via `joblib`. |

---

## Tech Stack

- **Python 3.11**
- **Scikit-learn** — Logistic Regression, Decision Tree, Random Forest
- **XGBoost** — Final tuned model (best F1 score)
- **Streamlit** — Interactive web dashboard
- **Plotly** — Gauge charts, bar charts, heatmaps, radar charts
- **Joblib** — Model serialization
- **Pandas / NumPy** — Data processing

---

## Setup & Installation

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

### 4. Run the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Model Training (Notebook)

The training pipeline is in `noteBooks/CLINICAL_APPOINTMENT_NO_SHOW_PREDICTION.ipynb`:

1. **Data Loading** — 50,000 hospital appointment records
2. **EDA** — Distributions, boxplots, correlation matrices
3. **Preprocessing** — Missing value imputation, one-hot encoding for categorical features, ordinal encoding for education level
4. **Feature Engineering** — 22 derived features:
   - `no_show_rate`, `is_new_patient`, `high_risk_patient`
   - `travel_burden`, `long_distance`, `high_travel_time`
   - `long_lead_time`, `short_lead_time`, `same_day`, `is_weekend`
   - `is_elderly`, `is_young_adult`, `has_chronic_condition`, `multiple_chronic`
   - `got_reminder`, `multiple_reminders`, `is_uninsured`, `is_unemployed`
   - `risk_distance`, `uninsured_distance`, `young_long_wait`, `rain_distance`
5. **Model Training** — Logistic Regression, Decision Tree, Random Forest, XGBoost
6. **Hyperparameter Tuning** — GridSearchCV on XGBoost
7. **Threshold Optimization** — Precision-Recall curve for optimal F1
8. **Export** — Models, scaler, features, and metrics saved to `models/`

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 67.5% | 89.0% | 66.0% | 75.8% | 0.764 |
| Decision Tree | 65.1% | 87.3% | 64.2% | 74.0% | 0.715 |
| Random Forest | 71.8% | 86.8% | 74.9% | 80.4% | 0.755 |
| **XGBoost (Tuned)** | **78.4%** | **80.2%** | **95.8%** | **87.3%** | **0.762** |

> XGBoost (Tuned) is the **primary model** used for predictions.

---

## Deployment

### Streamlit Community Cloud

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Set **Branch** to your deployment branch
6. Click **Deploy**

The `runtime.txt` ensures Python 3.11 is used. The `requirements.txt` handles all dependencies automatically.

---

## Team

- **Shaik Junaid Sami , Abhinay and Srikar**

---

## License

This project is developed as part of an academic capstone project.
