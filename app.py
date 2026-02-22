import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital No-Show Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Dark theme with white/cyan accents, NO purple ─────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #111827 50%, #0a0a0a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1a1a2e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #94a3b8 !important;
    }

    /* Hero Header */
    .hero-header {
        padding: 2rem 2.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 40%, #14b8a6 100%);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(14, 165, 233, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { transform: translate(-10%, -10%); }
        50% { transform: translate(10%, 10%); }
    }
    .hero-header h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0;
        position: relative;
        font-weight: 300;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .metric-card:hover {
        border-color: rgba(14, 165, 233, 0.5);
        box-shadow: 0 8px 32px rgba(14, 165, 233, 0.15);
        transform: translateY(-2px);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Risk Result Cards */
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .risk-description {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(14, 165, 233, 0.3);
        margin: 2rem 0 1rem;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* Sidebar nav buttons */
    .nav-button {
        display: block;
        width: 100%;
        padding: 0.85rem 1.2rem;
        margin-bottom: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
    }
    .nav-button:hover {
        background: rgba(14, 165, 233, 0.15);
        border-color: rgba(14, 165, 233, 0.4);
    }
    .nav-active {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(20, 184, 166, 0.2)) !important;
        border-color: rgba(14, 165, 233, 0.5) !important;
        color: #0ea5e9 !important;
        font-weight: 600;
    }

    /* Model comparison best badge */
    .best-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Override Streamlit form submit button */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
    }
    .stFormSubmitButton > button:hover {
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.5) !important;
        transform: translateY(-1px);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(20, 184, 166, 0.2));
        color: #0ea5e9;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Input styling */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Models & Data ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {
        "XGBoost (Tuned)": joblib.load("models/xgboost_model.pkl"),
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
        "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    }
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    metrics = joblib.load("models/metrics.pkl")
    return models, scaler, feature_columns, metrics


@st.cache_data
def load_dataset():
    df = pd.read_csv("Data/hospital_appointment_no_show_50000.csv")
    return df


models, scaler, feature_columns, metrics = load_models()
df = load_dataset()
primary_model = models["XGBoost (Tuned)"]

# ─── Feature column list for reference ────────────────────────────────────────
feature_cols_list = list(feature_columns)


# ─── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Navigation")
    st.markdown("---")

    page = st.radio(
        "Go to",
        ["🔮 Prediction", "📊 Model Performance", "💡 Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#475569; font-size:0.8rem; margin-top:1rem;'>
            <p style='color:#94a3b8;font-weight:600;'>Clinical No-Show Predictor</p>
            <p>Milestone 1 — ML System</p>
            <p style='margin-top:0.5rem;'>Built with Streamlit & Scikit-learn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Prediction":
    st.markdown(
        """
        <div class="hero-header">
            <h1>🔮 Appointment No-Show Prediction</h1>
            <p>Enter patient details to predict the probability of missing a scheduled appointment</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📋 Patient Information</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            city_type = st.selectbox("City Type", ["Urban", "Suburban", "Other"])
        with col2:
            distance_km = st.number_input("Distance from Hospital (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
            travel_time_min = st.number_input("Travel Time (min)", min_value=0.0, max_value=300.0, value=30.0, step=5.0)
            department = st.selectbox("Department", ["Cardiology", "Dermatology", "General", "Orthopedics", "Pediatrics"])
        with col3:
            lead_time = st.number_input("Lead Time / Waiting Days", min_value=0, max_value=60, value=5, step=1)
            previous_appointments = st.number_input("Previous Appointments", min_value=0, max_value=30, value=3, step=1)
            previous_no_shows = st.number_input("Previous No-Shows", min_value=0, max_value=20, value=0, step=1)

        st.markdown('<div class="section-header">🩺 Medical & Contact Details</div>', unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        with col4:
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            chronic_disease = st.selectbox("Chronic Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with col5:
            sms_reminder = st.selectbox("SMS Reminder Sent", [0, 1], format_func=lambda x: "Yes" if x else "No")
            email_reminder = st.selectbox("Email Reminder Sent", [0, 1], format_func=lambda x: "Yes" if x else "No")
            num_reminders = st.number_input("Total Reminders", min_value=0, max_value=5, value=1, step=1)
        with col6:
            employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Other"])
            education_level = st.selectbox("Education Level", ["Primary", "Secondary", "Higher"])
            insurance_status = st.selectbox("Insurance Status", ["Insured", "Uninsured"])

        st.markdown('<div class="section-header">📅 Appointment Details</div>', unsafe_allow_html=True)

        col7, col8, col9 = st.columns(3)
        with col7:
            appointment_day = st.selectbox("Appointment Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        with col8:
            appointment_time_slot = st.selectbox("Time Slot", ["Morning", "Afternoon", "Evening"])
        with col9:
            rainy_day = st.selectbox("Rainy Day", [0, 1], format_func=lambda x: "Yes" if x else "No")
            public_holiday = st.selectbox("Public Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🔍 Predict No-Show Risk", use_container_width=True)

    if submitted:
        # Build feature vector matching the training columns (55 features)
        education_map = {"Primary": 1, "Secondary": 2, "Higher": 3}

        # ── Derived / engineered features (matching notebook exactly) ──
        no_show_rate = previous_no_shows / previous_appointments if previous_appointments > 0 else 0
        is_new_patient = 1 if previous_appointments == 0 else 0
        high_risk_patient = 1 if previous_no_shows >= 2 else 0
        travel_burden = distance_km * travel_time_min
        long_distance = 1 if distance_km > 15 else 0
        high_travel_time = 1 if travel_time_min > 45 else 0
        long_lead_time = 1 if lead_time > 21 else 0
        short_lead_time = 1 if lead_time <= 3 else 0
        same_day = 1 if lead_time == 0 else 0
        is_weekend = 1 if appointment_day in ["Saturday", "Sunday"] else 0
        is_elderly = 1 if age >= 65 else 0
        is_young_adult = 1 if 18 <= age <= 30 else 0
        has_chronic_condition = 1 if (diabetes == 1 or hypertension == 1 or chronic_disease == 1) else 0
        multiple_chronic = 1 if (diabetes + hypertension + chronic_disease) >= 2 else 0
        got_reminder = 1 if (sms_reminder == 1 or email_reminder == 1) else 0
        multiple_reminders = 1 if num_reminders >= 2 else 0
        is_uninsured = 1 if insurance_status == "Uninsured" else 0
        is_unemployed = 1 if employment_status == "Unemployed" else 0
        risk_distance = high_risk_patient * long_distance
        uninsured_distance = is_uninsured * long_distance
        young_long_wait = is_young_adult * long_lead_time
        rain_distance = rainy_day * long_distance

        input_data = {
            # ── Raw features ──
            "age": age,
            "distance_km": distance_km,
            "travel_time_min": travel_time_min,
            "lead_time": lead_time,
            "previous_appointments": previous_appointments,
            "previous_no_shows": previous_no_shows,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "chronic_disease": chronic_disease,
            "sms_reminder": sms_reminder,
            "email_reminder": email_reminder,
            "num_reminders": num_reminders,
            "education_level": education_map[education_level],
            "rainy_day": rainy_day,
            "public_holiday": public_holiday,
            # ── Engineered features ──
            "no_show_rate": no_show_rate,
            "is_new_patient": is_new_patient,
            "high_risk_patient": high_risk_patient,
            "travel_burden": travel_burden,
            "long_distance": long_distance,
            "high_travel_time": high_travel_time,
            "long_lead_time": long_lead_time,
            "short_lead_time": short_lead_time,
            "same_day": same_day,
            "is_weekend": is_weekend,
            "is_elderly": is_elderly,
            "is_young_adult": is_young_adult,
            "has_chronic_condition": has_chronic_condition,
            "multiple_chronic": multiple_chronic,
            "got_reminder": got_reminder,
            "multiple_reminders": multiple_reminders,
            "is_uninsured": is_uninsured,
            "is_unemployed": is_unemployed,
            "risk_distance": risk_distance,
            "uninsured_distance": uninsured_distance,
            "young_long_wait": young_long_wait,
            "rain_distance": rain_distance,
            # ── One-hot encoded columns ──
            "gender_Male": 1 if gender == "Male" else 0,
            "city_type_Suburban": 1 if city_type == "Suburban" else 0,
            "city_type_Urban": 1 if city_type == "Urban" else 0,
            "appointment_day_Monday": 1 if appointment_day == "Monday" else 0,
            "appointment_day_Saturday": 1 if appointment_day == "Saturday" else 0,
            "appointment_day_Sunday": 1 if appointment_day == "Sunday" else 0,
            "appointment_day_Thursday": 1 if appointment_day == "Thursday" else 0,
            "appointment_day_Tuesday": 1 if appointment_day == "Tuesday" else 0,
            "appointment_day_Wednesday": 1 if appointment_day == "Wednesday" else 0,
            "appointment_time_slot_Evening": 1 if appointment_time_slot == "Evening" else 0,
            "appointment_time_slot_Morning": 1 if appointment_time_slot == "Morning" else 0,
            "department_Dermatology": 1 if department == "Dermatology" else 0,
            "department_General": 1 if department == "General" else 0,
            "department_Orthopedics": 1 if department == "Orthopedics" else 0,
            "department_Pediatrics": 1 if department == "Pediatrics" else 0,
            "employment_status_Other": 1 if employment_status == "Other" else 0,
            "employment_status_Student": 1 if employment_status == "Student" else 0,
            "employment_status_Unemployed": 1 if employment_status == "Unemployed" else 0,
            "insurance_status_Uninsured": 1 if insurance_status == "Uninsured" else 0,
        }

        input_df = pd.DataFrame([input_data])
        # Ensure columns match training order
        input_df = input_df.reindex(columns=feature_cols_list, fill_value=0)

        # Predict using XGBoost (tree-based, no scaling needed)
        prob = primary_model.predict_proba(input_df)[0][1]
        risk_pct = prob * 100

        # Determine risk category
        if risk_pct < 40:
            risk_cat = "Low Risk"
            risk_color = "#10b981"
            risk_class = "risk-low"
            risk_emoji = "✅"
            risk_desc = "Patient is likely to attend the appointment."
        elif risk_pct < 70:
            risk_cat = "Medium Risk"
            risk_color = "#f59e0b"
            risk_class = "risk-medium"
            risk_emoji = "⚠️"
            risk_desc = "Patient may miss the appointment. Consider sending a reminder."
        else:
            risk_cat = "High Risk"
            risk_color = "#ef4444"
            risk_class = "risk-high"
            risk_emoji = "🚨"
            risk_desc = "Patient is very likely to miss the appointment. Immediate intervention recommended."

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

        # Gauge Chart
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                number={"suffix": "%", "font": {"size": 48, "color": "white"}},
                title={"text": "No-Show Risk Score", "font": {"size": 18, "color": "#94a3b8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#334155",
                             "tickfont": {"color": "#94a3b8"}},
                    "bar": {"color": risk_color, "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16, 185, 129, 0.15)"},
                        {"range": [40, 70], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": risk_pct,
                    },
                },
            ))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=320,
                margin=dict(l=30, r=30, t=60, b=30),
            )
            st.plotly_chart(gauge, use_container_width=True)

        with res_col2:
            st.markdown(
                f"""
                <div class="{risk_class}" style="margin-top:1rem;">
                    <p style="font-size:3rem; margin:0;">{risk_emoji}</p>
                    <p class="risk-label" style="color:{risk_color};">{risk_cat}</p>
                    <p class="risk-description">{risk_desc}</p>
                    <p style="margin-top:1rem; font-size:2rem; font-weight:800; color:{risk_color};">{risk_pct:.1f}%</p>
                    <p style="color:#64748b;font-size:0.8rem;">Probability of No-Show</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Top contributing factors
        st.markdown('<div class="section-header">🔍 Top Contributing Factors</div>', unsafe_allow_html=True)

        if hasattr(primary_model, "feature_importances_"):
            importances = primary_model.feature_importances_
            feat_imp = pd.DataFrame({
                "Feature": feature_cols_list,
                "Importance": importances,
            }).sort_values("Importance", ascending=False).head(10)

            # Map to user-friendly names
            friendly_names = {
                "age": "Age", "distance_km": "Distance (km)", "travel_time_min": "Travel Time",
                "lead_time": "Lead Time (Days)", "previous_appointments": "Prev. Appointments",
                "previous_no_shows": "Prev. No-Shows", "diabetes": "Diabetes",
                "hypertension": "Hypertension", "chronic_disease": "Chronic Disease",
                "sms_reminder": "SMS Reminder", "email_reminder": "Email Reminder",
                "num_reminders": "Total Reminders", "education_level": "Education Level",
                "rainy_day": "Rainy Day", "public_holiday": "Public Holiday",
                "no_show_rate": "No-Show Rate", "is_new_patient": "New Patient",
                "high_risk_patient": "High-Risk Patient", "travel_burden": "Travel Burden",
                "long_distance": "Long Distance", "high_travel_time": "High Travel Time",
                "long_lead_time": "Long Lead Time", "short_lead_time": "Short Lead Time",
                "same_day": "Same-Day Appt", "is_weekend": "Weekend Appt",
                "is_elderly": "Elderly Patient", "is_young_adult": "Young Adult",
                "has_chronic_condition": "Has Chronic Condition", "multiple_chronic": "Multiple Chronic",
                "got_reminder": "Got Reminder", "multiple_reminders": "Multiple Reminders",
                "is_uninsured": "Uninsured", "is_unemployed": "Unemployed",
                "risk_distance": "Risk × Distance", "uninsured_distance": "Uninsured × Distance",
                "young_long_wait": "Young × Long Wait", "rain_distance": "Rain × Distance",
                "gender_Male": "Gender: Male",
                "city_type_Suburban": "City: Suburban", "city_type_Urban": "City: Urban",
                "appointment_day_Monday": "Day: Monday", "appointment_day_Saturday": "Day: Saturday",
                "appointment_day_Sunday": "Day: Sunday", "appointment_day_Thursday": "Day: Thursday",
                "appointment_day_Tuesday": "Day: Tuesday", "appointment_day_Wednesday": "Day: Wednesday",
                "appointment_time_slot_Evening": "Time: Evening",
                "appointment_time_slot_Morning": "Time: Morning",
                "department_Dermatology": "Dept: Dermatology", "department_General": "Dept: General",
                "department_Orthopedics": "Dept: Orthopedics", "department_Pediatrics": "Dept: Pediatrics",
                "employment_status_Other": "Employment: Other",
                "employment_status_Student": "Employment: Student",
                "employment_status_Unemployed": "Employment: Unemployed",
                "insurance_status_Uninsured": "Insurance: Uninsured",
            }
            feat_imp["Feature"] = feat_imp["Feature"].map(lambda x: friendly_names.get(x, x))

            fig_factors = px.bar(
                feat_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#0ea5e9", "#14b8a6"],
            )
            fig_factors.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Importance Score"),
                yaxis=dict(title="", autorange="reversed"),
                coloraxis_showscale=False,
                height=400,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_factors, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown(
        """
        <div class="hero-header">
            <h1>📊 Model Performance Dashboard</h1>
            <p>Compare evaluation metrics across all trained classification models</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metrics summary cards ──
    best_model_name = max(metrics, key=lambda k: metrics[k].get("F1", 0))
    best = metrics[best_model_name]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{best['Accuracy']*100:.1f}%</p>
                <p class="metric-label">Best Accuracy</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{best['Precision']*100:.1f}%</p>
                <p class="metric-label">Best Precision</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{best['Recall']*100:.1f}%</p>
                <p class="metric-label">Best Recall</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{best['F1']*100:.1f}%</p>
                <p class="metric-label">Best F1 Score</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="text-align:center;margin:1rem 0;">
            <span class="best-badge">🏆 Best Model: {best_model_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Comparison Table ──
    st.markdown('<div class="section-header">📋 Model Comparison Table</div>', unsafe_allow_html=True)

    table_data = []
    for name, m in metrics.items():
        table_data.append({
            "Model": name,
            "Accuracy": f"{m['Accuracy']*100:.2f}%",
            "Precision": f"{m['Precision']*100:.2f}%",
            "Recall": f"{m['Recall']*100:.2f}%",
            "F1 Score": f"{m['F1']*100:.2f}%",
            "ROC-AUC": f"{m['ROC-AUC']:.4f}",
        })

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )

    # ── Bar Chart Comparison ──
    st.markdown('<div class="section-header">📊 Metric Comparison Charts</div>', unsafe_allow_html=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    model_names = list(metrics.keys())
    colors = ["#0ea5e9", "#14b8a6", "#06b6d4", "#22d3ee"]

    fig_comparison = make_subplots(
        rows=1, cols=5,
        subplot_titles=metric_names,
        horizontal_spacing=0.05,
    )

    for idx, metric_name in enumerate(metric_names):
        values = [metrics[m][metric_name] for m in model_names]
        fig_comparison.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=10),
                showlegend=False,
            ),
            row=1, col=idx + 1,
        )

    fig_comparison.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=11),
        height=400,
        margin=dict(l=10, r=10, t=40, b=80),
    )
    fig_comparison.update_xaxes(
        tickangle=-45,
        gridcolor="rgba(255,255,255,0.03)",
        tickfont=dict(size=9),
    )
    fig_comparison.update_yaxes(
        range=[0, 1.15],
        gridcolor="rgba(255,255,255,0.05)",
    )
    for annotation in fig_comparison['layout']['annotations']:
        annotation['font'] = dict(color='#94a3b8', size=13)

    st.plotly_chart(fig_comparison, use_container_width=True)

    # ── Confusion Matrices ──
    st.markdown('<div class="section-header">🔢 Confusion Matrices</div>', unsafe_allow_html=True)

    cm_cols = st.columns(len(metrics))
    cm_colors = [
        [[0, "rgba(14,165,233,0.05)"], [1, "rgba(14,165,233,0.7)"]],
        [[0, "rgba(20,184,166,0.05)"], [1, "rgba(20,184,166,0.7)"]],
        [[0, "rgba(6,182,212,0.05)"], [1, "rgba(6,182,212,0.7)"]],
        [[0, "rgba(34,211,238,0.05)"], [1, "rgba(34,211,238,0.7)"]],
    ]

    for i, (name, m) in enumerate(metrics.items()):
        with cm_cols[i]:
            cm = np.array(m["Confusion Matrix"])
            labels = ["Show", "No-Show"]

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                colorscale=cm_colors[i % len(cm_colors)],
                showscale=False,
            ))
            fig_cm.update_layout(
                title=dict(text=name, font=dict(size=13, color="#e2e8f0"), x=0.5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                xaxis=dict(title="Predicted", side="bottom"),
                yaxis=dict(title="Actual", autorange="reversed"),
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Radar Chart ──
    st.markdown('<div class="section-header">🎯 Model Radar Comparison</div>', unsafe_allow_html=True)

    radar_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig_radar = go.Figure()
    radar_colors = ["#0ea5e9", "#14b8a6", "#06b6d4", "#22d3ee"]

    for i, (name, m) in enumerate(metrics.items()):
        values = [m[metric] for metric in radar_metrics]
        values.append(values[0])  # close the polygon

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            name=name,
            line=dict(color=radar_colors[i % len(radar_colors)]),
            fillcolor=radar_colors[i % len(radar_colors)].replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in radar_colors[i % len(radar_colors)] else None,
            opacity=0.8,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#64748b"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#94a3b8", size=12),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        ),
        height=450,
        margin=dict(l=60, r=60, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.markdown(
        """
        <div class="hero-header">
            <h1>💡 Data Insights & Analysis</h1>
            <p>Explore patterns and trends in the hospital appointment dataset</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary Stats ──
    total = len(df)
    no_show_count = df["no_show"].sum()
    show_count = total - no_show_count
    no_show_rate = no_show_count / total * 100

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{total:,}</p>
                <p class="metric-label">Total Records</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{show_count:,}</p>
                <p class="metric-label">Showed Up</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{no_show_count:,}</p>
                <p class="metric-label">No-Shows</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{no_show_rate:.1f}%</p>
                <p class="metric-label">No-Show Rate</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Feature Importance ──
    st.markdown('<div class="section-header">🏆 Feature Importance (XGBoost)</div>', unsafe_allow_html=True)

    if hasattr(primary_model, "feature_importances_"):
        importances = primary_model.feature_importances_
        friendly_names = {
            "age": "Age", "distance_km": "Distance (km)", "travel_time_min": "Travel Time",
            "lead_time": "Lead Time (Days)", "previous_appointments": "Prev. Appointments",
            "previous_no_shows": "Prev. No-Shows", "diabetes": "Diabetes",
            "hypertension": "Hypertension", "chronic_disease": "Chronic Disease",
            "sms_reminder": "SMS Reminder", "email_reminder": "Email Reminder",
            "num_reminders": "Total Reminders", "education_level": "Education Level",
            "rainy_day": "Rainy Day", "public_holiday": "Public Holiday",
            "no_show_rate": "No-Show Rate", "is_new_patient": "New Patient",
            "high_risk_patient": "High-Risk Patient", "travel_burden": "Travel Burden",
            "long_distance": "Long Distance", "high_travel_time": "High Travel Time",
            "long_lead_time": "Long Lead Time", "short_lead_time": "Short Lead Time",
            "same_day": "Same-Day Appt", "is_weekend": "Weekend Appt",
            "is_elderly": "Elderly Patient", "is_young_adult": "Young Adult",
            "has_chronic_condition": "Has Chronic Condition", "multiple_chronic": "Multiple Chronic",
            "got_reminder": "Got Reminder", "multiple_reminders": "Multiple Reminders",
            "is_uninsured": "Uninsured", "is_unemployed": "Unemployed",
            "risk_distance": "Risk × Distance", "uninsured_distance": "Uninsured × Distance",
            "young_long_wait": "Young × Long Wait", "rain_distance": "Rain × Distance",
            "gender_Male": "Gender: Male",
            "city_type_Suburban": "City: Suburban", "city_type_Urban": "City: Urban",
            "appointment_day_Monday": "Day: Monday", "appointment_day_Saturday": "Day: Saturday",
            "appointment_day_Sunday": "Day: Sunday", "appointment_day_Thursday": "Day: Thursday",
            "appointment_day_Tuesday": "Day: Tuesday", "appointment_day_Wednesday": "Day: Wednesday",
            "appointment_time_slot_Evening": "Time: Evening",
            "appointment_time_slot_Morning": "Time: Morning",
            "department_Dermatology": "Dept: Dermatology", "department_General": "Dept: General",
            "department_Orthopedics": "Dept: Orthopedics", "department_Pediatrics": "Dept: Pediatrics",
            "employment_status_Other": "Employment: Other",
            "employment_status_Student": "Employment: Student",
            "employment_status_Unemployed": "Employment: Unemployed",
            "insurance_status_Uninsured": "Insurance: Uninsured",
        }
        feat_imp_full = pd.DataFrame({
            "Feature": [friendly_names.get(c, c) for c in feature_cols_list],
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            feat_imp_full.tail(15),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#0ea5e9", "#14b8a6"],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Importance Score"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── No-Show by Department ──
    st.markdown('<div class="section-header">🏥 No-Show Rate by Department</div>', unsafe_allow_html=True)

    dept_stats = df.groupby("department")["no_show"].mean().sort_values(ascending=False).reset_index()
    dept_stats.columns = ["Department", "No-Show Rate"]
    dept_stats["No-Show Rate"] = dept_stats["No-Show Rate"] * 100

    fig_dept = px.bar(
        dept_stats,
        x="Department",
        y="No-Show Rate",
        color="No-Show Rate",
        color_continuous_scale=["#14b8a6", "#0ea5e9"],
        text=dept_stats["No-Show Rate"].apply(lambda x: f"{x:.1f}%"),
    )
    fig_dept.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
    fig_dept.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="No-Show Rate (%)"),
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_dept, use_container_width=True)

    # ── Two-column: Day & Time Slot ──
    i_col1, i_col2 = st.columns(2)

    with i_col1:
        st.markdown('<div class="section-header">📅 No-Show Rate by Day</div>', unsafe_allow_html=True)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_stats = df.groupby("appointment_day")["no_show"].mean().reindex(day_order).reset_index()
        day_stats.columns = ["Day", "No-Show Rate"]
        day_stats["No-Show Rate"] = day_stats["No-Show Rate"] * 100

        fig_day = px.line(
            day_stats,
            x="Day",
            y="No-Show Rate",
            markers=True,
            color_discrete_sequence=["#0ea5e9"],
        )
        fig_day.update_traces(
            line=dict(width=3),
            marker=dict(size=10, color="#14b8a6", line=dict(width=2, color="#0ea5e9")),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.1)",
        )
        fig_day.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="No-Show Rate (%)"),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_day, use_container_width=True)

    with i_col2:
        st.markdown('<div class="section-header">🕐 No-Show Rate by Time Slot</div>', unsafe_allow_html=True)
        time_stats = df.groupby("appointment_time_slot")["no_show"].mean().reset_index()
        time_stats.columns = ["Time Slot", "No-Show Rate"]
        time_stats["No-Show Rate"] = time_stats["No-Show Rate"] * 100

        fig_time = px.bar(
            time_stats,
            x="Time Slot",
            y="No-Show Rate",
            color="Time Slot",
            color_discrete_sequence=["#0ea5e9", "#14b8a6", "#06b6d4"],
            text=time_stats["No-Show Rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_time.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
        fig_time.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="No-Show Rate (%)"),
            showlegend=False,
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Age Distribution ──
    st.markdown('<div class="section-header">👤 Age Distribution by Show/No-Show</div>', unsafe_allow_html=True)

    fig_age = go.Figure()
    fig_age.add_trace(go.Histogram(
        x=df[df["no_show"] == 0]["age"],
        name="Showed Up",
        marker_color="rgba(14, 165, 233, 0.6)",
        opacity=0.7,
    ))
    fig_age.add_trace(go.Histogram(
        x=df[df["no_show"] == 1]["age"],
        name="No-Show",
        marker_color="rgba(239, 68, 68, 0.6)",
        opacity=0.7,
    ))
    fig_age.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="Age", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # ── SMS Reminder Impact ──
    st.markdown('<div class="section-header">📱 Impact of SMS Reminders</div>', unsafe_allow_html=True)

    sms_col1, sms_col2 = st.columns(2)

    with sms_col1:
        sms_stats = df.groupby("sms_reminder")["no_show"].mean().reset_index()
        sms_stats.columns = ["SMS Reminder", "No-Show Rate"]
        sms_stats["SMS Reminder"] = sms_stats["SMS Reminder"].map({0: "No SMS", 1: "SMS Sent"})
        sms_stats["No-Show Rate"] = sms_stats["No-Show Rate"] * 100

        fig_sms = px.bar(
            sms_stats,
            x="SMS Reminder",
            y="No-Show Rate",
            color="SMS Reminder",
            color_discrete_sequence=["#ef4444", "#10b981"],
            text=sms_stats["No-Show Rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_sms.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
        fig_sms.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="No-Show Rate (%)"),
            showlegend=False,
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sms, use_container_width=True)

    with sms_col2:
        # Insurance impact
        ins_stats = df.groupby("insurance_status")["no_show"].mean().reset_index()
        ins_stats.columns = ["Insurance", "No-Show Rate"]
        ins_stats["No-Show Rate"] = ins_stats["No-Show Rate"] * 100

        fig_ins = px.bar(
            ins_stats,
            x="Insurance",
            y="No-Show Rate",
            color="Insurance",
            color_discrete_sequence=["#0ea5e9", "#f59e0b"],
            text=ins_stats["No-Show Rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_ins.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
        fig_ins.update_layout(
            title=dict(text="Insurance Status Impact", font=dict(color="#94a3b8", size=14), x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="No-Show Rate (%)"),
            showlegend=False,
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_ins, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown('<div class="section-header">🔗 Feature Correlation Heatmap</div>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0, "#0ea5e9"], [0.5, "#111827"], [1, "#14b8a6"]],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9, "color": "#e2e8f0"},
        showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8")),
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=10),
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Key Findings ──
    st.markdown('<div class="section-header">📌 Key Findings</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="padding:1rem;">
                    <h4 style="color:#0ea5e9; margin:0 0 0.5rem;">🔬 Model Insights</h4>
                    <ul style="color:#94a3b8; margin:0; padding-left:1.2rem;">
                        <li>XGBoost achieves the best overall performance with highest F1 score</li>
                        <li>Random Forest provides strong generalization</li>
                        <li>Logistic Regression serves as a solid interpretable baseline</li>
                        <li>Class balancing techniques improve recall for no-show detection</li>
                    </ul>
                </div>
                <div style="padding:1rem;">
                    <h4 style="color:#14b8a6; margin:0 0 0.5rem;">📊 Data Insights</h4>
                    <ul style="color:#94a3b8; margin:0; padding-left:1.2rem;">
                        <li>Previous no-show history is a strong predictor</li>
                        <li>Lead time (waiting days) significantly impacts attendance</li>
                        <li>Distance and travel time influence no-show probability</li>
                        <li>SMS reminders show measurable impact on attendance rates</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        <p>🏥 Hospital Appointment No-Show Prediction System — Milestone 1 | Built with Streamlit, Scikit-learn & Plotly</p>
    </div>
    """,
    unsafe_allow_html=True,
)