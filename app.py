import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Hospital No-Show Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #e65100;
        border: 2px solid #e65100;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'best_no_show_model.pkl' and 'scaler.pkl' are in the same directory.")
        st.stop()

# Feature engineering function (MUST match training!)
def create_features(input_df):
    """
    Create all engineered features - MUST match exactly what was used in training
    """
    df = input_df.copy()
    
    # 1. BEHAVIORAL FEATURES
    df['no_show_rate'] = df['previous_no_shows'] / (df['previous_appointments'] + 1)
    df['reliability_score'] = (df['previous_appointments'] - df['previous_no_shows']) / (df['previous_appointments'] + 1)
    df['is_new_patient'] = (df['previous_appointments'] == 0).astype(int)
    df['is_frequent_patient'] = (df['previous_appointments'] > 5).astype(int)
    df['has_previous_no_show'] = (df['previous_no_shows'] > 0).astype(int)
    
    # 2. DISTANCE & TRAVEL FEATURES
    df['distance_km_filled'] = df['distance_km'].fillna(df['distance_km'].median())
    df['travel_time_filled'] = df['travel_time_min'].fillna(df['travel_time_min'].median())
    df['is_far'] = (df['distance_km_filled'] > df['distance_km_filled'].median()).astype(int)
    df['is_very_far'] = (df['distance_km_filled'] > df['distance_km_filled'].quantile(0.75)).astype(int)
    df['long_travel'] = (df['travel_time_filled'] > df['travel_time_filled'].median()).astype(int)
    df['very_long_travel'] = (df['travel_time_filled'] > 60).astype(int)
    df['distance_time_ratio'] = df['distance_km_filled'] / (df['travel_time_filled'] + 1)
    
    # 3. WAITING TIME FEATURES
    df['long_wait'] = (df['waiting_days'] > 7).astype(int)
    df['very_long_wait'] = (df['waiting_days'] > 14).astype(int)
    df['short_wait'] = (df['waiting_days'] <= 3).astype(int)
    
    # 4. TIMING FEATURES
    df['is_weekend'] = df['appointment_day'].isin(['Sat', 'Sun']).astype(int)
    df['is_monday'] = (df['appointment_day'] == 'Mon').astype(int)
    df['is_friday'] = (df['appointment_day'] == 'Fri').astype(int)
    df['is_midweek'] = df['appointment_day'].isin(['Tue', 'Wed', 'Thu']).astype(int)
    df['is_morning'] = (df['appointment_time_slot'] == 'Morning').astype(int)
    df['is_afternoon'] = (df['appointment_time_slot'] == 'Afternoon').astype(int)
    df['is_evening'] = (df['appointment_time_slot'] == 'Evening').astype(int)
    
    # 5. HEALTH CONDITION FEATURES
    df['num_conditions'] = df['diabetes'] + df['hypertension'] + df['chronic_disease']
    df['has_any_condition'] = (df['num_conditions'] > 0).astype(int)
    df['multiple_conditions'] = (df['num_conditions'] > 1).astype(int)
    
    # 6. REMINDER FEATURES
    df['email_reminder_filled'] = df['email_reminder'].fillna(0)
    df['total_reminders'] = df['sms_reminder'] + df['email_reminder_filled']
    df['has_multiple_reminders'] = (df['num_reminders'] > 1).astype(int)
    df['no_reminder'] = (df['num_reminders'] == 0).astype(int)
    df['has_any_reminder'] = (df['num_reminders'] > 0).astype(int)
    
    # 7. AGE FEATURES
    df['age_filled'] = df['age'].fillna(df['age'].median())
    df['is_child'] = (df['age_filled'] < 18).astype(int)
    df['is_young_adult'] = ((df['age_filled'] >= 18) & (df['age_filled'] < 30)).astype(int)
    df['is_adult'] = ((df['age_filled'] >= 30) & (df['age_filled'] < 50)).astype(int)
    df['is_senior'] = ((df['age_filled'] >= 50) & (df['age_filled'] < 65)).astype(int)
    df['is_elderly'] = (df['age_filled'] >= 65).astype(int)
    
    # 8. SOCIOECONOMIC FEATURES
    df['is_uninsured'] = (df['insurance_status'] == 'Uninsured').astype(int)
    df['is_insured'] = (df['insurance_status'] == 'Insured').astype(int)
    df['is_employed'] = (df['employment_status'] == 'Employed').astype(int)
    df['is_unemployed'] = (df['employment_status'] == 'Unemployed').astype(int)
    df['is_student'] = (df['employment_status'] == 'Student').astype(int)
    
    # 9. RISK COMBINATIONS
    df['unemployed_uninsured'] = (df['is_unemployed'] & df['is_uninsured']).astype(int)
    df['far_and_uninsured'] = (df['is_far'] & df['is_uninsured']).astype(int)
    df['no_show_history_far'] = (df['has_previous_no_show'] & df['is_far']).astype(int)
    df['long_wait_far'] = (df['long_wait'] & df['is_far']).astype(int)
    df['rainy_and_far'] = (df['rainy_day'] & df['is_far']).astype(int)
    df['new_patient_uninsured'] = (df['is_new_patient'] & df['is_uninsured']).astype(int)
    
    # 10. RISK SCORE
    df['risk_score'] = (
        df['has_previous_no_show'] * 3 +
        df['is_uninsured'] * 2 +
        df['is_unemployed'] * 1 +
        df['long_wait'] * 1 +
        df['is_far'] * 1 +
        df['no_reminder'] * 1 +
        df['rainy_day'] * 1
    )
    
    # 11. WEATHER & HOLIDAY
    df['bad_timing'] = (df['rainy_day'] | df['public_holiday']).astype(int)
    df['holiday_weekend'] = (df['public_holiday'] & df['is_weekend']).astype(int)
    
    # 12. DEPARTMENT FEATURES
    df['is_emergency_dept'] = df['department'].isin(['Cardiology', 'General Medicine']).astype(int)
    df['is_specialty'] = df['department'].isin(['Dermatology', 'Orthopedics']).astype(int)
    
    return df

# Preprocessing function
def preprocess_input(df, scaler):
    """
    Preprocess input data to match training format
    """
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values
    df = df.fillna(df.median())
    
    # Scale features
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df_scaled

# Main app
def main():
    # Header
    st.title("🏥 Hospital Appointment No-Show Predictor")
    st.markdown("### AI-Powered Patient Attendance Prediction System")
    st.markdown("---")
    
    # Load model
    model, scaler = load_model()
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Select Mode:", 
                            ["Single Prediction", "Batch Prediction", "Analytics Dashboard"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this tool:**
    
    This AI model predicts the likelihood of patients missing their scheduled appointments. 
    
    **Accuracy:** 68-75%
    
    **Models Used:**
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Voting Ensemble
    """)
    
    # SINGLE PREDICTION MODE
    if page == "Single Prediction":
        st.header("🔍 Single Patient Prediction")
        st.markdown("Enter patient details to predict no-show probability")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Patient Information")
            age = st.number_input("Age", min_value=0, max_value=120, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            city_type = st.selectbox("City Type", ["Urban", "Suburban", "Rural"])
            
        with col2:
            st.subheader("📍 Location & Travel")
            distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            travel_time_min = st.number_input("Travel Time (min)", min_value=0, max_value=300, value=20)
            
        with col3:
            st.subheader("📅 Appointment Details")
            appointment_day = st.selectbox("Appointment Day", 
                                          ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            appointment_time_slot = st.selectbox("Time Slot", 
                                                 ["Morning", "Afternoon", "Evening"])
            department = st.selectbox("Department", 
                                     ["General Medicine", "Cardiology", "Orthopedics", 
                                      "Pediatrics", "Dermatology"])
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.subheader("⏰ Scheduling")
            waiting_days = st.number_input("Waiting Days", min_value=0, max_value=90, value=7)
            
        with col5:
            st.subheader("📊 Patient History")
            previous_appointments = st.number_input("Previous Appointments", min_value=0, max_value=50, value=2)
            previous_no_shows = st.number_input("Previous No-Shows", min_value=0, 
                                               max_value=previous_appointments, value=0)
        
        with col6:
            st.subheader("🏥 Health Status")
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
            chronic_disease = st.checkbox("Other Chronic Disease")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            st.subheader("📱 Reminders")
            sms_reminder = st.checkbox("SMS Reminder Sent")
            email_reminder = st.checkbox("Email Reminder Sent")
            num_reminders = st.number_input("Total Reminders Sent", min_value=0, max_value=5, 
                                           value=int(sms_reminder) + int(email_reminder))
        
        with col8:
            st.subheader("💼 Socioeconomic")
            employment_status = st.selectbox("Employment Status", 
                                            ["Employed", "Unemployed", "Student", ""])
            education_level = st.selectbox("Education Level", 
                                          ["Primary", "Secondary", "Higher", ""])
            insurance_status = st.selectbox("Insurance Status", ["Insured", "Uninsured"])
        
        with col9:
            st.subheader("🌤️ External Factors")
            rainy_day = st.checkbox("Rainy Day Forecast")
            public_holiday = st.checkbox("Public Holiday")
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict No-Show Probability", type="primary", use_container_width=True):
            # Create input dataframe
            input_data = {
                'age': [age],
                'gender': [gender],
                'city_type': [city_type],
                'distance_km': [distance_km],
                'travel_time_min': [travel_time_min],
                'appointment_day': [appointment_day],
                'appointment_time_slot': [appointment_time_slot],
                'department': [department],
                'waiting_days': [waiting_days],
                'previous_appointments': [previous_appointments],
                'previous_no_shows': [previous_no_shows],
                'diabetes': [int(diabetes)],
                'hypertension': [int(hypertension)],
                'chronic_disease': [int(chronic_disease)],
                'sms_reminder': [int(sms_reminder)],
                'email_reminder': [int(email_reminder)],
                'num_reminders': [num_reminders],
                'employment_status': [employment_status],
                'education_level': [education_level],
                'insurance_status': [insurance_status],
                'rainy_day': [int(rainy_day)],
                'public_holiday': [int(public_holiday)]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Create features
            input_df = create_features(input_df)
            
            # Preprocess
            input_processed = preprocess_input(input_df, scaler)
            
            # Predict
            prediction_proba = model.predict_proba(input_processed)[0]
            no_show_probability = prediction_proba[1] * 100
            show_probability = prediction_proba[0] * 100
            
            # Determine risk level
            if no_show_probability >= 70:
                risk_level = "HIGH RISK"
                risk_class = "high-risk"
                risk_emoji = "🔴"
                recommendation = "**Action Required:** Contact patient immediately. Send additional reminders. Consider follow-up call."
            elif no_show_probability >= 50:
                risk_level = "MEDIUM RISK"
                risk_class = "medium-risk"
                risk_emoji = "🟡"
                recommendation = "**Suggested Action:** Send extra reminder 24 hours before appointment. Monitor closely."
            else:
                risk_level = "LOW RISK"
                risk_class = "low-risk"
                risk_emoji = "🟢"
                recommendation = "**Status:** Patient likely to attend. Standard procedure applies."
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            # Main prediction box
            st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    {risk_emoji} {risk_level}<br>
                    No-Show Probability: {no_show_probability:.1f}%
                </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            col_left, col_right = st.columns(2)
            
            with col_left:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=no_show_probability,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "No-Show Probability"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if no_show_probability >= 70 else "orange" if no_show_probability >= 50 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                # Probability breakdown
                st.markdown("#### Probability Breakdown")
                st.metric("Show Probability", f"{show_probability:.1f}%", 
                         delta=None, delta_color="normal")
                st.metric("No-Show Probability", f"{no_show_probability:.1f}%", 
                         delta=None, delta_color="inverse")
                
                # Risk factors
                st.markdown("#### Key Risk Factors")
                risk_factors = []
                if previous_no_shows > 0:
                    risk_factors.append(f"• Previous no-shows: {previous_no_shows}")
                if insurance_status == "Uninsured":
                    risk_factors.append("• Uninsured patient")
                if waiting_days > 10:
                    risk_factors.append(f"• Long wait time: {waiting_days} days")
                if distance_km > 10:
                    risk_factors.append(f"• Far distance: {distance_km} km")
                if num_reminders == 0:
                    risk_factors.append("• No reminders sent")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.markdown("• No major risk factors detected")
            
            # Recommendation
            st.markdown("---")
            st.markdown("### 💡 Recommended Action")
            st.info(recommendation)
    
    # BATCH PREDICTION MODE
    elif page == "Batch Prediction":
        st.header("📊 Batch Prediction")
        st.markdown("Upload a CSV file with multiple patient records for bulk prediction")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} records found.")
            
            # Show first few rows
            with st.expander("📋 Preview Data (First 5 rows)"):
                st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Create features
                    df_features = create_features(df.copy())
                    
                    # Drop patient_id if exists and target column
                    cols_to_drop = ['patient_id', 'no_show'] if 'patient_id' in df_features.columns else ['no_show'] if 'no_show' in df_features.columns else []
                    if cols_to_drop:
                        X = df_features.drop(cols_to_drop, axis=1)
                    else:
                        X = df_features
                    
                    # Preprocess
                    X_processed = preprocess_input(X, scaler)
                    
                    # Predict
                    predictions = model.predict(X_processed)
                    probabilities = model.predict_proba(X_processed)[:, 1] * 100
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['No_Show_Probability'] = probabilities
                    results_df['Predicted_No_Show'] = predictions
                    results_df['Risk_Level'] = pd.cut(probabilities, 
                                                      bins=[0, 50, 70, 100],
                                                      labels=['Low', 'Medium', 'High'])
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(results_df))
                with col2:
                    high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                    st.metric("High Risk", high_risk, 
                             delta=f"{high_risk/len(results_df)*100:.1f}%")
                with col3:
                    medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium'])
                    st.metric("Medium Risk", medium_risk,
                             delta=f"{medium_risk/len(results_df)*100:.1f}%")
                with col4:
                    low_risk = len(results_df[results_df['Risk_Level'] == 'Low'])
                    st.metric("Low Risk", low_risk,
                             delta=f"{low_risk/len(results_df)*100:.1f}%")
                
                # Risk distribution chart
                st.markdown("#### Risk Distribution")
                fig = px.histogram(results_df, x='No_Show_Probability', 
                                  nbins=20,
                                  title='Distribution of No-Show Probabilities',
                                  labels={'No_Show_Probability': 'No-Show Probability (%)'})
                fig.add_vline(x=50, line_dash="dash", line_color="orange", 
                             annotation_text="Medium Risk Threshold")
                fig.add_vline(x=70, line_dash="dash", line_color="red", 
                             annotation_text="High Risk Threshold")
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("#### Detailed Results")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    risk_filter = st.multiselect("Filter by Risk Level", 
                                                ['Low', 'Medium', 'High'],
                                                default=['High'])
                with col2:
                    sort_by = st.selectbox("Sort by", 
                                          ['No_Show_Probability', 'patient_id', 'age'])
                
                # Filter and sort
                filtered_df = results_df[results_df['Risk_Level'].isin(risk_filter)]
                filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
                
                # Display
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name=f"no_show_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # ANALYTICS DASHBOARD
    else:
        st.header("📈 Analytics Dashboard")
        st.markdown("Upload data to view insights and statistics")
        
        uploaded_file = st.file_uploader("Upload CSV file with historical data", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Basic stats
            st.markdown("### 📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Appointments", len(df))
            with col2:
                if 'no_show' in df.columns:
                    no_shows = df['no_show'].sum()
                    st.metric("No-Shows", no_shows, 
                             delta=f"{no_shows/len(df)*100:.1f}%")
            with col3:
                if 'no_show' in df.columns:
                    shows = len(df) - df['no_show'].sum()
                    st.metric("Shows", shows,
                             delta=f"{shows/len(df)*100:.1f}%")
            with col4:
                st.metric("Features", df.shape[1])
            
            # Visualizations
            if 'no_show' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # No-show by day
                    st.markdown("#### No-Show Rate by Day of Week")
                    day_stats = df.groupby('appointment_day')['no_show'].mean() * 100
                    fig = px.bar(x=day_stats.index, y=day_stats.values,
                                labels={'x': 'Day', 'y': 'No-Show Rate (%)'},
                                title='Average No-Show Rate by Day')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # No-show by department
                    st.markdown("#### No-Show Rate by Department")
                    dept_stats = df.groupby('department')['no_show'].mean() * 100
                    fig = px.bar(x=dept_stats.index, y=dept_stats.values,
                                labels={'x': 'Department', 'y': 'No-Show Rate (%)'},
                                title='Average No-Show Rate by Department')
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age distribution
                    st.markdown("#### Age Distribution")
                    fig = px.histogram(df, x='age', color='no_show',
                                      title='Age Distribution by Show/No-Show',
                                      labels={'no_show': 'No-Show', 'age': 'Age'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Insurance status
                    st.markdown("#### No-Show Rate by Insurance Status")
                    ins_stats = df.groupby('insurance_status')['no_show'].mean() * 100
                    fig = px.pie(values=ins_stats.values, names=ins_stats.index,
                                title='No-Show Distribution by Insurance')
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()