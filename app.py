import streamlit as st
import joblib as jb

st.set_page_config(
    page_title="Hospital No-Show Prediction",
    layout="wide",
)

st.markdown(
    """
    <style>
        .header{
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(90deg, #1f77b4, #17becf);
            color: white;
            font-weight: bold;
            text-align: center;
            font-size: 28px;
        }


        div.stButton > button {
            height: 55px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 10px;
        }
    </style>

    """
, unsafe_allow_html=True)

model = jb.load("models/model.pkl")
scaler = jb.load("models/scaler.pkl")


if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")
if st.sidebar.button("Home",use_container_width=True):
    st.session_state.page = "Home"
if st.sidebar.button("Predict",use_container_width=True):
    st.session_state.page = "Predict"

page = st.session_state.page

if page == "Home":
    st.title("Hospital No-Show Prediction")
    st.markdown("Predict whether a patient will show up for their appointment based on various features.")

elif page == "Predict":
    st.title("Make a Prediction")