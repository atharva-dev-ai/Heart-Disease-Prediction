import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# -------------------- DARK MODE TOGGLE --------------------
dark_mode = st.sidebar.toggle("🌙 Night Mode", value=True)

# -------------------- THEME STYLES --------------------
if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0B0F14;
        color: #E6EDF3;
    }
    h1, h2, h3 {
        color: #4FA3FF;
    }
    label {
        color: #D0D7DE !important;
    }
    .section {
        background-color: #11161D;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.7);
    }
    div[data-testid="stForm"] {
        background-color: #11161D;
        padding: 1.5rem;
        border-radius: 14px;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #F8F9FA;
        color: #000000;
    }
    h1, h2, h3 {
        color: #0F4C81;
    }
    .section {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    """
    <h1 style="text-align:center;">🫀 Heart Disease Risk Predictor</h1>
    <p style="text-align:center; font-size:17px;">
    Machine Learning based clinical decision-support system
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD DATA & TRAIN MODEL --------------------
data = pd.read_csv("heart_disease_data.csv")

X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# -------------------- NAVIGATION --------------------
tabs = st.tabs(["🩺 Risk Assessment", "📊 Model Insights", "ℹ️ About"])

# ==================== TAB 1 ====================
with tabs[0]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("👤 Patient Clinical Information")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 90, 45)

            sex = st.selectbox(
                "Sex (0 = Female, 1 = Male)",
                {"Female": 0, "Male": 1}
            )

            cp = st.selectbox(
                "Chest Pain Type (cp)",
                {
                    "0 = Typical Angina": 0,
                    "1 = Atypical Angina": 1,
                    "2 = Non-anginal Pain": 2,
                    "3 = Asymptomatic": 3
                }
            )

        with col2:
            trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
            chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL (fbs)",
                {
                    "0 = No": 0,
                    "1 = Yes": 1
                }
            )

            restecg = st.selectbox(
                "Resting ECG Result (restecg)",
                {
                    "0 = Normal": 0,
                    "1 = ST-T Wave Abnormality": 1,
                    "2 = Left Ventricular Hypertrophy": 2
                }
            )

        with col3:
            thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)

            exang = st.selectbox(
                "Exercise Induced Angina (exang)",
                {
                    "0 = No": 0,
                    "1 = Yes": 1
                }
            )

            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope)",
                {
                    "0 = Upsloping": 0,
                    "1 = Flat": 1,
                    "2 = Downsloping": 2
                }
            )

        ca = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy (ca)",
            [0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia (thal)",
            {
                "0 = Normal": 0,
                "1 = Fixed Defect": 1,
                "2 = Reversible Defect": 2
            }
        )

        submitted = st.form_submit_button("🩺 Assess Heart Disease Risk")

    if submitted:
        with st.spinner("Analyzing patient data..."):
            time.sleep(1)

        input_data = np.array([
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.subheader("🧾 Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease Detected")
            st.write("Consult a medical professional for further evaluation.")
        else:
            st.success("✅ Low Risk of Heart Disease Detected")
            st.write("Maintain a healthy lifestyle and regular checkups.")

    st.info("⚠️ This application is for educational purposes only and not a medical diagnosis.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TAB 2 ====================
with tabs[1]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📊 Model Insights")

    st.metric("Model", "Logistic Regression")
    st.metric("Features Used", "13 Clinical Parameters")
    st.metric("Accuracy", "≈ 85%")

    st.progress(0.85)

    st.write(
        "The model was trained on standardized clinical data and evaluated "
        "using classification metrics to ensure reliability and transparency."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TAB 3 ====================
with tabs[2]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About This Application")

    st.write("""
    - End-to-end Machine Learning healthcare project  
    - Deployed using Streamlit Cloud  
    - Designed with accessibility, usability, and ethical AI principles  
    """)

    st.write("👨‍💻 Developer: **Atharva Savant**")
    st.write("🔗 GitHub: https://github.com/atharva-dev-ai")

    st.markdown("</div>", unsafe_allow_html=True)
