import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="wide"
)

# --------------------------------------------------
# SAFE Load model & scaler (NO CRASH)
# --------------------------------------------------
@st.cache_resource
def load_artifacts_safe():
    if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
        return None, None
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_artifacts_safe()

# --------------------------------------------------
# HARD STOP if model not available
# --------------------------------------------------
if model is None or scaler is None:
    st.error("⚠️ Model files not found or failed to load.")
    st.info("The prediction system is temporarily unavailable. Please contact the developer.")
    st.stop()

# --------------------------------------------------
# Session state for reports
# --------------------------------------------------
if "report" not in st.session_state:
    st.session_state.report = None

# --------------------------------------------------
# Custom CSS (glass + dark medical UI)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

h1 {
    color: #60a5fa;
    font-weight: 800;
}

.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    color: #9ca3af;
}

.stTabs [aria-selected="true"] {
    color: #ef4444;
    border-bottom: 2px solid #ef4444;
}

label {
    font-weight: 600;
    color: #e5e7eb !important;
}

.stButton > button {
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 25px;
}

.glass {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 25px;
    margin-top: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.result-high {
    background-color: #450a0a;
    padding: 18px;
    border-radius: 12px;
    color: #fecaca;
    font-size: 18px;
    font-weight: 700;
}

.result-low {
    background-color: #052e16;
    padding: 18px;
    border-radius: 12px;
    color: #bbf7d0;
    font-size: 18px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    "<h1>❤️ Heart Disease Risk Predictor</h1>"
    "<p>ML-based clinical decision support system</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tabs = st.tabs(
    ["🩺 Risk Assessment", "📄 Reports", "📊 Model Insights", "ℹ️ About"]
)

# ==================================================
# TAB 0: RISK ASSESSMENT
# ==================================================
with tabs[0]:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    patient_name = st.text_input("Patient Name")

    age = st.slider("Age (age)", 29, 77, 45)
    sex = st.selectbox("Sex (sex)", ["Female", "Male"])
    sex_val = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        ["0 = Typical Angina", "1 = Atypical Angina", "2 = Non-anginal Pain", "3 = Asymptomatic"]
    )
    cp = int(cp.split("=")[0])

    trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])

    if st.button("Assess Risk"):

        input_data = np.array([[
            age, sex_val, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]])

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        st.session_state.report = {
            "name": patient_name,
            "age": age,
            "sex": sex,
            "prediction": prediction,
            "probability": probability,
            "time": datetime.now().strftime("%d %b %Y • %I:%M %p")
        }

        if prediction == 1:
            st.markdown(
                f"<div class='result-high'>⚠️ High Risk of Heart Disease ({probability:.2f}%)</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-low'>✅ Low Risk of Heart Disease ({probability:.2f}%)</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 1: REPORTS
# ==================================================
with tabs[1]:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📄 Risk Assessment Report")

    if st.session_state.report is None:
        st.info("No report generated yet.")
    else:
        r = st.session_state.report

        st.markdown(f"""
        **Patient Name:** {r["name"] or "N/A"}  
        **Age:** {r["age"]}  
        **Sex:** {r["sex"]}  
        **Generated On:** {r["time"]}  

        **Prediction:** {"High Risk" if r["prediction"] == 1 else "Low Risk"}  
        **Risk Probability:** {r["probability"]:.2f}%
        """)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 2: MODEL INSIGHTS
# ==================================================
with tabs[2]:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📊 Model Insights")

    st.write("""
    - **Algorithm:** Logistic Regression  
    - **Preprocessing:** StandardScaler  
    - **Prediction Type:** Binary Classification  
    - **Output:** Probability-based risk score  
    - **Dataset:** UCI Heart Disease Dataset
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 3: ABOUT
# ==================================================
with tabs[3]:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("ℹ️ About This Project")

    st.write("""
    **Project Highlights**
    - End-to-end Machine Learning healthcare system  
    - Probability-based heart disease risk prediction  
    - Explainable AI with feature importance visualization  
    - Automated PDF medical report generation  
    - Premium medical-grade UI/UX using Streamlit  

    ⚠️ *This application is for educational purposes only and should not be used as a medical diagnosis.*
    """)

    st.markdown("---")

    st.subheader("👨‍💻 Developer Information")

    st.write("""
    **Name:** Atharva Savant  
    **Email:** atharvasavant2506@gmail.com  
    **GitHub:** https://github.com/atharva-dev-ai  

    Passionate about Machine Learning, Data Science, and building
    real-world, production-ready AI applications.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
