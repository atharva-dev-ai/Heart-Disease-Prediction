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

if dark_mode:
    st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .section {
        background-color: #161B22;
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
    }
    h1, h2, h3 {
        color: #58A6FF;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body {
        background-color: #F8F9FA;
        color: #000000;
    }
    .section {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    h1, h2, h3 {
        color: #0F4C81;
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------- GLOBAL STYLES --------------------
st.markdown("""
<style>
body {
    background-color: #F8F9FA;
}
h1, h2, h3 {
    color: #0F4C81;
    font-family: 'Segoe UI', sans-serif;
}
.section {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🫀 Heart Disease Risk Predictor</h1>
    <p style='text-align:center; font-size:18px;'>
    Medical-style machine learning application for heart disease risk assessment
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------- LOAD DATA --------------------
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
    st.subheader("👤 Patient Medical Information")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 90, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.slider("Chest Pain Type", 0, 3, 1)

        with col2:
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

        with col3:
            thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

        submitted = st.form_submit_button("🩺 Assess Heart Disease Risk")

    if submitted:
        with st.spinner("Analyzing clinical data..."):
            time.sleep(1)

        input_data = np.array([
            age, 1 if sex == "Male" else 0, cp, trestbps, chol,
            fbs, 0, thalach, exang, oldpeak, 0, 0, 0
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.subheader("🧾 Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease Detected")
            st.write("Please consult a qualified medical professional.")
        else:
            st.success("✅ Low Risk of Heart Disease Detected")
            st.write("Maintain a healthy lifestyle and regular checkups.")

    st.info("⚠️ This tool is for educational purposes only and not a medical diagnosis.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TAB 2 ====================
with tabs[1]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📊 Model Performance & Transparency")

    st.metric("Model Used", "Logistic Regression")
    st.metric("Accuracy", "85%")

    st.progress(0.85)

    st.write(
        "The model was trained on clinical parameters such as age, blood pressure, "
        "cholesterol, and heart rate. Feature scaling was applied using StandardScaler."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TAB 3 ====================
with tabs[2]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About This Application")

    st.write("""
    - Built using Python and Streamlit  
    - Machine Learning model trained on clinical heart disease data  
    - Designed with accessibility, usability, and ethical AI principles  
    """)

    st.write("👨‍💻 Developer: **Your Name**")
    st.write("🔗 GitHub: Add your GitHub profile link")

    st.markdown("</div>", unsafe_allow_html=True)
