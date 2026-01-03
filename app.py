# ============================================================
# Heart Disease Risk Predictor
# Premium Medical-Grade ML Web Application
# ============================================================

# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import numpy as np
import time
import io

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Visualization
import matplotlib.pyplot as plt

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# ============================================================
# PREMIUM DARK UI + HERO ANIMATION (CSS)
# ============================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a 0%, #020617 65%);
    color: #e5e7eb;
    font-family: "Segoe UI", sans-serif;
}

@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #60a5fa;
    text-align: center;
    animation: fadeSlide 1s ease-out;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    animation: fadeSlide 1.4s ease-out;
}

.glass {
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.08),
        rgba(255,255,255,0.02)
    );
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.65);
    border: 1px solid rgba(255,255,255,0.08);
}

label {
    color: #c7d2fe !important;
    font-weight: 600;
}

div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #ef4444, #f97316);
}

button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #38bdf8) !important;
    color: white !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    padding: 0.8rem 1.6rem !important;
    box-shadow: 0 12px 35px rgba(56,189,248,0.6);
}
button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 20px 55px rgba(56,189,248,0.9);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div class="hero-title">🫀 Heart Disease Risk Predictor</div>
<div class="hero-subtitle">
AI-powered clinical decision support system
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA & TRAIN MODEL
# ============================================================
data = pd.read_csv("heart_disease_data.csv")

X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# ============================================================
# NAVIGATION
# ============================================================
tabs = st.tabs(["🩺 Risk Assessment", "📊 Model Insights", "📄 Report", "ℹ️ About"])

# ============================================================
# TAB 1: RISK ASSESSMENT
# ============================================================
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("👤 Patient Clinical Information")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        # ---------- Column 1 ----------
        with c1:
            age = st.slider("Age", 18, 90, 45)

            sex_label = st.selectbox("Sex (0 = Female, 1 = Male)", ["Female", "Male"])
            sex = 0 if sex_label == "Female" else 1

            cp_label = st.selectbox(
                "Chest Pain Type (cp)",
                ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
            )
            cp = {
                "Typical Angina": 0,
                "Atypical Angina": 1,
                "Non-anginal Pain": 2,
                "Asymptomatic": 3
            }[cp_label]

        # ---------- Column 2 ----------
        with c2:
            trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
            chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)

            fbs_label = st.selectbox("Fasting Blood Sugar >120 (fbs)", ["No", "Yes"])
            fbs = 0 if fbs_label == "No" else 1

            restecg_label = st.selectbox(
                "Resting ECG (restecg)",
                ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
            )
            restecg = {
                "Normal": 0,
                "ST-T Abnormality": 1,
                "Left Ventricular Hypertrophy": 2
            }[restecg_label]

        # ---------- Column 3 ----------
        with c3:
            thalach = st.slider("Maximum Heart Rate (thalach)", 70, 210, 150)

            exang_label = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
            exang = 0 if exang_label == "No" else 1

            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

            slope_label = st.selectbox(
                "Slope of ST Segment (slope)",
                ["Upsloping", "Flat", "Downsloping"]
            )
            slope = {
                "Upsloping": 0,
                "Flat": 1,
                "Downsloping": 2
            }[slope_label]

        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])

        # 🔴 FIXED: Thalassemia has 4 categories
        thal_label = st.selectbox(
            "Thalassemia (thal)",
            ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"]
        )
        thal = {
            "Normal": 0,
            "Fixed Defect": 1,
            "Reversible Defect": 2,
            "Unknown": 3
        }[thal_label]

        submitted = st.form_submit_button("🩺 Assess Heart Disease Risk", type="primary")

    # ---------- PREDICTION ----------
    if submitted:
        with st.spinner("Analyzing patient data..."):
            time.sleep(1)

        input_data = np.array([
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        risk_prob = model.predict_proba(input_scaled)[0][1] * 100

        st.subheader("📈 Risk Probability")
        st.progress(int(risk_prob))
        st.metric("Predicted Risk (%)", f"{risk_prob:.2f}%")

        if risk_prob >= 50:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Low Risk Detected")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2: FEATURE IMPORTANCE
# ============================================================
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Feature Importance")

    importance = pd.Series(model.coef_[0], index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="#60a5fa")
    ax.set_title("Feature Influence on Heart Disease Prediction")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3: PDF REPORT
# ============================================================
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📄 Download Medical Report")

    if submitted:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica", 12)

        c.drawString(50, 800, "Heart Disease Risk Assessment Report")
        c.drawString(50, 770, f"Predicted Risk: {risk_prob:.2f}%")
        c.drawString(50, 740, "Generated using Machine Learning.")

        c.save()
        buffer.seek(0)

        st.download_button(
            "⬇️ Download PDF Report",
            data=buffer,
            file_name="heart_disease_report.pdf",
            mime="application/pdf"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 4: ABOUT
# ============================================================


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
