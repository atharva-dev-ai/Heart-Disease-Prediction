# ============================================================
# Heart Disease Risk Predictor
# Premium Medical-Grade ML Web Application
# ============================================================

# -------------------- IMPORTS --------------------
# Core libraries
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
# Sets browser tab title, icon, and layout
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

/* --------- GLOBAL BACKGROUND --------- */
.stApp {
    background: radial-gradient(circle at top, #0f172a 0%, #020617 65%);
    color: #e5e7eb;
    font-family: "Segoe UI", sans-serif;
}

/* --------- HERO ANIMATION --------- */
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* --------- HEADERS --------- */
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

/* --------- GLASS CARDS --------- */
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

/* --------- LABELS --------- */
label {
    color: #c7d2fe !important;
    font-weight: 600;
}

/* --------- SLIDERS --------- */
div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #ef4444, #f97316);
}

/* --------- PRIMARY BUTTON --------- */
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #38bdf8) !important;
    color: white !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    padding: 0.8rem 1.6rem !important;
    box-shadow: 0 12px 35px rgba(56,189,248,0.6);
    transition: all 0.25s ease;
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

# Load dataset
data = pd.read_csv("heart_disease_data.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# ============================================================
# NAVIGATION TABS
# ============================================================
tabs = st.tabs(["🩺 Risk Assessment", "📊 Model Insights", "📄 Report", "ℹ️ About"])

# ============================================================
# TAB 1: RISK ASSESSMENT
# ============================================================
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("👤 Patient Clinical Information")

    # --------- INPUT FORM ---------
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        # Column 1
        with c1:
            age = st.slider("Age", 18, 90, 45)
            sex = st.selectbox("Sex (0 = Female, 1 = Male)", {"Female": 0, "Male": 1})
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                {
                    "0 = Typical Angina": 0,
                    "1 = Atypical Angina": 1,
                    "2 = Non-anginal Pain": 2,
                    "3 = Asymptomatic": 3
                }
            )

        # Column 2
        with c2:
            trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
            chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)
            fbs = st.selectbox("Fasting Blood Sugar >120 (fbs)", {"0 = No": 0, "1 = Yes": 1})
            restecg = st.selectbox(
                "Resting ECG (restecg)",
                {
                    "0 = Normal": 0,
                    "1 = ST-T Abnormality": 1,
                    "2 = LV Hypertrophy": 2
                }
            )

        # Column 3
        with c3:
            thalach = st.slider("Max Heart Rate (thalach)", 70, 210, 150)
            exang = st.selectbox("Exercise Induced Angina (exang)", {"0 = No": 0, "1 = Yes": 1})
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
            slope = st.selectbox(
                "ST Segment Slope (slope)",
                {
                    "0 = Upsloping": 0,
                    "1 = Flat": 1,
                    "2 = Downsloping": 2
                }
            )

        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia (thal)",
            {
                "0 = Normal": 0,
                "1 = Fixed Defect": 1,
                "2 = Reversible Defect": 2
            }
        )

        submitted = st.form_submit_button("🩺 Assess Heart Disease Risk", type="primary")

    # --------- PREDICTION ---------
    if submitted:
        with st.spinner("Analyzing patient data..."):
            time.sleep(1)

        # Prepare input in correct order
        input_data = np.array([
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Probability prediction
        risk_prob = model.predict_proba(input_scaled)[0][1] * 100

        st.subheader("📈 Risk Probability")

        # --------- ANIMATED GAUGE ---------
        st.progress(int(risk_prob))

        st.metric("Predicted Risk (%)", f"{risk_prob:.2f}%")

        if risk_prob > 50:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Low Risk Detected")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2: MODEL INSIGHTS (FEATURE IMPORTANCE)
# ============================================================
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📊 Feature Importance")

    # Logistic regression coefficients
    importance = pd.Series(
        model.coef_[0],
        index=X.columns
    ).sort_values()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="#60a5fa")
    ax.set_title("Feature Influence on Prediction")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3: PDF MEDICAL REPORT
# ============================================================
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📄 Download Medical Report")

    def generate_pdf():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica", 12)

        c.drawString(50, 800, "Heart Disease Risk Assessment Report")
        c.drawString(50, 770, f"Predicted Risk: {risk_prob:.2f}%")
        c.drawString(50, 740, "This report is generated by an ML model.")

        c.save()
        buffer.seek(0)
        return buffer

    if submitted:
        pdf = generate_pdf()
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf,
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
