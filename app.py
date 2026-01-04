# ============================================================
# Heart Disease Risk Predictor
# End-to-End Machine Learning Web Application
# ============================================================

# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
from datetime import datetime

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
# PREMIUM DARK UI (STABLE & CLEAN)
# ============================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a 0%, #020617 65%);
    color: #e5e7eb;
    font-family: "Segoe UI", sans-serif;
}
h1 {
    color: #60a5fa;
    text-align: center;
    font-weight: 800;
}
h2, h3 {
    color: #93c5fd;
}
label {
    color: #c7d2fe !important;
    font-weight: 600;
}
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.65);
}
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #38bdf8) !important;
    color: white !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    padding: 0.8rem 1.6rem !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>ML-based clinical decision support system</p>",
    unsafe_allow_html=True
)

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

        # ---------- COLUMN 1 ----------
        with c1:
            patient_name = st.text_input("Patient Name")
            age = st.slider("Age (age)", 18, 90, 45)

            sex_label = st.selectbox(
                "Sex (sex) → 0 = Female, 1 = Male",
                ["Female", "Male"]
            )
            sex = 0 if sex_label == "Female" else 1

            cp_label = st.selectbox(
                "Chest Pain Type (cp)",
                [
                    "0 = Typical Angina",
                    "1 = Atypical Angina",
                    "2 = Non-anginal Pain",
                    "3 = Asymptomatic"
                ]
            )
            cp = int(cp_label[0])

        # ---------- COLUMN 2 ----------
        with c2:
            trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
            chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)

            fbs_label = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL (fbs) → 1 = Yes, 0 = No",
                ["No", "Yes"]
            )
            fbs = 0 if fbs_label == "No" else 1

            restecg_label = st.selectbox(
                "Resting ECG Result (restecg)",
                [
                    "0 = Normal",
                    "1 = ST-T Wave Abnormality",
                    "2 = Left Ventricular Hypertrophy"
                ]
            )
            restecg = int(restecg_label[0])

        # ---------- COLUMN 3 ----------
        with c3:
            thalach = st.slider(
                "Maximum Heart Rate Achieved (thalach)",
                70, 210, 150
            )

            exang_label = st.selectbox(
                "Exercise Induced Angina (exang) → 1 = Yes, 0 = No",
                ["No", "Yes"]
            )
            exang = 0 if exang_label == "No" else 1

            oldpeak = st.slider(
                "ST Depression (oldpeak)",
                0.0, 6.0, 1.0
            )

            slope_label = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope)",
                [
                    "0 = Upsloping",
                    "1 = Flat",
                    "2 = Downsloping"
                ]
            )
            slope = int(slope_label[0])

        ca = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy (ca)",
            [0, 1, 2, 3, 4]
        )

        thal_label = st.selectbox(
            "Thalassemia (thal)",
            [
                "0 = Normal",
                "1 = Fixed Defect",
                "2 = Reversible Defect",
                "3 = Unknown"
            ]
        )
        thal = int(thal_label[0])

        submitted = st.form_submit_button(
            "🩺 Assess Heart Disease Risk",
            type="primary"
        )

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

        # Save everything to session state for PDF
        st.session_state["risk_prob"] = risk_prob
        st.session_state["patient_data"] = {
            "Patient Name": patient_name,
            "Age": age,
            "Sex": sex_label,
            "Chest Pain (cp)": cp,
            "Resting BP (trestbps)": trestbps,
            "Cholesterol (chol)": chol,
            "Fasting Blood Sugar (fbs)": fbs,
            "Resting ECG (restecg)": restecg,
            "Max Heart Rate (thalach)": thalach,
            "Exercise Angina (exang)": exang,
            "ST Depression (oldpeak)": oldpeak,
            "Slope (slope)": slope,
            "Major Vessels (ca)": ca,
            "Thalassemia (thal)": thal
        }

        st.subheader("📈 Risk Probability")
        st.progress(int(risk_prob))
        st.metric("Predicted Risk (%)", f"{risk_prob:.2f}%")

        if risk_prob >= 50:
            st.error("⚠️ High Risk of Heart Disease Detected")
        else:
            st.success("✅ Low Risk of Heart Disease Detected")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2: MODEL INSIGHTS
# ============================================================
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Feature Importance")

    importance = pd.Series(model.coef_[0], index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature Influence on Heart Disease Prediction")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3: PDF REPORT
# ============================================================
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📄 Download Medical Report")

    if "risk_prob" in st.session_state and "patient_data" in st.session_state:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica", 11)

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 820, "Heart Disease Risk Assessment Report")

        c.setFont("Helvetica", 11)
        c.drawString(50, 800, f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
        c.drawString(50, 785, "-" * 90)

        y = 760
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "Patient Details")
        y -= 20

        c.setFont("Helvetica", 11)
        for k, v in st.session_state["patient_data"].items():
            c.drawString(60, y, f"{k}: {v}")
            y -= 16

        y -= 10
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "Prediction Result")
        y -= 20

        c.setFont("Helvetica", 11)
        c.drawString(
            60,
            y,
            f"Predicted Risk of Heart Disease: {st.session_state['risk_prob']:.2f}%"
        )

        y -= 30
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(
            50,
            y,
            "Disclaimer: This report is generated using a machine learning model and"
        )
        c.drawString(
            50,
            y - 12,
            "is for educational purposes only. It is not a medical diagnosis."
        )

        c.save()
        buffer.seek(0)

        st.download_button(
            "⬇️ Download PDF Report",
            data=buffer,
            file_name="heart_disease_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Please assess heart disease risk first.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 4: ABOUT
# ============================================================
with tabs[3]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.write("""
    **Project Highlights**
    - End-to-end Machine Learning healthcare system  
    - Explicit feature–column traceability  
    - Probability-based prediction  
    - Explainable AI & PDF medical reporting  

    **Developer**
    - Name: Atharva  
    - GitHub: https://github.com/atharva-dev-ai  

    ⚠️ For educational purposes only. Not a medical diagnosis.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
