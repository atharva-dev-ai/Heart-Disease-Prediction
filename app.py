# ============================================================
# Heart Disease Risk Predictor – Medical-grade ML Application
# ============================================================

# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import green, red, orange, black

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# ============================================================
# UI THEME
# ============================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a 0%, #020617 65%);
    color: #e5e7eb;
}
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.65);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def interpret_risk(risk):
    if risk <= 10:
        return "Very Low Risk (Normal)", green
    elif risk <= 25:
        return "Low Risk", green
    elif risk <= 50:
        return "Moderate Risk", orange
    elif risk <= 75:
        return "High Risk", red
    else:
        return "Very High Risk (Critical)", red


def doctor_recommendation(risk):
    if risk <= 10:
        return "Maintain a healthy lifestyle. Routine annual checkups recommended."
    elif risk <= 25:
        return "Preventive monitoring and healthy diet advised."
    elif risk <= 50:
        return "Lifestyle modifications and periodic cardiology review advised."
    elif risk <= 75:
        return "Medical consultation with a cardiologist is strongly recommended."
    else:
        return "Immediate cardiology consultation required. High clinical risk detected."

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;color:#60a5fa;'>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>ML-based clinical decision support system</p>", unsafe_allow_html=True)

# ============================================================
# LOAD DATA & MODEL
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

    with st.form("patient_form"):
        patient_name = st.text_input("Patient Name")

        age = st.slider("Age (age)", 18, 90, 45)
        sex_label = st.selectbox("Sex (sex) → 0 = Female, 1 = Male", ["Female", "Male"])
        sex = 0 if sex_label == "Female" else 1

        cp_label = st.selectbox(
            "Chest Pain Type (cp)",
            ["0 = Typical Angina", "1 = Atypical Angina", "2 = Non-anginal Pain", "3 = Asymptomatic"]
        )
        cp = int(cp_label[0])

        trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
        chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)

        fbs_label = st.selectbox("Fasting Blood Sugar >120 mg/dL (fbs)", ["No", "Yes"])
        fbs = 1 if fbs_label == "Yes" else 0

        restecg_label = st.selectbox(
            "Resting ECG Result (restecg)",
            ["0 = Normal", "1 = ST-T Abnormality", "2 = LV Hypertrophy"]
        )
        restecg = int(restecg_label[0])

        thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)

        exang_label = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
        exang = 1 if exang_label == "Yes" else 0

        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

        slope_label = st.selectbox(
            "Slope of Peak Exercise ST Segment (slope)",
            ["0 = Upsloping", "1 = Flat", "2 = Downsloping"]
        )
        slope = int(slope_label[0])

        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])

        thal_label = st.selectbox(
            "Thalassemia (thal)",
            ["0 = Normal", "1 = Fixed Defect", "2 = Reversible Defect", "3 = Unknown"]
        )
        thal = int(thal_label[0])

        submitted = st.form_submit_button("🩺 Assess Heart Disease Risk")

    if submitted:
        input_data = np.array([
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]).reshape(1, -1)

        risk_prob = model.predict_proba(scaler.transform(input_data))[0][1] * 100

        risk_text, _ = interpret_risk(risk_prob)

        st.session_state.setdefault("report_history", [])
        st.session_state["report_history"].append({
            "time": datetime.now(),
            "risk": risk_prob
        })

        st.session_state["risk_prob"] = risk_prob
        st.session_state["patient_data"] = {
            "Patient Name": patient_name or "Not Provided",
            "Age (age)": age,
            "Sex (sex)": sex_label,
            "Chest Pain (cp)": cp_label,
            "Resting BP (trestbps)": trestbps,
            "Cholesterol (chol)": chol,
            "Fasting Blood Sugar (fbs)": fbs_label,
            "Resting ECG (restecg)": restecg_label,
            "Max Heart Rate (thalach)": thalach,
            "Exercise Angina (exang)": exang_label,
            "ST Depression (oldpeak)": oldpeak,
            "Slope (slope)": slope_label,
            "Major Vessels (ca)": ca,
            "Thalassemia (thal)": thal_label
        }

        st.metric("Predicted Risk (%)", f"{risk_prob:.2f}%")
        st.info(f"Clinical Interpretation: **{risk_text}**")
        st.warning(doctor_recommendation(risk_prob))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3: PDF REPORT
# ============================================================
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    if "risk_prob" in st.session_state:

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, height - 50, "Heart Disease Risk Assessment Report")

        y = height - 120
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Patient Clinical Parameters")
        y -= 20

        risk_text, color = interpret_risk(st.session_state["risk_prob"])

        for k, v in st.session_state["patient_data"].items():
            c.setFillColor(black)
            c.rect(45, y - 5, 500, 20)
            c.drawString(60, y, f"{k}")
            c.drawString(350, y, f"{v}")
            y -= 20

        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y - 10, f"Risk Level: {risk_text}")
        c.setFillColor(black)

        y -= 40
        c.drawString(50, y, f"Predicted Risk: {st.session_state['risk_prob']:.2f}%")
        y -= 20
        c.drawString(50, y, "Doctor Recommendation:")
        y -= 15
        c.drawString(60, y, doctor_recommendation(st.session_state["risk_prob"]))

        c.setFont("Helvetica", 9)
        c.drawCentredString(
            width / 2, 40,
            "Developed by Atharva Savant | atharvasavant2506@gmail.com"
        )

        c.save()
        buffer.seek(0)

        st.download_button(
            "⬇️ Download Medical PDF Report",
            buffer,
            "heart_disease_report.pdf",
            "application/pdf"
        )

        st.info(f"Reports generated this session: {len(st.session_state['report_history'])}")

    else:
        st.warning("Please assess heart disease risk first.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ABOUT
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
    - Name: Atharva Savant
   
    - Email: atharvasavant2506@gmail.com
   
    - GitHub: https://github.com/atharva-dev-ai  

    ⚠️ For educational purposes only. Not a medical diagnosis.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
