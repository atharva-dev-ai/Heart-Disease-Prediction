# ============================================================
# Heart Disease Risk Predictor
# Medical-grade ML Web Application
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
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #050816, #020617);
        color: #f8fafc;
    }

    /* Section headers */
    h1, h2, h3 {
        color: #60a5fa;
        font-weight: 700;
    }

    /* Labels */
    label {
        color: #e5e7eb !important;
        font-size: 15px;
        font-weight: 600;
    }

    /* Input boxes */
    input, select, textarea {
        background-color: #020617 !important;
        color: #f9fafb !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }

    /* Sliders */
    .stSlider > div {
        color: #f9fafb;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 10px 20px;
    }

    /* Result cards */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 18px;
        font-weight: 700;
    }

    .low-risk {
        background-color: #052e16;
        color: #22c55e;
    }

    .high-risk {
        background-color: #450a0a;
        color: #ef4444;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
        return "Preventive monitoring and a balanced diet are advised."
    elif risk <= 50:
        return "Lifestyle modification and periodic cardiology review recommended."
    elif risk <= 75:
        return "Medical consultation with a cardiologist is strongly advised."
    else:
        return "Immediate cardiologist consultation required. High clinical risk."


def generate_pdf(report):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50,
                         "Heart Disease Risk Assessment Report")

    c.setFont("Helvetica", 10)
    c.drawCentredString(
        width / 2, height - 70,
        f"Generated on: {report['timestamp'].strftime('%d %B %Y, %H:%M')}"
    )

    y = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Clinical Parameters")
    y -= 20

    c.setFont("Helvetica", 10)
    for k, v in report["patient_data"].items():
        c.rect(45, y - 5, 500, 20)
        c.drawString(60, y, str(k))
        c.drawString(350, y, str(v))
        y -= 20
        if y < 120:
            c.showPage()
            y = height - 120
            c.setFont("Helvetica", 10)

    risk_text, color = interpret_risk(report["risk"])

    y -= 10
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Risk Level: {risk_text}")
    c.setFillColor(black)

    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Predicted Risk: {report['risk']}%")
    y -= 15
    c.drawString(50, y, f"Doctor Recommendation: {doctor_recommendation(report['risk'])}")

    c.setFont("Helvetica", 9)
    c.drawCentredString(
        width / 2, 40,
        "Developed by Atharva Savant | atharvasavant2506@gmail.com"
    )

    c.save()
    buffer.seek(0)
    return buffer

# ============================================================
# HEADER
# ============================================================
st.markdown(
    "<h1 style='text-align:center;color:#60a5fa;'>🫀 Heart Disease Risk Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>ML-based clinical decision support system</p>",
    unsafe_allow_html=True
)

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
tabs = st.tabs(["🩺 Risk Assessment", "📄 Reports", "📊 Model Insights", "ℹ️ About"])

# ============================================================
# TAB 1: RISK ASSESSMENT
# ============================================================
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    with st.form("patient_form"):
        patient_name = st.text_input("Patient Name")

        age = st.number_input(
           "Age (years)",
            min_value=29,
            max_value=77,
            value=45,
            step=1
)



        sex_label = st.selectbox("Sex (sex)", ["Female", "Male"])
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
            "patient_name": patient_name or "Not Provided",
            "timestamp": datetime.now(),
            "risk": round(risk_prob, 2),
            "risk_text": risk_text,
            "patient_data": {
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
        })

        st.metric("Predicted Risk (%)", f"{risk_prob:.2f}%")
        st.info(f"Clinical Interpretation: **{risk_text}**")
        st.warning(doctor_recommendation(risk_prob))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2: REPORTS (LAST 10 + INDIVIDUAL DOWNLOAD)
# ============================================================
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🗂️ Recent Reports (Last 10)")

    if "report_history" in st.session_state and len(st.session_state["report_history"]) > 0:
        recent_reports = st.session_state["report_history"][-10:][::-1]

        for idx, report in enumerate(recent_reports):
            with st.expander(
                f"🧑 {report['patient_name']} | "
                f"{report['timestamp'].strftime('%d-%m-%Y %H:%M')} | "
                f"{report['risk']}% | {report['risk_text']}"
            ):
                pdf_buffer = generate_pdf(report)
                st.download_button(
                    "⬇️ Download This Report (PDF)",
                    pdf_buffer,
                    file_name=f"{report['patient_name'].replace(' ', '_')}_heart_report.pdf",
                    mime="application/pdf",
                    key=f"dl_{idx}"
                )
    else:
        st.info("No reports generated in this session yet.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3: MODEL INSIGHTS
# ============================================================
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Feature Importance")

    importance = pd.Series(model.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax)
    st.pyplot(fig)

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
    - Name: Atharva Savant
   
    - Email: atharvasavant2506@gmail.com
   
    - GitHub: https://github.com/atharva-dev-ai  

    ⚠️ For educational purposes only. Not a medical diagnosis.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
