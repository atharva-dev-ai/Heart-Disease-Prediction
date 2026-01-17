# ============================================================
# Heart Disease Risk Predictor
# Internship-Ready Medical ML Web Application
# ============================================================

# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

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
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def interpret_risk(risk):
    if risk <= 10:
        return "Very Low Risk", green
    elif risk <= 25:
        return "Low Risk", green
    elif risk <= 50:
        return "Moderate Risk", orange
    elif risk <= 75:
        return "High Risk", red
    else:
        return "Very High Risk", red


def doctor_recommendation(risk):
    if risk <= 25:
        return "Maintain healthy lifestyle."
    elif risk <= 50:
        return "Lifestyle changes advised."
    elif risk <= 75:
        return "Consult a cardiologist."
    else:
        return "Immediate medical attention required."


def generate_pdf(report):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "Heart Disease Risk Report")

    c.setFont("Helvetica", 10)
    c.drawCentredString(
        width / 2, height - 70,
        report["timestamp"].strftime("%d %B %Y, %H:%M")
    )

    y = height - 120
    for k, v in report["patient_data"].items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 15

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 20, f"Risk: {report['risk']}%")

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
    "<p style='text-align:center;'>ML-based Clinical Decision Support System</p>",
    unsafe_allow_html=True
)

# ============================================================
# LOAD DATA & TRAIN MODEL (CACHED)
# ============================================================
@st.cache_resource
def train_model():
    data = pd.read_csv("heart_disease_data.csv")

    X = data.drop("target", axis=1)
    y = data["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # Evaluation metrics
    y_pred = model.predict(X_scaled)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "confusion": confusion_matrix(y, y_pred)
    }

    return model, scaler, X, metrics


model, scaler, X, metrics = train_model()

# ============================================================
# TABS
# ============================================================
tabs = st.tabs(["🩺 Risk Assessment", "📄 Reports", "📊 Model Insights", "ℹ️ About"])

# ============================================================
# TAB 1: RISK ASSESSMENT
# ============================================================
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    mode = st.radio(
        "Select Mode",
        ["🧑 Patient Mode", "🩺 Doctor Mode"],
        horizontal=True
    )

    with st.form("form"):
        age = st.slider("Age", 18, 90, 45)
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex = 0 if sex == "Female" else 1
        trestbps = st.slider("Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 400, 200)
        thalach = st.slider("Max Heart Rate", 70, 210, 150)

        cp = st.selectbox("Chest Pain", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting Sugar > 120", [0, 1])
        restecg = st.selectbox("ECG", [0, 1, 2])
        exang = st.selectbox("Exercise Angina", [0, 1])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = np.array([
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]).reshape(1, -1)

        risk = model.predict_proba(scaler.transform(input_data))[0][1] * 100
        risk_text, _ = interpret_risk(risk)

        st.metric("Risk Probability", f"{risk:.2f}%")
        st.info(risk_text)
        st.warning(doctor_recommendation(risk))

        # Explainable AI
        contributions = (
            pd.Series(model.coef_[0], index=X.columns)
            * scaler.transform(input_data)[0]
        ).abs().sort_values(ascending=False)

        st.markdown("### 🔍 Top Risk Contributors")
        for f in contributions.head(5).index:
            st.write(f"• {f}")

        if mode == "🩺 Doctor Mode":
            st.markdown("### Clinical Input Vector")
            st.json(input_data.tolist())

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2: REPORTS
# ============================================================
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    if submit:
        st.session_state.history.append({
            "timestamp": datetime.now(),
            "risk": round(risk, 2),
            "patient_data": dict(zip(X.columns, input_data[0]))
        })

    for i, r in enumerate(st.session_state.history[-10:][::-1]):
        with st.expander(f"Report {i+1} | {r['risk']}%"):
            pdf = generate_pdf(r)
            st.download_button("Download PDF", pdf, key=i)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3: MODEL INSIGHTS
# ============================================================
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    col4.metric("F1 Score", f"{metrics['f1']:.2f}")

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics["confusion"], annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 4: ABOUT
# ============================================================
with tabs[3]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.write("""
    **Heart Disease Risk Predictor**

    • End-to-end ML system  
    • Model evaluation & Explainable AI  
    • Medical-grade reporting  
    • Role-based UI design  

    **Developer:** Atharva Savant  
    **GitHub:** https://github.com/atharva-dev-ai  

    ⚠️ Educational use only.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
