import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #050816, #020617);
    color: #f8fafc;
}
h1, h2, h3 {
    color: #60a5fa;
    font-weight: 700;
}
label {
    color: #e5e7eb !important;
    font-weight: 600;
}
input, select {
    background-color: #020617 !important;
    color: #f9fafb !important;
    border: 1px solid #334155 !important;
    border-radius: 8px;
}
.stButton > button {
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 20px;
}
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
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD & TRAIN MODEL (CACHED)
# --------------------------------------------------
@st.cache_resource
def train_pipeline():
    df = pd.read_csv("heart_disease_data.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    return pipeline

pipeline = train_pipeline()

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("❤️ Heart Disease Risk Assessment")
st.markdown(
    "A **preliminary screening tool** using Machine Learning. "
    "**Not a medical diagnosis.**"
)

# --------------------------------------------------
# Mode Selection
# --------------------------------------------------
mode = st.radio(
    "Select Mode",
    ["👤 General User", "🩺 Doctor / Clinical Mode"]
)

# ==================================================
# USER MODE (MINIMISED FEATURES)
# ==================================================
if mode == "👤 General User":

    st.subheader("🧾 Basic Information")

    age = st.number_input("Age (years)", min_value=29, max_value=77, value=45)
    sex = st.selectbox("Gender", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    st.subheader("🩺 Health Information")

    bp_choice = st.selectbox(
        "Blood Pressure",
        ["Normal (~120)", "Elevated (~130)", "High (140+)", "I don't know"]
    )
    trestbps = {
        "Normal (~120)": 120,
        "Elevated (~130)": 130,
        "High (140+)": 145,
        "I don't know": 130
    }[bp_choice]

    chol_choice = st.selectbox(
        "Cholesterol Level",
        ["Normal (<200)", "Borderline (200–239)", "High (240+)", "I don't know"]
    )
    chol = {
        "Normal (<200)": 180,
        "Borderline (200–239)": 220,
        "High (240+)": 260,
        "I don't know": 220
    }[chol_choice]

    stamina_choice = st.selectbox(
        "Physical Stamina",
        ["Low", "Moderate", "High", "I don't know"]
    )
    thalach = {
        "Low": 120,
        "Moderate": 150,
        "High": 175,
        "I don't know": 150
    }[stamina_choice]

    # Defaults for hidden clinical features
    cp = 0
    fbs = 0
    restecg = 1
    exang = 0
    oldpeak = 1.0
    slope = 1
    ca = 0
    thal = 2

    if st.button("Generate Report"):

        input_data = np.array([[
            age, sex_val, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]])

        prediction = pipeline.predict(input_data)[0]
        report_time = datetime.now().strftime("%d %b %Y • %I:%M %p")

        st.subheader("📄 Risk Assessment Report")
        st.markdown(f"🕒 **Report Generated On:** `{report_time}`")

        if prediction == 1:
            st.markdown(
                "<div class='result-box high-risk'>⚠️ High Risk of Heart Disease</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box low-risk'>✅ Low Risk of Heart Disease</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            "_Based on limited self-reported inputs. "
            "For screening purposes only._"
        )

# ==================================================
# DOCTOR MODE (FULL FEATURES)
# ==================================================
else:

    st.subheader("🩺 Clinical Input (Doctor Mode)")
    st.markdown("For trained healthcare professionals only.")

    age = st.number_input("Age", 29, 77, 50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])

    if st.button("Generate Clinical Report"):

        input_data = np.array([[
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]])

        prediction = pipeline.predict(input_data)[0]
        report_time = datetime.now().strftime("%d %b %Y • %I:%M %p")

        st.subheader("📄 Clinical Report")
        st.markdown(f"🕒 **Report Generated On:** `{report_time}`")

        if prediction == 1:
            st.markdown(
                "<div class='result-box high-risk'>⚠️ High Risk Detected</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box low-risk'>✅ Low Risk Detected</div>",
                unsafe_allow_html=True
            )
