import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="centered"
)

# --------------------------------------------------
# CSS (minimal – no fancy breakage)
# --------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #020617; color: #f8fafc; }
h1, h2, h3 { color: #60a5fa; }
label { color: #e5e7eb !important; font-weight: 600; }
.stButton > button {
    background-color: #dc2626;
    color: white;
    font-weight: 700;
    border-radius: 8px;
}
.report-box {
    background-color: #020617;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 20px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD + TRAIN MODEL
# --------------------------------------------------
@st.cache_resource
def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return pipeline, accuracy

pipeline, model_accuracy = train_model()

# --------------------------------------------------
# ABOUT SECTION (RESTORED)
# --------------------------------------------------
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
### 📘 About This Project
This is a **machine learning–based heart disease risk screening system**.

- Designed for **general users** with simplified inputs  
- Includes **doctor mode** for full clinical parameters  
- Uses **Logistic Regression with feature scaling**
- Intended for **educational and screening purposes only**

⚠️ This is **not a medical diagnosis tool**.
""")

st.divider()

# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------
mode = st.radio(
    "Select Usage Mode",
    ["👤 General User", "🩺 Doctor Mode"]
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.subheader("🧾 Input Details")

age = st.number_input("Age (years)", 29, 77, 45)
sex = st.selectbox("Gender", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

if mode == "👤 General User":
    trestbps = st.selectbox(
        "Blood Pressure",
        [120, 130, 145],
        format_func=lambda x: "Normal" if x == 120 else "Elevated" if x == 130 else "High"
    )
    chol = st.selectbox("Cholesterol Level", [180, 220, 260])
    thalach = st.selectbox("Physical Stamina", [120, 150, 175])

    # defaults for hidden clinical features
    cp, fbs, restecg, exang, oldpeak, slope, ca, thal = 0, 0, 1, 0, 1.0, 1, 0, 2

else:
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

# --------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------
if st.button("Generate Report"):

    input_data = np.array([[
        age, sex_val, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]])

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1] * 100
    timestamp = datetime.now().strftime("%d %b %Y • %I:%M %p")

    st.divider()
    st.subheader("📄 Prediction Report")

    st.markdown(f"""
    <div class="report-box">
    🕒 <b>Report Generated:</b> {timestamp}<br><br>
    <b>Predicted Risk:</b> {"High Risk" if prediction else "Low Risk"}<br>
    <b>Risk Probability:</b> {probability:.2f}%<br>
    <b>Model Accuracy:</b> {model_accuracy*100:.2f}%
    </div>
    """, unsafe_allow_html=True)
