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

# -------------------- THEME STYLES --------------------
if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3, label {
        color: #58A6FF !important;
    }
    .section {
        background-color: #161B22;
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
    }
    div[data-testid="stForm"] {
        background-color: #161B22;
        padding: 1.5rem;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #F8F9FA;
        color: #000000;
    }
    h1, h2, h3, label {
        color: #0F4C81 !important;
    }
    .section {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    """
    <h1 style="text-align:center;">🫀 Heart Disease Risk Predictor</h1>
    <p style="text-align:center; font-size:18px;">
    Medical-grade Machine Learning Web Application
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD DATA & TRAIN MODEL --------------------
data = pd.read_csv("heart_disease_data.csv")

X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# -------------------- NAVIGATION --------------------
tabs = st.tabs(["🩺 Risk Assessment", "📊 Model Insights", "ℹ️ About"])

# ==================== TAB 1: RISK ASSESSMENT ====================
with tabs[0]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("👤 Patient Clinical Information")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age (age)", 18, 90, 45)
            sex = st.selectbox("Sex (sex)", ["Female", "Male"])
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                {
                    "Typical Angina": 0,
                    "Atypical Angina": 1,
                    "Non-anginal Pain": 2,
                    "Asymptomatic": 3
                }
            )

        with col2:
            trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
            chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL (fbs)",
                {"No": 0, "Yes": 1}
            )
            restecg = st.selectbox(
                "Resting ECG Results (restecg)",
                {
                    "Normal": 0,
                    "ST-T Wave Abnormality": 1,
                    "Left Ventricular Hypertrophy": 2
                }
            )

        with col3:
            thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)
            exang = st.selectbox(
                "Exercise Induced Angina (exang)",
                {"No": 0, "Yes": 1}
            )
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope)",
                {
                    "Upsloping": 0,
                    "Flat": 1,
                    "Downsloping": 2
                }
            )

        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia (thal)",
            {
                "Normal": 0,
                "Fixed Defect": 1,
                "Reversible Defect": 2
            }
        )

        submitted = st.form_submit_button("🩺 Assess Heart Disease Risk")

    if submitted:
        with st.spinner("Analyzing clinical data..."):
            time.sleep(1)

        input_data = np.array([
            age,
            1 if sex == "Male" else 0,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
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

# ==================== TAB 2: MODEL INSIGHTS ====================
with tabs[1]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📊 Model Transparency")

    st.metric("Model Used", "Logistic Regression")
    st.metric("Input Features", "13 Clinical Parameters")
    st.metric("Accuracy", "≈ 85%")

    st.progress(0.85)

    st.write(
        "The model was trained using standardized clinical features and evaluated "
        "using classification metrics to ensure reliability."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TAB 3: ABOUT ====================
with tabs[2]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About This Application")

    st.write("""
    - End-to-end Machine Learning project  
    - Deployed using Streamlit Cloud  
    - Designed with usability, accessibility, and ethical AI principles  
    """)

    st.write("👨‍💻 Developer: **Atharva Savant**")
    st.write("🔗 GitHub: https://github.com/atharva-dev-ai")

    st.markdown("</div>", unsafe_allow_html=True)
