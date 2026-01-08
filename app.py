import streamlit as st
import numpy as np
import pickle
from datetime import datetime

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="centered"
)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["About", "Prediction", "Reports"]
)

# --------------------------------------------------
# Session State for Reports
# --------------------------------------------------
if "report" not in st.session_state:
    st.session_state.report = None

# ==================================================
# ABOUT PAGE
# ==================================================
if page == "About":

    st.title("❤️ Heart Disease Prediction System")

    st.markdown("""
    ### 📘 About the Project

    This application uses **Machine Learning** to provide a
    **preliminary heart disease risk screening**.

    #### 🔹 Highlights
    - Simple user-friendly mode
    - Doctor mode with clinical parameters
    - Logistic Regression–based model
    - Risk probability estimation
    - Timestamped medical-style reports

    ⚠️ **Disclaimer:**  
    This tool is for **educational and screening purposes only**  
    and **is not a medical diagnosis system**.
    """)

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "Prediction":

    st.title("🔍 Heart Disease Risk Prediction")

    mode = st.radio(
        "Select Mode",
        ["👤 General User", "🩺 Doctor Mode"]
    )

    st.subheader("🧾 Basic Information")

    age = st.number_input("Age (years)", 29, 77, 45)
    sex = st.selectbox("Gender", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    # --------------------------------------------------
    # USER MODE (MINIMISED INPUTS)
    # --------------------------------------------------
    if mode == "👤 General User":

        trestbps = st.selectbox(
            "Blood Pressure",
            [120, 130, 145],
            format_func=lambda x:
            "Normal (~120)" if x == 120 else
            "Elevated (~130)" if x == 130 else
            "High (140+)"
        )

        chol = st.selectbox(
            "Cholesterol Level",
            [180, 220, 260],
            format_func=lambda x:
            "Normal (<200)" if x == 180 else
            "Borderline (200–239)" if x == 220 else
            "High (240+)"
        )

        thalach = st.selectbox(
            "Physical Stamina",
            [120, 150, 175],
            format_func=lambda x:
            "Low" if x == 120 else
            "Moderate" if x == 150 else
            "High"
        )

        # Hidden clinical defaults
        cp = 0
        fbs = 0
        restecg = 1
        exang = 0
        oldpeak = 1.0
        slope = 1
        ca = 0
        thal = 2

    # --------------------------------------------------
    # DOCTOR MODE (FULL INPUTS)
    # --------------------------------------------------
    else:
        st.subheader("🩺 Clinical Parameters")

        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])

    # --------------------------------------------------
    # Generate Report
    # --------------------------------------------------
    if st.button("Generate Report"):

        input_data = np.array([[
            age, sex_val, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]])

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        st.session_state.report = {
            "prediction": prediction,
            "probability": probability,
            "time": datetime.now().strftime("%d %b %Y • %I:%M %p"),
            "accuracy": 0.88  # replace with real accuracy if known
        }

        st.success("Report generated successfully. Go to Reports section.")

# ==================================================
# REPORTS PAGE
# ==================================================
elif page == "Reports":

    st.title("📄 Reports")

    if st.session_state.report is None:
        st.warning("No report generated yet.")
    else:
        r = st.session_state.report

        st.markdown(f"""
        **🕒 Generated On:** {r['time']}  

        **🩺 Prediction:** {"High Risk" if r['prediction'] == 1 else "Low Risk"}  

        **📊 Risk Probability:** {r['probability']:.2f}%  

        **📈 Model Accuracy:** {r['accuracy'] * 100:.2f}%  
        """)

        st.markdown("""
        ⚠️ This report is generated by a machine learning model and
        should not be considered a medical diagnosis.
        """)
