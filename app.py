# ============================================================
# Heart Disease Risk Predictor
# Medical-grade ML Web Application
# ============================================================

# -------------------- IMPORTS --------------------

# Streamlit is used to build the web-based UI
import streamlit as st

# Pandas is used to load and manipulate the dataset
import pandas as pd

# NumPy is used for numerical computations
import numpy as np

# time module (optional utility)
import time

# io is used to create in-memory PDF files
import io

# datetime is used to add timestamps to reports
from datetime import datetime

# StandardScaler normalizes input features
from sklearn.preprocessing import StandardScaler

# Logistic Regression is the ML classification model
from sklearn.linear_model import LogisticRegression

# Matplotlib is used for feature importance plotting
import matplotlib.pyplot as plt

# ReportLab is used to generate PDF reports
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import green, red, orange, black


# -------------------- PAGE CONFIG --------------------

# Set page title, icon, and layout
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# ============================================================
# UI THEME (Custom CSS)
# ============================================================

# Inject custom CSS for dark medical UI + glass effect
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

# Converts numerical risk percentage into human-readable risk level
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


# Provides medical-style recommendations based on risk level
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


# Generates a downloadable PDF medical report
def generate_pdf(report):
    # Create an in-memory buffer to store the PDF
    buffer = io.BytesIO()

    # Create a PDF canvas with A4 size
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Report title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50,
                        "Heart Disease Risk Assessment Report")

    # Timestamp below the title
    c.setFont("Helvetica", 10)
    c.drawCentredString(
        width / 2, height - 70,
        f"Generated on: {report['timestamp'].strftime('%d %B %Y, %H:%M')}"
    )

    # Starting Y position for patient details
    y = height - 120

    # Section heading
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Clinical Parameters")
    y -= 20

    # Loop through patient data and draw rows
    c.setFont("Helvetica", 10)
    for k, v in report["patient_data"].items():
        c.rect(45, y - 5, 500, 20)
        c.drawString(60, y, str(k))
        c.drawString(350, y, str(v))
        y -= 20

        # Start new page if space runs out
        if y < 120:
            c.showPage()
            y = height - 120
            c.setFont("Helvetica", 10)

    # Interpret risk for display
    risk_text, color = interpret_risk(report["risk"])

    # Display risk level
    y -= 10
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Risk Level: {risk_text}")
    c.setFillColor(black)

    # Display risk percentage and recommendation
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Predicted Risk: {report['risk']}%")
    y -= 15
    c.drawString(50, y, f"Doctor Recommendation: {doctor_recommendation(report['risk'])}")

    # Footer information
    c.setFont("Helvetica", 9)
    c.drawCentredString(
        width / 2, 40,
        "Developed by Atharva Savant | atharvasavant2506@gmail.com"
    )

    # Save the PDF and return it
    c.save()
    buffer.seek(0)
    return buffer


# ============================================================
# HEADER
# ============================================================

# Display main title
st.markdown(
    "<h1 style='text-align:center;color:#60a5fa;'>🫀 Heart Disease Risk Predictor</h1>",
    unsafe_allow_html=True
)

# Display subtitle
st.markdown(
    "<p style='text-align:center;'>ML-based clinical decision support system</p>",
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA & TRAIN MODEL (CACHED)
# ============================================================

# Cache the training so it runs only once
@st.cache_resource
def train_model():
    # Load dataset from CSV
    data = pd.read_csv("heart_disease_data.csv")

    # Separate features and target column
    X = data.drop("target", axis=1)
    y = data["target"]

    # Scale features for better ML performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler, X


# Retrieve trained model, scaler, and feature names
model, scaler, X = train_model()


# ============================================================
# NAVIGATION TABS
# ============================================================

# Create 4 navigation tabs
tabs = st.tabs([
    "🩺 Risk Assessment",
    "📄 Reports",
    "📊 Model Insights",
    "ℹ️ About"
])


# ============================================================
# TAB 1: RISK ASSESSMENT
# ============================================================

with tabs[0]:

    # Glass container for styling
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

     # --------------------------------------------------
    # Help section: How users can get medical values
    # --------------------------------------------------
    with st.expander("ℹ️ How do I get these medical values?"):
        st.markdown("""
        ### 🩺 Basic Measurements

        **trestbps (Resting Blood Pressure)**  
        • Measured using a **BP machine** at home or clinic  

        **chol (Serum Cholesterol)**  
        • Obtained from a **blood test (lipid profile)**  

        **thalach (Maximum Heart Rate Achieved)**  
        • Measured during a **stress test / treadmill test**

        **fbs (Fasting Blood Sugar)**  
        • Blood sugar test after **8–12 hours of fasting**

        ---

        ### ❤️ Heart-related Clinical Tests

        **cp (Chest Pain Type)**  
        • Determined by a **doctor based on symptoms**  
        • Includes angina type, pain during exertion, or no pain  

        **restecg (Resting ECG Result)**  
        • Obtained from an **ECG (Electrocardiogram) test**  

        **exang (Exercise Induced Angina)**  
        • Identified during a **stress test or physical activity**  
        • Indicates chest pain during exercise  

        **oldpeak (ST Depression)**  
        • Measured from an **ECG during stress testing**  
        • Indicates stress on the heart  

        **slope (Slope of ST Segment)**  
        • Derived from **ECG graph patterns** during exercise  

        **ca (Number of Major Vessels)**  
        • Determined using **angiography (imaging test)**  

        **thal (Thalassemia / Blood Disorder Indicator)**  
        • Determined via **blood tests or nuclear imaging**
        """)


    # Form groups all patient inputs together
    with st.form("patient_form"):

        # Patient name input
        patient_name = st.text_input("Patient Name")

        # Age slider
        age = st.slider("Age (age)", 18, 90, 45)

        # Sex selection (encoded later)
        sex_label = st.selectbox("Sex (sex)", ["Female", "Male"])
        sex = 0 if sex_label == "Female" else 1

        # Chest pain type selection
        cp_label = st.selectbox(
            "Chest Pain Type (cp)",
            ["0 = Typical Angina", "1 = Atypical Angina",
             "2 = Non-anginal Pain", "3 = Asymptomatic"]
        )
        cp = int(cp_label[0])

        # Blood pressure input
        trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)

        # Cholesterol input
        chol = st.slider("Serum Cholesterol (chol)", 100, 400, 200)

        # Fasting blood sugar
        fbs_label = st.selectbox("Fasting Blood Sugar >120 mg/dL (fbs)", ["No", "Yes"])
        fbs = 1 if fbs_label == "Yes" else 0

        # ECG result
        restecg_label = st.selectbox(
            "Resting ECG Result (restecg)",
            ["0 = Normal", "1 = ST-T Abnormality", "2 = LV Hypertrophy"]
        )
        restecg = int(restecg_label[0])

        # Maximum heart rate
        thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)

        # Exercise induced angina
        exang_label = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
        exang = 1 if exang_label == "Yes" else 0

        # ST depression value
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

        # Slope of ST segment
        slope_label = st.selectbox(
            "Slope of Peak Exercise ST Segment (slope)",
            ["0 = Upsloping", "1 = Flat", "2 = Downsloping"]
        )
        slope = int(slope_label[0])

        # Number of major vessels
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])

        # Thalassemia type
        thal_label = st.selectbox(
            "Thalassemia (thal)",
            ["0 = Normal", "1 = Fixed Defect",
             "2 = Reversible Defect", "3 = Unknown"]
        )
        thal = int(thal_label[0])

        # Submit button
        submitted = st.form_submit_button("🩺 Assess Heart Disease Risk")

    # Run prediction after form submission
    if submitted:

        # Convert inputs to NumPy array
        input_data = np.array([
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]).reshape(1, -1)

        # Predict probability of heart disease
        risk_prob = model.predict_proba(
            scaler.transform(input_data)
        )[0][1] * 100

        # Interpret risk level
        risk_text, _ = interpret_risk(risk_prob)

        # Store report in session state
        st.session_state.setdefault("report_history", [])
        st.session_state["report_history"].append({
            "patient_name": patient_name or "Not Provided",
            "timestamp": datetime.now(),
            "risk": round(risk_prob, 2),
            "risk_text": risk_text,
            "patient_data": {
                "Patient Name": patient_name or "Not Provided",
                "Age": age,
                "Sex": sex_label
            }
        })

        # Display results
        st.metric("Predicted Risk (%)", f"{risk_prob:.2f}%")
        st.info(f"Clinical Interpretation: **{risk_text}**")
        st.warning(doctor_recommendation(risk_prob))

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 2: REPORTS
# ============================================================

with tabs[1]:

    # Styled container
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    # Section title
    st.subheader("🗂️ Recent Reports (Last 10)")

    # Check if any reports exist
    if "report_history" in st.session_state:

        # Show last 10 reports
        for i, report in enumerate(st.session_state["report_history"][-10:][::-1]):

            # Expandable section per report
            with st.expander(report["patient_name"]):

                # Generate PDF
                pdf = generate_pdf(report)

                # Download button
                st.download_button(
                    "⬇️ Download PDF",
                    pdf,
                    file_name="heart_report.pdf",
                    mime="application/pdf",
                    key=i
                )
    else:
        # No reports message
        st.info("No reports generated yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 3: MODEL INSIGHTS
# ============================================================

with tabs[2]:

    # Styled container
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    # Compute feature importance from model coefficients
    importance = pd.Series(model.coef_[0], index=X.columns).sort_values()

    # Plot horizontal bar chart
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)

    # Display plot
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 4: ABOUT
# ============================================================

with tabs[3]:

    # Styled container
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    # About content
    st.write("""
    **Project Highlights**
    - End-to-end Machine Learning healthcare system  
    - Probability-based prediction  
    - Explainable AI  
    - Automated PDF reports  

    **Developer:** Atharva Savant  
    **Email:** atharvasavant2506@gmail.com  
    **GitHub:** https://github.com/atharva-dev-ai  

    ⚠️ Educational use only. Not a medical diagnosis.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
