import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("🫀 Heart Disease Prediction App")
st.write("Predict the likelihood of heart disease using patient medical data.")

# Load dataset
data = pd.read_csv("heart_disease_data.csv")

X = data.drop("target", axis=1)
y = data["target"]

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Sidebar inputs
st.sidebar.header("Enter Patient Details")

def user_input():
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(col, float(X[col].min()), float(X[col].max()))
    return pd.DataFrame(inputs, index=[0])

input_df = user_input()

# Prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

st.subheader("Prediction Result")

if prediction == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")
