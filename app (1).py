import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details:")

gender = st.selectbox("Gender (0=Female,1=Male)", [0,1])
age = st.number_input("Age")
hypertension = st.selectbox("Hypertension", [0,1])
heart_disease = st.selectbox("Heart Disease", [0,1])
smoking_history = st.selectbox("Smoking History", [0,1,2,3,4,5])
bmi = st.number_input("BMI")
HbA1c_level = st.number_input("HbA1c Level")
blood_glucose_level = st.number_input("Blood Glucose Level")

st.write("Model Accuracy: ~96%")

if st.button("Predict"):
    data = np.array([[gender, age, hypertension, heart_disease,
                      smoking_history, bmi, HbA1c_level,
                      blood_glucose_level]])

    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")
