import streamlit as st
import numpy as np
import joblib

# Load the trained XGBoost model
model = joblib.load("best_model.pkl")

# Set page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# Title
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("This app uses a machine learning model to predict the likelihood of diabetes based on health indicators.")

# Sidebar for input
st.sidebar.header("Input Health Details")

def user_input_features():
    st.sidebar.subheader("🔹 High Blood Pressure")
    bp_input_mode = st.sidebar.radio("Input Method for BP", ["Yes/No", "Numeric"], key="bp_mode")
    if bp_input_mode == "Yes/No":
        HighBP = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
    else:
        systolic = st.sidebar.number_input("Systolic (mm Hg)", min_value=50, max_value=250, value=120)
        diastolic = st.sidebar.number_input("Diastolic (mm Hg)", min_value=30, max_value=150, value=80)
        HighBP = 1 if systolic >= 130 or diastolic >= 80 else 0

    st.sidebar.subheader("🔹 High Cholesterol")
    chol_input_mode = st.sidebar.radio("Input Method for Cholesterol", ["Yes/No", "Numeric"], key="chol_mode")
    if chol_input_mode == "Yes/No":
        HighChol = st.sidebar.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "Yes" if x else "No")
    else:
        cholesterol_level = st.sidebar.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
        HighChol = 1 if cholesterol_level >= 200 else 0

    BMI = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
    Stroke = st.sidebar.selectbox("Ever had a Stroke", [0, 1], format_func=lambda x: "Yes" if x else "No")
    HeartDiseaseorAttack = st.sidebar.selectbox("Heart Disease or Heart Attack", [0, 1], format_func=lambda x: "Yes" if x else "No")
    PhysActivity = st.sidebar.selectbox("Physical Activity in Past 30 Days", [0, 1], format_func=lambda x: "Yes" if x else "No")
    GenHlth = st.sidebar.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
    MentHlth = st.sidebar.slider("Mental Health (Days unwell in last 30)", 0, 30, 5)
    PhysHlth = st.sidebar.slider("Physical Health (Days unwell in last 30)", 0, 30, 5)
    DiffWalk = st.sidebar.selectbox("Difficulty Walking or Climbing Stairs", [0, 1], format_func=lambda x: "Yes" if x else "No")
    Age = st.sidebar.slider("Age Category", 1, 13, 5)
    Income = st.sidebar.slider("Income Category (1=Low, 8=High)", 1, 8, 4)

    features = np.array([[HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack, PhysActivity,
                          GenHlth, MentHlth, PhysHlth, DiffWalk, Age, Income]])
    return features

# Predict button
input_data = user_input_features()
if st.button("Predict Diabetes Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes detected.\nProbability: {prob:.2f}%")
    else:
        st.success(f"✅ Low risk of diabetes.\nProbability: {prob:.2f}%")

# Footer
st.markdown("---")
st.markdown("📊 **Model**: Tuned XGBoost | 🔧 Trained on BRFSS 2015 Dataset")
