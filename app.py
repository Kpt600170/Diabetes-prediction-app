import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Feature names
features_list = ['HighBP', 'HighChol', 'BMI', 'Stroke', 'HeartDiseaseorAttack',
                 'PhysActivity', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Income']

# Initialize session state for chatbot
if 'chat_stage' not in st.session_state:
    st.session_state.chat_stage = 0
if 'chat_data' not in st.session_state:
    st.session_state.chat_data = []
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = False

# Title and toggle
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("This app predicts diabetes risk using a machine learning model.")
st.session_state.chat_mode = st.toggle("💬 Switch to Chatbot Mode", value=st.session_state.chat_mode)

if not st.session_state.chat_mode:
    # -------- Classic Sidebar Input Mode --------
    st.sidebar.header("Input Health Details")
    def user_input_features():
        HighBP = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
        HighChol = st.sidebar.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "Yes" if x else "No")
        BMI = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
        Stroke = st.sidebar.selectbox("Ever had a Stroke", [0, 1], format_func=lambda x: "Yes" if x else "No")
        HeartDiseaseorAttack = st.sidebar.selectbox("Heart Disease/Attack", [0, 1], format_func=lambda x: "Yes" if x else "No")
        PhysActivity = st.sidebar.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
        GenHlth = st.sidebar.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
        MentHlth = st.sidebar.slider("Mental Health Days (0–30)", 0, 30, 5)
        PhysHlth = st.sidebar.slider("Physical Health Days (0–30)", 0, 30, 5)
        DiffWalk = st.sidebar.selectbox("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x else "No")
        Age = st.sidebar.slider("Age Category (1–13)", 1, 13, 5)
        Income = st.sidebar.slider("Income Category (1–8)", 1, 8, 4)

        return np.array([[HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack, PhysActivity,
                          GenHlth, MentHlth, PhysHlth, DiffWalk, Age, Income]])

    input_data = user_input_features()
    if st.button("Predict Diabetes Risk"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        if prediction == 1:
            st.error(f"High risk of diabetes detected.\nProbability: {prob:.2f}%")
        else:
            st.success(f"Low risk of diabetes.\nProbability: {prob:.2f}%")

else:
    # -------- Chatbot Mode --------
    questions = [
        "Do you have high blood pressure?",
        "Do you have high cholesterol?",
        "What is your BMI? (e.g., 23.5)",
        "Have you ever had a stroke?",
        "Do you have heart disease or had a heart attack?",
        "Have you done any physical activity in the past 30 days?",
        "Rate your general health (1=Excellent to 5=Poor):",
        "In the past 30 days, how many days was your mental health not good? (0–30)",
        "In the past 30 days, how many days was your physical health not good? (0–30)",
        "Do you have difficulty walking or climbing stairs?",
        "Select your age category (1–13):",
        "Select your income category (1–8):"
    ]

    # Reset Chat
    if st.button("🔄 Start Over"):
        st.session_state.chat_stage = 0
        st.session_state.chat_data = []
        st.experimental_rerun()

    # Show chat history
    for i in range(st.session_state.chat_stage):
        st.chat_message("assistant").write(questions[i])
        st.chat_message("user").write(st.session_state.chat_data[i])

    if st.session_state.chat_stage < len(questions):
        q = questions[st.session_state.chat_stage]

        with st.chat_message("assistant"):
            st.write(q)

        def record_response(answer):
            st.session_state.chat_data.append(str(answer).strip().lower())
            st.session_state.chat_stage += 1
            st.experimental_rerun()

        # Input options based on question type
        if st.session_state.chat_stage in [0, 1, 3, 4, 5, 9]:  # Yes/No questions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes"):
                    record_response("yes")
            with col2:
                if st.button("No"):
                    record_response("no")

        elif st.session_state.chat_stage == 2:  # BMI
            bmi = st.slider("Select your BMI:", 10.0, 60.0, 25.0)
            if st.button("Submit"):
                record_response(bmi)

        elif st.session_state.chat_stage == 6:  # General Health
            gen = st.selectbox("Rate your general health:", list(range(1, 6)))
            if st.button("Submit"):
                record_response(gen)

        elif st.session_state.chat_stage in [7, 8]:  # Mental/Physical Health Days
            days = st.slider("Number of days (0–30):", 0, 30, 5)
            if st.button("Submit"):
                record_response(days)

        elif st.session_state.chat_stage == 10:  # Age
            age = st.selectbox("Select your age category (1–13):", list(range(1, 14)))
            if st.button("Submit"):
                record_response(age)

        elif st.session_state.chat_stage == 11:  # Income
            inc = st.selectbox("Select your income category (1–8):", list(range(1, 9)))
            if st.button("Submit"):
                record_response(inc)

    # After all responses
    if st.session_state.chat_stage == len(questions):
        def parse_input(idx, value):
            yes_vals = ['yes', '1', 'y', 'yeah', 'yep']
            if idx in [0, 1, 3, 4, 5, 9]:
                return 1 if value in yes_vals else 0
            elif idx == 2:
                return float(value)
            elif idx in [6, 10, 11]:
                return int(value)
            elif idx in [7, 8]:
                return min(max(int(value), 0), 30)
            return 0

        final_input = np.array([[parse_input(i, val) for i, val in enumerate(st.session_state.chat_data)]])
        prediction = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][1] * 100

        with st.chat_message("assistant"):
            if prediction == 1:
                st.error(f"⚠️ High risk of diabetes detected.\nProbability: {prob:.2f}%")
            else:
                st.success(f"✅ Low risk of diabetes.\nProbability: {prob:.2f}%")
