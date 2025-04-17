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
        "Do you have high blood pressure? (yes/no)",
        "Do you have high cholesterol? (yes/no)",
        "What is your BMI? (e.g., 23.5)",
        "Have you ever had a stroke? (yes/no)",
        "Do you have heart disease or had a heart attack? (yes/no)",
        "Have you done any physical activity in the past 30 days? (yes/no)",
        "How would you rate your general health? (1=Excellent to 5=Poor)",
        "In the past 30 days, how many days was your mental health not good? (0–30)",
        "In the past 30 days, how many days was your physical health not good? (0–30)",
        "Do you have difficulty walking or climbing stairs? (yes/no)",
        "Select your age category (1–13):",
        "Select your income category (1–8):"
    ]

    # Show previous chat messages
    for i in range(st.session_state.chat_stage):
        st.chat_message("assistant").write(questions[i])
        st.chat_message("user").write(st.session_state.chat_data[i])

    # Ask current question
    if st.session_state.chat_stage < len(questions):
        question = questions[st.session_state.chat_stage]
        with st.chat_message("assistant"):
            st.write(question)

        user_input = st.chat_input("Your answer:")
        if user_input:
            st.chat_message("user").write(user_input)
            st.session_state.chat_data.append(user_input.strip().lower())

            # Improved input parser
            def parse_input(idx, value):
                value = value.strip().lower()
                yes_vals = ['yes', '1', 'y', 'yeah', 'yep']
                no_vals = ['no', '0', 'n', 'nope']

                if idx in [0, 1, 3, 4, 5, 9]:
                    if value in yes_vals:
                        return 1
                    elif value in no_vals:
                        return 0
                    else:
                        return 0  # default fallback

                elif idx == 2:  # BMI
                    try:
                        return float(value)
                    except:
                        return 25.0

                elif idx in [6, 10, 11]:  # GenHlth, Age, Income
                    try:
                        return int(value)
                    except:
                        return 3

                elif idx in [7, 8]:  # MentHlth, PhysHlth
                    try:
                        return min(max(int(value), 0), 30)
                    except:
                        return 5

                return 0

            if len(st.session_state.chat_data) == st.session_state.chat_stage + 1:
                st.session_state.chat_stage += 1

            # When all answers are collected
            if st.session_state.chat_stage == len(questions):
                final_input = np.array([[parse_input(i, ans) for i, ans in enumerate(st.session_state.chat_data)]])
                prediction = model.predict(final_input)[0]
                prob = model.predict_proba(final_input)[0][1] * 100
                with st.chat_message("assistant"):
                    if prediction == 1:
                        st.error(f"⚠️ High risk of diabetes detected.\nProbability: {prob:.2f}%")
                    else:
                        st.success(f"✅ Low risk of diabetes.\nProbability: {prob:.2f}%")
