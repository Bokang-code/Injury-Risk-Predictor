import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load('RFmodel.pkl')

# Streamlit page config
st.set_page_config(page_title="Injury Risk Predictor", page_icon="🩹", layout="centered")

# App Title
st.title('🩹 Injury Risk Predictors')

st.markdown("Fill in the player’s information below:")

# Inputs for all features
gender = st.selectbox('Gender', ('Male', 'Female'))
gender_display = gender 
gender = 0 if gender == 'Male' else 1


weight_kg = st.number_input('Weight (kg)', min_value=30, max_value=150, value=75)

position = st.selectbox('Position', ('Guard', 'Forward', 'Center'))
position_display = position
position_mapping = {'Guard': 0, 'Forward': 1, 'Center': 2}
position = position_mapping[position]

training_intensity = st.slider('Training Intensity (1-10)', 1, 10, 5)
training_hours_per_week = st.slider('Training Hours per Week', 0, 20, 10)
recovery_days_per_week = st.slider('Recovery Days per Week', 0, 7, 2)
match_count_per_week = st.slider('Match Count per Week', 0, 5, 2)
rest_between_events_days = st.slider('Rest Between Events (days)', 0, 5, 2)
fatigue_score = st.slider('Fatigue Score (1-10)', 1, 10, 5)
performance_score = st.slider('Performance Score (0-100)', 0, 100, 70)
team_contribution_score = st.slider('Team Contribution Score (0-100)', 0, 100, 70)


# Combine all features into one array
features = np.array([[gender, weight_kg, position, training_intensity,
                      training_hours_per_week, recovery_days_per_week, match_count_per_week,
                      rest_between_events_days, fatigue_score, performance_score,
                      team_contribution_score]])

# Predict when user clicks button
if st.button('Predict Injury Risk'):
    with st.spinner('Predicting Injury Risk...'):
        probability = model.predict_proba(features)[0][1]
        risk_percentage = probability * 100

        # Apply manual adjustments
        if fatigue_score >= 8:
            risk_percentage += 20  # Very high fatigue boosts risk
        if recovery_days_per_week <= 3:
            risk_percentage += 20  # Very low recovery boosts risk
        if training_hours_per_week >= 15:
            risk_percentage += 10   # Overtraining slightly boosts risk
        
        # Clip risk percentage to maximum 100
        risk_percentage = min(risk_percentage, 100)

        # Show the result
        st.subheader("Prediction Result:")
        
        if risk_percentage < 20:
            st.success(f"✅ Low Risk of Injury ({risk_percentage:.2f}%)")
        elif 20 <= risk_percentage < 40:
            st.warning(f"⚠️ Medium Risk of Injury ({risk_percentage:.2f}%)")
        elif 40 <= risk_percentage < 60:
            st.warning(f"⚠️ High Risk of Injury ({risk_percentage:.2f}%)")
        else:
            st.error(f"‼️ Very High Risk of Injury ({risk_percentage:.2f}%)")
