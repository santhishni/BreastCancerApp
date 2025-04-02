
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split data
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'breast_cancer_model.pkl')

# Streamlit UI
st.set_page_config(page_title='Breast Cancer Predictor', page_icon='🎗', layout='centered')
st.markdown("""
    <style>
        .main { background-color: #ffe6f2; }
        .stButton>button { background-color: #ff6699; color: white; font-size: 18px; }
        .stTextInput>label { color: #ff3366; }
    </style>
""", unsafe_allow_html=True)

st.title('Breast Cancer Prediction App 🎗')
st.write("Early detection saves lives. Enter the details to get a prediction.")

# Initialize session state for resetting
if "inputs" not in st.session_state:
    st.session_state.inputs = [0.0] * 30  # Default values

# Input fields
user_input = []
for i, feature in enumerate(data.feature_names[:30]):
    user_value = st.number_input(f'Enter {feature}', min_value=0.0, key=f"input_{i}", value=st.session_state.inputs[i])
    user_input.append(user_value)

if st.button('Predict'):
    model = joblib.load('breast_cancer_model.pkl')
    
    probabilities = model.predict_proba([user_input])[0]
    malignant_prob = probabilities[0] * 100  # Probability of Malignant
    benign_prob = probabilities[1] * 100    # Probability of Benign

    if malignant_prob > benign_prob:
        result = 'Malignant (Cancerous)'
        confidence = malignant_prob
    else:
        result = 'Benign (Non-cancerous)'
        confidence = benign_prob

    st.subheader(f'Result: {result}')
    st.write(f'Confidence: {confidence:.2f}%')

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(['Benign', 'Malignant'], [benign_prob, malignant_prob], color=['green', 'red'])
    ax.set_ylabel('Probability (%)')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

# Reset button functionality
if st.button('Reset'):
    st.session_state.clear()  # Clears all session state variables
    st.rerun()  # Refreshes the page