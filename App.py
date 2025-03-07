import streamlit as st
import pandas as pd
import numpy as np
import base64
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
#from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import ReLU



# Load trained LSTM model and scaler
model = load_model("CNC_LSTM_model.h5")
scaler = joblib.load("scaler.pkl")


# Function to add background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("CNC.jpg")

# Define numerical columns used during training
numerical_cols = [
    "X1_ActualVelocity", "X1_ActualAcceleration", "X1_OutputPower",
    "Y1_ActualVelocity", "Y1_ActualAcceleration", "Y1_OutputPower",
    "Z1_ActualVelocity", "Z1_ActualAcceleration", "Z1_OutputVoltage",
    "S1_ActualVelocity", "S1_ActualAcceleration", "S1_OutputPower",
    "M1_CURRENT_FEEDRATE"
]

# Streamlit UI
st.title("🔧 CNC Milling Performance Prediction")
st.write("Upload an experiment CSV file to predict Tool Condition, Machining Finalized, and Passed Visual Inspection.")

# User inputs
feedrate = st.number_input("Enter Feed Rate (mm/s):", min_value=0.0, step=0.1)
clamp_pressure = st.number_input("Enter Clamp Pressure (bar):", min_value=0.0, step=0.1)
material = st.selectbox("Select Material:", ["Wax", "Other"])
material_encoded = 1 if material == "Wax" else 0

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Check if required columns exist
    missing_cols = [col for col in numerical_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
    else:
        # Extract mean values for each feature
        df_mean = df[numerical_cols].mean().to_frame().T
        
        # Add manual inputs
        df_mean["feedrate"] = feedrate
        df_mean["clamp_pressure"] = clamp_pressure
        df_mean["material"] = material_encoded
        
        # Normalize using saved MinMaxScaler
        df_mean_scaled = scaler.transform(df_mean)
        
        # Reshape for LSTM input
        X = df_mean_scaled.reshape(1, 1, df_mean.shape[1])
        
        # Make predictions
        predictions = model.predict(X)
        tool_condition = "Worn" if predictions[0][0] > 0.5 else "Unworn"
        machining_finalized = "Completed" if predictions[0][1] > 0.5 else "Not Completed"
        passed_visual_inspection = "Passed" if predictions[0][2] > 0.5 else "Failed"
        
        # Display results
        st.subheader("🔍 Prediction Results:")
        st.write(f"🛠 **Tool Condition:** {tool_condition}")
        st.write(f"⚙ **Machining Finalized:** {machining_finalized}")
        st.write(f"🔎 **Passed Visual Inspection:** {passed_visual_inspection}")

