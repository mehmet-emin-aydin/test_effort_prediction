import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# Streamlit title
st.title("Effort Estimation Application")

# Collect data from the user
st.header("Please enter the data")

# Numeric inputs (data collected from the user)
rely = st.number_input("Required Software Reliability (rely)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
data = st.number_input("Database Size (data)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
cplx = st.number_input("Process Complexity (cplx)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
time = st.number_input("Time Constraint for CPU (time)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
stor = st.number_input("Main Memory Constraint (stor)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
virt = st.number_input("Machine Volatility (virt)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
turn = st.number_input("Turnaround Time (turn)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
acap = st.number_input("Analysts Capability (acap)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
aexp = st.number_input("Application Experience (aexp)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
pcap = st.number_input("Programmers Capability (pcap)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
vexp = st.number_input("Virtual Machine Experience (vexp)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
lexp = st.number_input("Language Experience (lexp)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
modp = st.number_input("Modern Programming Practices (modp)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
tool = st.number_input("Use of Software Tools (tool)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
sced = st.number_input("Schedule Constraint (sced)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
loc = st.number_input("Lines of Code (loc)", min_value=0, max_value=100000, step=10, value=25)

# Categorical input for dev_mode
dev_mode = st.selectbox("Dev Mode", options=["embedded", "organic", "semidetached"], index=1)  # Default 'organic'

# Encoding dev_mode
dev_mode_dict = {"embedded": [1, 0, 0], "organic": [0, 1, 0], "semidetached": [0, 0, 1]}
dev_mode_encoded = dev_mode_dict[dev_mode]

# Scaling numeric data
scaler = joblib.load('numerical_scaler.pkl')
scaled_data = scaler.transform([[rely, data, cplx, time, stor, virt, turn, acap, aexp, pcap, vexp, lexp, modp, tool, sced, loc]])

# Combine user data into a single format
user_input = np.concatenate([scaled_data[0], dev_mode_encoded])

# Define column names
column_names = [
    "rely", "data", "cplx", "time", "stor", "virt", "turn", "acap", "aexp", "pcap", "vexp",
    "lexp", "modp", "tool", "sced", "loc", "dev_mode_embedded", "dev_mode_organic", "dev_mode_semidetached"
]

# Convert data to DataFrame for visualization
user_input_df = pd.DataFrame([user_input], columns=column_names)

# Load models
svr_model = joblib.load('svr_model.pkl')
ann_model = joblib.load('ann_model.pkl')
dt_model = joblib.load('dt_model.pkl')

# Model selection
st.header("Model Selection")
models = st.multiselect(
    "Which models would you like to use?",
    options=["SVR", "ANN", "DecisionTreeRegressor"],
    default=["SVR"]  # Default selection
)

# Estimate button
if models:
    estimate_button = st.button("Estimate")
else:
    estimate_button = False

scaler_y = joblib.load('scaler_y.pkl')

# Perform predictions and visualize results
if estimate_button:
    # Predictions based on selected models
    predictions = {}
    if "SVR" in models:
        svr_prediction = svr_model.predict([user_input])
        svr_prediction = scaler_y.inverse_transform(svr_prediction.reshape(-1, 1))
        predictions["SVR"] = svr_prediction[0][0]
    if "ANN" in models:
        ann_prediction = ann_model.predict([user_input])
        ann_prediction = scaler_y.inverse_transform(ann_prediction.reshape(-1, 1))
        predictions["ANN"] = ann_prediction[0][0]
    if "DecisionTreeRegressor" in models:
        dt_prediction = dt_model.predict([user_input])
        dt_prediction = scaler_y.inverse_transform(dt_prediction.reshape(-1, 1))
        predictions["DecisionTreeRegressor"] = dt_prediction[0][0]

    # Display predictions
    st.header("Prediction Results")
    st.write(predictions)

    # Bar chart visualization
    fig, ax = plt.subplots()
    ax.bar(predictions.keys(), predictions.values(), color='skyblue')
    ax.set_xlabel('Model')
    ax.set_ylabel('Prediction Result')
    ax.set_title('Prediction Results for Selected Models')
    st.pyplot(fig)
