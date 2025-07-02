import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("PowerPulse: Household Energy Usage Forecast")

# Project Overview
st.markdown("""
### ✅ The main goals are:
- Enable **better energy planning** for households and providers
- Support **cost reduction** through smart usage insights
- Help **energy companies forecast demand** more accurately

By analyzing past energy usage, this model gives actionable insights that benefit both consumers and providers.

---

### 🧾 Input Features Explanation

1. **Hour**
   - Represents the hour of the day from 0 to 23.
   - Example: 21 = 9 PM, typically peak usage time in households.

2. **Day of Month**
   - The calendar day of the month (1 to 31).
   - Some days (e.g., holidays, salary dates) may have higher consumption.

3. **Weekday**
   - Day of the week: 0 = Monday, 6 = Sunday.
   - Weekends often have higher usage since people stay home more.

4. **Global Reactive Power**
   - Power that is stored and released by appliances (not directly consumed).
   - Affects overall power patterns.

5. **Voltage**
   - The electric potential in volts (usually around 220V–240V).
   - Fluctuations in voltage can impact appliance performance.

6. **Global Intensity**
   - The current drawn in amperes.
   - Higher current usually indicates more power consumption.

---
""")

# Sidebar Inputs
st.sidebar.header("Enter Input Features")

hour = st.sidebar.slider("Hour (0-23)", 0, 23, 18)
day = st.sidebar.slider("Day of Month (1-31)", 1, 31, 15)
weekday = st.sidebar.selectbox("Weekday (0=Mon, 6=Sun)", list(range(7)))

global_reactive_power = st.sidebar.number_input("Global Reactive Power (kW)", min_value=0.0, max_value=5.0, value=0.1)
voltage = st.sidebar.number_input("Voltage (V)", min_value=200.0, max_value=260.0, value=240.0)
global_intensity = st.sidebar.number_input("Global Intensity (ampere)", min_value=0.0, max_value=30.0, value=10.0)

# Prepare input for prediction
input_data = pd.DataFrame({
    'hour': [hour],
    'day': [day],
    'weekday': [weekday],
    'Global_reactive_power': [global_reactive_power],
    'Voltage': [voltage],
    'Global_intensity': [global_intensity]
})

# Debug: show input
st.markdown("### 🔍 Input Preview")
st.dataframe(input_data)

# Load pre-trained model
@st.cache_resource
def load_model():
    path = "power_model.pkl"
    return joblib.load(path)

model = load_model()

# Predict
if st.button("Predict Energy Usage"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Global Active Power: {prediction:.3f} kW")
