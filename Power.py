import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    file_id = '1trgBEubuq7bKp7tmL_MufkXnG6wGuqE2'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Read CSV - confirmed tab-separated
    df = pd.read_csv(url, sep='\t', encoding='utf-8', engine='python')

    # ✅ Convert Date + Time ➝ Datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract useful features
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day

    return df   
df = load_data()

# Sidebar - Model Selector
model_name = st.sidebar.selectbox(
    "Select Your Model",
    ("Linear Regression", "Random Forest", "Gradient Boosting")
)

st.title("🏠 PowerPulse Dashboard")
st.subheader("📊 Forecasting home electricity usage")

# Show basic data info
if st.checkbox("View Data"):
    st.dataframe(df.head())

# Prepare Data
X = df[['Year', 'Month', 'Day']]
y = df['Global_active_power']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train selected model
if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Random Forest":
    model = RandomForestRegressor(random_state=42)
else:
    model = GradientBoostingRegressor(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("### 📈 Evaluation metrics:")
st.write(f"**RMSE**: {rmse:.2f}")
st.write(f"**MAE**: {mae:.2f}")
st.write(f"**R² Score**: {r2:.3f}")

# Plot: Actual vs Predicted
st.markdown("### 📉 Actual vs Predicted")
st.line_chart(pd.DataFrame({'Actual': y_test.values[:100], 'Predicted': y_pred[:100]}))

# Feature Importance
if model_name != "Linear Regression":
    st.markdown("### 🔍 Feature Importance")
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(X.columns, importances, color='skyblue')
    ax.set_title("Feature Importance")
    st.pyplot(fig)
