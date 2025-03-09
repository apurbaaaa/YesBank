import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 🔹 Load the trained XGBoost model
# ================================
model_filename = 'best_xgb_model.pkl'  # Ensure this is the correct model filename
with open(model_filename, 'rb') as file:
    loaded_xgb = pickle.load(file)

# ================================
# 🔹 Load stock dataset
# ================================
df = pd.read_csv("data_YesBank_StockPrices.csv", parse_dates=["Date"], index_col="Date")

# Ensure DateTime Index is properly formatted
df.index = pd.to_datetime(df.index, errors='coerce')  # Convert index to datetime
df = df[df.index.notna()]  # Remove NaT (invalid timestamps)

# Handle cases where the last date is missing or too old
if df.index.empty or pd.isnull(df.index[-1]) or df.index[-1].year < 2000:
    last_valid_date = pd.Timestamp("2023-12-31")  # Default to recent date if missing
else:
    last_valid_date = df.index[-1]

# ================================
# 🔹 Define trained features
# ================================
trained_features = ['Avg_Price', 'Range', 'Prev_Close_1', 'Prev_Close_3', 'Prev_Close_6']

# ================================
# 🔹 Streamlit UI
# ================================
st.title("📈 Stock Price Prediction App")
st.write("This app predicts the future closing prices of a stock using a trained XGBoost model.")

# User input: Number of months to predict
future_steps = st.slider("Select the number of months to predict", min_value=1, max_value=24, value=12)

# ================================
# 🔹 Generate Future Data
# ================================
future_dates = pd.date_range(start=last_valid_date, periods=future_steps+1, freq='ME')[1:]

# Create DataFrame for future predictions
unseen_df = pd.DataFrame(index=future_dates, columns=trained_features)

# Copy last known values for features
for col in trained_features:
    if col in df.columns:
        unseen_df[col] = df[col].iloc[-1]  # Use last known values

# Fill missing values
unseen_df.fillna(0, inplace=True)

# ================================
# 🔹 Predict Future Prices (Recursive Forecasting)
# ================================
predictions = []
for date in future_dates:
    # Ensure only trained features are passed
    input_features = unseen_df.loc[[date], trained_features]

    # Predict closing price
    pred = loaded_xgb.predict(input_features)[0]
    predictions.append(pred)

    # Update 'Prev_Close_1' for next prediction
    if 'Prev_Close_1' in unseen_df.columns:
        unseen_df.loc[date, 'Prev_Close_1'] = pred

# Store predictions in the DataFrame
unseen_df['XGB_Predicted_Close'] = predictions

# ================================
# 🔹 Display Results in Streamlit
# ================================
st.subheader("📊 Predicted Stock Prices")
st.dataframe(unseen_df[['XGB_Predicted_Close']])

# ================================
# 🔹 Plot Predictions
# ================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, predictions, marker='o', linestyle='-', color='red', label="Predicted Close Price")
ax.set_title("Future Stock Price Predictions")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Closing Price")
ax.legend()
ax.grid(True)

# Show plot in Streamlit
st.pyplot(fig)

st.success("✅ Prediction completed! Adjust parameters to explore more.")
