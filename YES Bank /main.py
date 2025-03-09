import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMAResults, ARIMAResultsWrapper  # Ensure compatibility

# ================================
# 🔹 Load ARIMA model from `.sav`
# ================================
model_filename = 'best_arima_1.sav'  # ✅ Using .sav format

@st.cache_resource
def load_model():
    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)  # ✅ Load model

        # ✅ Convert ARIMAResultsWrapper to ARIMAResults
        if isinstance(model, ARIMAResultsWrapper):  
            model = model._results  # Extract the actual ARIMAResults

        st.write(f"✅ Model Loaded Type (after conversion): {type(model)}")  # Debugging output

        if not isinstance(model, ARIMAResults):
            st.error(f"❌ Incorrect model type: {type(model)}. Expected ARIMAResults.")
            st.stop()

        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Ensure 'best_arima.sav' exists in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load model
loaded_arima = load_model()

# ================================
# 🔹 Load stock dataset
# ================================
df = pd.read_csv("data_YesBank_StockPrices.csv", parse_dates=["Date"], index_col="Date")
df.index = pd.to_datetime(df.index, errors='coerce')  # Ensure DateTime format
df = df[df.index.notna()]  # Remove NaT values

# Handle missing or incorrect last date
if df.index.empty or pd.isnull(df.index[-1]):
    last_valid_date = pd.Timestamp("2023-12-31")  # Default fallback
else:
    last_valid_date = df.index[-1]

# ================================
# 🔹 Streamlit UI
# ================================
st.title("📈 ARIMA Stock Price Prediction App")
st.write("This app predicts future stock closing prices using an ARIMA model.")

# User selects the number of months to predict
future_steps = st.slider("Select months to predict", min_value=1, max_value=24, value=12)

# ================================
# 🔹 Generate Future Dates
# ================================
future_dates = pd.date_range(start=last_valid_date, periods=future_steps, freq='M')

# ================================
# 🔹 Predict Future Prices
# ================================
try:
    # ✅ Make prediction (Removed .fit())
    predictions = loaded_arima.predict(start=len(df), end=len(df) + future_steps - 1)

    # Handle NaN values in predictions
    if predictions.isna().any():
        st.error("❌ ARIMA model returned NaN values. Ensure the model was trained correctly.")
        st.stop()

except Exception as e:
    st.error(f"❌ Error during forecasting: {e}")
    st.stop()

# Store predictions in DataFrame
results_df = pd.DataFrame({'Date': future_dates, 'ARIMA_Predicted_Close': predictions.values})
results_df.set_index('Date', inplace=True)

# ================================
# 🔹 Display Predictions in Streamlit
# ================================
st.subheader("📊 Predicted Stock Prices")
st.dataframe(results_df)

# ================================
# 🔹 Plot Predictions
# ================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, predictions, marker='o', linestyle='-', color='blue', label="ARIMA Predicted Close")
ax.set_title("ARIMA Future Stock Price Predictions")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Closing Price")
ax.legend()
ax.grid(True)

# Show plot in Streamlit
st.pyplot(fig)
st.success("✅ Prediction completed! Adjust parameters to explore more.")
