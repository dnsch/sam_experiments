import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb


# Load the AirPassengers dataset
data = sm.datasets.get_rdataset("AirPassengers", "datasets", cache=True).data

# Convert the data to a time series
data["time"] = pd.date_range(start="1949-01-01", periods=len(data), freq="M")
data.set_index("time", inplace=True)

# Inspect the first few rows
print(data.head())

# Plot the original time series
plt.figure(figsize=(12, 6))
plt.plot(data["value"], label="Monthly Passengers", color="blue")
plt.title("AirPassengers Time Series")
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()

# Fit a Seasonal ARIMA model to the data
# (p, d, q) x (P, D, Q, s) where s=12 for monthly data
mod = sm.tsa.SARIMAX(data["value"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = mod.fit()

# Forecast the next steps
forecast_steps = 12  # Forecast for the next year
forecast = res.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(
    start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq="M"
)

# Plot the original time series and predictions
plt.figure(figsize=(12, 6))
plt.plot(data["value"], label="Original", color="blue")
plt.plot(
    forecast_index,
    forecast.predicted_mean,
    label="Forecast",
    color="orange",
    linestyle="--",
)

# Plot confidence intervals
conf_int = forecast.conf_int()
plt.fill_between(
    forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="orange", alpha=0.3
)

# Add labels and legend
plt.title("SARIMA Model Forecast for AirPassengers")
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
