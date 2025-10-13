import pandas as pd
import matplotlib.pyplot as plt
from darts.datasets import AirPassengersDataset
from darts.models import ARIMA, AutoARIMA
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pdb

# # Load the dataset
# series = AirPassengersDataset().load()
#
# # Optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
# future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
#
# # Define ARIMA parameters
# # model = ARIMA(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12))
# model = AutoARIMA(season_length=12)
# pdb.set_trace()
# model.fit(series, future_covariates=future_cov)
#
# # Make predictions with confidence intervals
# forecast_steps = 6
# pred = model.predict(
#     forecast_steps, future_covariates=future_cov
# )  # set n to the number of simulations
#
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

df = AirPassengersDF

# Split data into train/test
train_size = len(df) - 12  # Use last 12 months as test
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

# Fit on training data
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq='ME',
)
sf.fit(train_df)

# Predict for the test period
pred = sf.predict(h=12, level=[95])

# Create matplotlib figure
plt.figure(figsize=(12, 6))

# Plot historical training data
plt.plot(train_df['ds'], train_df['y'], label="Training Data", color="blue", linewidth=2)

# Plot actual test values
plt.plot(test_df['ds'], test_df['y'], label="Actual Test Values", color="green", linewidth=2)

# Plot predictions
plt.plot(pred['ds'], pred['AutoARIMA'], label="Forecast", color="red", linestyle="--", linewidth=2)

# Plot confidence intervals if available
if 'AutoARIMA-lo-95' in pred.columns and 'AutoARIMA-hi-95' in pred.columns:
    plt.fill_between(pred['ds'], pred['AutoARIMA-lo-95'], pred['AutoARIMA-hi-95'], 
                     alpha=0.3, color="red", label="95% Confidence Interval")

plt.title("ARIMA Forecast vs Actual Test Values")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Save the plot
plt.savefig("arima_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
pdb.set_trace()
# plt.show()

# Prepare the forecast index
forecast_index = pd.date_range(
    start=series.time_index[-1] + pd.DateOffset(months=1),
    periods=forecast_steps,
    freq="M",
)

# Plotting
plt.figure(figsize=(12, 6))

# Plot original time series
plt.plot(series.time_index, series.values(), label="Original", color="blue")

# Plot forecasted values
plt.plot(
    forecast_index, pred.values(), label="Forecast", color="orange", linestyle="--"
)

# Add labels and legend
plt.title("ARIMA Model Forecast for AirPassengers")
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
