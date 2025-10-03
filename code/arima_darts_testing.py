import pandas as pd
import matplotlib.pyplot as plt
from darts.datasets import AirPassengersDataset
from darts.models import ARIMA
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pdb

# Load the dataset
series = AirPassengersDataset().load()

# Optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)

# Define ARIMA parameters
model = ARIMA(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12))
model.fit(series, future_covariates=future_cov)

# Make predictions with confidence intervals
forecast_steps = 6
pred = model.predict(
    forecast_steps, future_covariates=future_cov
)  # set n to the number of simulations

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
