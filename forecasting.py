# modules/forecasting.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Optional Prophet support
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def prepare_time_series(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly sales time series."""
    if "InvoiceDate" not in data.columns:
        raise ValueError("InvoiceDate column required for time series.")
    df = (
        data.groupby(pd.Grouper(key="InvoiceDate", freq="M"))["TotalRevenue"]
        .sum()
        .reset_index()
        .rename(columns={"InvoiceDate": "ds", "TotalRevenue": "y"})
    )
    return df


def decompose_trend(ts_df: pd.DataFrame):
    """Decompose sales trend and seasonality."""
    ts_df = ts_df.set_index("ds")
    result = seasonal_decompose(ts_df["y"], model="additive", period=12)
    return result


def arima_forecast(ts_df: pd.DataFrame, steps: int = 6):
    """Simple ARIMA baseline forecast."""
    model = ARIMA(ts_df["y"], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast_df = pd.DataFrame({
        "ds": pd.date_range(ts_df["ds"].iloc[-1] + pd.offsets.MonthBegin(), periods=steps, freq="M"),
        "yhat": forecast.values,
    })
    return forecast_df


def prophet_forecast(ts_df: pd.DataFrame, steps: int = 6):
    """Prophet-based forecast (if available)."""
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Install it with `pip install prophet`.")

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=steps, freq="M")
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return forecast


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute forecast evaluation metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
