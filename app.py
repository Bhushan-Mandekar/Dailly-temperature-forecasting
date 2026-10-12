import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Daily Temperature Forecast",
    page_icon="🌡️",
    layout="wide"
)

# Project directory
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# TITLE
# ============================================================

st.title("🌡️ Daily Minimum Temperature Forecasting")

st.markdown(
    """
    **Time-series forecasting using SARIMAX with Fourier annual seasonality**

    This dashboard explores historical daily minimum temperatures
    and provides forecasts for the next 30, 90, 180, or 365 days
    using the final **SARIMAX(3,0,1) + Fourier** model.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    csv_path = BASE_DIR / "daily_minimum_temperature_cleaned.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {"Date", "Temperature"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The dataset must contain 'Date' and 'Temperature' columns."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Temperature"] = pd.to_numeric(
        df["Temperature"], errors="coerce"
    )

    df = df.dropna(subset=["Date", "Temperature"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")

    if df.empty:
        raise ValueError("The dataset contains no valid observations.")

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load the dataset: {e}")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = BASE_DIR / "sarimax_fourier_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    return SARIMAXResults.load(str(model_path))


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained SARIMAX model: {e}")
    st.info(
        "Check that sarimax_fourier_model.pkl is present in the "
        "root of your GitHub repository and was saved as a "
        "statsmodels SARIMAX results object."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

forecast_days = st.sidebar.selectbox(
    "Forecast horizon",
    [30, 90, 180, 365],
    index=3
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: SARIMAX(3,0,1) + Fourier seasonality"
)


# ============================================================
# KEY METRICS
# ============================================================

st.subheader("📌 Project Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Observations",
    f"{len(df):,}"
)

col2.metric(
    "Mean Temperature",
    f"{df['Temperature'].mean():.2f} °C"
)

col3.metric(
    "Minimum Temperature",
    f"{df['Temperature'].min():.1f} °C"
)

col4.metric(
    "Maximum Temperature",
    f"{df['Temperature'].max():.1f} °C"
)


# ============================================================
# HISTORICAL DATA
# ============================================================

st.subheader("📈 Historical Daily Minimum Temperature")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df.index,
    df["Temperature"],
    linewidth=0.8,
    label="Daily Minimum Temperature"
)

ax.set_xlabel("Date")
ax.set_ylabel("Minimum Temperature (°C)")
ax.set_title("Historical Daily Minimum Temperature")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

st.subheader("📊 Historical Summary Statistics")

summary = df["Temperature"].describe().to_frame(
    "Temperature (°C)"
)

st.dataframe(
    summary,
    use_container_width=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🤖 Final Model Performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Final Model",
    "SARIMAX(3,0,1)"
)

metric2.metric(
    "MAE",
    "1.97 °C"
)

metric3.metric(
    "RMSE",
    "2.56 °C"
)

st.info(
    "The final model uses 5 Fourier harmonics to represent "
    "annual seasonality with a period of 365.25 days."
)

st.caption(
    "The MAE and RMSE displayed above are the reported evaluation "
    "values. Verify that they match your final 80–20 train-test "
    "evaluation results."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("🏆 Model Comparison")

comparison_path = (
    BASE_DIR / "forecast_model_comparison.csv"
)

if comparison_path.exists():

    try:
        comparison = pd.read_csv(comparison_path)

        st.dataframe(
            comparison,
            use_container_width=True
        )

        if {"Model", "RMSE"}.issubset(comparison.columns):

            comparison["RMSE"] = pd.to_numeric(
                comparison["RMSE"],
                errors="coerce"
            )

            valid_comparison = comparison.dropna(
                subset=["Model", "RMSE"]
            )

            if not valid_comparison.empty:

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.bar(
                    valid_comparison["Model"].astype(str),
                    valid_comparison["RMSE"]
                )

                ax.set_ylabel("RMSE (°C)")
                ax.set_xlabel("Model")
                ax.set_title("Forecasting Model Comparison")

                plt.xticks(
                    rotation=45,
                    ha="right"
                )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning(
                    "The comparison file has no valid RMSE values."
                )

        else:
            st.warning(
                "The comparison CSV must contain 'Model' and 'RMSE' "
                "columns to display the comparison chart."
            )

    except Exception as e:
        st.warning(
            f"Unable to display model comparison: {e}"
        )

else:
    st.warning(
        "Model comparison file not found. To display the comparison, "
        "add outputs/forecast_model_comparison.csv to your repository."
    )


# ============================================================
# FORECAST
# ============================================================

st.subheader("🔮 Temperature Forecast")

# Generate future dates
last_date = df.index.max()

future_index = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)


# ============================================================
# FOURIER FEATURES
# ============================================================

def create_fourier_terms(index, period=365.25, K=5):
    """
    Create future Fourier terms using a continuous time index.

    IMPORTANT:
    The time-index convention, period, number of harmonics,
    and column order must match those used during model training.
    """

    combined_index = df.index.append(index)

    t = np.arange(len(combined_index))

    fourier = pd.DataFrame(index=combined_index)

    for k in range(1, K + 1):
        fourier[f"sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )

        fourier[f"cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    return fourier.loc[index]


try:
    X_future = create_fourier_terms(
        future_index,
        period=365.25,
        K=5
    )

    # Generate forecast
    forecast_result = model.get_forecast(
        steps=forecast_days,
        exog=X_future
    )

    forecast = forecast_result.predicted_mean
    confidence = forecast_result.conf_int()

    # Align the forecast output to the future dates
    forecast_values = np.asarray(forecast).reshape(-1)
    lower_values = confidence.iloc[:, 0].to_numpy()
    upper_values = confidence.iloc[:, 1].to_numpy()

    if not (
        len(forecast_values)
        == len(lower_values)
        == len(upper_values)
        == len(future_index)
    ):
        raise ValueError(
            "The model forecast length does not match the selected horizon."
        )

except Exception as e:
    st.error(f"Unable to generate the forecast: {e}")
    st.info(
        "Check that the saved model was trained with five Fourier "
        "harmonics using the same period, feature names, column order, "
        "and time-index convention as this application."
    )
    st.stop()


# ============================================================
# FORECAST CHART
# ============================================================

st.subheader(
    f"📉 Next {forecast_days} Days Forecast"
)

fig, ax = plt.subplots(figsize=(14, 6))

# Recent historical observations
recent_history = df.tail(730)

ax.plot(
    recent_history.index,
    recent_history["Temperature"],
    label="Historical",
    linewidth=1
)

# Forecast
ax.plot(
    future_index,
    forecast_values,
    label="Forecast",
    linewidth=2
)

# Confidence interval
ax.fill_between(
    future_index,
    lower_values,
    upper_values,
    alpha=0.2,
    label="95% Prediction Interval"
)

ax.axvline(
    last_date,
    linestyle="--",
    label="Forecast Start"
)

ax.set_xlabel("Date")
ax.set_ylabel("Minimum Temperature (°C)")
ax.set_title(
    f"{forecast_days}-Day Daily Minimum Temperature Forecast"
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ============================================================
# FORECAST SUMMARY
# ============================================================

st.subheader("📌 Forecast Summary")

fc1, fc2, fc3, fc4 = st.columns(4)

fc1.metric(
    "Mean Forecast",
    f"{np.mean(forecast_values):.2f} °C"
)

fc2.metric(
    "Minimum Forecast",
    f"{np.min(forecast_values):.2f} °C"
)

fc3.metric(
    "Maximum Forecast",
    f"{np.max(forecast_values):.2f} °C"
)

fc4.metric(
    "Forecast SD",
    f"{np.std(forecast_values):.2f} °C"
)


# ============================================================
# FORECAST TABLE
# ============================================================

forecast_table = pd.DataFrame({
    "Date": future_index,
    "Forecast (°C)": forecast_values,
    "Lower 95% (°C)": lower_values,
    "Upper 95% (°C)": upper_values
})

st.subheader("📋 Forecast Values")

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD FORECAST
# ============================================================

csv_data = forecast_table.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv_data,
    file_name="temperature_forecast.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Model: SARIMAX(3,0,1) + 5 Fourier harmonics | "
    "Annual seasonal period: 365.25 days"
)

st.caption(
    "Forecasts are estimates based on historical data and the "
    "trained model. Prediction intervals represent model uncertainty."
)