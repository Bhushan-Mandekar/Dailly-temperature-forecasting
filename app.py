import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Daily Temperature Forecast",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌡️ Daily Minimum Temperature Forecasting")

st.markdown(
    """
    **Time-series forecasting using SARIMAX with Fourier annual seasonality**

    This dashboard explores historical daily minimum temperatures
    and provides a 365-day forecast using the final
    **SARIMAX(3,0,1) + Fourier** model.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "daily_minimum_temperature_cleaned.csv"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df = df.set_index("Date")

    return df


df = load_data()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = SARIMAXResults.load(
        "models/sarimax_fourier_model.pkl"
    )

    return model


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

forecast_days = st.sidebar.selectbox(
    "Forecast horizon",
    [30, 90, 180, 365],
    index=3
)


# ============================================================
# KEY METRICS
# ============================================================

st.subheader("Project Overview")

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
    "Minimum",
    f"{df['Temperature'].min():.1f} °C"
)

col4.metric(
    "Maximum",
    f"{df['Temperature'].max():.1f} °C"
)


# ============================================================
# HISTORICAL DATA
# ============================================================

st.subheader("📈 Historical Daily Minimum Temperature")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df.index,
    df["Temperature"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Minimum Temperature (°C)")
ax.set_title("Historical Daily Minimum Temperature")

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

st.subheader("📊 Historical Summary Statistics")

summary = df["Temperature"].describe()

st.dataframe(
    summary.to_frame("Temperature (°C)"),
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


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("🏆 Model Comparison")

comparison = pd.read_csv(
    "outputs/forecast_model_comparison.csv"
)

st.dataframe(
    comparison,
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    comparison["Model"],
    comparison["RMSE"]
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


# ============================================================
# FORECAST
# ============================================================

st.subheader("🔮 Temperature Forecast")

# Create future dates

last_date = df.index.max()

future_index = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)


# Fourier function

def create_fourier_terms(
    index,
    period=365.25,
    K=5
):

    # Need continuous time index beginning with
    # the historical observations

    combined_index = df.index.append(index)

    t = np.arange(len(combined_index))

    fourier = pd.DataFrame(
        index=combined_index
    )

    for k in range(1, K + 1):

        fourier[f"sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )

        fourier[f"cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    return fourier.loc[index]


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


# ============================================================
# FORECAST CHART
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

# Recent historical data

recent_history = df.tail(730)

ax.plot(
    recent_history.index,
    recent_history["Temperature"],
    label="Historical"
)

ax.plot(
    future_index,
    forecast,
    label="Forecast"
)

ax.fill_between(
    future_index,
    confidence.iloc[:, 0],
    confidence.iloc[:, 1],
    alpha=0.2,
    label="95% Prediction Interval"
)

ax.axvline(
    last_date,
    linestyle="--"
)

ax.set_xlabel("Date")
ax.set_ylabel("Minimum Temperature (°C)")
ax.set_title(
    f"{forecast_days}-Day Temperature Forecast"
)

ax.legend()

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# FORECAST METRICS
# ============================================================

st.subheader("Forecast Summary")

fc1, fc2, fc3, fc4 = st.columns(4)

fc1.metric(
    "Mean Forecast",
    f"{forecast.mean():.2f} °C"
)

fc2.metric(
    "Minimum Forecast",
    f"{forecast.min():.2f} °C"
)

fc3.metric(
    "Maximum Forecast",
    f"{forecast.max():.2f} °C"
)

fc4.metric(
    "Forecast SD",
    f"{forecast.std():.2f} °C"
)


# ============================================================
# FORECAST TABLE
# ============================================================

forecast_table = pd.DataFrame({
    "Date": future_index,
    "Forecast (°C)": forecast.values,
    "Lower 95%": confidence.iloc[:, 0].values,
    "Upper 95%": confidence.iloc[:, 1].values
})

st.subheader("Forecast Values")

st.dataframe(
    forecast_table,
    use_container_width=True
)


# ============================================================
# DOWNLOAD
# ============================================================

csv = forecast_table.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv,
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