# 🌡️ Melbourne Daily Minimum Temperature Forecasting

An interactive time-series forecasting application built using **Python, Streamlit, and SARIMAX** to analyze historical daily minimum temperatures in Melbourne, Australia, and predict future temperatures.

## 📌 Project Overview

This project uses historical daily minimum temperature data from Melbourne, Australia, covering the period from 1981 to 1990. It applies time-series forecasting techniques to analyze temperature patterns and generate future predictions.

The application provides an interactive dashboard where users can explore historical temperature data, evaluate forecasting performance, and generate future temperature forecasts.

## 🎯 Objectives

- Analyze historical daily minimum temperature data.
- Clean and preprocess time-series data.
- Visualize historical temperature trends.
- Split the dataset into training and testing sets.
- Train a SARIMAX forecasting model.
- Evaluate model performance using MAE and RMSE.
- Predict future daily minimum temperatures.
- Display forecast confidence intervals.
- Download cleaned data, evaluation results, and future predictions.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Statsmodels** – SARIMAX time-series forecasting
- **Scikit-learn** – Model evaluation metrics
- **Streamlit** – Interactive web application
- **Jupyter Notebook** – Data analysis and experimentation
- **Git and GitHub** – Version control and project hosting

## 📂 Project Structure

```text
melbourne-temperature-forecasting/
│
├── app.py
├── daily-minimum-temperatures-in-me.csv
├── Capstone 2_Bhushan_Mandekar.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

**Dataset:** Daily Minimum Temperatures in Melbourne, Australia

**Period:** 1981–1990

The dataset contains daily observations with two main fields:

- `Date` – Date of observation
- `Daily minimum temperatures in Melbourne, Australia, 1981-1990` – Recorded minimum temperature in degrees Celsius

The application preprocesses the dataset by converting dates, handling missing values, removing duplicate dates, and organizing observations into daily frequency.

Dataset source: [Daily Minimum Temperatures in Melbourne — Kaggle](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne)

## 🤖 Machine Learning Approach

The project uses the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)** model for time-series forecasting.

Fourier features are incorporated to represent yearly seasonal patterns.

### Workflow

1. Load the historical temperature dataset.
2. Clean and preprocess the data.
3. Visualize historical temperature trends.
4. Divide observations into training and testing periods.
5. Generate Fourier features to capture annual seasonality.
6. Train the SARIMAX model.
7. Evaluate predictions using MAE and RMSE.
8. Forecast future daily minimum temperatures.
9. Visualize predictions and confidence intervals.

## 📈 Model Evaluation

The application evaluates the forecasting model using the following metrics:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted temperatures.

**Root Mean Squared Error (RMSE)**

Measures the square root of the average squared prediction errors and penalizes larger errors more heavily.

Lower MAE and RMSE values generally indicate better forecasting performance on the test data.

The actual metric values depend on the model settings and the selected test period.

## ✨ Application Features

- Interactive temperature dashboard
- Historical temperature visualization
- Dataset statistics and data table
- Configurable SARIMAX parameters
- Adjustable testing period
- Fourier-based yearly seasonality features
- Model evaluation with MAE and RMSE
- Actual versus predicted temperature plots
- Future forecasts for up to 365 days
- 95% forecast confidence intervals
- CSV downloads for cleaned data and predictions
- Optional CSV dataset upload

## ⚙️ Installation and Setup

### Prerequisites

Install Python and Visual Studio Code. Ensure Python is available from your Windows Command Prompt.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/melbourne-temperature-forecasting.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Open the project directory

```bash
cd melbourne-temperature-forecasting
```

### Step 3: Create a virtual environment

```bash
python -m venv myenv
```

### Step 4: Activate the virtual environment on Windows

```bat
myenv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 📦 Requirements

The `requirements.txt` file should contain:

```text
streamlit
pandas
numpy
matplotlib
scikit-learn
statsmodels
```

## 🚀 How to Use

1. Launch the application using Streamlit.
2. Load the provided CSV dataset or upload another compatible CSV file.
3. Explore historical temperature trends and dataset statistics.
4. Configure the SARIMAX parameters in the sidebar.
5. Select the test period and click **Train Model and Evaluate**.
6. Review MAE, RMSE, and actual versus predicted temperatures.
7. Select the forecast horizon and click **Generate Future Forecast**.
8. Download the forecast results as a CSV file.

## 🔮 Future Improvements

- Compare SARIMAX with ARIMA, Holt-Winters, and machine learning models.
- Add automated hyperparameter tuning.
- Include interactive Plotly visualizations.
- Improve forecast accuracy through additional seasonal modeling.
- Deploy the application to Streamlit Community Cloud.
- Add automated model performance reporting.

## ⚠️ Limitations

- Forecast accuracy depends on the historical data and model parameters.
- The dataset covers 1981–1990 and may not represent current climate conditions.
- Long-term predictions become increasingly uncertain.
- Forecasts are estimates and should not be interpreted as guaranteed temperatures.

## 👨‍💻 Author

**Bhushan Mandekar**

## 📄 License

This project is intended for educational and portfolio purposes. Add an appropriate license file if you plan to distribute the source code publicly.

---

⭐ If you find this project useful, consider giving the repository a star on GitHub.
