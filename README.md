# ₿ Bitcoin Price Forecasting Portal

An interactive Streamlit web application for BTC time-series analysis and price forecasting using **Prophet** and **ARIMA**.

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/MohamedElmansy2/BTC_Forecasting.git
cd btc_forecaster
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows / conda users**: Prophet requires `pystan`. If you encounter issues, install via conda:
> ```bash
> conda install -c conda-forge prophet
> ```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Dataset

**Recommended Kaggle dataset:**  
[Bitcoin Historical Data 2014–2024](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)

Download the CSV and upload it directly through the app's sidebar.

The app auto-detects common Kaggle BTC CSV formats including:
- `Date` / `Timestamp` columns
- `Close` / `Open` / `High` / `Low` price columns (with or without `$` signs or comma separators)

---

## 🏗️ App Structure

```
btc_forecaster/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🤖 Model Explanations

### Prophet (Facebook / Meta)

Prophet is a decomposable time-series model designed for business forecasting with strong seasonal patterns.

**How it handles crypto volatility:**

| Feature | Behaviour |
|---|---|
| Changepoint detection | Automatically detects BTC price regime shifts (e.g., bull/bear transitions) |
| `changepoint_prior_scale=0.15` | Moderately flexible trend — not too rigid, not overfitting |
| Weekly + yearly seasonality | Captures BTC's known day-of-week and halving-cycle patterns |
| Uncertainty intervals | Widen naturally further into the future, reflecting compounding uncertainty |

**Limitations:** Prophet doesn't model volatility clustering (GARCH effects) or market microstructure noise. Treat forecasts as directional guidance.

---

### ARIMA (5, 1, 0)

AutoRegressive Integrated Moving Average — a classical statistical time-series model.

**How it handles crypto volatility:**

| Parameter | Meaning |
|---|---|
| `d=1` (Integration) | First-differencing removes non-stationarity from BTC prices; the model forecasts *returns*, not raw prices |
| `p=5` (Autoregression) | Captures 5-day autocorrelation (one trading week of momentum) |
| `q=0` (Moving Average) | No MA component — keeps the model parsimonious |

**Limitations:** ARIMA assumes linear relationships and constant variance. BTC exhibits heteroskedasticity (volatility clustering) — for production use, consider a GARCH model on residuals or a machine learning approach.

---

## 📐 Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average USD deviation of backtest predictions |
| **RMSE** | Root Mean Square Error — penalises large errors more heavily |
| **MAPE** | Mean Absolute Percentage Error — scale-free accuracy measure |

Backtesting uses an 80/20 train-test split on historical data.

---

## 🎛️ Features

- **File Upload** — Drag-and-drop Kaggle BTC CSV with automatic format detection
- **Model Selection** — Prophet or ARIMA with tunable parameters
- **Forecast Horizon** — 7 to 180 days
- **Confidence Intervals** — 80%, 90%, 95%, 99%
- **Technical Indicators** — SMA-20, EMA-20 overlays
- **Interactive Charts** — Plotly with range slider, hover tooltips, and forecast uncertainty bands
- **Backtesting Metrics** — MAE and RMSE in USD terms
- **Error Handling** — Clear feedback for incompatible CSVs or parsing failures

---

## ⚙️ Tech Stack

| Tool | Version | Role |
|---|---|---|
| Streamlit | ≥1.35 | Web UI framework |
| Plotly | ≥5.22 | Interactive visualisations |
| Prophet | ≥1.1.5 | Seasonal time-series forecasting |
| statsmodels | ≥0.14 | ARIMA model |
| pandas | ≥2.2 | Data manipulation |
| NumPy | ≥1.26 | Numerical computation |
| scikit-learn | ≥1.4 | Metrics utilities |

---
