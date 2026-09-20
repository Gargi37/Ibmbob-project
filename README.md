# Walmart Store Sales Forecasting

An interactive Python forecasting project for the Walmart weekly sales dataset. It combines historical sales with store metadata, regional economic signals, holiday flags, and markdown events, then exposes the results through a Streamlit dashboard.

## Project plan

1. **Data layer:** join `train.csv`, `test.csv`, `features.csv`, and `stores.csv` on store/date keys; clean markdown nulls and create calendar features.
2. **Model layer:** train a chronological holdout baseline using `HistGradientBoostingRegressor`, with a signed log-transformed sales target and non-negative predictions.
3. **Frontend:** use Streamlit for store and department filters, historical-versus-forecast charts, model error, forecast detail, and CSV export.
4. **Next iterations:** compare LightGBM/XGBoost, add lag and rolling sales features, tune by department, and add WMAE and holiday-specific error slices.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

## Data source

The dataset was obtained from Kaggle: [Walmart Sales Forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast?resource=download).

The CSV files are expected beside `app.py`: `train.csv`, `test.csv`, `features.csv`, and `stores.csv`.

## Project files

- `app.py` contains both the Python backend data/model pipeline and the Streamlit frontend.
- `requirements.txt` lists the Python dependencies.
- `README.md` contains setup, plan, and data-source information.
- `UI_OUTPUT.md` documents the dashboard output and user controls.