from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


DATA_DIR = Path(__file__).resolve().parent
MARKDOWN_COLUMNS = [f"MarkDown{i}" for i in range(1, 6)]
FEATURE_COLUMNS = [
    "Store", "Dept", "TypeCode", "Size", "Temperature", "Fuel_Price",
    *MARKDOWN_COLUMNS, "CPI", "Unemployment", "IsHoliday", "Year", "Month",
    "Week", "Quarter",
]


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["Date"])
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    return prepare_dataset(train, features, stores), prepare_dataset(test, features, stores)


def prepare_dataset(
    rows: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame
) -> pd.DataFrame:
    data = rows.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    data = data.merge(stores, on="Store", how="left")
    data["TypeCode"] = data["Type"].map({"A": 0, "B": 1, "C": 2}).fillna(-1)
    data["IsHoliday"] = data["IsHoliday"].astype(int)
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Week"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Quarter"] = data["Date"].dt.quarter
    for column in MARKDOWN_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "IsHoliday"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(0)
    return data


def signed_log1p(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(np.abs(values))


def signed_expm1(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.expm1(np.abs(values))


def train_model(train: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, float]:
    ordered = train.sort_values("Date")
    split_date = ordered["Date"].max() - pd.Timedelta(weeks=8)
    fit_rows = ordered[ordered["Date"] < split_date]
    validation_rows = ordered[ordered["Date"] >= split_date]
    model = HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(fit_rows[FEATURE_COLUMNS], signed_log1p(fit_rows["Weekly_Sales"]))
    validation_prediction = signed_expm1(model.predict(validation_rows[FEATURE_COLUMNS]))
    mae = mean_absolute_error(validation_rows["Weekly_Sales"], validation_prediction)
    model.fit(ordered[FEATURE_COLUMNS], signed_log1p(ordered["Weekly_Sales"]))
    return model, float(mae)


def forecast(model: HistGradientBoostingRegressor, test: pd.DataFrame) -> pd.DataFrame:
    predictions = np.maximum(0, signed_expm1(model.predict(test[FEATURE_COLUMNS])))
    result = test[["Store", "Dept", "Date"]].copy()
    result["Forecast"] = predictions
    return result


st.set_page_config(page_title="Walmart Sales Forecasting", page_icon="📈", layout="wide")


@st.cache_data
def get_data():
    return load_data(Path(__file__).resolve().parent)


@st.cache_resource
def get_model(train):
    return train_model(train)


st.title("Walmart Sales Forecasting")
st.caption("A practical weekly demand view for 45 stores and their departments")

with st.spinner("Preparing the forecasting model..."):
    train, test = get_data()
    model, holdout_mae = get_model(train)
    predictions = forecast(model, test)

store_options = ["All stores"] + sorted(predictions["Store"].unique().tolist())
selected_store = st.sidebar.selectbox("Store", store_options)
department_options = ["All departments"] + sorted(predictions["Dept"].unique().tolist())
selected_department = st.sidebar.selectbox("Department", department_options)

filtered_train = train.copy()
filtered_forecast = predictions.copy()
if selected_store != "All stores":
    filtered_train = filtered_train[filtered_train["Store"] == selected_store]
    filtered_forecast = filtered_forecast[filtered_forecast["Store"] == selected_store]
if selected_department != "All departments":
    filtered_train = filtered_train[filtered_train["Dept"] == selected_department]
    filtered_forecast = filtered_forecast[filtered_forecast["Dept"] == selected_department]

total_sales = filtered_train["Weekly_Sales"].sum()
forecast_total = filtered_forecast["Forecast"].sum()
latest_week = filtered_train["Date"].max().strftime("%d %b %Y")
metric_columns = st.columns(4)
metric_columns[0].metric("Historical sales", f"${total_sales:,.0f}")
metric_columns[1].metric("Forecast horizon", f"${forecast_total:,.0f}")
metric_columns[2].metric("Model holdout MAE", f"${holdout_mae:,.0f}")
metric_columns[3].metric("Latest history", latest_week)

history = filtered_train.groupby("Date", as_index=False)["Weekly_Sales"].sum()
history = history.rename(columns={"Weekly_Sales": "Sales"})
future = filtered_forecast.groupby("Date", as_index=False)["Forecast"].sum()
future = future.rename(columns={"Forecast": "Sales"})
history["Series"] = "Historical"
future["Series"] = "Forecast"
chart_data = history._append(future, ignore_index=True)
fig = px.line(
    chart_data,
    x="Date",
    y="Sales",
    color="Series",
    color_discrete_map={"Historical": "#1f6f5b", "Forecast": "#d36b3c"},
    title="Weekly sales trend",
)
fig.update_layout(hovermode="x unified", yaxis_title="Weekly sales ($)", xaxis_title=None)
st.plotly_chart(fig, width="stretch")

st.subheader("Forecast detail")
display_forecast = filtered_forecast.sort_values("Date").copy()
display_forecast["Date"] = display_forecast["Date"].dt.strftime("%d %b %Y")
display_forecast["Forecast"] = display_forecast["Forecast"].map(lambda value: f"${value:,.2f}")
st.dataframe(display_forecast, width="stretch", hide_index=True)

csv_data = predictions.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download full forecast CSV",
    data=csv_data,
    file_name="walmart_sales_forecast.csv",
    mime="text/csv",
)