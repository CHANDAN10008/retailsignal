"""RetailSignal interactive portfolio dashboard."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from retailsignal.churn import train_churn_models
from retailsignal.data_generation import generate_demo_data
from retailsignal.demand import forecast_demand
from retailsignal.pipeline import run_etl

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(page_title="RetailSignal | Retail Analytics", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Preparing the synthetic demo and modelling outputs…")
def prepare_demo() -> None:
    customers = RAW / "customers.csv"
    transactions = RAW / "transactions.csv"
    if not customers.exists() or not transactions.exists():
        generate_demo_data(RAW)
    modelled = PROCESSED / "modelled_customers.csv"
    monthly = PROCESSED / "monthly_demand.csv"
    if not modelled.exists() or not monthly.exists():
        run_etl(customers, transactions, PROCESSED)
    if not (REPORTS / "churn_model_metrics.csv").exists():
        train_churn_models(modelled, REPORTS, MODELS)
    if not (REPORTS / "demand_forecast.csv").exists():
        forecast_demand(monthly, REPORTS)


@st.cache_data
def read_csv(path: str, dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=dates or [])


st.markdown("<div class='eyebrow'>RETAIL ANALYTICS · SYNTHETIC DEMO</div>", unsafe_allow_html=True)
st.title("RetailSignal")
st.write("A reproducible pipeline for retail data quality, customer churn, and demand planning.")
st.markdown("""
<style>
.block-container {padding-top: 2.0rem; padding-bottom: 2rem; max-width: 1440px;}
.eyebrow {color:#68a8ff; font-size:0.76rem; letter-spacing:0.18em; font-weight:700;}
[data-testid="stMetric"] {background:#111b2b; border:1px solid #22324a; padding:16px; border-radius:12px;}
div[data-testid="stPlotlyChart"] {border:1px solid #22324a; border-radius:12px; padding:8px; background:#0c1420;}
</style>
""", unsafe_allow_html=True)

try:
    prepare_demo()
except Exception as exc:
    st.error("The demo pipeline could not be prepared. Check the README setup instructions and app logs.")
    st.exception(exc)
    st.stop()

customers = read_csv(str(PROCESSED / "modelled_customers.csv"))
transactions = read_csv(str(PROCESSED / "clean_transactions.csv"), ["order_date"])
forecast = read_csv(str(REPORTS / "demand_forecast.csv"), ["order_month"])
churn_metrics = read_csv(str(REPORTS / "churn_model_metrics.csv"))
importance = read_csv(str(REPORTS / "churn_feature_importance.csv"))
demand_metrics = read_csv(str(REPORTS / "demand_model_metrics.csv"))
quality_path = PROCESSED / "data_quality_report.json"
quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}

with st.sidebar:
    st.header("Explore the demo")
    st.caption("All records are generated and contain no real customer information.")
    regions = sorted(customers["region"].dropna().unique().tolist())
    selected_regions = st.multiselect("Region", regions, default=regions)
    categories = sorted(transactions["product_category"].dropna().unique().tolist())
    selected_category = st.selectbox("Demand category", categories)
    if st.button("Rebuild synthetic demo", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        for path in [RAW / "customers.csv", RAW / "transactions.csv", PROCESSED / "modelled_customers.csv",
                     PROCESSED / "monthly_demand.csv", REPORTS / "churn_model_metrics.csv",
                     REPORTS / "demand_forecast.csv"]:
            path.unlink(missing_ok=True)
        st.rerun()

filtered = customers[customers["region"].isin(selected_regions)]
filtered_tx = transactions[transactions["region"].isin(selected_regions)]
churn_rate = float(filtered["churned"].mean() * 100) if len(filtered) else 0.0
sales_total = float(filtered_tx["net_sales"].sum()) if len(filtered_tx) else 0.0
total_orders = int(filtered_tx["transaction_id"].nunique())
duplicates_removed = quality.get("duplicate_transactions_removed", 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(filtered):,}")
k2.metric("Observed churn", f"{churn_rate:.1f}%")
k3.metric("Net sales", f"${sales_total:,.0f}")
k4.metric("Duplicate rows removed", f"{duplicates_removed:,}")

tab_overview, tab_churn, tab_demand, tab_quality = st.tabs(["Overview", "Churn models", "Demand planning", "Data quality"])

with tab_overview:
    left, right = st.columns([1.35, 1])
    with left:
        sales_by_month = filtered_tx.groupby(filtered_tx["order_date"].dt.to_period("M").dt.to_timestamp(), as_index=False)["net_sales"].sum()
        sales_by_month = sales_by_month.rename(columns={"order_date": "month"})
        fig = px.area(sales_by_month, x="month", y="net_sales", title="Monthly net sales trend",
                      color_discrete_sequence=["#51b6ff"])
        fig.update_layout(xaxis_title="Month", yaxis_title="Net sales ($)", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        segment = filtered.groupby("membership_tier", as_index=False).agg(churn_rate=("churned", "mean"), customers=("customer_id", "count"))
        segment["churn_rate"] *= 100
        fig = px.bar(segment, x="membership_tier", y="churn_rate", color="membership_tier",
                     text=segment["churn_rate"].map(lambda value: f"{value:.1f}%"),
                     title="Observed churn by membership tier", color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(showlegend=False, xaxis_title="Tier", yaxis_title="Churn rate (%)", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Transaction count in selected regions: {total_orders:,}. Churn is the synthetic label provided with the generated customer records.")

with tab_churn:
    st.subheader("Model comparison")
    st.caption("A stratified 25% holdout split is used. Metrics are illustrative and depend on the synthetic data generator.")
    metric_left, metric_right = st.columns([1.3, 1])
    with metric_left:
        chart = churn_metrics.melt(id_vars="model", value_vars=["precision", "recall", "f1", "roc_auc"],
                                  var_name="metric", value_name="score")
        fig = px.bar(chart, x="model", y="score", color="metric", barmode="group", title="Holdout classification metrics",
                     color_discrete_sequence=["#63c7b2", "#ffd166", "#ff7b89", "#7b8cff"])
        fig.update_layout(yaxis_range=[0, 1], xaxis_title="Model", yaxis_title="Score", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with metric_right:
        top = importance.head(12).sort_values("importance")
        fig = px.bar(top, x="importance", y="feature", orientation="h", title="Top features in selected model",
                     color="importance", color_continuous_scale="Blues")
        fig.update_layout(xaxis_title="Relative importance", yaxis_title="", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(churn_metrics.style.format({"accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "roc_auc": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
    st.info("Feature importance describes model associations in this generated dataset. It does not establish causation or support real customer decisions.")

with tab_demand:
    st.subheader(f"Monthly unit demand · {selected_category}")
    category_forecast = forecast[forecast["product_category"] == selected_category]
    fig = px.line(category_forecast, x="order_month", y="demand_units", color="series_type", markers=True,
                  title="Observed history and next three month forecast",
                  color_discrete_map={"Actual": "#51b6ff", "Forecast": "#ffbf69"})
    fig.update_layout(xaxis_title="Month", yaxis_title="Units", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The final six observed months were held out for evaluation. The forecast recursively uses prior actual/forecast values and is a demonstration, not a production inventory plan.")
    st.dataframe(demand_metrics.style.format({"mae": "{:.1f}", "rmse": "{:.1f}", "mape_pct": "{:.1f}%"}), use_container_width=True, hide_index=True)

with tab_quality:
    st.subheader("ETL quality controls")
    quality_left, quality_right = st.columns(2)
    with quality_left:
        st.write("Input / output row counts and repair counts")
        st.json(quality)
    with quality_right:
        st.write("Clean transaction preview")
        st.dataframe(filtered_tx.head(25), use_container_width=True, hide_index=True)
    st.download_button("Download modelled customer table", customers.to_csv(index=False), "modelled_customers.csv", "text/csv")
    st.download_button("Download clean transactions", transactions.to_csv(index=False), "clean_transactions.csv", "text/csv")

st.divider()
st.caption("RetailSignal · Portfolio demonstration · Synthetic data · Built with Python, scikit-learn, pandas, SQL, R, and Streamlit")

