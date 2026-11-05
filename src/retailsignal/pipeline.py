"""Validation, cleaning, enrichment, and modelling-table construction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_etl(customers_path: str | Path, transactions_path: str | Path,
            output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path)
    report = {
        "source_customers": int(len(customers)),
        "source_transactions": int(len(transactions)),
        "duplicate_transactions_removed": int(transactions.duplicated("transaction_id").sum()),
        "missing_units_filled": int(transactions["units"].isna().sum()),
        "missing_categories_filled": int(transactions["product_category"].isna().sum()),
    }

    customers = customers.drop_duplicates("customer_id", keep="last")
    transactions = transactions.drop_duplicates("transaction_id", keep="first")
    transactions["order_date"] = pd.to_datetime(transactions["order_date"], errors="coerce")
    transactions = transactions.dropna(subset=["transaction_id", "customer_id", "order_date"])
    transactions["units"] = pd.to_numeric(transactions["units"], errors="coerce")
    transactions["units"] = transactions["units"].fillna(transactions["units"].median()).clip(lower=1)
    transactions["unit_price"] = pd.to_numeric(transactions["unit_price"], errors="coerce")
    transactions["unit_price"] = transactions["unit_price"].fillna(transactions["unit_price"].median()).clip(lower=0)
    transactions["discount_pct"] = pd.to_numeric(transactions["discount_pct"], errors="coerce").fillna(0).clip(0, 0.9)
    transactions["product_category"] = transactions["product_category"].fillna("Unknown").astype(str)
    transactions["channel"] = transactions["channel"].fillna("Unknown").astype(str)
    transactions["region"] = transactions["region"].fillna("Unknown").astype(str)
    transactions["net_sales"] = transactions["units"] * transactions["unit_price"] * (1 - transactions["discount_pct"])
    transactions["order_month"] = transactions["order_date"].dt.to_period("M").dt.to_timestamp()

    numeric_customer_cols = ["age", "tenure_months", "visits_90d", "returns_12m", "support_tickets_90d", "marketing_opt_in", "churned"]
    for column in numeric_customer_cols:
        customers[column] = pd.to_numeric(customers[column], errors="coerce")
        customers[column] = customers[column].fillna(customers[column].median())
    customers["churned"] = customers["churned"].round().clip(0, 1).astype(int)
    for column in ["membership_tier", "region", "preferred_channel"]:
        customers[column] = customers[column].fillna("Unknown").astype(str)

    aggregates = transactions.groupby("customer_id").agg(
        order_count=("transaction_id", "nunique"),
        total_units=("units", "sum"),
        lifetime_net_sales=("net_sales", "sum"),
        average_order_value=("net_sales", "mean"),
        average_discount=("discount_pct", "mean"),
        last_order_date=("order_date", "max"),
    ).reset_index()
    modelled = customers.merge(aggregates, on="customer_id", how="left", validate="one_to_one")
    modelled["order_count"] = modelled["order_count"].fillna(0)
    modelled["total_units"] = modelled["total_units"].fillna(0)
    modelled["lifetime_net_sales"] = modelled["lifetime_net_sales"].fillna(0)
    modelled["average_order_value"] = modelled["average_order_value"].fillna(0)
    modelled["average_discount"] = modelled["average_discount"].fillna(0)
    reference_date = transactions["order_date"].max()
    modelled["days_since_last_order"] = (reference_date - modelled["last_order_date"]).dt.days.fillna(1095)
    modelled = modelled.drop(columns="last_order_date")

    monthly = transactions.groupby(["order_month", "product_category"], as_index=False).agg(
        demand_units=("units", "sum"), net_sales=("net_sales", "sum"), orders=("transaction_id", "nunique")
    ).sort_values(["product_category", "order_month"])
    modelled_path = output_dir / "modelled_customers.csv"
    transactions_path_out = output_dir / "clean_transactions.csv"
    monthly_path = output_dir / "monthly_demand.csv"
    report_path = output_dir / "data_quality_report.json"
    modelled.to_csv(modelled_path, index=False)
    transactions.to_csv(transactions_path_out, index=False, date_format="%Y-%m-%d")
    monthly.to_csv(monthly_path, index=False, date_format="%Y-%m-%d")
    report.update({
        "clean_transactions": int(len(transactions)),
        "modelled_customers": int(len(modelled)),
        "orphan_customer_transactions": int((~transactions["customer_id"].isin(customers["customer_id"])).sum()),
        "rows_with_non_positive_sales": int((transactions["net_sales"] <= 0).sum()),
    })
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"customers": modelled_path, "transactions": transactions_path_out,
            "monthly_demand": monthly_path, "quality_report": report_path}

