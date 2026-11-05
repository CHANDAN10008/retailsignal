"""Command-line entry point for the complete reproducible demo pipeline."""

import argparse
from pathlib import Path

from retailsignal.churn import train_churn_models
from retailsignal.data_generation import generate_demo_data
from retailsignal.demand import forecast_demand
from retailsignal.pipeline import run_etl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RetailSignal synthetic-data analytics pipeline")
    parser.add_argument("command", choices=["generate", "etl", "train", "forecast", "all"], nargs="?", default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    data_dir, reports_dir, models_dir = Path(args.data_dir), Path(args.reports_dir), Path(args.models_dir)
    raw_dir, processed_dir = data_dir / "raw", data_dir / "processed"
    customers = raw_dir / "customers.csv"
    transactions = raw_dir / "transactions.csv"
    if args.command in {"generate", "all"}:
        generate_demo_data(raw_dir)
        print(f"Synthetic inputs written to {raw_dir}")
        if args.command == "generate":
            return
    if args.command in {"etl", "all"}:
        outputs = run_etl(customers, transactions, processed_dir)
        print(f"ETL complete: {outputs}")
        if args.command == "etl":
            return
    modelled = processed_dir / "modelled_customers.csv"
    monthly = processed_dir / "monthly_demand.csv"
    if args.command in {"train", "all"}:
        outputs = train_churn_models(modelled, reports_dir, models_dir)
        print(f"Churn modelling complete: {outputs}")
        if args.command == "train":
            return
    if args.command in {"forecast", "all"}:
        outputs = forecast_demand(monthly, reports_dir)
        print(f"Demand forecasting complete: {outputs}")


if __name__ == "__main__":
    main()

