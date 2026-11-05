"""Temporal holdout evaluation and recursive category demand forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _feature_row(history: list[float], month: pd.Timestamp, trend: int) -> dict[str, float]:
    return {
        "month_number": float(month.month),
        "trend": float(trend),
        "lag_1": float(history[-1]),
        "lag_3_mean": float(np.mean(history[-3:])),
        "lag_12": float(history[-12]),
    }


def forecast_demand(monthly_demand_path: str | Path, output_dir: str | Path,
                    horizon: int = 3) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(monthly_demand_path, parse_dates=["order_month"])
    metrics_rows = []
    forecast_rows = []
    for category, group in monthly.groupby("product_category"):
        series = group.groupby("order_month")["demand_units"].sum().sort_index()
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="MS")
        series = series.reindex(full_index, fill_value=0).astype(float)
        if len(series) < 18:
            continue
        values = series.to_numpy()
        dates = series.index
        split = max(12, len(series) - 6)
        rows, targets, row_dates = [], [], []
        for i in range(12, len(values)):
            rows.append(_feature_row(values[:i].tolist(), dates[i], i))
            targets.append(values[i])
            row_dates.append(dates[i])
        X = pd.DataFrame(rows)
        y = np.asarray(targets)
        train_mask = np.array(row_dates) < dates[split]
        test_mask = ~train_mask
        model = GradientBoostingRegressor(n_estimators=140, learning_rate=0.04, max_depth=2, loss="huber", random_state=42)
        model.fit(X.loc[train_mask], y[train_mask])
        predicted = model.predict(X.loc[test_mask])
        actual = y[test_mask]
        mae = mean_absolute_error(actual, predicted)
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mape = float(np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100)
        metrics_rows.append({"product_category": category, "mae": mae, "rmse": rmse, "mape_pct": mape,
                             "holdout_months": int(test_mask.sum()), "baseline": "Seasonal lag features; final 6 months held out"})

        history = values.tolist()
        last_date = dates[-1]
        for step in range(1, horizon + 1):
            next_date = last_date + pd.offsets.MonthBegin(step)
            features = pd.DataFrame([_feature_row(history, next_date, len(history))])
            estimate = max(0.0, float(model.predict(features)[0]))
            forecast_rows.append({"order_month": next_date, "product_category": category,
                                  "demand_units": estimate, "series_type": "Forecast"})
            history.append(estimate)
    history_rows = monthly.groupby(["order_month", "product_category"], as_index=False)["demand_units"].sum()
    history_rows["series_type"] = "Actual"
    output = pd.concat([history_rows, pd.DataFrame(forecast_rows)], ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    forecast_path = output_dir / "demand_forecast.csv"
    metrics_path = output_dir / "demand_model_metrics.csv"
    output.to_csv(forecast_path, index=False, date_format="%Y-%m-%d")
    metrics.to_csv(metrics_path, index=False)
    return {"forecast": forecast_path, "metrics": metrics_path}

