"""Create deterministic synthetic retail data for a safe, repeatable demo."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(output_dir: str | Path, seed: int = 42) -> tuple[Path, Path]:
    """Write intentionally imperfect synthetic customers and transactions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_customers = 1800
    customer_ids = np.array([f"C{i:05d}" for i in range(1, n_customers + 1)])
    age = np.clip(rng.normal(39, 12, n_customers).round(), 18, 78).astype(int)
    tenure = rng.integers(1, 73, n_customers)
    visits = np.clip(rng.poisson(5, n_customers) + (tenure > 24), 0, 30)
    returns = rng.poisson(1.2, n_customers)
    tickets = rng.poisson(0.7, n_customers)
    marketing = rng.choice([0, 1], n_customers, p=[0.38, 0.62])
    tier = rng.choice(["Standard", "Silver", "Gold"], n_customers, p=[0.58, 0.29, 0.13])
    region = rng.choice(["North", "South", "East", "West"], n_customers)
    channel = rng.choice(["Online", "Store", "Mobile"], n_customers, p=[0.43, 0.34, 0.23])

    logit = (-2.0 + 0.075 * tickets + 0.035 * returns - 0.055 * visits
             - 0.012 * tenure - 0.32 * marketing + 0.35 * (tier == "Standard"))
    churn_probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_probability)
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "tenure_months": tenure,
        "visits_90d": visits,
        "returns_12m": returns,
        "support_tickets_90d": tickets,
        "marketing_opt_in": marketing,
        "membership_tier": tier,
        "region": region,
        "preferred_channel": channel,
        "churned": churned,
    })

    n_orders = 15000
    start = np.datetime64("2023-01-01")
    day_offsets = rng.integers(0, 1095, n_orders)
    order_dates = start + day_offsets.astype("timedelta64[D]")
    category = rng.choice(["Grocery", "Apparel", "Home", "Electronics", "Beauty"], n_orders,
                          p=[0.30, 0.20, 0.18, 0.14, 0.18]).astype(object)
    base_prices = {"Grocery": 18, "Apparel": 46, "Home": 72, "Electronics": 165, "Beauty": 29}
    unit_price = np.array([base_prices[c] for c in category]) * rng.lognormal(0, 0.38, n_orders)
    transactions = pd.DataFrame({
        "transaction_id": [f"T{i:07d}" for i in range(1, n_orders + 1)],
        "customer_id": rng.choice(customer_ids, n_orders),
        "order_date": order_dates.astype(str),
        "product_category": category,
        "units": rng.integers(1, 6, n_orders),
        "unit_price": unit_price.round(2),
        "discount_pct": rng.choice([0, 0.05, 0.10, 0.15, 0.20], n_orders,
                                   p=[0.35, 0.22, 0.22, 0.14, 0.07]),
        "channel": rng.choice(["Online", "Store", "Mobile"], n_orders),
        "region": rng.choice(["North", "South", "East", "West"], n_orders),
    })
    # Add realistic source-system defects so the ETL demonstrates its controls.
    missing_units = rng.choice(n_orders, size=45, replace=False)
    transactions.loc[missing_units, "units"] = np.nan
    missing_categories = rng.choice(n_orders, size=35, replace=False)
    transactions.loc[missing_categories, "product_category"] = None
    transactions = pd.concat([transactions, transactions.iloc[:30]], ignore_index=True)

    customer_path = output_dir / "customers.csv"
    transaction_path = output_dir / "transactions.csv"
    customers.to_csv(customer_path, index=False)
    transactions.to_csv(transaction_path, index=False)
    return customer_path, transaction_path

