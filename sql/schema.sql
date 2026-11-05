-- PostgreSQL analytical schema for the cleaned RetailSignal output.
CREATE SCHEMA IF NOT EXISTS retailsignal;

CREATE TABLE IF NOT EXISTS retailsignal.customers (
    customer_id TEXT PRIMARY KEY,
    age SMALLINT,
    tenure_months SMALLINT,
    visits_90d SMALLINT,
    returns_12m SMALLINT,
    support_tickets_90d SMALLINT,
    marketing_opt_in BOOLEAN,
    membership_tier TEXT,
    region TEXT,
    preferred_channel TEXT,
    churned BOOLEAN,
    order_count INTEGER,
    total_units NUMERIC(12,2),
    lifetime_net_sales NUMERIC(14,2),
    average_order_value NUMERIC(12,2),
    average_discount NUMERIC(6,4),
    days_since_last_order INTEGER
);

CREATE TABLE IF NOT EXISTS retailsignal.transactions (
    transaction_id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES retailsignal.customers(customer_id),
    order_date DATE NOT NULL,
    product_category TEXT,
    units NUMERIC(8,2),
    unit_price NUMERIC(12,2),
    discount_pct NUMERIC(6,4),
    channel TEXT,
    region TEXT,
    net_sales NUMERIC(14,2) NOT NULL,
    order_month DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS retailsignal.monthly_demand (
    order_month DATE NOT NULL,
    product_category TEXT NOT NULL,
    demand_units NUMERIC(12,2),
    net_sales NUMERIC(14,2),
    orders INTEGER,
    PRIMARY KEY (order_month, product_category)
);

CREATE INDEX IF NOT EXISTS idx_transactions_order_date ON retailsignal.transactions(order_date);
CREATE INDEX IF NOT EXISTS idx_transactions_region_category ON retailsignal.transactions(region, product_category);

