-- Monthly demand and sales by category
SELECT order_month, product_category, SUM(demand_units) AS units, SUM(net_sales) AS net_sales
FROM retailsignal.monthly_demand
GROUP BY order_month, product_category
ORDER BY order_month, product_category;

-- Churn rate and customer count by region / membership tier
SELECT region, membership_tier, COUNT(*) AS customers,
       AVG(churned::int)::numeric(6,4) AS churn_rate
FROM retailsignal.customers
GROUP BY region, membership_tier
ORDER BY churn_rate DESC;

-- Revenue concentration by channel
SELECT channel, COUNT(DISTINCT transaction_id) AS orders,
       SUM(net_sales)::numeric(14,2) AS net_sales,
       AVG(net_sales)::numeric(12,2) AS average_order_value
FROM retailsignal.transactions
GROUP BY channel
ORDER BY net_sales DESC;

-- Recent customers with higher observed support load and churn label.
-- This is descriptive exploration, not an intervention recommendation.
SELECT membership_tier, AVG(support_tickets_90d)::numeric(8,2) AS mean_tickets,
       AVG(churned::int)::numeric(6,4) AS observed_churn_rate,
       COUNT(*) AS customers
FROM retailsignal.customers
GROUP BY membership_tier
ORDER BY mean_tickets DESC;

