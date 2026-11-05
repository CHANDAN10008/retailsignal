# Statistical exploration of the ETL output. Run from the repository root:
# Rscript r/eda.R

input_path <- "data/processed/modelled_customers.csv"
output_dir <- "reports/r_analysis"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(input_path)) {
  stop("Processed table not found. Run: python -m retailsignal.cli all")
}

customers <- read.csv(input_path, stringsAsFactors = FALSE)
customers$churned <- as.integer(customers$churned)

summary_table <- data.frame(
  customers = nrow(customers),
  observed_churn_rate = mean(customers$churned, na.rm = TRUE),
  mean_tenure_months = mean(customers$tenure_months, na.rm = TRUE),
  median_lifetime_sales = median(customers$lifetime_net_sales, na.rm = TRUE),
  mean_orders = mean(customers$order_count, na.rm = TRUE)
)
write.csv(summary_table, file.path(output_dir, "summary.csv"), row.names = FALSE)

# Welch two-sample test compares average lifetime net sales for generated
# churn labels. Report it as an association, not a causal effect.
sales_by_churn <- split(customers$lifetime_net_sales, customers$churned)
sales_test <- t.test(sales_by_churn[["0"]], sales_by_churn[["1"]], var.equal = FALSE)
write.csv(data.frame(
  comparison = "Lifetime net sales: churned vs retained",
  estimate_retained = unname(sales_test$estimate[[1]]),
  estimate_churned = unname(sales_test$estimate[[2]]),
  confidence_low = sales_test$conf.int[[1]],
  confidence_high = sales_test$conf.int[[2]],
  p_value = sales_test$p.value,
  method = sales_test$method
), file.path(output_dir, "welch_sales_test.csv"), row.names = FALSE)

correlation <- cor(customers[c("tenure_months", "visits_90d", "returns_12m",
                               "support_tickets_90d", "order_count", "lifetime_net_sales",
                               "days_since_last_order", "churned")],
                   use = "pairwise.complete.obs", method = "spearman")
write.csv(correlation, file.path(output_dir, "spearman_correlations.csv"))
cat("R analysis written to", output_dir, "\n")

