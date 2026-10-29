import pandas as pd

data = pd.read_csv("../data/processed/campaigns_processed.csv")

# Create ₹5,000 budget buckets
data["budget_bucket"] = (
    (data["planned_budget"] // 5000) * 5000
)

summary = (
    data.groupby(["platform", "budget_bucket"])
    .agg(
        campaigns=("revenue", "count"),
        average_budget=("planned_budget", "mean"),
        average_revenue=("revenue", "mean")
    )
    .reset_index()
)

print("\nHISTORICAL BUDGET → REVENUE")
print("=" * 80)

for platform in data["platform"].unique():

    print(f"\n{platform}")
    print("-" * 80)

    platform_data = summary[
        summary["platform"] == platform
    ]

    print(
        platform_data.to_string(
            index=False,
            formatters={
                "average_budget": "₹{:,.2f}".format,
                "average_revenue": "₹{:,.2f}".format
            }
        )
    )