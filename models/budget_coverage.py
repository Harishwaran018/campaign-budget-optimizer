import pandas as pd

data = pd.read_csv("../data/processed/campaigns_processed.csv")

# Create ₹5,000 budget ranges
data["budget_range"] = pd.cut(
    data["planned_budget"],
    bins=[0, 5000, 10000, 15000, 20000, 25000, 30000, 40000],
    labels=[
        "₹0–5k",
        "₹5k–10k",
        "₹10k–15k",
        "₹15k–20k",
        "₹20k–25k",
        "₹25k–30k",
        "₹30k–40k"
    ],
    include_lowest=True
)

summary = (
    data.groupby(
        ["platform", "budget_range"],
        observed=False
    )
    .agg(
        campaigns=("revenue", "count"),
        avg_budget=("planned_budget", "mean"),
        avg_revenue=("revenue", "mean")
    )
    .reset_index()
)

print("\nBUDGET COVERAGE BY PLATFORM")
print("=" * 90)

for platform in data["platform"].unique():

    print(f"\n{platform}")
    print("-" * 90)

    platform_data = summary[
        summary["platform"] == platform
    ].copy()

    print(
        platform_data.to_string(
            index=False,
            formatters={
                "avg_budget": "₹{:,.2f}".format,
                "avg_revenue": "₹{:,.2f}".format
            }
        )
    )