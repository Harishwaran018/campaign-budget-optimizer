import pandas as pd

df = pd.read_csv("../data/processed/campaigns_processed.csv")

print("=" * 70)
print("          BUDGET vs REVENUE DATA DIAGNOSTIC")
print("=" * 70)

print("\nOverall correlation:")
print(f"{df['planned_budget'].corr(df['revenue']):.4f}")

print("\nBudget ranges and average revenue:")
df["budget_range"] = pd.qcut(
    df["planned_budget"],
    q=10,
    duplicates="drop"
)

summary = df.groupby("budget_range", observed=True).agg(
    campaigns=("revenue", "count"),
    avg_budget=("planned_budget", "mean"),
    avg_revenue=("revenue", "mean"),
    median_revenue=("revenue", "median")
)

print(summary.to_string())

print("\nPlatform-level budget/revenue correlation:")

for platform in df["platform"].unique():
    platform_df = df[df["platform"] == platform]

    correlation = platform_df["planned_budget"].corr(
        platform_df["revenue"]
    )

    print(
        f"{platform:<15} → "
        f"Correlation: {correlation:.4f}"
    )

print("\n" + "=" * 70)