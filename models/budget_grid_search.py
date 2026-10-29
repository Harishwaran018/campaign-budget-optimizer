import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("saved_models/campaign_revenue_model.pkl")

TOTAL_BUDGET = 50000
MIN_BUDGET = 5000
MAX_BUDGET = 30000

platforms = ["Google Ads", "Meta Ads", "TikTok Ads"]

best_allocation = None
best_revenue = -np.inf

results = []

# Test every ₹1,000 allocation
for google in range(MIN_BUDGET, MAX_BUDGET + 1, 1000):

    for meta in range(MIN_BUDGET, MAX_BUDGET + 1, 1000):

        tiktok = TOTAL_BUDGET - google - meta

        # Check TikTok constraints
        if tiktok < MIN_BUDGET or tiktok > MAX_BUDGET:
            continue

        budgets = [google, meta, tiktok]

        test_data = pd.DataFrame({
            "platform": platforms,
            "campaign_type": ["Search"] * 3,
            "industry": ["E-commerce"] * 3,
            "country": ["India"] * 3,
            "month": [6] * 3,
            "day_of_week": [2] * 3,
            "is_weekend": [0] * 3,
            "planned_budget": budgets
        })

        predicted_log = model.predict(test_data)
        predictions = np.expm1(predicted_log)

        total_revenue = predictions.sum()

        results.append({
            "Google": google,
            "Meta": meta,
            "TikTok": tiktok,
            "Revenue": total_revenue
        })

        if total_revenue > best_revenue:
            best_revenue = total_revenue
            best_allocation = budgets


# Sort results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Revenue",
    ascending=False
)


print("=" * 70)
print("             BUDGET GRID SEARCH")
print("=" * 70)

print("\nTOP 10 ALLOCATIONS:\n")

for _, row in results_df.head(10).iterrows():

    print(
        f"Google: ₹{row['Google']:,.0f} | "
        f"Meta: ₹{row['Meta']:,.0f} | "
        f"TikTok: ₹{row['TikTok']:,.0f} "
        f"→ Predicted Revenue: ₹{row['Revenue']:,.2f}"
    )

print("\n" + "=" * 70)

print("BEST ALLOCATION")
print("=" * 70)

print(f"Google Ads : ₹{best_allocation[0]:,.0f}")
print(f"Meta Ads   : ₹{best_allocation[1]:,.0f}")
print(f"TikTok Ads : ₹{best_allocation[2]:,.0f}")

print("-" * 70)

print(f"Total Budget: ₹{sum(best_allocation):,.0f}")
print(f"Predicted Revenue: ₹{best_revenue:,.2f}")

print("=" * 70)