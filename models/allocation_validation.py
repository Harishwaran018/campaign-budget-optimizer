import joblib
import pandas as pd
import numpy as np

model = joblib.load("saved_models/campaign_revenue_model.pkl")

platforms = ["Google Ads", "Meta Ads", "TikTok Ads"]

allocations = [
    [5000, 22000, 23000],
    [6000, 22000, 22000],
    [7000, 21000, 22000],
    [8000, 20000, 22000],
    [6000, 23000, 21000],
    [6000, 21000, 23000],
    [7000, 22000, 21000],
    [5000, 23000, 22000],
]

results = []

for budgets in allocations:

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

    results.append({
        "Google": budgets[0],
        "Meta": budgets[1],
        "TikTok": budgets[2],
        "Revenue": predictions.sum()
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Revenue", ascending=False)

print("=" * 70)
print("             ALLOCATION VALIDATION")
print("=" * 70)

for _, row in results_df.iterrows():
    print(
        f"Google: ₹{row['Google']:,.0f} | "
        f"Meta: ₹{row['Meta']:,.0f} | "
        f"TikTok: ₹{row['TikTok']:,.0f} "
        f"→ Predicted Revenue: ₹{row['Revenue']:,.2f}"
    )

print("=" * 70)