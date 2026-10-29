import joblib
import pandas as pd
import numpy as np

model = joblib.load("saved_models/campaign_revenue_model.pkl")

platforms = ["Google Ads", "Meta Ads", "TikTok Ads"]

budgets = list(range(5000, 30001, 1000))

print("=" * 90)
print("                 BUDGET RESPONSE CURVE")
print("=" * 90)

for platform in platforms:

    print(f"\n{platform}")
    print("-" * 50)

    test_data = pd.DataFrame({
        "platform": [platform] * len(budgets),
        "campaign_type": ["Search"] * len(budgets),
        "industry": ["E-commerce"] * len(budgets),
        "country": ["India"] * len(budgets),
        "month": [6] * len(budgets),
        "day_of_week": [2] * len(budgets),
        "is_weekend": [0] * len(budgets),
        "planned_budget": budgets
    })

    predicted_log = model.predict(test_data)
    predictions = np.expm1(predicted_log)

    for budget, revenue in zip(budgets, predictions):
        print(
            f"Budget: ₹{budget:,.0f}"
            f" → Predicted Revenue: ₹{revenue:,.2f}"
        )

print("\n" + "=" * 90)