import pandas as pd
import numpy as np

from train_monotonic_model import model, preprocessor


def predict_revenue(platform, budget):
    data = pd.DataFrame([{
        "platform": platform,
        "campaign_type": "Search",
        "industry": "E-commerce",
        "country": "India",
        "month": 6,
        "day_of_week": 2,
        "is_weekend": 0,
        "planned_budget": budget
    }])

    X_processed = preprocessor.transform(data)

    prediction_log = model.predict(X_processed)[0]

    return np.expm1(prediction_log)


budgets = range(5000, 30001, 1000)

platforms = [
    "Google Ads",
    "Meta Ads",
    "TikTok Ads"
]

print("\nMONOTONIC MODEL — BUDGET CURVE")
print("=" * 70)

for platform in platforms:

    print(f"\n{platform}")
    print("-" * 40)

    previous = None
    decreasing = False

    for budget in budgets:

        revenue = predict_revenue(platform, budget)

        if previous is not None and revenue < previous:
            decreasing = True

        print(f"₹{budget:>6,} → ₹{revenue:>12,.2f}")

        previous = revenue

    print(f"Decreasing anywhere: {'YES ❌' if decreasing else 'NO ✅'}")