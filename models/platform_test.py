import joblib
import pandas as pd
import numpy as np

model = joblib.load("saved_models/campaign_revenue_model.pkl")

platforms = ["Google Ads", "Meta Ads", "TikTok Ads"]

test_data = pd.DataFrame({
    "platform": platforms,
    "campaign_type": ["Search"] * 3,
    "industry": ["E-commerce"] * 3,
    "country": ["India"] * 3,
    "month": [6] * 3,
    "day_of_week": [2] * 3,
    "is_weekend": [0] * 3,
    "planned_budget": [10000] * 3
})

predicted_log = model.predict(test_data)
predictions = np.expm1(predicted_log)

print("=" * 50)
print("       PLATFORM PREDICTION TEST")
print("=" * 50)

for platform, revenue in zip(platforms, predictions):
    print(f"{platform:<15} → Predicted Revenue: ₹{revenue:,.2f}")

print("=" * 50)