import joblib
import pandas as pd
import numpy as np

model = joblib.load("saved_models/campaign_revenue_model.pkl")

budgets = [1000, 2000, 5000, 10000, 20000]

test_data = pd.DataFrame({
    "platform": ["Google Ads"] * 5,
    "campaign_type": ["Search"] * 5,
    "industry": ["E-commerce"] * 5,
    "country": ["India"] * 5,
    "month": [6] * 5,
    "day_of_week": [2] * 5,
    "is_weekend": [0] * 5,
    "planned_budget": budgets
})

predicted_log = model.predict(test_data)
predictions = np.expm1(predicted_log)

print("=" * 45)
print("       BUDGET SENSITIVITY TEST")
print("=" * 45)

for budget, revenue in zip(budgets, predictions):
    print(f"Budget: ₹{budget:,.0f}  →  Predicted Revenue: ₹{revenue:,.2f}")

print("=" * 45)