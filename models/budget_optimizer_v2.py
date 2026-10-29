import joblib
import pandas as pd
import numpy as np


# Load trained model
model = joblib.load("saved_models/campaign_revenue_model.pkl")


PLATFORMS = ["Google Ads", "Meta Ads", "TikTok Ads"]

MIN_BUDGET = 5000
MAX_BUDGET = 30000
STEP = 1000


def predict_revenue(budgets):

    test_data = pd.DataFrame({
        "platform": PLATFORMS,
        "campaign_type": ["Search"] * 3,
        "industry": ["E-commerce"] * 3,
        "country": ["India"] * 3,
        "month": [6] * 3,
        "day_of_week": [2] * 3,
        "is_weekend": [0] * 3,
        "planned_budget": budgets
    })

    predicted_log = model.predict(test_data)

    return np.expm1(predicted_log)


def optimize_budget(total_budget):

    # Validate total budget
    if total_budget < MIN_BUDGET * 3:
        raise ValueError(
            f"Total budget must be at least ₹{MIN_BUDGET * 3:,}."
        )

    if total_budget > MAX_BUDGET * 3:
        raise ValueError(
            f"Total budget cannot exceed ₹{MAX_BUDGET * 3:,}."
        )

    best_allocation = None
    best_predictions = None
    best_revenue = -np.inf

    for google in range(MIN_BUDGET, MAX_BUDGET + 1, STEP):

        for meta in range(MIN_BUDGET, MAX_BUDGET + 1, STEP):

            tiktok = total_budget - google - meta

            if tiktok < MIN_BUDGET or tiktok > MAX_BUDGET:
                continue

            budgets = [google, meta, tiktok]

            predictions = predict_revenue(budgets)

            total_revenue = predictions.sum()

            if total_revenue > best_revenue:
                best_revenue = total_revenue
                best_allocation = budgets
                best_predictions = predictions

    return best_allocation, best_predictions, best_revenue


# Test the optimizer
if __name__ == "__main__":

    total_budget = 50000

    allocation, predictions, revenue = optimize_budget(total_budget)

    print("=" * 60)
    print("           BUDGET OPTIMIZATION")
    print("=" * 60)

    for platform, budget, predicted in zip(
        PLATFORMS,
        allocation,
        predictions
    ):
        print(
            f"{platform:<15} "
            f"Budget: ₹{budget:,.0f} "
            f"→ Predicted Revenue: ₹{predicted:,.2f}"
        )

    print("-" * 60)

    print(f"Total Budget: ₹{sum(allocation):,.0f}")
    print(f"Predicted Revenue: ₹{revenue:,.2f}")

    print("=" * 60)