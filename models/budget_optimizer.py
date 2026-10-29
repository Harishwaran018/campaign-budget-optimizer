import joblib
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------
# 1. LOAD TRAINED REVENUE MODEL
# --------------------------------------------------

model = joblib.load("saved_models/campaign_revenue_model.pkl")


# --------------------------------------------------
# 2. OPTIMIZATION SETTINGS
# --------------------------------------------------

TOTAL_BUDGET = 50000

MIN_BUDGET = 5000
MAX_BUDGET = 30000

platforms = ["Google Ads", "Meta Ads", "TikTok Ads"]


# --------------------------------------------------
# 3. FUNCTION TO PREDICT TOTAL REVENUE
# --------------------------------------------------

def predict_total_revenue(budgets):

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

    return predictions.sum()


# --------------------------------------------------
# 4. OBJECTIVE FUNCTION
# --------------------------------------------------

def objective(budgets):

    # Optimizer minimizes functions.
    # We want maximum revenue,
    # so we return negative revenue.

    return -predict_total_revenue(budgets)


# --------------------------------------------------
# 5. BUDGET CONSTRAINT
# --------------------------------------------------

def budget_constraint(budgets):

    return np.sum(budgets) - TOTAL_BUDGET


# --------------------------------------------------
# 6. INITIAL GUESS
# --------------------------------------------------

initial_guess = np.array([
    TOTAL_BUDGET / 3,
    TOTAL_BUDGET / 3,
    TOTAL_BUDGET / 3
])


# --------------------------------------------------
# 7. BOUNDS
# --------------------------------------------------

bounds = [
    (MIN_BUDGET, MAX_BUDGET),
    (MIN_BUDGET, MAX_BUDGET),
    (MIN_BUDGET, MAX_BUDGET)
]


# --------------------------------------------------
# 8. RUN OPTIMIZATION
# --------------------------------------------------

result = minimize(
    objective,
    initial_guess,
    method="SLSQP",
    bounds=bounds,
    constraints={
        "type": "eq",
        "fun": budget_constraint
    }
)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

if result.success:

    optimal_budgets = result.x

    predicted_revenues = []

    for platform, budget in zip(platforms, optimal_budgets):

        test_data = pd.DataFrame({
            "platform": [platform],
            "campaign_type": ["Search"],
            "industry": ["E-commerce"],
            "country": ["India"],
            "month": [6],
            "day_of_week": [2],
            "is_weekend": [0],
            "planned_budget": [budget]
        })

        predicted_log = model.predict(test_data)[0]

        revenue = np.expm1(predicted_log)

        predicted_revenues.append(revenue)

    print("=" * 60)
    print("           OPTIMAL BUDGET ALLOCATION")
    print("=" * 60)

    for platform, budget, revenue in zip(
        platforms,
        optimal_budgets,
        predicted_revenues
    ):

        print(
            f"{platform:<15} "
            f"Budget: ₹{budget:,.2f} "
            f"→ Predicted Revenue: ₹{revenue:,.2f}"
        )

    print("-" * 60)

    print(
        f"Total Budget: ₹{optimal_budgets.sum():,.2f}"
    )

    print(
        f"Total Predicted Revenue: "
        f"₹{sum(predicted_revenues):,.2f}"
    )

    print("=" * 60)

else:

    print("Optimization failed.")

    print(result.message)