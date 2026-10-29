import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load processed data
data = pd.read_csv("../data/processed/campaigns_processed.csv")


# Diminishing-returns function
def revenue_curve(budget, alpha, beta, gamma):
    return alpha * np.log1p(beta * budget) + gamma


print("\nBUDGET RESPONSE CURVE FITTING")
print("=" * 90)

platforms = ["Google Ads", "Meta Ads", "TikTok Ads"]

for platform in platforms:

    platform_data = data[data["platform"] == platform].copy()

    x = platform_data["planned_budget"].values
    y = platform_data["revenue"].values

    # Fit curve
    params, _ = curve_fit(
        revenue_curve,
        x,
        y,
        p0=[30000, 0.001, 0],
        bounds=(
            [0, 0, -np.inf],
            [np.inf, np.inf, np.inf]
        ),
        maxfev=20000
    )

    alpha, beta, gamma = params

    predictions = revenue_curve(x, alpha, beta, gamma)

    # R²
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    print(f"\n{platform}")
    print("-" * 90)

    print(f"Campaigns       : {len(platform_data)}")
    print(f"Alpha           : {alpha:.4f}")
    print(f"Beta            : {beta:.8f}")
    print(f"Gamma           : {gamma:.4f}")
    print(f"R²              : {r2:.4f}")

    print("\nPredicted revenue at selected budgets:")

    for budget in [5000, 10000, 15000, 20000, 25000]:

        prediction = revenue_curve(
            budget,
            alpha,
            beta,
            gamma
        )

        print(
            f"₹{budget:,.0f} → "
            f"₹{prediction:,.2f}"
        )