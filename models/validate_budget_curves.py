import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Load data
# --------------------------------------------------

data = pd.read_csv(
    "../data/processed/campaigns_processed.csv"
)


# --------------------------------------------------
# Diminishing-returns function
# --------------------------------------------------

def revenue_curve(budget, alpha, beta, gamma):

    return (
        alpha * np.log1p(beta * budget)
        + gamma
    )


# --------------------------------------------------
# Platforms
# --------------------------------------------------

platforms = [
    "Google Ads",
    "Meta Ads",
    "TikTok Ads"
]


print("\nBUDGET RESPONSE CURVE VALIDATION")
print("=" * 90)


# --------------------------------------------------
# Validate each platform
# --------------------------------------------------

for platform in platforms:

    platform_data = data[
        data["platform"] == platform
    ].copy()

    X = platform_data[
        "planned_budget"
    ].values

    y = platform_data[
        "revenue"
    ].values


    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


    # --------------------------------------------------
    # Fit curve using TRAINING data only
    # --------------------------------------------------

    params, _ = curve_fit(
        revenue_curve,
        X_train,
        y_train,
        p0=[
            30000,
            0.001,
            0
        ],
        bounds=(
            [
                0,
                0,
                -np.inf
            ],
            [
                np.inf,
                np.inf,
                np.inf
            ]
        ),
        maxfev=20000
    )


    alpha, beta, gamma = params


    # --------------------------------------------------
    # Predictions on unseen TEST data
    # --------------------------------------------------

    y_pred = revenue_curve(
        X_test,
        alpha,
        beta,
        gamma
    )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    r2 = r2_score(
        y_test,
        y_pred
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(f"\n{platform}")
    print("-" * 90)

    print(
        f"Training campaigns : {len(X_train)}"
    )

    print(
        f"Testing campaigns  : {len(X_test)}"
    )

    print(
        f"Alpha              : {alpha:.4f}"
    )

    print(
        f"Beta               : {beta:.8f}"
    )

    print(
        f"Gamma              : {gamma:.4f}"
    )

    print(
        f"\nTest R²            : {r2:.4f}"
    )

    print(
        f"Test MAE           : ₹{mae:,.2f}"
    )

    print(
        f"Test RMSE          : ₹{rmse:,.2f}"
    )


print("\n" + "=" * 90)
print("Validation completed.")