import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------
# Platform-specific fitted curve parameters
# --------------------------------------------------

PARAMS = {
    "Google Ads": {
        "alpha": 38833.0620,
        "beta": 0.00015731,
        "gamma": 343.0009,
        "min_budget": 5000,
        "max_budget": 25000
    },

    "Meta Ads": {
        "alpha": 21685.8715,
        "beta": 0.00055104,
        "gamma": -789.4299,
        "min_budget": 5000,
        "max_budget": 15000
    },

    "TikTok Ads": {
        "alpha": 125092.3500,
        "beta": 0.00006146,
        "gamma": 8969.1745,
        "min_budget": 5000,
        "max_budget": 20000
    }
}


# --------------------------------------------------
# Revenue response curve
# --------------------------------------------------

def revenue_curve(budget, alpha, beta, gamma):

    return alpha * np.log1p(beta * budget) + gamma


# --------------------------------------------------
# Predict revenue for a platform
# --------------------------------------------------

def predicted_revenue(platform, budget):

    p = PARAMS[platform]

    return revenue_curve(
        budget,
        p["alpha"],
        p["beta"],
        p["gamma"]
    )


# --------------------------------------------------
# Budget optimization
# --------------------------------------------------

def optimize_budget(total_budget, market="Global"):

    # ----------------------------------------------
    # Select platforms based on market
    # ----------------------------------------------

    if market == "India":

        platforms = [
            "Google Ads",
            "Meta Ads"
        ]

    else:

        platforms = [
            "Google Ads",
            "Meta Ads",
            "TikTok Ads"
        ]


    # ----------------------------------------------
    # Get platform budget limits
    # ----------------------------------------------

    minimums = np.array([
        PARAMS[p]["min_budget"]
        for p in platforms
    ])

    maximums = np.array([
        PARAMS[p]["max_budget"]
        for p in platforms
    ])


    # ----------------------------------------------
    # Check feasible budget range
    # ----------------------------------------------

    minimum_total = minimums.sum()
    maximum_total = maximums.sum()

    if total_budget < minimum_total:

        raise ValueError(
            f"Minimum feasible total budget for {market} "
            f"is ₹{minimum_total:,.0f}"
        )


    if total_budget > maximum_total:

        raise ValueError(
            f"Maximum supported total budget for {market} "
            f"is ₹{maximum_total:,.0f}"
        )


    # ----------------------------------------------
    # Objective function
    # ----------------------------------------------

    def objective(budgets):

        total_revenue = sum(
            predicted_revenue(
                platform,
                budget
            )
            for platform, budget in zip(
                platforms,
                budgets
            )
        )

        # SLSQP minimizes,
        # so return negative revenue
        return -total_revenue


    # ----------------------------------------------
    # Initial feasible allocation
    # ----------------------------------------------

    initial = minimums.astype(float)

    remaining = (
        total_budget
        - minimum_total
    )


    for i in range(len(platforms)):

        available = (
            maximums[i]
            - minimums[i]
        )

        addition = min(
            remaining,
            available
        )

        initial[i] += addition

        remaining -= addition

        if remaining <= 0:
            break


    # ----------------------------------------------
    # Run constrained optimization
    # ----------------------------------------------

    result = minimize(

        objective,

        initial,

        method="SLSQP",

        bounds=list(
            zip(
                minimums,
                maximums
            )
        ),

        constraints={
            "type": "eq",
            "fun": lambda x:
                np.sum(x) - total_budget
        }
    )


    # ----------------------------------------------
    # Check optimization result
    # ----------------------------------------------

    if not result.success:

        raise RuntimeError(
            f"Optimization failed: {result.message}"
        )


    # ----------------------------------------------
    # Create allocation result
    # ----------------------------------------------

    allocation = dict(
        zip(
            platforms,
            result.x
        )
    )


    # ----------------------------------------------
    # Calculate predicted revenue
    # ----------------------------------------------

    predictions = {

        platform:
        predicted_revenue(
            platform,
            allocation[platform]
        )

        for platform in platforms
    }


    # ----------------------------------------------
    # Total predicted revenue
    # ----------------------------------------------

    total_predicted_revenue = sum(
        predictions.values()
    )


    return (
        allocation,
        predictions,
        total_predicted_revenue
    )


# --------------------------------------------------
# Test optimizer
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("BUDGET RESPONSE OPTIMIZATION")
    print("=" * 80)


    # ----------------------------------------------
    # Global market tests
    # ----------------------------------------------

    print()
    print("GLOBAL MARKET")
    print("=" * 80)


    for total_budget in [
        20000,
        30000,
        40000,
        50000,
        60000
    ]:

        allocation, predictions, total_revenue = (
            optimize_budget(
                total_budget,
                market="Global"
            )
        )


        print()
        print(
            f"Total Budget: ₹{total_budget:,.0f}"
        )

        print("-" * 80)


        for platform in allocation:

            print(
                f"{platform:<15} "
                f"Budget: ₹{allocation[platform]:,.2f}  "
                f"Predicted Revenue: "
                f"₹{predictions[platform]:,.2f}"
            )


        print(
            f"Total Predicted Revenue: "
            f"₹{total_revenue:,.2f}"
        )


    # ----------------------------------------------
    # India market tests
    # ----------------------------------------------

    print()
    print()
    print("INDIA MARKET")
    print("=" * 80)


    for total_budget in [
        10000,
        20000,
        30000,
        40000
    ]:

        allocation, predictions, total_revenue = (
            optimize_budget(
                total_budget,
                market="India"
            )
        )


        print()
        print(
            f"Total Budget: ₹{total_budget:,.0f}"
        )

        print("-" * 80)


        for platform in allocation:

            print(
                f"{platform:<15} "
                f"Budget: ₹{allocation[platform]:,.2f}  "
                f"Predicted Revenue: "
                f"₹{predictions[platform]:,.2f}"
            )


        print(
            f"Total Predicted Revenue: "
            f"₹{total_revenue:,.2f}"
        )