import numpy as np

from budget_response_optimizer import (
    PARAMS,
    predicted_revenue,
    optimize_budget
)


def grid_search(total_budget, step=100):

    best_allocation = None
    best_revenue = -np.inf

    google_min = PARAMS["Google Ads"]["min_budget"]
    google_max = PARAMS["Google Ads"]["max_budget"]

    meta_min = PARAMS["Meta Ads"]["min_budget"]
    meta_max = PARAMS["Meta Ads"]["max_budget"]

    tiktok_min = PARAMS["TikTok Ads"]["min_budget"]
    tiktok_max = PARAMS["TikTok Ads"]["max_budget"]

    for google in np.arange(
        google_min,
        google_max + step,
        step
    ):

        for meta in np.arange(
            meta_min,
            meta_max + step,
            step
        ):

            tiktok = total_budget - google - meta

            # Check TikTok constraints
            if tiktok < tiktok_min:
                continue

            if tiktok > tiktok_max:
                continue

            # Calculate predicted total revenue
            revenue = (
                predicted_revenue("Google Ads", google)
                + predicted_revenue("Meta Ads", meta)
                + predicted_revenue("TikTok Ads", tiktok)
            )

            # Keep the best allocation
            if revenue > best_revenue:

                best_revenue = revenue

                best_allocation = {
                    "Google Ads": google,
                    "Meta Ads": meta,
                    "TikTok Ads": tiktok
                }

    return best_allocation, best_revenue


if __name__ == "__main__":

    print("\nOPTIMIZER VALIDATION")
    print("=" * 90)

    for total_budget in [20000, 30000, 40000, 50000, 60000]:

        # --------------------------------------------------
        # SLSQP optimizer
        # --------------------------------------------------

        optimizer_allocation, optimizer_predictions, optimizer_revenue = (
            optimize_budget(total_budget)
        )

        # --------------------------------------------------
        # Independent grid search
        # --------------------------------------------------

        grid_allocation, grid_revenue = grid_search(
            total_budget,
            step=100
        )

        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        print(
            f"\nTotal Budget: ₹{total_budget:,.0f}"
        )

        print("-" * 90)

        print("SLSQP allocation:")

        for platform, budget in optimizer_allocation.items():

            print(
                f"  {platform:<12}: ₹{budget:,.2f}"
            )

        print(
            f"  Revenue       : ₹{optimizer_revenue:,.2f}"
        )

        print("\nGrid-search allocation:")

        for platform, budget in grid_allocation.items():

            print(
                f"  {platform:<12}: ₹{budget:,.2f}"
            )

        print(
            f"  Revenue       : ₹{grid_revenue:,.2f}"
        )

        # --------------------------------------------------
        # Compare both methods
        # --------------------------------------------------

        revenue_difference = abs(
            optimizer_revenue - grid_revenue
        )

        print(
            f"\nRevenue difference: "
            f"₹{revenue_difference:,.2f}"
        )

        # Difference as percentage
        if grid_revenue != 0:

            percentage_difference = (
                revenue_difference / grid_revenue
            ) * 100

            print(
                f"Percentage difference: "
                f"{percentage_difference:.4f}%"
            )

        # --------------------------------------------------
        # Validation result
        # --------------------------------------------------

        if revenue_difference < 100:

            print(
                "Validation: PASS ✅"
            )

        else:

            print(
                "Validation: CHECK ⚠️"
            )

    print("\n" + "=" * 90)
    print("Validation completed.")