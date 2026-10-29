from budget_optimizer_v2 import optimize_budget


budgets_to_test = [
    30000,
    50000,
    75000,
    90000
]


print("=" * 70)
print("              MULTI-BUDGET VALIDATION")
print("=" * 70)


for total_budget in budgets_to_test:

    allocation, predictions, revenue = optimize_budget(total_budget)

    print(f"\nTotal Budget: ₹{total_budget:,.0f}")

    print(
        f"Google Ads: ₹{allocation[0]:,.0f} | "
        f"Meta Ads: ₹{allocation[1]:,.0f} | "
        f"TikTok Ads: ₹{allocation[2]:,.0f}"
    )

    print(f"Predicted Revenue: ₹{revenue:,.2f}")


print("\n" + "=" * 70)