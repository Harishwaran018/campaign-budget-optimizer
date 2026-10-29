import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
df = pd.read_csv("../data/processed/campaigns_processed.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# Pre-campaign features only
features = [
    "platform",
    "campaign_type",
    "industry",
    "country",
    "month",
    "day_of_week",
    "is_weekend",
    "planned_budget"
]

target = "revenue"

X = df[features]
y = np.log1p(df[target])

# First 80% = past
# Last 20% = future
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Categorical and numerical columns
categorical_features = [
    "platform",
    "campaign_type",
    "industry",
    "country"
]

numerical_features = [
    "month",
    "day_of_week",
    "is_weekend",
    "planned_budget"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

# Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train
pipeline.fit(X_train, y_train)

# Predict
pred_log = pipeline.predict(X_test)

# Convert back to revenue
pred_revenue = np.expm1(pred_log)
actual_revenue = np.expm1(y_test)

# Metrics
mae = mean_absolute_error(actual_revenue, pred_revenue)
rmse = np.sqrt(mean_squared_error(actual_revenue, pred_revenue))
r2 = r2_score(actual_revenue, pred_revenue)

print("=" * 45)
print("       TIME-BASED MODEL VALIDATION")
print("=" * 45)

print(f"Training campaigns : {len(X_train)}")
print(f"Testing campaigns  : {len(X_test)}")

print(f"Training dates     : {df['date'].iloc[0].date()} → {df['date'].iloc[split_index-1].date()}")
print(f"Testing dates      : {df['date'].iloc[split_index].date()} → {df['date'].iloc[-1].date()}")

print()
print(f"R²   : {r2:.4f}")
print(f"MAE  : ₹{mae:.2f}")
print(f"RMSE : ₹{rmse:.2f}")

print("=" * 45)