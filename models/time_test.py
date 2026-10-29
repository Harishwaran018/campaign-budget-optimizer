import pandas as pd
import numpy as np
import os
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from feature_pipeline import PRE_CAMPAIGN_FEATURES, transform_campaign_features

# Load data
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "campaigns_processed.csv")
df = pd.read_csv(DATA_PATH)

# Convert date & sort chronologically (identical time split)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Pre-campaign features only (no post-campaign leakage)
features = PRE_CAMPAIGN_FEATURES
target = "revenue"

X = df[features]
y = np.log1p(df[target])

# First 80% = past (training)
# Last 20% = future (testing)
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

cat_cols = [
    "platform",
    "campaign_type",
    "industry",
    "country",
    "plat_camp",
    "plat_country",
    "plat_ind"
]

num_cols = [
    "planned_budget",
    "log_budget",
    "sqrt_budget",
    "sin_month",
    "cos_month",
    "month",
    "day_of_week",
    "is_weekend"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        ),
        (
            "num",
            StandardScaler(),
            num_cols
        )
    ]
)

# Robust Gradient Boosting with Huber Loss & engineered pre-campaign features
model = GradientBoostingRegressor(
    loss="huber",
    n_estimators=180,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=11,
    subsample=0.8,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("feature_engineering", FunctionTransformer(transform_campaign_features, validate=False)),
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train on historical 80%
pipeline.fit(X_train, y_train)

# Predict on future 20%
pred_log = pipeline.predict(X_test)

# Convert back to revenue units
pred_revenue = np.clip(np.expm1(pred_log), 0, None)
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
