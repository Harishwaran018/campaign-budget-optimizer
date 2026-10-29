import pandas as pd
import numpy as np
import os
import sys
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ensure local module imports work from either models/ or root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from feature_pipeline import PRE_CAMPAIGN_FEATURES, transform_campaign_features

# Load data
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "campaigns_processed.csv")
df = pd.read_csv(DATA_PATH)

features = PRE_CAMPAIGN_FEATURES
target = "revenue"

X = df[features]
y = np.log1p(df[target])

# Train/test split (exact same seed and split as baseline)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature definitions for post-engineering transformation
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

# Robust column transformer
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

# Tuned Robust Gradient Boosting Model
# Huber loss provides resistance against high-revenue outliers
# Learning rate 0.03, leaf size 11, and subsample 0.8 prevent overfitting
model = GradientBoostingRegressor(
    loss="huber",
    n_estimators=180,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=11,
    subsample=0.8,
    random_state=42
)

# Complete end-to-end pipeline (guarantees training and inference use identical transformations)
pipeline = Pipeline(
    steps=[
        ("feature_engineering", FunctionTransformer(transform_campaign_features, validate=False)),
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train model
pipeline.fit(X_train, y_train)

# Test model
predicted_log = pipeline.predict(X_test)
y_pred = np.clip(np.expm1(predicted_log), 0, None)
y_actual = np.expm1(y_test)

# Metrics in original revenue units
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

# Save directory
SAVED_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVED_DIR, "campaign_revenue_model.pkl")
METRICS_PATH = os.path.join(SAVED_DIR, "model_metrics.json")

# Save model
joblib.dump(pipeline, MODEL_PATH)

# Save metrics
metrics = {
    "model": "Robust Gradient Boosting Regressor (Huber Loss & Engineered Pre-Campaign Features)",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n======================================")
print("       IMPROVED REVENUE MODEL")
print("======================================")
print("Model: Robust Gradient Boosting Regressor (Huber Loss)")
print(f"R²   : {r2:.4f}")
print(f"MAE  : ₹{mae:.2f}")
print(f"RMSE : ₹{rmse:.2f}")

print("\nModel saved to:")
print(MODEL_PATH)

print("\nMetrics saved to:")
print(METRICS_PATH)

print("\n======================================")
print("       TRAINING COMPLETED!")
print("======================================")
