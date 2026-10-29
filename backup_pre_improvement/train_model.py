import pandas as pd
import numpy as np
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load data
df = pd.read_csv("../data/processed/campaigns_processed.csv")

# Pre-campaign features
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


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Feature types
categorical_features = [
    "platform",
    "campaign_type",
    "industry",
    "country"
]

numeric_features = [
    "month",
    "day_of_week",
    "is_weekend",
    "planned_budget"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# Final model
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train final model
pipeline.fit(X_train, y_train)


# Test final model
predicted_log = pipeline.predict(X_test)

y_pred = np.expm1(predicted_log)
y_actual = np.expm1(y_test)


# Metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)


# Create save folder
os.makedirs("saved_models", exist_ok=True)


# Save model
joblib.dump(
    pipeline,
    "saved_models/campaign_revenue_model.pkl"
)


# Save metrics
metrics = {
    "model": "Gradient Boosting Regressor",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

with open("saved_models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("\n======================================")
print("       FINAL PHASE 2 MODEL")
print("======================================")
print("Model: Gradient Boosting Regressor")
print(f"R²   : {r2:.4f}")
print(f"MAE  : ₹{mae:.2f}")
print(f"RMSE : ₹{rmse:.2f}")

print("\nModel saved to:")
print("saved_models/campaign_revenue_model.pkl")

print("\nMetrics saved to:")
print("saved_models/model_metrics.json")

print("\n======================================")
print("       PHASE 2 COMPLETED!")
print("======================================")