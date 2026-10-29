import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load data
data = pd.read_csv("../data/processed/campaigns_processed.csv")


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

X = data[features]
y = np.log1p(data["revenue"])


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


# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


X_processed = preprocessor.fit_transform(X)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42
)


# Identify the column corresponding to planned_budget
feature_names = preprocessor.get_feature_names_out()

monotonic_constraints = []

for feature in feature_names:

    if feature == "numeric__planned_budget":
        monotonic_constraints.append(1)
    else:
        monotonic_constraints.append(0)


# Monotonic Gradient Boosting
model = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42,
    monotonic_cst=monotonic_constraints
)


model.fit(X_train, y_train)


# Predictions
predicted_log = model.predict(X_test)

predictions = np.expm1(predicted_log)
actual = np.expm1(y_test)


# Metrics
r2 = r2_score(actual, predictions)
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))


print("=" * 60)
print("       MONOTONIC GRADIENT BOOSTING MODEL")
print("=" * 60)

print(f"R²   : {r2:.4f}")
print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")

print("=" * 60)