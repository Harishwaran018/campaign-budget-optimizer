import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load data
data = pd.read_csv("../data/processed/campaigns_processed.csv")


# Features
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


# One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            PolynomialFeatures(degree=2, include_bias=False),
            numeric_features
        )
    ]
)


# Smooth nonlinear regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=10.0))
])


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Predict
predicted_log = model.predict(X_test)

predictions = np.expm1(predicted_log)
actual = np.expm1(y_test)


# Metrics
r2 = r2_score(actual, predictions)
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))


print("=" * 60)
print("       SMOOTH BUDGET-RESPONSE MODEL")
print("=" * 60)

print(f"R²   : {r2:.4f}")
print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")

print("=" * 60)