import pandas as pd
import numpy as np

# Valid pre-campaign features only (guarantees zero data leakage)
PRE_CAMPAIGN_FEATURES = [
    "platform",
    "campaign_type",
    "industry",
    "country",
    "month",
    "day_of_week",
    "is_weekend",
    "planned_budget"
]

def transform_campaign_features(X_in):
    """
    Transforms raw pre-campaign features into enriched features for model training and inference.
    
    Transformations:
    - log1p(planned_budget) & sqrt(planned_budget) for non-linear advertising response.
    - Cyclical month representations (sin_month, cos_month).
    - Cross-feature interactions (platform x campaign_type, platform x country, platform x industry).
    
    Zero data leakage: relies solely on pre-campaign inputs.
    """
    if not isinstance(X_in, pd.DataFrame):
        X_in = pd.DataFrame(X_in, columns=PRE_CAMPAIGN_FEATURES)
    
    X_out = X_in.copy()
    
    # Non-linear budget representations
    X_out["log_budget"] = np.log1p(X_out["planned_budget"])
    X_out["sqrt_budget"] = np.sqrt(X_out["planned_budget"])
    
    # Cyclical seasonal features
    X_out["sin_month"] = np.sin(2 * np.pi * X_out["month"] / 12.0)
    X_out["cos_month"] = np.cos(2 * np.pi * X_out["month"] / 12.0)
    
    # Platform context interactions
    X_out["plat_camp"] = X_out["platform"].astype(str) + "_" + X_out["campaign_type"].astype(str)
    X_out["plat_country"] = X_out["platform"].astype(str) + "_" + X_out["country"].astype(str)
    X_out["plat_ind"] = X_out["platform"].astype(str) + "_" + X_out["industry"].astype(str)
    
    return X_out
