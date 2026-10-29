import os
import pandas as pd
import numpy as np

# Set directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'global_ads_performance_dataset.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PROCESSED_CSV = os.path.join(PROCESSED_DIR, 'campaigns_processed.csv')

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load raw untouched dataset
df = pd.read_csv(RAW_CSV)

print("="*65)
print("PHASE 1: DATASET VERIFICATION & EXPLORATORY DATA ANALYSIS (EDA)")
print("="*65)

# 1. Exact file name and shape
print(f"1. Exact File Name: {os.path.basename(RAW_CSV)}")
print(f"2. Total Row Count: {len(df):,}")
print(f"3. Total Column Count: {len(df.columns)}")
print(f"4. Exact Column Names: {list(df.columns)}")

# 2. Data types
print("\n5. Column Data Types:")
for col, dtype in df.dtypes.items():
    print(f"   - {col:15s} : {dtype}")

# 3. Missing values
print("\n6. Missing Values per Column:")
nulls = df.isnull().sum()
for col, val in nulls.items():
    print(f"   - {col:15s} : {val} ({val/len(df)*100:.2f}%)")

# 4. Duplicates
print(f"\n7. Duplicate Rows Count: {df.duplicated().sum()}")

# 5. Unique platforms & frequencies
print("\n8. Unique Platforms & Frequencies:")
for plat, count in df['platform'].value_counts().items():
    print(f"   - {plat:15s} : {count:,} ({count/len(df)*100:.2f}%)")

# 6. Campaign context features
print("\n9. Campaign Context Features Distribution:")
print("   Campaign Types:")
for ct, count in df['campaign_type'].value_counts().items():
    print(f"     * {ct:12s} : {count:,} ({count/len(df)*100:.2f}%)")
print("   Industries:")
for ind, count in df['industry'].value_counts().items():
    print(f"     * {ind:12s} : {count:,} ({count/len(df)*100:.2f}%)")
print("   Countries:")
for cntry, count in df['country'].value_counts().items():
    print(f"     * {cntry:12s} : {count:,} ({count/len(df)*100:.2f}%)")

# 7. Date range
df['date_parsed'] = pd.to_datetime(df['date'])
min_date = df['date_parsed'].min()
max_date = df['date_parsed'].max()
print(f"\n10. Date Range:")
print(f"    - Earliest Date   : {min_date.strftime('%Y-%m-%d')}")
print(f"    - Latest Date     : {max_date.strftime('%Y-%m-%d')}")
print(f"    - Duration / Span : {(max_date - min_date).days} days (~1 full calendar year)")

# 8. Numeric statistics
print("\n11. Ad Spend Statistics ($):")
for k, v in df['ad_spend'].describe().items():
    print(f"    - {k:8s} : ${v:,.2f}" if isinstance(v, (int, float)) else f"    - {k:8s} : {v}")

print("\n12. Conversions Statistics (Acquisitions Count):")
for k, v in df['conversions'].describe().items():
    print(f"    - {k:8s} : {v:,.2f}" if isinstance(v, (int, float)) else f"    - {k:8s} : {v}")

print("\n13. Revenue Statistics ($):")
for k, v in df['revenue'].describe().items():
    print(f"    - {k:8s} : ${v:,.2f}" if isinstance(v, (int, float)) else f"    - {k:8s} : {v}")

# 9. Exploratory Analysis: Spend vs Conversions & Spend vs Revenue by Platform
print("\n" + "="*65)
print("EXPLORATORY ANALYSIS: SPEND VS CONVERSIONS & REVENUE BY PLATFORM")
print("="*65)

summary = df.groupby('platform').agg(
    Campaigns=('ad_spend', 'count'),
    Total_Spend=('ad_spend', 'sum'),
    Mean_Spend=('ad_spend', 'mean'),
    Median_Spend=('ad_spend', 'median'),
    Total_Conversions=('conversions', 'sum'),
    Mean_Conversions=('conversions', 'mean'),
    Total_Revenue=('revenue', 'sum'),
    Mean_Revenue=('revenue', 'mean'),
    Overall_ROAS=('revenue', lambda r: r.sum() / df.loc[r.index, 'ad_spend'].sum()),
    Overall_CPA=('ad_spend', lambda s: s.sum() / df.loc[s.index, 'conversions'].sum()),
    Corr_Spend_Conversions=('ad_spend', lambda s: s.corr(df.loc[s.index, 'conversions'])),
    Corr_Spend_Revenue=('ad_spend', lambda s: s.corr(df.loc[s.index, 'revenue']))
)
print(summary.to_string())

# 10. Diminishing returns inspection across spend tiers
df['spend_quartile'] = pd.qcut(df['ad_spend'], q=4, labels=['Q1 (Low)', 'Q2 (Mid)', 'Q3 (High)', 'Q4 (Peak)'])
quartile_summary = df.groupby(['platform', 'spend_quartile'], observed=True).agg(
    Avg_Spend=('ad_spend', 'mean'),
    Avg_Conversions=('conversions', 'mean'),
    Avg_Revenue=('revenue', 'mean'),
    Tier_ROAS=('revenue', lambda r: r.sum() / df.loc[r.index, 'ad_spend'].sum()),
    Tier_CPA=('ad_spend', lambda s: s.sum() / df.loc[s.index, 'conversions'].sum())
)
print("\nDiminishing Marginal Returns Analysis by Spend Quartile:")
print(quartile_summary.to_string())

# 11. Create Processed Dataset
# Preserve raw untouched; add calendar features and explicit planned_budget alias
df_proc = df.drop(columns=['date_parsed', 'spend_quartile']).copy()
df_proc['date'] = pd.to_datetime(df_proc['date'])
df_proc['year'] = df_proc['date'].dt.year
df_proc['month'] = df_proc['date'].dt.month
df_proc['day'] = df_proc['date'].dt.day
df_proc['day_of_week'] = df_proc['date'].dt.dayofweek
df_proc['day_name'] = df_proc['date'].dt.day_name()
df_proc['is_weekend'] = df_proc['day_of_week'].isin([5, 6]).astype(int)

# Planned budget alias for explicit planning
df_proc['planned_budget'] = df_proc['ad_spend']

df_proc.to_csv(PROCESSED_CSV, index=False)
print("\n" + "="*65)
print(f"Cleaned processed dataset successfully created: {PROCESSED_CSV}")
print(f"Processed shape: {df_proc.shape}")
print("="*65)
