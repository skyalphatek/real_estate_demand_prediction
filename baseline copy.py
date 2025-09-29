import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import os

def custom_mape_score(y_true, y_pred):
    """
    Custom two-stage MAPE evaluation metric from competition rules:
    
    Stage 1: If over 30% of samples have absolute percentage errors > 100%, score = 0
    Stage 2: Otherwise, calculate MAPE with samples having APE <= 1, 
             divide by fraction of samples with APE <= 1, final score = 1 - scaled_MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    if len(y_true_filtered) == 0:
        return 0
    
    # Calculate absolute percentage errors
    ape = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
    
    # Stage 1: Check if over 30% have APE > 100% (> 1.0)
    high_error_fraction = np.mean(ape > 1.0)
    if high_error_fraction > 0.3:
        return 0
    
    # Stage 2: Calculate MAPE with samples having APE <= 1
    valid_mask = ape <= 1.0
    if np.sum(valid_mask) == 0:
        return 0
    
    # MAPE on valid samples
    mape = np.mean(ape[valid_mask])
    
    # Fraction of valid samples
    valid_fraction = np.mean(valid_mask)
    
    # Scaled MAPE
    scaled_mape = mape / valid_fraction
    
    # Final score
    score = 1 - scaled_mape
    
    return max(0, score)  # Ensure score is non-negative

# Configuration
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
OUTPUT_FILE = "submission.csv"
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

print("Loading and processing all training data...")

# Load all data files
def load_all_data():
    # Main transaction data
    df_new = pl.read_csv(os.path.join(TRAIN_DIR, "new_house_transactions.csv"), infer_schema_length=10000)
    df_pre = pl.read_csv(os.path.join(TRAIN_DIR, "pre_owned_house_transactions.csv"), infer_schema_length=10000)
    df_land = pl.read_csv(os.path.join(TRAIN_DIR, "land_transactions.csv"), infer_schema_length=10000)
    
    # Nearby sector data
    df_new_nearby = pl.read_csv(os.path.join(TRAIN_DIR, "new_house_transactions_nearby_sectors.csv"), infer_schema_length=10000)
    df_pre_nearby = pl.read_csv(os.path.join(TRAIN_DIR, "pre_owned_house_transactions_nearby_sectors.csv"), infer_schema_length=10000)
    df_land_nearby = pl.read_csv(os.path.join(TRAIN_DIR, "land_transactions_nearby_sectors.csv"), infer_schema_length=10000)
    
    # Static features
    df_poi = pl.read_csv(os.path.join(TRAIN_DIR, "sector_POI.csv"), infer_schema_length=10000)
    df_city = pl.read_csv(os.path.join(TRAIN_DIR, "city_indexes.csv"), infer_schema_length=10000)
    
    return df_new, df_pre, df_land, df_new_nearby, df_pre_nearby, df_land_nearby, df_poi, df_city

df_new, df_pre, df_land, df_new_nearby, df_pre_nearby, df_land_nearby, df_poi, df_city = load_all_data()

# Process main new house transactions (target data)
def process_main_data(df_new):
    df_new = df_new.with_columns([
        pl.col("sector").str.extract(r"sector (\d+)").cast(pl.Int64).alias("sector_id"),
        pl.col("month").str.strptime(pl.Date, "%Y-%b").alias("date")
    ]).drop("sector")
    return df_new

# Process pre-owned house data
def process_pre_owned(df_pre):
    df_pre = df_pre.with_columns([
        pl.col("sector").str.extract(r"sector (\d+)").cast(pl.Int64).alias("sector_id"),
        pl.col("month").str.strptime(pl.Date, "%Y-%b").alias("date")
    ]).drop(["sector", "month"]).select([
        "date", "sector_id", "area_pre_owned_house_transactions", 
        "amount_pre_owned_house_transactions", "num_pre_owned_house_transactions",
        "price_pre_owned_house_transactions"
    ])
    return df_pre

# Process land transaction data  
def process_land(df_land):
    df_land = df_land.with_columns([
        pl.col("sector").str.extract(r"sector (\d+)").cast(pl.Int64).alias("sector_id"),
        pl.col("month").str.strptime(pl.Date, "%Y-%b").alias("date")
    ]).drop(["sector", "month"]).select([
        "date", "sector_id", "num_land_transactions", "construction_area",
        "planned_building_area", "transaction_amount"
    ]).rename({"transaction_amount": "land_transaction_amount"})
    return df_land

# Process nearby sector data
def process_nearby_data(df, prefix):
    df = df.with_columns([
        pl.col("sector").str.extract(r"sector (\d+)").cast(pl.Int64).alias("sector_id"),
        pl.col("month").str.strptime(pl.Date, "%Y-%b").alias("date")
    ]).drop(["sector", "month"])
    
    # Rename columns to avoid conflicts
    rename_dict = {}
    for col in df.columns:
        if col not in ["date", "sector_id"]:
            rename_dict[col] = f"{prefix}_{col}"
    
    if rename_dict:
        df = df.rename(rename_dict)
    return df

# Process POI data
def process_poi(df_poi):
    df_poi = df_poi.with_columns(
        pl.col("sector").str.extract(r"sector (\d+)").cast(pl.Int64).alias("sector_id")
    ).drop("sector")
    return df_poi

# Process city index data
def process_city(df_city):
    df_city = df_city.with_columns(
        pl.col("city_indicator_data_year").cast(pl.Int32).alias("year")
    ).drop("city_indicator_data_year")
    return df_city

print("Processing all datasets...")

# Process all data
df_new_proc = process_main_data(df_new)
df_pre_proc = process_pre_owned(df_pre)
df_land_proc = process_land(df_land)

df_new_nearby_proc = process_nearby_data(df_new_nearby, "nearby_new")
df_pre_nearby_proc = process_nearby_data(df_pre_nearby, "nearby_pre")
df_land_nearby_proc = process_nearby_data(df_land_nearby, "nearby_land")

df_poi_proc = process_poi(df_poi)
df_city_proc = process_city(df_city)

print("Merging all datasets...")

# Merge all data
data = df_new_proc
data = data.join(df_pre_proc, on=["date", "sector_id"], how="left")
data = data.join(df_land_proc, on=["date", "sector_id"], how="left")
data = data.join(df_new_nearby_proc, on=["date", "sector_id"], how="left")
data = data.join(df_pre_nearby_proc, on=["date", "sector_id"], how="left")
data = data.join(df_land_nearby_proc, on=["date", "sector_id"], how="left")
data = data.join(df_poi_proc, on="sector_id", how="left")

# Add year for city index merge
data = data.with_columns(pl.col("date").dt.year().alias("year"))
data = data.join(df_city_proc, on="year", how="left")

print("Creating features...")

# Feature engineering
data = data.with_columns([
    pl.col("date").dt.month().alias("month_num"),
    pl.col("date").dt.quarter().alias("quarter"),
    pl.col("date").dt.ordinal_day().alias("day_of_year"),
    (pl.col("date") - pl.date(2019, 1, 1)).dt.total_days().alias("days_from_start")
])

# Calculate rolling features for key metrics
data = data.sort(["sector_id", "date"])

# Create lag features for target variable by sector
for lag in [1, 2, 3, 6, 12]:
    data = data.with_columns(
        pl.col("amount_new_house_transactions")
        .shift(lag)
        .over("sector_id")
        .alias(f"amount_lag_{lag}")
    )

# Calculate rolling averages 
for window in [3, 6, 12]:
    data = data.with_columns(
        pl.col("amount_new_house_transactions")
        .rolling_mean(window_size=window)
        .over("sector_id")
        .alias(f"amount_rolling_mean_{window}")
    )

print("Preparing data for modeling...")

# Fill missing values
data = data.fill_null(0)

# Define target and features
target = "amount_new_house_transactions"
exclude_cols = ["date", "month", "sector_id", target, "year"]
features = [col for col in data.columns if col not in exclude_cols]

print(f"Using {len(features)} features for modeling")

# Convert to pandas for sklearn/lightgbm
data_pd = data.to_pandas()

# Convert object columns to numeric, filling with 0 for non-convertible values
for col in data_pd.columns:
    if data_pd[col].dtype == 'object':
        data_pd[col] = pd.to_numeric(data_pd[col], errors='coerce').fillna(0)

# Split data: use 2019-2023 for training, 2024 for validation
train_data = data_pd[data_pd['date'] < '2024-01-01'].copy()
val_data = data_pd[data_pd['date'] >= '2024-01-01'].copy()

X_train = train_data[features]
y_train = train_data[target]
X_val = val_data[features]
y_val = val_data[target]

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# Train LightGBM model
print("Training LightGBM model...")

lgb_params = {
    'objective': 'regression',
    'metric': 'mape',  # Use MAPE instead of MAE for better alignment
    'boosting_type': 'gbdt',
    'num_leaves': 150,  # Increased for more complexity
    'learning_rate': 0.03,  # Lower learning rate for better convergence
    'feature_fraction': 0.9,  # Use more features
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 15,  # More flexibility
    'min_child_weight': 0.001,
    'lambda_l1': 0.1,  # L1 regularization
    'lambda_l2': 0.1,  # L2 regularization
    'random_state': 42,
    'verbose': -1
}

train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

model = lgb.train(
    lgb_params,
    train_set,
    num_boost_round=2000,  # More rounds with lower learning rate
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)],  # More patience
    valid_sets=[train_set, val_set],
    valid_names=['train', 'val']
)

# Validate model
val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
val_custom_score = custom_mape_score(y_val, val_pred)
print(f"Validation MAE: {val_mae:.2f}")
print(f"Validation Custom MAPE Score: {val_custom_score:.4f}")

# Additional validation metrics
ape = np.abs((y_val - val_pred) / y_val)
high_error_pct = np.mean(ape > 1.0) * 100
print(f"Percentage of predictions with APE > 100%: {high_error_pct:.2f}%")

print("Generating predictions for submission...")

# Load submission template and generate predictions
submission_df = pl.read_csv(SAMPLE_SUBMISSION_FILE)

# Parse submission IDs 
submission_df = submission_df.with_columns([
    pl.col("id").str.extract(r"sector (\d+)").cast(pl.Int64).alias("sector_id"),
    pl.col("id").str.slice(0, 8).str.strptime(pl.Date, "%Y %b").alias("date")
])

# Add year for city index merge
submission_df = submission_df.with_columns(pl.col("date").dt.year().alias("year"))

# Get the latest available data for each sector to use as base for predictions
latest_data = data.group_by("sector_id").last()

# Join submission requirements with latest sector data
pred_data = submission_df.join(
    latest_data.drop(["date", "month", target]), 
    on="sector_id", 
    how="left"
)

# Update temporal features for prediction dates
pred_data = pred_data.with_columns([
    pl.col("date").dt.month().alias("month_num"),
    pl.col("date").dt.quarter().alias("quarter"), 
    pl.col("date").dt.ordinal_day().alias("day_of_year"),
    (pl.col("date") - pl.date(2019, 1, 1)).dt.total_days().alias("days_from_start")
])

# Join with city data for prediction year
pred_data = pred_data.join(df_city_proc, on="year", how="left")
pred_data = pred_data.fill_null(0)

# Convert to pandas and make predictions
pred_data_pd = pred_data.to_pandas()

# Convert object columns to numeric, filling with 0 for non-convertible values
for col in pred_data_pd.columns:
    if pred_data_pd[col].dtype == 'object':
        pred_data_pd[col] = pd.to_numeric(pred_data_pd[col], errors='coerce').fillna(0)

X_pred = pred_data_pd[features]

predictions = model.predict(X_pred)

# Create submission file
submission = pd.DataFrame({
    'id': submission_df.to_pandas()['id'],
    'new_house_transaction_amount': predictions
})

# Save submission
submission.to_csv(OUTPUT_FILE, index=False)

print(f"Submission file created: {OUTPUT_FILE}")
print(f"Sample predictions:")
print(submission.head(10))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 15 most important features:")
print(feature_importance.head(15))