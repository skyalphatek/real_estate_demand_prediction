import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import os
import logging
from datetime import datetime
import json
import optuna
from optuna.samplers import TPESampler

# Setup logging
def setup_logging():
    """Setup comprehensive logging for tracking model improvements"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"model_experiments_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return log_filename

def purged_time_series_cv(data, n_splits=5, test_size=0.2, embargo_pct=0.01):
    """
    Purged Time Series Cross-Validation to prevent data leakage
    
    Args:
        data: DataFrame with 'date' column
        n_splits: Number of CV folds
        test_size: Size of test set as fraction
        embargo_pct: Embargo period as fraction of total data to prevent leakage
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    data_sorted = data.sort_values('date').reset_index(drop=True)
    n_samples = len(data_sorted)
    embargo_samples = int(embargo_pct * n_samples)
    test_samples = int(test_size * n_samples)
    
    splits = []
    
    for i in range(n_splits):
        # Calculate test period for this fold
        test_start = int(n_samples * (0.5 + 0.1 * i))  # Staggered test periods
        test_end = min(test_start + test_samples, n_samples)
        
        # Training data: everything before test period minus embargo
        train_end = max(0, test_start - embargo_samples)
        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, test_end))
        
        if len(train_indices) > 100 and len(test_indices) > 50:  # Minimum samples
            splits.append((train_indices, test_indices))
    
    return splits

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

# Setup logging
log_filename = setup_logging()
logging.info("=" * 80)
logging.info("REAL ESTATE DEMAND PREDICTION - MODEL TRAINING SESSION")
logging.info("=" * 80)
logging.info(f"Log file: {log_filename}")

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

logging.info(f"Total features available: {len(features)}")
logging.info(f"Feature list: {features[:20]}...")  # Log first 20 features

print(f"Using {len(features)} features for modeling")

# Convert to pandas for sklearn/lightgbm
data_pd = data.to_pandas()

# Convert object columns to numeric, filling with 0 for non-convertible values
for col in data_pd.columns:
    if data_pd[col].dtype == 'object':
        data_pd[col] = pd.to_numeric(data_pd[col], errors='coerce').fillna(0)

# Proper time series split: Train/Test/Validation (70/20/10)
# Respecting temporal order to avoid data leakage
data_sorted = data_pd.sort_values('date').reset_index(drop=True)
n_total = len(data_sorted)

# Calculate split indices
train_end_idx = int(0.70 * n_total)
test_end_idx = int(0.90 * n_total)  # 70% + 20% = 90%

# Get corresponding dates for splits
train_end_date = data_sorted.iloc[train_end_idx]['date']
test_end_date = data_sorted.iloc[test_end_idx]['date']

print(f"Data range: {data_sorted['date'].min()} to {data_sorted['date'].max()}")
print(f"Training cutoff: {train_end_date}")
print(f"Test cutoff: {test_end_date}")

# Create splits based on dates (maintains temporal order)
train_data = data_pd[data_pd['date'] < train_end_date].copy()
test_data = data_pd[(data_pd['date'] >= train_end_date) & (data_pd['date'] < test_end_date)].copy()
val_data = data_pd[data_pd['date'] >= test_end_date].copy()

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]
X_val = val_data[features] 
y_val = val_data[target]

print(f"Training set (70%): {len(X_train)} samples ({len(X_train)/n_total*100:.1f}%)")
print(f"Test set (20%): {len(X_test)} samples ({len(X_test)/n_total*100:.1f}%)")
print(f"Validation set (10%): {len(X_val)} samples ({len(X_val)/n_total*100:.1f}%)")

logging.info(f"Data splits - Train: {len(X_train)}, Test: {len(X_test)}, Val: {len(X_val)}")
logging.info(f"Date ranges - Train: {train_data['date'].min()} to {train_data['date'].max()}")
logging.info(f"Test: {test_data['date'].min()} to {test_data['date'].max()}")
logging.info(f"Validation: {val_data['date'].min()} to {val_data['date'].max()}")

# Perform Purged Cross-Validation for robust MAPE evaluation
print("\nPerforming Purged Cross-Validation...")
logging.info("Starting Purged Cross-Validation for robust model evaluation")

# Use training + test data for CV (validation set remains untouched)
cv_data = pd.concat([train_data, test_data]).reset_index(drop=True)
cv_splits = purged_time_series_cv(cv_data, n_splits=5, test_size=0.15, embargo_pct=0.02)

logging.info(f"Generated {len(cv_splits)} CV folds with purging")

# LightGBM parameters (defined once for consistency)
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

cv_scores = []
cv_mape_scores = []

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    fold_start_time = datetime.now()
    
    # Prepare fold data
    X_fold_train = cv_data.iloc[train_idx][features]
    y_fold_train = cv_data.iloc[train_idx][target]
    X_fold_val = cv_data.iloc[val_idx][features]
    y_fold_val = cv_data.iloc[val_idx][target]
    
    # Train model for this fold
    fold_train_set = lgb.Dataset(X_fold_train, label=y_fold_train)
    fold_val_set = lgb.Dataset(X_fold_val, label=y_fold_val, reference=fold_train_set)
    
    fold_model = lgb.train(
        {**lgb_params, 'verbose': -1},  # Suppress output for CV
        fold_train_set,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100)],
        valid_sets=[fold_val_set],
        valid_names=['val']
    )
    
    # Evaluate fold
    fold_pred = fold_model.predict(X_fold_val)
    fold_custom_score = custom_mape_score(y_fold_val, fold_pred)
    fold_ape = np.abs((y_fold_val - fold_pred) / y_fold_val)
    fold_mape = np.mean(fold_ape[fold_ape <= 1.0])
    
    cv_scores.append(fold_custom_score)
    cv_mape_scores.append(fold_mape)
    
    fold_time = (datetime.now() - fold_start_time).total_seconds()
    
    print(f"Fold {fold + 1}: Custom Score = {fold_custom_score:.4f}, MAPE = {fold_mape:.4f}")
    logging.info(f"Fold {fold + 1} - Custom Score: {fold_custom_score:.4f}, MAPE: {fold_mape:.4f}, Time: {fold_time:.1f}s")

cv_mean_score = np.mean(cv_scores)
cv_std_score = np.std(cv_scores)
cv_mean_mape = np.mean(cv_mape_scores)

print(f"\nCross-Validation Results:")
print(f"Mean Custom Score: {cv_mean_score:.4f} ± {cv_std_score:.4f}")
print(f"Mean MAPE: {cv_mean_mape:.4f}")

logging.info(f"CV Results - Mean Custom Score: {cv_mean_score:.4f} ± {cv_std_score:.4f}")
logging.info(f"CV Mean MAPE: {cv_mean_mape:.4f}")
logging.info(f"Individual CV Scores: {[f'{s:.4f}' for s in cv_scores]}")

# Hyperparameter Optimization with Optuna
print("\nStarting Hyperparameter Optimization with Optuna...")
logging.info("Starting hyperparameter optimization with Optuna")

def objective(trial):
    """Optuna objective function for LightGBM hyperparameter optimization"""
    
    # Suggest hyperparameters - NARROWER search around excellent baseline
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 100, 200),  # Narrowed around baseline 150
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.05),  # Narrowed around baseline 0.03
        'feature_fraction': trial.suggest_float('feature_fraction', 0.85, 0.95),  # Narrowed around baseline 0.9
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 0.9),  # Narrowed around baseline 0.85
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),  # Narrowed around baseline 5
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 25),  # Narrowed around baseline 15
        'min_child_weight': trial.suggest_float('min_child_weight', 0.0005, 0.005, log=True),  # Narrowed
        'lambda_l1': trial.suggest_float('lambda_l1', 0.05, 0.2),  # Narrowed around baseline 0.1
        'lambda_l2': trial.suggest_float('lambda_l2', 0.05, 0.2),  # Narrowed around baseline 0.1
        'max_depth': -1,  # Keep unlimited depth (LightGBM default)
        'random_state': 42,
        'verbose': -1
    }
    
    # Perform cross-validation with suggested parameters
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        # Prepare fold data
        X_fold_train = cv_data.iloc[train_idx][features]
        y_fold_train = cv_data.iloc[train_idx][target]
        X_fold_val = cv_data.iloc[val_idx][features]
        y_fold_val = cv_data.iloc[val_idx][target]
        
        # Train model with suggested parameters
        fold_train_set = lgb.Dataset(X_fold_train, label=y_fold_train)
        fold_val_set = lgb.Dataset(X_fold_val, label=y_fold_val, reference=fold_train_set)
        
        fold_model = lgb.train(
            params,
            fold_train_set,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100)],
            valid_sets=[fold_val_set],
            valid_names=['val']
        )
        
        # Evaluate fold
        fold_pred = fold_model.predict(X_fold_val)
        fold_custom_score = custom_mape_score(y_fold_val, fold_pred)
        fold_scores.append(fold_custom_score)
    
    # Return mean cross-validation score
    mean_score = np.mean(fold_scores)
    return mean_score

# Configure Optuna study
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna's output
study = optuna.create_study(
    direction='maximize',  # Maximize custom MAPE score
    sampler=TPESampler(seed=42),
    study_name='lightgbm_hpo'
)

# Run optimization with narrower search space
# Note: Previous wide search made performance slightly worse because:
# 1. Original parameters were already excellent
# 2. Wide search led to overfitting to CV folds
# 3. Now using narrower ranges around proven baseline values

n_trials = 30  # Reduced trials for focused search
print(f"Running {n_trials} trials for FOCUSED hyperparameter optimization...")
print("Using narrower search space around excellent baseline parameters...")
logging.info(f"Starting focused Optuna optimization with {n_trials} trials")
logging.info("Using narrower search ranges around baseline values to avoid overfitting")

optimization_start_time = datetime.now()
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
optimization_time = (datetime.now() - optimization_start_time).total_seconds()

# Get best parameters
best_params = study.best_params
best_score = study.best_value

print(f"\nOptuna Optimization Results:")
print(f"Best CV Score: {best_score:.4f}")
print(f"Optimization Time: {optimization_time:.1f} seconds")
print(f"Best Parameters: {best_params}")

logging.info(f"Optuna optimization completed in {optimization_time:.1f}s")
logging.info(f"Best CV Score: {best_score:.4f}")
logging.info(f"Best Parameters: {best_params}")

# Update LightGBM parameters with optimized values
optimized_lgb_params = {
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
    **best_params  # Add optimized parameters
}

# Train final model with optimized parameters
print("\nTraining Final LightGBM Model with Optimized Parameters...")
logging.info("Starting final model training with optimized hyperparameters")
logging.info(f"Optimized LightGBM parameters: {optimized_lgb_params}")

train_set = lgb.Dataset(X_train, label=y_train)
test_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

model = lgb.train(
    optimized_lgb_params,
    train_set,
    num_boost_round=2000,  # More rounds with lower learning rate
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)],  # More patience
    valid_sets=[train_set, test_set],
    valid_names=['train', 'test']
)

# Test model on test set (20%) - Used during training for early stopping
test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)
test_custom_score = custom_mape_score(y_test, test_pred)
print(f"\n=== TEST RESULTS (20%) ===")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test Custom MAPE Score: {test_custom_score:.4f}")

# Additional test metrics
ape_test = np.abs((y_test - test_pred) / y_test)
high_error_pct_test = np.mean(ape_test > 1.0) * 100
print(f"Percentage of predictions with APE > 100%: {high_error_pct_test:.2f}%")

# Validate model on validation set (10%) - Final unbiased evaluation
val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
val_custom_score = custom_mape_score(y_val, val_pred)
print(f"\n=== VALIDATION RESULTS (10%) ===")
print(f"Validation MAE: {val_mae:.2f}")
print(f"Validation Custom MAPE Score: {val_custom_score:.4f}")

# Additional validation metrics
ape_val = np.abs((y_val - val_pred) / y_val)
high_error_pct_val = np.mean(ape_val > 1.0) * 100
print(f"Percentage of predictions with APE > 100%: {high_error_pct_val:.2f}%")

# Performance summary
print(f"\n=== PERFORMANCE SUMMARY ===")
print(f"Baseline CV Score: {cv_mean_score:.4f} ± {cv_std_score:.4f}")
print(f"Optimized CV Score: {best_score:.4f}")
print(f"HPO Improvement: {(best_score - cv_mean_score):.4f} ({((best_score - cv_mean_score) / cv_mean_score * 100):+.2f}%)")
print(f"Test Score (20%): {test_custom_score:.4f}")
print(f"Validation Score (10%): {val_custom_score:.4f}")
print(f"Score Difference: {abs(val_custom_score - test_custom_score):.4f}")

# Comprehensive logging of results
experiment_results = {
    "timestamp": datetime.now().isoformat(),
    "cross_validation": {
        "baseline_mean_score": float(cv_mean_score),
        "baseline_std_score": float(cv_std_score),
        "baseline_mean_mape": float(cv_mean_mape),
        "individual_scores": [float(s) for s in cv_scores]
    },
    "hyperparameter_optimization": {
        "best_cv_score": float(best_score),
        "improvement": float(best_score - cv_mean_score),
        "optimization_time_seconds": float(optimization_time),
        "n_trials": n_trials,
        "best_params": best_params
    },
    "test_results": {
        "mae": float(test_mae),
        "custom_score": float(test_custom_score),
        "high_error_pct": float(high_error_pct_test)
    },
    "validation_results": {
        "mae": float(val_mae), 
        "custom_score": float(val_custom_score),
        "high_error_pct": float(high_error_pct_val)
    },
    "model_config": {
        "features_count": len(features),
        "baseline_lgb_params": lgb_params,
        "optimized_lgb_params": optimized_lgb_params,
        "train_samples": len(X_train),
        "test_samples": len(X_test), 
        "val_samples": len(X_val)
    }
}

logging.info("=" * 50)
logging.info("FINAL EXPERIMENT RESULTS")
logging.info("=" * 50)
logging.info(f"Baseline CV Score: {cv_mean_score:.4f} ± {cv_std_score:.4f}")
logging.info(f"Optimized CV Score: {best_score:.4f}")
logging.info(f"HPO Improvement: {(best_score - cv_mean_score):.4f} ({((best_score - cv_mean_score) / cv_mean_score * 100):+.2f}%)")
logging.info(f"Test Score (20%): {test_custom_score:.4f}")
logging.info(f"Validation Score (10%): {val_custom_score:.4f}")
logging.info(f"Model generalization (score diff): {abs(val_custom_score - test_custom_score):.4f}")
logging.info(f"Competition readiness: {'EXCELLENT' if val_custom_score > 0.9 else 'GOOD' if val_custom_score > 0.8 else 'NEEDS IMPROVEMENT'}")

# Save detailed results to JSON for tracking
results_filename = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_filename, 'w') as f:
    json.dump(experiment_results, f, indent=2)

logging.info(f"Detailed results saved to: {results_filename}")

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

# Log feature importance
logging.info("Top 15 Feature Importance:")
for idx, row in feature_importance.head(15).iterrows():
    logging.info(f"{row['feature']}: {row['importance']}")

logging.info(f"Submission file generated: {OUTPUT_FILE}")
logging.info(f"Session completed successfully at {datetime.now()}")
logging.info("=" * 80)