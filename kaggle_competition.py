import os
# packages-EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# packages- time series and models
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

from pathlib import Path

import optuna
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Define base directory and data folder
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train"


# ---------------------------
#  Month mapping
# ---------------------------
def build_month_codes():
    return {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }


# ---------------------------
#  Parse id into month text and sector string
# ---------------------------
def split_test_id_column(df):
    parts = df.id.str.split('_', expand=True)
    df['month_text'] = parts[0]
    df['sector'] = parts[1]
    return df


# ---------------------------
#  Add parsed time fields to a dataframe
# ---------------------------
def add_time_and_sector_fields(df, month_codes):
    if 'sector' in df.columns:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
    if 'month' not in df.columns:
        df['month'] = df['month_text'].str.slice(5, None).map(month_codes)
        df['year'] = df['month_text'].str.slice(0, 4).astype(int)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    else:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    return df


# ---------------------------
#  Load competition tables used for submission
# ---------------------------
def load_competition_data():

# Target variable data
    train_nht = pd.read_csv(DATA_PATH / "new_house_transactions.csv")

# Test data
    test = pd.read_csv("test.csv")
    return train_nht, test

# ---------------------------
#  Build training matrix: amount_new_house_transactions [time x sector_id]
# ---------------------------
def build_amount_matrix(train_nht, month_codes):
    train_nht= add_time_and_sector_fields(train_nht.copy(), month_codes)
    pivot = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
    pivot = pivot.fillna(0)
    all_sectors = np.arange(1, 97)
    for s in all_sectors:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[all_sectors]
    return pivot


# ---------------------------
#  NEW: Create seasonal features
# ---------------------------
def create_seasonal_features(time_index):
    """Create cyclical seasonal features"""
    features = pd.DataFrame(index=time_index)
    
    # Month-based seasonality
    month_in_year = (time_index % 12) + 1
    features['month_sin'] = np.sin(2 * np.pi * month_in_year / 12)
    features['month_cos'] = np.cos(2 * np.pi * month_in_year / 12)
    
    # Quarter-based seasonality
    quarter = ((month_in_year - 1) // 3) + 1
    features['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
    features['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    # Year trend
    #features['year_trend'] = time_index / 12.0
    #features['year_trend_sq'] = features['year_trend'] ** 2
    
    return features


# ---------------------------
#  NEW: Create lag and rolling features
# ---------------------------
def create_lag_features(data, n_lags=6, rolling_windows=[3, 6, 12]):
    """Create lag and rolling window features for each sector"""
    features_dict = {}
    
    for sector in data.columns:
        sector_data = data[sector]
        sector_features = pd.DataFrame(index=data.index)
        
        # Lag features
        for lag in range(1, n_lags + 1):
            sector_features[f'lag_{lag}'] = sector_data.shift(lag)
        
        # Rolling statistics
        for window in rolling_windows:
            if len(sector_data) >= window:
                sector_features[f'roll_mean_{window}'] = sector_data.rolling(window=window, min_periods=1).mean()
                sector_features[f'roll_std_{window}'] = sector_data.rolling(window=window, min_periods=1).std().fillna(0)
                sector_features[f'roll_max_{window}'] = sector_data.rolling(window=window, min_periods=1).max()
                sector_features[f'roll_min_{window}'] = sector_data.rolling(window=window, min_periods=1).min()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            sector_features[f'ema_{alpha}'] = sector_data.ewm(alpha=alpha, adjust=False).mean()
        
        # Growth rates
        sector_features['growth_1m'] = sector_data.pct_change(1).fillna(0)
        sector_features['growth_3m'] = sector_data.pct_change(3).fillna(0)
        sector_features['growth_12m'] = sector_data.pct_change(12).fillna(0)
        
        features_dict[sector] = sector_features.fillna(0)
    
    return features_dict


# ---------------------------
#  NEW: Cross-sector features
# ---------------------------
def create_cross_sector_features(data):
    """Create features based on cross-sector relationships"""
    features = pd.DataFrame(index=data.index)
    
    # Market-wide statistics
    features['market_total'] = data.sum(axis=1)
    features['market_mean'] = data.mean(axis=1)
    features['market_std'] = data.std(axis=1).fillna(0)
    features['market_max'] = data.max(axis=1)
    features['market_min'] = data.min(axis=1)
    
    # Active sectors count
    features['active_sectors'] = (data > 0).sum(axis=1)
    
    # Market concentration (Herfindahl index)
    market_shares = data.div(data.sum(axis=1), axis=0).fillna(0)
    features['herfindahl_index'] = (market_shares ** 2).sum(axis=1)
    
    return features


# ---------------------------
#  NEW: Enhanced sector grouping
# ---------------------------
def create_sector_groups():
    """Create sector groupings based on clustering results"""
    groups = {
        'cluster0': [10, 11, 12, 18,19, 2,27,35,39, 46, 49, 5, 56, 58, 59, 60, 69, 7, 70, 71,72, 78, 8, 87, 96],
        'cluster1': [14, 15, 30, 32, 33, 37, 44, 65, 76, 81, 84, 89, 9],
        'cluster2': [1, 13, 16, 17,20, 21, 22, 23, 24, 26, 28, 29, 3, 36, 38, 40, 43, 48, 52, 53, 54, 62, 63, 64, 73, 74, 77, 82, 83,88, 91],
        'cluster3': [25, 31, 4, 45, 47, 50, 51, 57, 6, 61, 66, 68, 79, 80, 86, 92, 93, 94 ],
        'cluster4': [55, 85]
    }
    return groups


# ---------------------------
#  NEW: Group-based features
# ---------------------------
def create_group_features(data, groups):
    """Create features based on sector groups"""
    group_features = pd.DataFrame(index=data.index)
    
    for group_name, sectors in groups.items():
        valid_sectors = [s for s in sectors if s in data.columns]
        if valid_sectors:
            group_data = data[valid_sectors]
            group_features[f'{group_name}_total'] = group_data.sum(axis=1)
            group_features[f'{group_name}_mean'] = group_data.mean(axis=1)
            group_features[f'{group_name}_std'] = group_data.std(axis=1).fillna(0)
            
            # Group momentum
            group_features[f'{group_name}_momentum_3m'] = group_features[f'{group_name}_total'].pct_change(3).fillna(0)
    
    return group_features


# ---------------------------
#  Enhanced December multipliers with sector grouping
# ---------------------------
def compute_enhanced_december_multipliers(a_tr, groups, eps=1e-9, min_dec_obs=1):
    """Enhanced December multipliers with group-based smoothing"""
    is_december = (a_tr.index.values % 12) == 11
    dec_means = a_tr[is_december].mean(axis=0)
    nondec_means = a_tr[~is_december].mean(axis=0)
    dec_counts = a_tr[is_december].notna().sum(axis=0)
    
    raw_mult = dec_means / (nondec_means + eps)
    
    # Group-based smoothing
    sector_to_mult = {}
    for group_name, sectors in groups.items():
        valid_sectors = [s for s in sectors if s in a_tr.columns]
        if valid_sectors:
            group_mults = raw_mult[valid_sectors]
            group_counts = dec_counts[valid_sectors]
            
            # Calculate group average for sectors with insufficient data
            reliable_mask = group_counts >= min_dec_obs
            if reliable_mask.sum() > 0:
                group_avg = group_mults[reliable_mask].mean()
            else:
                group_avg = 1.0
            
            for sector in valid_sectors:
                if dec_counts[sector] >= min_dec_obs:
                    mult = raw_mult[sector]
                else:
                    mult = group_avg
                
                # Clip to reasonable bounds
                mult = np.clip(mult, 0.7, 1.8)
                mult = mult if not np.isnan(mult) and np.isfinite(mult) else 1.0
                sector_to_mult[sector] = mult
    
    # Handle any remaining sectors
    overall_mult = float(dec_means.mean() / (nondec_means.mean() + eps))
    for sector in a_tr.columns:
        if sector not in sector_to_mult:
            sector_to_mult[sector] = np.clip(overall_mult, 0.7, 1.8)
    
    return sector_to_mult


# ---------------------------
#  NEW: Ensemble prediction using multiple models
# ---------------------------
def predict_with_ensemble(features, target, test_features, model_type='ridge', **model_params):
    """Train ensemble model and make predictions"""
    # Remove any inf or extreme values
    features = features.replace([np.inf, -np.inf], 0)
    test_features = test_features.replace([np.inf, -np.inf], 0)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    test_features_scaled = scaler.transform(test_features)
    
    # Select model
    if model_type == 'ridge':
        model = Ridge(alpha=model_params.get('alpha', 1.0))
    elif model_type == 'elastic':
        model = ElasticNet(alpha=model_params.get('alpha', 1.0), 
                          l1_ratio=model_params.get('l1_ratio', 0.5))
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=42
        )
    else:
        model = Ridge(alpha=1.0)  # Default
    
    # Train and predict
    model.fit(features_scaled, target)
    predictions = model.predict(test_features_scaled)
    
    return np.maximum(predictions, 0)  # Ensure non-negative predictions

#where is your ensamble?? -- Daniel is going to fix it !
# if daniel doesn't know how to do ensamble (i would be surprised) but
#i can still do the stacking(emsamble)


# ---------------------------
#  NEW: Advanced prediction function
# ---------------------------
def predict_advanced_horizon(a_tr, model_params, groups):
    """Advanced horizon prediction with multiple techniques"""
    prediction_horizon = np.arange(67, 79)
    sector_predictions = {}
    
    # Create features for training data
    seasonal_features = create_seasonal_features(a_tr.index)
    cross_sector_features = create_cross_sector_features(a_tr)
    group_features = create_group_features(a_tr, groups)
    lag_features_dict = create_lag_features(a_tr, n_lags=model_params.get('n_lags', 6))
    
    # Create features for prediction horizon
    pred_seasonal_features = create_seasonal_features(prediction_horizon)
    
    for sector in a_tr.columns:
        if a_tr[sector].sum() == 0:
            sector_predictions[sector] = np.zeros(len(prediction_horizon))
            continue
        
        # Combine features for this sector
        sector_lag_features = lag_features_dict[sector]
        combined_features = pd.concat([
            seasonal_features,
            cross_sector_features,
            group_features,
            sector_lag_features
        ], axis=1).fillna(0)
        
        # Use only data where we have target values
        valid_mask = a_tr[sector].notna() & (combined_features.index < a_tr.index.max() - 6)
        if valid_mask.sum() < 10:  # Need minimum data points
            sector_predictions[sector] = np.zeros(len(prediction_horizon))
            continue
        
        train_features = combined_features[valid_mask]
        train_target = a_tr[sector][valid_mask]
        
        # Create test features (simplified - using last known values and seasonal info)
        last_values = combined_features.iloc[-1:].copy()
        test_features_list = []
        
        for i, pred_time in enumerate(prediction_horizon):
            test_row = last_values.copy()
            test_row.index = [pred_time]
            
            # Update seasonal features
            seasonal_row = pred_seasonal_features.iloc[i:i+1]
            for col in seasonal_row.columns:
                if col in test_row.columns:
                    test_row[col] = seasonal_row[col].iloc[0]
            
            test_features_list.append(test_row)
        
        test_features = pd.concat(test_features_list, axis=0)
        
        # Make predictions
        try:
            predictions = predict_with_ensemble(
                train_features, train_target, test_features,
                model_type=model_params.get('model_type', 'ridge'),
                **model_params
            )
            sector_predictions[sector] = predictions
        except:
            # Fallback to simple method
            recent_mean = a_tr[sector].tail(6).mean()
            sector_predictions[sector] = np.full(len(prediction_horizon), 
                                               max(recent_mean, 0))
    
    # Convert to DataFrame
    result_df = pd.DataFrame(sector_predictions, index=prediction_horizon)
    result_df.index.name = 'time'
    return result_df


# ---------------------------
#  Enhanced evaluation function
# ---------------------------
def evaluate_advanced_params(a_tr_full, model_params, groups, val_len=6):
    """Evaluate parameters using time series cross-validation"""
    times = a_tr_full.index.values
    if len(times) < 20:  # Need sufficient data
        return 1e12
    
    val_times = times[-val_len:]
    rmse_list = []
    
    for t in val_times:
        # Split data
        train_mask = a_tr_full.index < t
        a_train = a_tr_full[train_mask]
        
        if len(a_train) < 12:  # Need sufficient history
            continue
        
        try:
            # Make prediction for time t
            pred_df = predict_advanced_horizon(a_train, model_params, groups)
            
            if t not in pred_df.index:
                continue
                
            y_pred = pred_df.loc[t]
            y_true = a_tr_full.loc[t]
            
            # Apply December adjustment if needed
            if (t % 12) == 11:
                sector_to_mult = compute_enhanced_december_multipliers(a_train, groups)
                for sector in y_pred.index:
                    mult = sector_to_mult.get(sector, 1.0)
                    y_pred.loc[sector] *= mult
            
            # Calculate RMSE
            diff = y_pred.values - y_true.values
            rmse = float(np.sqrt(np.mean(diff * diff)))
            rmse_list.append(rmse)
            
        except Exception as e:
            continue
    
    if len(rmse_list) == 0:
        return 1e12
    
    return float(np.mean(rmse_list))


# ---------------------------
#  Enhanced Optuna objective
# ---------------------------
def enhanced_optuna_objective(trial, a_tr, groups):
    """Enhanced Optuna objective with more parameters"""
    model_params = {
        'n_lags': trial.suggest_int('n_lags', 3, 12),
        'model_type': trial.suggest_categorical('model_type', ['ridge', 'elastic']),
        'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
    }
    
    if model_params['model_type'] == 'elastic':
        model_params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
    
    loss = evaluate_advanced_params(a_tr, model_params, groups, val_len=6)
    return loss


# ---------------------------
#  Enhanced submission generation
# ---------------------------
def generate_enhanced_submission(model_params, groups):
    """Generate submission using enhanced methods"""
    month_codes = build_month_codes()
    train_nht, test = load_competition_data()
    a_tr = build_amount_matrix(train_nht, month_codes)
    
    # Generate predictions
    a_pred = predict_advanced_horizon(a_tr, model_params, groups)
    
    # Apply December multipliers
    sector_to_mult = compute_enhanced_december_multipliers(a_tr, groups)
    dec_rows = [t for t in a_pred.index.values if (t % 12) == 11]
    
    for r in dec_rows:
        for sector in a_pred.columns:
            mult = sector_to_mult.get(sector, 1.0)
            a_pred.loc[r, sector] *= mult
    
    # Build submission
    test = split_test_id_column(test.copy())
    test = add_time_and_sector_fields(test, month_codes)
    lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    merged = test.merge(lookup, how='left', on=['time', 'sector_id'])
    merged['pred'] = merged['pred'].fillna(0.0)
    submission = merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    
    return a_tr, a_pred, submission


# ---------------------------
#  Enhanced main function
# ---------------------------
def main():
    """Enhanced main function with improved methodology"""
    print("Loading data and setting up enhanced prediction system...")
    
    month_codes = build_month_codes()
    train_nht, _ = load_competition_data()
    a_tr = build_amount_matrix(train_nht, month_codes)
    groups = create_sector_groups()
    
    print(f"Training data shape: {a_tr.shape}")
    print("Running enhanced hyperparameter optimization...")
    
    # Run optimization
    sampler = optuna.samplers.TPESampler(seed=1337)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    objective_fn = partial(enhanced_optuna_objective, a_tr=a_tr, groups=groups)
    study.optimize(objective_fn, n_trials=200, show_progress_bar=False)  # Reduced trials for faster execution
    
    best_params = study.best_trial.params
    print(f"Best parameters: {best_params}")
    print(f"Best validation score: {study.best_value:.5f}")
    
    # Generate final submission
    print("Generating enhanced submission...")
    a_tr, a_pred, submission = generate_enhanced_submission(best_params, groups)
    
    # Save submission
    submission.to_csv('china-real-estate-demand-prediction/submission.csv', index=False)
    print("Enhanced submission saved to /kaggle/working/submission.csv")
    
    # Additional diagnostics
    print(f"Prediction statistics:")
    print(f"  Mean prediction: {submission['new_house_transaction_amount'].mean():.2f}")
    print(f"  Std prediction: {submission['new_house_transaction_amount'].std():.2f}")
    print(f"  Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()}")
    
    return submission


# ---------------------------
#  Entry point
# ---------------------------
if __name__ == '__main__':
    main()

