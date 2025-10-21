"""
Data preprocessing for dengue forecasting pipeline.

Loads data, cleans it, creates features, splits into train/val/test.
Author: Mustafa
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# Seed for reproducible results
np.random.seed(42)

def load_and_concatenate_data(data_folder="data"):
    """Load and concatenate all CSV files from the data folder."""
    data_folder = Path(data_folder)
    csv_files = list(data_folder.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Keep only required columns, ignore extras
        required_cols = ['date', 'year', 'week', 'dengue_cases', 'rainfall', 
                        'rain_days_ge1mm', 'temperature', 'humidity', 'vpd_kpa']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def parse_and_validate_dates(df):
    """Parse dates and validate weekly cadence."""
    # Parse date column - handle different formats
    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    except ValueError:
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        except ValueError:
            df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    
    # Sort by date and remove duplicates
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    
    # Validate that dates are Mondays with 7-day steps
    corrected_dates = []
    for i, date in enumerate(df['date']):
        # Check if it's a Monday (weekday 0)
        if date.weekday() != 0:
            # Coerce to nearest Monday
            days_to_monday = date.weekday()
            corrected_date = date - timedelta(days=days_to_monday)
            warnings.warn(f"Date {date.strftime('%Y-%m-%d')} is not a Monday, "
                         f"corrected to {corrected_date.strftime('%Y-%m-%d')}")
            corrected_dates.append(corrected_date)
        else:
            corrected_dates.append(date)
    
    df['date'] = corrected_dates
    
    # Check for 7-day steps
    if len(df) > 1:
        date_diffs = df['date'].diff().dt.days
        non_weekly = date_diffs[(date_diffs != 7) & (date_diffs.notna())]
        if len(non_weekly) > 0:
            warnings.warn(f"Found {len(non_weekly)} date gaps that are not 7-day steps")
    
    return df

def handle_missing_values(df):
    """Handle missing values according to the specification."""
    df = df.copy()
    
    # Climate variables for forward-fill and rolling mean imputation
    climate_vars = ['rainfall', 'rain_days_ge1mm', 'temperature', 'humidity', 'vpd_kpa']
    
    for var in climate_vars:
        if var in df.columns:
            # Forward-fill short gaps (up to 2 weeks)
            df[var] = df[var].ffill(limit=2)
            
            # For remaining missing values, use 4-week rolling mean
            rolling_mean = df[var].rolling(window=4, min_periods=2, center=False).mean()
            df[var] = df[var].fillna(rolling_mean)
    
    # For dengue_cases, do not impute - will drop rows with missing values later
    return df

def create_calendar_features(df):
    """Create calendar-based features."""
    df = df.copy()
    
    # Week of year (1-53)
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Month (1-12)
    df['month'] = df['date'].dt.month
    
    # Is monsoon season (May to October: months 5-10)
    df['is_monsoon'] = (df['month'].isin([5, 6, 7, 8, 9, 10])).astype(int)
    
    return df

def create_lagged_features(df):
    """Create lagged predictors without future leakage."""
    df = df.copy()
    
    # Dengue cases lags
    for lag in [1, 2, 3, 4]:
        df[f'y_lag{lag}'] = df['dengue_cases'].shift(lag)
    
    # Climate lags
    if 'rainfall' in df.columns:
        df['rainfall_lag2'] = df['rainfall'].shift(2)
        df['rainfall_lag4'] = df['rainfall'].shift(4)
    
    if 'humidity' in df.columns:
        df['humidity_lag1'] = df['humidity'].shift(1)
        df['humidity_lag2'] = df['humidity'].shift(2)
    
    if 'temperature' in df.columns:
        df['temperature_lag1'] = df['temperature'].shift(1)
    
    if 'vpd_kpa' in df.columns:
        df['vpd_lag1'] = df['vpd_kpa'].shift(1)
        df['vpd_lag2'] = df['vpd_kpa'].shift(2)
    
    return df

def create_rolling_features(df):
    """Create rolling mean features."""
    df = df.copy()
    
    # 4-week rolling means (center=False, min_periods=2)
    if 'rainfall' in df.columns:
        df['rain_roll4'] = df['rainfall'].rolling(window=4, min_periods=2, center=False).mean()
    
    if 'humidity' in df.columns:
        df['hum_roll4'] = df['humidity'].rolling(window=4, min_periods=2, center=False).mean()
    
    if 'temperature' in df.columns:
        df['temp_roll4'] = df['temperature'].rolling(window=4, min_periods=2, center=False).mean()
    
    return df

def split_data_by_time(df):
    """Split data by time: adaptive splits based on available data."""
    # Drop rows with missing dengue_cases
    df_clean = df.dropna(subset=['dengue_cases']).reset_index(drop=True)
    
    n_total = len(df_clean)
    
    # Adaptive split strategy based on data size
    if n_total < 20:
        raise ValueError(f"Insufficient data: only {n_total} weeks available, need at least 20 weeks")
    elif n_total < 40:
        # Very small dataset: use 60% train, 20% val, 20% test
        test_size = max(4, int(n_total * 0.2))
        val_size = max(4, int(n_total * 0.2))
        warnings.warn(f"Small dataset ({n_total} weeks): using adaptive splits")
    elif n_total < 104:
        # Small dataset: use ~70% train, 15% val, 15% test
        test_size = max(8, int(n_total * 0.15))
        val_size = max(8, int(n_total * 0.15))
        warnings.warn(f"Limited data ({n_total} weeks): using adaptive splits")
    else:
        # Standard split: last 52 weeks test, 52 weeks val
        test_size = 52
        val_size = 52
    
    # Calculate split indices
    test_start_idx = n_total - test_size
    val_start_idx = test_start_idx - val_size
    
    # Ensure training set has at least 10 weeks
    if val_start_idx < 10:
        val_start_idx = 10
        test_start_idx = val_start_idx + val_size
        test_size = n_total - test_start_idx
        warnings.warn(f"Adjusted splits to ensure minimum training data")
    
    # Create splits
    train_df = df_clean.iloc[:val_start_idx].copy()
    val_df = df_clean.iloc[val_start_idx:test_start_idx].copy()
    test_df = df_clean.iloc[test_start_idx:].copy()
    
    print(f"Data splits:")
    print(f"Training: {len(train_df)} weeks ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Validation: {len(val_df)} weeks ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test: {len(test_df)} weeks ({test_df['date'].min()} to {test_df['date'].max()})")
    
    return train_df, val_df, test_df

def prepare_features_and_target(df, feature_columns=None):
    """Prepare feature matrix and target variable."""
    if feature_columns is None:
        # Define feature columns
        feature_columns = [
            # Calendar features
            'week_of_year', 'month', 'is_monsoon',
            # Lagged features
            'y_lag1', 'y_lag2', 'y_lag3', 'y_lag4',
            'rainfall_lag2', 'rainfall_lag4',
            'humidity_lag1', 'humidity_lag2',
            'temperature_lag1', 'vpd_lag1', 'vpd_lag2',
            # Rolling features
            'rain_roll4', 'hum_roll4', 'temp_roll4',
            # Current climate (for some models)
            'rainfall', 'rain_days_ge1mm', 'temperature', 'humidity', 'vpd_kpa'
        ]
    
    # Keep only available columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features].copy()
    y = df['dengue_cases'].copy()
    
    return X, y, available_features

def scale_features(X_train, X_val, X_test, continuous_features=None):
    """Scale continuous features using StandardScaler."""
    if continuous_features is None:
        # Identify continuous features (exclude categorical/binary ones)
        categorical_features = ['week_of_year', 'month', 'is_monsoon', 'rain_days_ge1mm']
        continuous_features = [col for col in X_train.columns if col not in categorical_features]
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    if continuous_features:
        X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
        X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
        X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, continuous_features

def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test, 
                          feature_columns, scaler, continuous_features, artifacts_dir="artifacts"):
    """Save preprocessed data and preprocessing artifacts."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save data splits
    X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
    y_train.to_csv(artifacts_dir / "y_train.csv", index=False)
    X_val.to_csv(artifacts_dir / "X_val.csv", index=False)
    y_val.to_csv(artifacts_dir / "y_val.csv", index=False)
    X_test.to_csv(artifacts_dir / "X_test.csv", index=False)
    y_test.to_csv(artifacts_dir / "y_test.csv", index=False)
    
    # Save column information
    columns_info = {
        "features": feature_columns,
        "target": "dengue_cases",
        "continuous_features": continuous_features
    }
    with open(artifacts_dir / "columns_used.json", "w") as f:
        json.dump(columns_info, f, indent=2)
    
    # Save preprocessor (scaler + column order)
    preprocessor_info = {
        "scaler": scaler,
        "feature_columns": feature_columns,
        "continuous_features": continuous_features
    }
    joblib.dump(preprocessor_info, artifacts_dir / "preprocessor.pkl")
    
    print(f"Preprocessed data saved to {artifacts_dir}")

def main():
    """Main preprocessing pipeline."""
    print("Starting data preprocessing pipeline...")
    
    # Create necessary directories
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    try:
        # 1. Load and concatenate data
        print("1. Loading data...")
        df = load_and_concatenate_data("data")
        print(f"Loaded {len(df)} rows")
        
        # 2. Parse and validate dates
        print("2. Parsing and validating dates...")
        df = parse_and_validate_dates(df)
        
        # 3. Handle missing values
        print("3. Handling missing values...")
        df = handle_missing_values(df)
        
        # 4. Create calendar features
        print("4. Creating calendar features...")
        df = create_calendar_features(df)
        
        # 5. Create lagged features
        print("5. Creating lagged features...")
        df = create_lagged_features(df)
        
        # 6. Create rolling features
        print("6. Creating rolling features...")
        df = create_rolling_features(df)
        
        # 7. Drop rows with any NaN after feature engineering
        print("7. Cleaning data after feature engineering...")
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        final_rows = len(df)
        print(f"Dropped {initial_rows - final_rows} rows with missing values")
        
        # 8. Split data by time
        print("8. Splitting data by time...")
        train_df, val_df, test_df = split_data_by_time(df)
        
        # 9. Prepare features and target
        print("9. Preparing features and target...")
        X_train, y_train, feature_columns = prepare_features_and_target(train_df)
        X_val, y_val, _ = prepare_features_and_target(val_df, feature_columns)
        X_test, y_test, _ = prepare_features_and_target(test_df, feature_columns)
        
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Features: {feature_columns}")
        
        # 10. Scale features
        print("10. Scaling features...")
        X_train_scaled, X_val_scaled, X_test_scaled, scaler, continuous_features = scale_features(
            X_train, X_val, X_test
        )
        
        # 11. Save preprocessed data
        print("11. Saving preprocessed data...")
        save_preprocessed_data(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            feature_columns, scaler, continuous_features
        )
        
        print("Preprocessing pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
