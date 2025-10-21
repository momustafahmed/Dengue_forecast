"""
Forecasting module for dengue predictions.

Does recursive multi-step forecasting with bootstrap CIs.
Author: Mustafa
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Seed for reproducibility
np.random.seed(42)

def compute_vpd_from_temp_humidity(temperature, humidity):
    """Calculate VPD from temp and humidity using Tetens equation."""
    # Saturation vapor pressure using Tetens formula
    es = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))  # in kPa
    # VPD calculation
    vpd = es * (1 - humidity / 100)
    return vpd

def load_model_and_artifacts(models_dir="models", artifacts_dir="artifacts"):
    """Load trained model and preprocessing artifacts."""
    models_dir = Path(models_dir)
    artifacts_dir = Path(artifacts_dir)
    
    # Load model
    model = joblib.load(models_dir / "best_model.pkl")
    
    # Load metadata
    with open(models_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load preprocessor
    preprocessor_info = joblib.load(artifacts_dir / "preprocessor.pkl")
    
    return model, metadata, preprocessor_info

def prepare_recent_data(recent_df):
    """Prepare recent data with feature engineering."""
    df = recent_df.copy()
    
    # Ensure date is datetime
    if 'date' not in df.columns:
        raise ValueError("recent_df must contain 'date' column")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Compute VPD if missing
    if 'vpd_kpa' not in df.columns:
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['vpd_kpa'] = compute_vpd_from_temp_humidity(df['temperature'], df['humidity'])
        else:
            raise ValueError("Either vpd_kpa or both temperature and humidity must be provided")
    
    # Create calendar features
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['is_monsoon'] = (df['month'].isin([5, 6, 7, 8, 9, 10])).astype(int)
    
    return df

def create_features_for_forecast(df, required_features, last_dengue_cases=None):
    """Create features needed for forecasting from recent data."""
    df = df.copy()
    
    # If we have dengue_cases for the most recent weeks, use them for lags
    if 'dengue_cases' in df.columns and last_dengue_cases is not None:
        df.loc[df.index[-1], 'dengue_cases'] = last_dengue_cases
    
    # Create lagged features
    for lag in [1, 2, 3, 4]:
        if f'y_lag{lag}' in required_features:
            if 'dengue_cases' in df.columns:
                df[f'y_lag{lag}'] = df['dengue_cases'].shift(lag)
            else:
                df[f'y_lag{lag}'] = np.nan  # Will be handled during forecasting
    
    # Climate lags
    if 'rainfall_lag2' in required_features and 'rainfall' in df.columns:
        df['rainfall_lag2'] = df['rainfall'].shift(2)
    if 'rainfall_lag4' in required_features and 'rainfall' in df.columns:
        df['rainfall_lag4'] = df['rainfall'].shift(4)
    if 'humidity_lag1' in required_features and 'humidity' in df.columns:
        df['humidity_lag1'] = df['humidity'].shift(1)
    if 'humidity_lag2' in required_features and 'humidity' in df.columns:
        df['humidity_lag2'] = df['humidity'].shift(2)
    if 'temperature_lag1' in required_features and 'temperature' in df.columns:
        df['temperature_lag1'] = df['temperature'].shift(1)
    if 'vpd_lag1' in required_features and 'vpd_kpa' in df.columns:
        df['vpd_lag1'] = df['vpd_kpa'].shift(1)
    if 'vpd_lag2' in required_features and 'vpd_kpa' in df.columns:
        df['vpd_lag2'] = df['vpd_kpa'].shift(2)
    
    # Rolling features
    if 'rain_roll4' in required_features and 'rainfall' in df.columns:
        df['rain_roll4'] = df['rainfall'].rolling(window=4, min_periods=2, center=False).mean()
    if 'hum_roll4' in required_features and 'humidity' in df.columns:
        df['hum_roll4'] = df['humidity'].rolling(window=4, min_periods=2, center=False).mean()
    if 'temp_roll4' in required_features and 'temperature' in df.columns:
        df['temp_roll4'] = df['temperature'].rolling(window=4, min_periods=2, center=False).mean()
    
    return df

def get_model_residuals(artifacts_dir="artifacts"):
    """Get model residuals for bootstrap prediction intervals with minimum uncertainty."""
    try:
        # Load training and validation data for better residual estimation
        X_train = pd.read_csv(Path(artifacts_dir) / "X_train.csv")
        y_train = pd.read_csv(Path(artifacts_dir) / "y_train.csv").iloc[:, 0]
        X_val = pd.read_csv(Path(artifacts_dir) / "X_val.csv")
        y_val = pd.read_csv(Path(artifacts_dir) / "y_val.csv").iloc[:, 0]
        
        # Load model and metadata
        model, metadata, preprocessor_info = load_model_and_artifacts()
        
        # Calculate residuals on training data (more robust for small datasets)
        if metadata["model_name"] == "SARIMAX":
            # For SARIMAX, use variance-based estimate
            residuals = np.random.normal(0, y_train.std() * 0.3, len(y_train))
        else:
            # Use training residuals as they're more robust
            y_train_pred = model.predict(X_train)
            train_residuals = (y_train - y_train_pred).values
            
            # Also calculate validation residuals
            y_val_pred = model.predict(X_val)
            val_residuals = (y_val - y_val_pred).values
            
            # Combine both for better coverage
            residuals = np.concatenate([train_residuals, val_residuals])
            
            # Ensure minimum uncertainty based on test/validation performance
            residual_std = np.std(residuals)
            if residual_std < 100:  # Very small residuals indicate overfitting
                # Use test MAE if available, otherwise validation MAE
                test_mae = metadata.get("test_metric", {}).get("MAE", None)
                
                # Handle validation_metric (can be float or dict)
                val_metric = metadata.get("validation_metric", 500)
                if isinstance(val_metric, dict):
                    val_mae = val_metric.get("MAE", 500)
                else:
                    val_mae = val_metric  # Already a float
                
                # Use test MAE preferentially, fall back to validation MAE
                mae = test_mae if test_mae is not None else val_mae
                
                # Ensure mae is a valid number
                if mae is None or mae <= 0:
                    mae = 500  # Safe default
                
                # Use 60% of MAE as uncertainty (conservative but realistic)
                uncertainty_std = float(mae * 0.6)
                residuals = np.random.normal(0, uncertainty_std, len(residuals))
        
        return residuals
        
    except Exception as e:
        warnings.warn(f"Could not load residuals: {str(e)}, using default residuals")
        # Return robust default based on typical dengue case variance
        return np.random.normal(0, 500, 50)

def forecast(horizon_weeks, recent_df, last_dengue_cases=None):
    """
    Generate recursive forecasts with prediction intervals.
    
    Parameters:
    - horizon_weeks: Number of weeks to forecast ahead
    - recent_df: DataFrame with recent climate data (date, rainfall, temperature, humidity, etc.)
    - last_dengue_cases: Optional, last observed dengue cases for lag features
    
    Returns:
    - DataFrame with columns: week_start, y_pred, lo80, hi80, lo95, hi95
    """
    
    # Load model and preprocessing info
    model, metadata, preprocessor_info = load_model_and_artifacts()
    scaler = preprocessor_info["scaler"]
    required_features = metadata["features"]
    continuous_features = preprocessor_info["continuous_features"]
    model_name = metadata["model_name"]
    
    # Prepare recent data
    recent_df = prepare_recent_data(recent_df)
    
    # Get residuals for bootstrap
    residuals = get_model_residuals()
    n_simulations = 500
    
    # Special handling for SARIMAX
    if model_name == "SARIMAX":
        # For SARIMAX, use its built-in forecast method with exogenous variables
        try:
            # Prepare future exogenous variables
            exog_features = [f for f in required_features if not f.startswith('y_lag')]
            
            # Create future climate scenarios (use recent averages)
            future_dates = [recent_df['date'].max() + timedelta(weeks=i+1) for i in range(horizon_weeks)]
            future_climate = []
            
            for future_date in future_dates:
                future_row = {}
                future_row['week_of_year'] = future_date.isocalendar()[1]
                future_row['month'] = future_date.month
                future_row['is_monsoon'] = int(future_date.month in [5, 6, 7, 8, 9, 10])
                
                # Use seasonal or overall averages for climate
                for col in ['rainfall', 'rain_days_ge1mm', 'temperature', 'humidity', 'vpd_kpa']:
                    if col in recent_df.columns:
                        seasonal_data = recent_df[recent_df['month'] == future_date.month][col]
                        future_row[col] = seasonal_data.mean() if len(seasonal_data) > 0 else recent_df[col].mean()
                
                # Climate lags (use recent values)
                if 'rainfall_lag2' in exog_features and 'rainfall' in recent_df.columns:
                    future_row['rainfall_lag2'] = recent_df['rainfall'].iloc[-min(2, len(recent_df))]
                if 'rainfall_lag4' in exog_features and 'rainfall' in recent_df.columns:
                    future_row['rainfall_lag4'] = recent_df['rainfall'].iloc[-min(4, len(recent_df))]
                if 'humidity_lag1' in exog_features and 'humidity' in recent_df.columns:
                    future_row['humidity_lag1'] = recent_df['humidity'].iloc[-1]
                if 'humidity_lag2' in exog_features and 'humidity' in recent_df.columns:
                    future_row['humidity_lag2'] = recent_df['humidity'].iloc[-min(2, len(recent_df))]
                if 'temperature_lag1' in exog_features and 'temperature' in recent_df.columns:
                    future_row['temperature_lag1'] = recent_df['temperature'].iloc[-1]
                if 'vpd_lag1' in exog_features and 'vpd_kpa' in recent_df.columns:
                    future_row['vpd_lag1'] = recent_df['vpd_kpa'].iloc[-1]
                if 'vpd_lag2' in exog_features and 'vpd_kpa' in recent_df.columns:
                    future_row['vpd_lag2'] = recent_df['vpd_kpa'].iloc[-min(2, len(recent_df))]
                
                # Rolling features
                if 'rain_roll4' in exog_features and 'rainfall' in recent_df.columns:
                    future_row['rain_roll4'] = recent_df['rainfall'].tail(4).mean()
                if 'hum_roll4' in exog_features and 'humidity' in recent_df.columns:
                    future_row['hum_roll4'] = recent_df['humidity'].tail(4).mean()
                if 'temp_roll4' in exog_features and 'temperature' in recent_df.columns:
                    future_row['temp_roll4'] = recent_df['temperature'].tail(4).mean()
                
                future_climate.append(future_row)
            
            # Create DataFrame with all features in correct order
            future_exog_df = pd.DataFrame(future_climate)
            future_exog_df = future_exog_df[exog_features]
            
            # Scale continuous features
            continuous_exog = [f for f in continuous_features if f in exog_features]
            if continuous_exog:
                future_exog_df[continuous_exog] = scaler.transform(future_exog_df[continuous_exog])
            
            # Get forecast from SARIMAX
            forecast_result = model.get_forecast(steps=horizon_weeks, exog=future_exog_df.values)
            y_preds = forecast_result.predicted_mean.values
            
            # Create results with bootstrap intervals
            forecast_results = []
            
            for i, (future_date, y_pred) in enumerate(zip(future_dates, y_preds)):
                # Bootstrap for intervals with growing uncertainty
                week = i + 1
                horizon_factor = 1.0 + (week - 1) * 0.10  # Natural uncertainty growth
                simulated_forecasts = []
                for _ in range(n_simulations):
                    noise = np.random.choice(residuals) * horizon_factor
                    simulated_forecasts.append(max(0, y_pred + noise))
                
                simulated_forecasts = np.array(simulated_forecasts)
                
                # Ensure non-negative prediction
                y_pred_safe = max(0, y_pred)
                
                # Calculate confidence intervals from bootstrap percentiles
                lo95 = np.percentile(simulated_forecasts, 2.5)
                hi95 = np.percentile(simulated_forecasts, 97.5)
                lo80 = np.percentile(simulated_forecasts, 10)
                hi80 = np.percentile(simulated_forecasts, 90)
                
                forecast_results.append({
                    'week_start': future_date,
                    'y_pred': y_pred_safe,
                    'lo80': lo80,
                    'hi80': hi80,
                    'lo95': lo95,
                    'hi95': hi95
                })
            
            return pd.DataFrame(forecast_results)
            
        except Exception as e:
            warnings.warn(f"SARIMAX forecasting failed: {str(e)}, falling back to simple model")
            # Fall through to use simple recursive approach
    
    # For non-SARIMAX models or if SARIMAX failed, use recursive forecasting
    # Create features
    feature_df = create_features_for_forecast(recent_df, required_features, last_dengue_cases)
    
    # Get the last complete row for starting forecasts
    last_complete_idx = feature_df.dropna(subset=required_features).index
    if len(last_complete_idx) == 0:
        # If no complete row, use the last row and fill missing values
        start_idx = len(feature_df) - 1
        # Fill missing lag features with recent values or means
        for col in required_features:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default value
            elif pd.isna(feature_df.loc[start_idx, col]):
                if 'lag' in col and 'y_' in col:
                    # For dengue lags, use a reasonable default
                    feature_df.loc[start_idx, col] = last_dengue_cases if last_dengue_cases else 500
                else:
                    # For other features, use column mean or forward fill
                    feature_df[col] = feature_df[col].ffill().fillna(feature_df[col].mean())
    else:
        start_idx = last_complete_idx[-1]
    
    # Get last complete date
    last_date = feature_df.loc[start_idx, 'date']
    
    # Initialize forecast results
    forecast_results = []
    
    # Recursive forecasting
    current_features = feature_df.loc[start_idx:start_idx].copy()  # Start with last complete observation
    
    for week in range(1, horizon_weeks + 1):
        # Calculate forecast date
        forecast_date = last_date + timedelta(weeks=week)
        
        # Create future row
        future_row = current_features.iloc[-1:].copy()
        future_row['date'] = forecast_date
        
        # Update calendar features
        future_row['week_of_year'] = forecast_date.isocalendar()[1]
        future_row['month'] = forecast_date.month
        future_row['is_monsoon'] = int(forecast_date.month in [5, 6, 7, 8, 9, 10])
        
        # For climate variables, use the last available values or interpolate
        # This is a simplification - in practice, you'd want climate forecasts
        climate_cols = ['rainfall', 'rain_days_ge1mm', 'temperature', 'humidity', 'vpd_kpa']
        for col in climate_cols:
            if col in recent_df.columns and len(recent_df) > 0:
                # Use seasonal average or recent trend
                seasonal_data = recent_df[recent_df['month'] == forecast_date.month][col]
                if len(seasonal_data) > 0:
                    future_row[col] = seasonal_data.mean()
                else:
                    future_row[col] = recent_df[col].mean()
        
        # Prepare feature vector - ensure it's a properly formatted DataFrame
        X_forecast = future_row[required_features].copy()
        
        # Scale features - maintain DataFrame structure with feature names
        X_forecast_scaled = X_forecast.copy()
        if continuous_features:
            available_continuous = [col for col in continuous_features if col in X_forecast.columns]
            if available_continuous:
                # Scale while maintaining DataFrame structure
                scaled_values = scaler.transform(X_forecast[available_continuous])
                X_forecast_scaled[available_continuous] = scaled_values
        
        # Make point prediction (only for non-SARIMAX models)
        # Pass as DataFrame to preserve feature names
        y_pred = model.predict(X_forecast_scaled)[0]
        
        # Bootstrap prediction intervals with growing uncertainty
        # Uncertainty should increase with forecast horizon
        horizon_factor = 1.0 + (week - 1) * 0.10  # Natural uncertainty growth
        simulated_forecasts = []
        for _ in range(n_simulations):
            # Add residual noise scaled by horizon
            noise = np.random.choice(residuals) * horizon_factor
            simulated_y = y_pred + noise
            simulated_forecasts.append(max(0, simulated_y))  # Ensure non-negative
        
        simulated_forecasts = np.array(simulated_forecasts)
        
        # Ensure non-negative prediction
        y_pred = max(0, y_pred)
        
        # Calculate confidence intervals from bootstrap percentiles
        lo95 = np.percentile(simulated_forecasts, 2.5)
        hi95 = np.percentile(simulated_forecasts, 97.5)
        lo80 = np.percentile(simulated_forecasts, 10)
        hi80 = np.percentile(simulated_forecasts, 90)
        
        
        # Store results
        forecast_results.append({
            'week_start': forecast_date,
            'y_pred': y_pred,
            'lo80': lo80,
            'hi80': hi80,
            'lo95': lo95,
            'hi95': hi95
        })
        
        # Update features for next iteration
        # Update lag features with the new prediction
        future_row['dengue_cases'] = y_pred  # Use prediction for future lags
        
        # Update lagged dengue features
        for lag in [1, 2, 3, 4]:
            if f'y_lag{lag}' in required_features:
                if lag == 1:
                    future_row[f'y_lag{lag}'] = current_features.iloc[-1]['dengue_cases'] if 'dengue_cases' in current_features.columns else y_pred
                else:
                    # Get previous lag value
                    prev_lag_col = f'y_lag{lag-1}'
                    if prev_lag_col in current_features.columns:
                        future_row[f'y_lag{lag}'] = current_features.iloc[-1][prev_lag_col]
        
        # Update climate lags
        if len(current_features) >= 2:
            if 'rainfall_lag2' in required_features:
                future_row['rainfall_lag2'] = current_features.iloc[-2]['rainfall'] if 'rainfall' in current_features.columns else future_row['rainfall']
        if len(current_features) >= 4:
            if 'rainfall_lag4' in required_features:
                future_row['rainfall_lag4'] = current_features.iloc[-4]['rainfall'] if 'rainfall' in current_features.columns else future_row['rainfall']
        
        # Update rolling features (simplified)
        for col_base in ['rain', 'hum', 'temp']:
            roll_col = f'{col_base}_roll4'
            if roll_col in required_features:
                source_col = {'rain': 'rainfall', 'hum': 'humidity', 'temp': 'temperature'}[col_base]
                if source_col in future_row.columns:
                    # Simple rolling mean approximation
                    if len(current_features) >= 3:
                        recent_values = current_features.iloc[-3:][source_col].tolist() + [future_row[source_col].iloc[0]]
                        future_row[roll_col] = np.mean(recent_values)
                    else:
                        future_row[roll_col] = future_row[source_col]
        
        # Append to current features for next iteration
        current_features = pd.concat([current_features, future_row], ignore_index=True)
    
    # Convert results to DataFrame
    forecast_df = pd.DataFrame(forecast_results)
    
    return forecast_df

def main():
    """Test the forecasting function."""
    # This is mainly for testing purposes
    print("Forecasting module loaded successfully!")
    
    # You can add test code here if needed
    pass

if __name__ == "__main__":
    main()
