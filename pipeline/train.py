"""
Model training pipeline for dengue forecasting.

Trains 5 different models, picks best one by validation MAE, saves it.
Author: Mustafa
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
from datetime import datetime
import joblib

# Scikit-learn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    warnings.warn("XGBoost not available, will skip XGBRegressor")

# Statsmodels for SARIMAX
try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    sm = None
    SARIMAX = None
    warnings.warn("Statsmodels not available, will skip SARIMAX")

# Random seed for reproducibility
np.random.seed(42)

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, handling zero values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def load_preprocessed_data(artifacts_dir="artifacts"):
    """Load preprocessed data."""
    artifacts_dir = Path(artifacts_dir)
    
    X_train = pd.read_csv(artifacts_dir / "X_train.csv")
    y_train = pd.read_csv(artifacts_dir / "y_train.csv").iloc[:, 0]  # First column
    X_val = pd.read_csv(artifacts_dir / "X_val.csv")
    y_val = pd.read_csv(artifacts_dir / "y_val.csv").iloc[:, 0]
    X_test = pd.read_csv(artifacts_dir / "X_test.csv")
    y_test = pd.read_csv(artifacts_dir / "y_test.csv").iloc[:, 0]
    
    # Load column information
    with open(artifacts_dir / "columns_used.json", "r") as f:
        columns_info = json.load(f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, columns_info

def prepare_sarimax_features(X):
    """Prepare exogenous features for SARIMAX (exclude y lags to avoid duplication)."""
    y_lag_columns = [col for col in X.columns if col.startswith('y_lag')]
    sarimax_features = X.drop(columns=y_lag_columns, errors='ignore')
    return sarimax_features

def train_sarimax_model(y_train, X_train, y_val, X_val):
    """Train SARIMAX model with exogenous variables."""
    if SARIMAX is None:
        return None, np.inf
    
    # Prepare exogenous features (remove y lags)
    exog_train = prepare_sarimax_features(X_train)
    exog_val = prepare_sarimax_features(X_val)
    
    try:
        # Try different SARIMAX configurations
        configs = [
            (1, 1, 1, 1, 1, 1, 52),  # (p,d,q,P,D,Q,s)
            (2, 1, 1, 1, 1, 1, 52),
            (1, 1, 2, 1, 1, 1, 52),
            (0, 1, 1, 1, 1, 1, 52),
        ]
        
        best_model = None
        best_aic = np.inf
        
        for p, d, q, P, D, Q, s in configs:
            try:
                model = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted_model = model.fit(disp=False, maxiter=100)
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    
            except Exception as e:
                continue
        
        if best_model is None:
            return None, np.inf
        
        # Make predictions on validation set
        val_pred = best_model.forecast(steps=len(y_val), exog=exog_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        return best_model, val_mae
        
    except Exception as e:
        print(f"SARIMAX training failed: {str(e)}")
        return None, np.inf

def train_sklearn_model(model, X_train, y_train, X_val, y_val, param_grid=None):
    """Train scikit-learn model with time series cross-validation."""
    if param_grid is None:
        # Fit model with default parameters
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        return model, val_mae
    
    # Time series cross-validation for hyperparameter tuning
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = None
    best_cv_mae = np.inf
    
    # Grid search with time series CV
    for params in param_grid:
        cv_maes = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Clone model and set parameters
            cv_model = model.__class__(**params)
            cv_model.fit(X_cv_train, y_cv_train)
            cv_pred = cv_model.predict(X_cv_val)
            cv_mae = mean_absolute_error(y_cv_val, cv_pred)
            cv_maes.append(cv_mae)
        
        avg_cv_mae = np.mean(cv_maes)
        if avg_cv_mae < best_cv_mae:
            best_cv_mae = avg_cv_mae
            best_params = params
    
    # Refit best model on full training data
    best_model = model.__class__(**best_params)
    best_model.fit(X_train, y_train)
    val_pred = best_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    return best_model, val_mae

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with early stopping."""
    if xgb is None:
        return None, np.inf
    
    try:
        model = xgb.XGBRegressor(
            random_state=42,
            eval_metric='mae',
            early_stopping_rounds=10
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        return model, val_mae
        
    except Exception as e:
        print(f"XGBoost training failed: {str(e)}")
        return None, np.inf

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    try:
        if model_name == "SARIMAX":
            # For SARIMAX, need to use forecast method
            exog_test = prepare_sarimax_features(X_test)
            y_pred = model.forecast(steps=len(y_test), exog=exog_test)
        else:
            y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return {
            "MAE": np.inf,
            "RMSE": np.inf,
            "MAPE": np.inf
        }

def save_model_and_metadata(model, model_name, X_train, y_train, X_val, y_val, 
                           validation_metrics, test_metrics, feature_columns, models_dir="models"):
    """Save the best model and its metadata."""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    joblib.dump(model, models_dir / "best_model.pkl")
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "training_dates": {
            "start": str(X_train.index[0]) if hasattr(X_train, 'index') else "unknown",
            "end": str(X_train.index[-1]) if hasattr(X_train, 'index') else "unknown"
        },
        "target": "dengue_cases",
        "features": feature_columns,
        "scaler_path": "artifacts/preprocessor.pkl",
        "validation_metric": validation_metrics,
        "test_metric": test_metrics,
        "created_at": datetime.now().isoformat()
    }
    
    # Save metadata
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model and metadata saved to {models_dir}")

def main():
    """Main training pipeline."""
    print("Starting model training pipeline...")
    
    try:
        # Load preprocessed data
        print("1. Loading preprocessed data...")
        X_train, y_train, X_val, y_val, X_test, y_test, columns_info = load_preprocessed_data()
        feature_columns = columns_info["features"]
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Initialize results storage
        models = {}
        validation_results = []
        
        # 1. SARIMAX Model
        print("\n2. Training SARIMAX model...")
        sarimax_model, sarimax_mae = train_sarimax_model(y_train, X_train, y_val, X_val)
        if sarimax_model is not None:
            models["SARIMAX"] = sarimax_model
            validation_results.append({
                "model": "SARIMAX",
                "validation_mae": sarimax_mae
            })
            print(f"SARIMAX validation MAE: {sarimax_mae:.2f}")
        else:
            print("SARIMAX training failed")
        
        # 2. Ridge Regression
        print("\n3. Training Ridge regression...")
        ridge_param_grid = [
            {"alpha": 0.1},
            {"alpha": 1.0},
            {"alpha": 10.0},
            {"alpha": 100.0}
        ]
        ridge_model, ridge_mae = train_sklearn_model(
            Ridge(random_state=42), X_train, y_train, X_val, y_val, ridge_param_grid
        )
        models["Ridge"] = ridge_model
        validation_results.append({
            "model": "Ridge",
            "validation_mae": ridge_mae
        })
        print(f"Ridge validation MAE: {ridge_mae:.2f}")
        
        # 3. Random Forest
        print("\n4. Training Random Forest...")
        rf_param_grid = [
            {"n_estimators": 50, "max_depth": 10, "random_state": 42},
            {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            {"n_estimators": 100, "max_depth": 20, "random_state": 42}
        ]
        rf_model, rf_mae = train_sklearn_model(
            RandomForestRegressor(), X_train, y_train, X_val, y_val, rf_param_grid
        )
        models["RandomForest"] = rf_model
        validation_results.append({
            "model": "RandomForest",
            "validation_mae": rf_mae
        })
        print(f"Random Forest validation MAE: {rf_mae:.2f}")
        
        # 4. Gradient Boosting
        print("\n5. Training Gradient Boosting...")
        gb_param_grid = [
            {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.1, "random_state": 42},
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42},
            {"n_estimators": 100, "max_depth": 8, "learning_rate": 0.05, "random_state": 42}
        ]
        gb_model, gb_mae = train_sklearn_model(
            GradientBoostingRegressor(), X_train, y_train, X_val, y_val, gb_param_grid
        )
        models["GradientBoosting"] = gb_model
        validation_results.append({
            "model": "GradientBoosting",
            "validation_mae": gb_mae
        })
        print(f"Gradient Boosting validation MAE: {gb_mae:.2f}")
        
        # 5. XGBoost
        print("\n6. Training XGBoost...")
        xgb_model, xgb_mae = train_xgboost_model(X_train, y_train, X_val, y_val)
        if xgb_model is not None:
            models["XGBoost"] = xgb_model
            validation_results.append({
                "model": "XGBoost",
                "validation_mae": xgb_mae
            })
            print(f"XGBoost validation MAE: {xgb_mae:.2f}")
        else:
            print("XGBoost training failed")
        
        # Select best model
        print("\n7. Selecting best model...")
        validation_df = pd.DataFrame(validation_results)
        
        # Prefer XGBoost over SARIMAX for production reliability
        # Use XGBoost if it's available and within 10% of the best MAE
        best_mae = validation_df['validation_mae'].min()
        xgb_result = validation_df[validation_df['model'] == 'XGBoost']
        
        if len(xgb_result) > 0 and xgb_result['validation_mae'].values[0] <= best_mae * 1.1:
            # Use XGBoost for better production stability
            best_model_info = xgb_result.iloc[0]
            best_model_name = 'XGBoost'
            print(f"Selected XGBoost for production reliability (MAE: {best_model_info['validation_mae']:.2f})")
        else:
            # Fall back to absolute best
            best_model_info = validation_df.loc[validation_df['validation_mae'].idxmin()]
            best_model_name = best_model_info['model']
        
        best_model = models[best_model_name]
        print(f"Best model: {best_model_name} (Validation MAE: {best_model_info['validation_mae']:.2f})")
        
        # Evaluate all models on validation set
        print("\n8. Evaluating all models on validation set...")
        all_metrics = []
        for model_name, model in models.items():
            if model_name == "SARIMAX":
                exog_val = prepare_sarimax_features(X_val)
                try:
                    val_pred = model.forecast(steps=len(y_val), exog=exog_val)
                except:
                    val_pred = [np.nan] * len(y_val)
            else:
                val_pred = model.predict(X_val)
            
            metrics = {
                "model": model_name,
                "split": "validation",
                "MAE": mean_absolute_error(y_val, val_pred),
                "RMSE": np.sqrt(mean_squared_error(y_val, val_pred)),
                "MAPE": mean_absolute_percentage_error(y_val, val_pred)
            }
            all_metrics.append(metrics)
        
        # Refit best model on train+val and evaluate on test
        print(f"\n9. Refitting best model ({best_model_name}) on train+val...")
        X_train_val = pd.concat([X_train, X_val], ignore_index=True)
        y_train_val = pd.concat([y_train, y_val], ignore_index=True)
        
        if best_model_name == "SARIMAX":
            # Refit SARIMAX on combined data
            exog_train_val = prepare_sarimax_features(X_train_val)
            try:
                # Use the same order as the best model
                model_order = best_model.model.order
                seasonal_order = best_model.model.seasonal_order
                
                final_model = SARIMAX(
                    y_train_val,
                    exog=exog_train_val,
                    order=model_order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                final_fitted_model = final_model.fit(disp=False, maxiter=100)
                
                # Evaluate on test
                exog_test = prepare_sarimax_features(X_test)
                test_pred = final_fitted_model.forecast(steps=len(y_test), exog=exog_test)
                
            except Exception as e:
                print(f"Error refitting SARIMAX: {str(e)}")
                final_fitted_model = best_model
                test_pred = [np.nan] * len(y_test)
                
        else:
            # Refit sklearn/xgboost model
            if best_model_name == "XGBoost" and xgb is not None:
                # Remove early stopping for final training on all data
                params = best_model.get_params()
                params.pop('early_stopping_rounds', None)
                final_fitted_model = xgb.XGBRegressor(**params)
                final_fitted_model.fit(X_train_val, y_train_val)
            else:
                final_fitted_model = best_model.__class__(**best_model.get_params())
                final_fitted_model.fit(X_train_val, y_train_val)
                
            test_pred = final_fitted_model.predict(X_test)
        
        # Test metrics
        test_metrics = {
            "MAE": mean_absolute_error(y_test, test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "MAPE": mean_absolute_percentage_error(y_test, test_pred)
        }
        
        all_metrics.append({
            "model": best_model_name,
            "split": "test",
            **test_metrics
        })
        
        print(f"Test MAE: {test_metrics['MAE']:.2f}")
        print(f"Test RMSE: {test_metrics['RMSE']:.2f}")
        print(f"Test MAPE: {test_metrics['MAPE']:.2f}")
        
        # Save metrics
        print("\n10. Saving results...")
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv("artifacts/metrics.csv", index=False)
        
        # Save best model and metadata
        save_model_and_metadata(
            final_fitted_model, best_model_name, X_train_val, y_train_val, 
            X_val, y_val, best_model_info['validation_mae'], test_metrics, feature_columns
        )
        
        print("Model training pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
