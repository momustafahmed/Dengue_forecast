# Dengue Early Warning System (Thailand)

## Overview

This project is a machine learning-based early warning system for predicting dengue transmission in Thailand. The system forecasts weekly dengue case counts using climate data (rainfall, temperature, humidity, vapor pressure deficit) and historical case patterns. 

The application trains and evaluates multiple forecasting models (SARIMAX, Ridge Regression, Random Forest, Gradient Boosting, XGBoost) and automatically selects the best performer based on validation metrics. An interactive Streamlit dashboard allows users to input recent climate data and visualize multi-week forecasts with confidence intervals.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Structure

**Pipeline-based architecture**: The system follows a three-stage pipeline design:
1. **Preprocessing** (`pipeline/preprocess.py`) - Data loading, cleaning, feature engineering, and train/validation/test splitting
2. **Training** (`pipeline/train.py`) - Multi-model training with automatic model selection based on validation performance
3. **Forecasting** (`pipeline/forecast.py`) - Recursive multi-step forecasting with bootstrap confidence intervals

**Rationale**: This modular pipeline allows independent execution of each stage, making it easy to retrain models with new data or adjust preprocessing steps without touching the entire workflow.

### Machine Learning Approach

**Time series forecasting with engineered features**: Rather than pure time series models, the system uses supervised learning with carefully crafted temporal features:
- Lag features (y_lag1 through y_lag4 for dengue cases)
- Climate variable lags (rainfall, humidity, temperature, VPD)
- Rolling window aggregations (4-week moving averages)
- Temporal indicators (week_of_year, month, monsoon season flag)

**Rationale**: This hybrid approach captures both autoregressive patterns and climate-disease relationships. Feature engineering makes the data suitable for both traditional ML models (Ridge, Random Forest) and statistical models (SARIMAX).

**Model ensemble through selection**: Trains 5 different model types and selects the best based on validation MAE:
- Ridge Regression (regularized linear model)
- Random Forest Regressor
- Gradient Boosting Regressor  
- XGBoost Regressor
- SARIMAX (statistical time series model)

**Rationale**: Different models capture different patterns. Automatic selection ensures the deployed model performs best on validation data without manual tuning.

### Feature Engineering

**Recursive forecasting architecture**: Uses lagged features and rolling statistics to enable multi-step ahead forecasting. For predictions beyond the input data, the system recursively updates features using its own predictions.

**VPD calculation**: Vapor Pressure Deficit is computed from temperature and humidity using the Tetens equation rather than being required as input.

**Rationale**: This makes the system more flexible - users can provide either VPD directly or just temperature/humidity, which are more commonly measured.

### Data Preprocessing

**StandardScaler normalization**: Continuous features are standardized (zero mean, unit variance) before model training.

**Rationale**: Normalization ensures features are on comparable scales, which improves convergence for linear models and distance-based algorithms.

**Train/Validation/Test splitting**: Data is split chronologically:
- Training: First 70% of data
- Validation: Next 15% 
- Test: Final 15%

**Rationale**: Time series data requires chronological splitting to prevent data leakage. The validation set is used for model selection, and the held-out test set provides unbiased performance estimates.

### Web Interface

**Streamlit-based dashboard**: Interactive UI built with Streamlit for data entry and visualization.

**Plotly visualizations**: Uses Plotly for interactive charts showing forecasts with confidence intervals and historical patterns.

**Rationale**: Streamlit provides rapid development of data science dashboards without frontend complexity. Plotly enables interactive exploration of time series forecasts.

### Artifact Management

**JSON metadata storage**: Model metadata, feature lists, and column information stored as JSON files in `artifacts/` and `models/` directories.

**Joblib serialization**: Trained models and preprocessors (StandardScaler) serialized using joblib for efficient loading.

**Rationale**: Separating metadata (JSON, human-readable) from binary artifacts (joblib) makes the system easier to debug and version control.

## External Dependencies

### Machine Learning Libraries
- **scikit-learn**: Ridge Regression, Random Forest, Gradient Boosting, StandardScaler, metrics
- **XGBoost**: Gradient boosting implementation (optional, gracefully degrades if unavailable)
- **statsmodels**: SARIMAX time series model (optional, gracefully degrades if unavailable)

### Data Processing
- **pandas**: DataFrame operations, time series handling, CSV I/O
- **numpy**: Numerical computations, array operations, random seed control

### Visualization & Web Interface
- **Streamlit**: Web dashboard framework
- **Plotly**: Interactive time series visualizations

### Utilities
- **joblib**: Model serialization and deserialization
- **pathlib**: Cross-platform file path handling
- **json**: Metadata storage and loading
- **datetime**: Date parsing and manipulation for time series

### Data Storage
- **CSV files**: Training data stored as CSV files in `data/` directory
- **No database**: Application uses file-based storage only (CSV for data, joblib/JSON for models and artifacts)