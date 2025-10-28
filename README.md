# Dengue EWS (Thailand)

Early warning system for dengue transmission in Thailand. Uses machine learning to forecast weekly case counts based on climate data. Trains multiple models (SARIMAX, Ridge, Random Forest, Gradient Boosting, XGBoost) and provides an interactive web interface with confidence intervals.

**Author:** Mustafa

## Data Schema

Input data should be a CSV file with weekly observations containing:

- `date`: Week start date (Monday, YYYY-MM-DD format)
- `year`: Year
- `week`: Week number
- `dengue_cases`: Weekly dengue case count (target variable)
- `rainfall`: Weekly total rainfall (mm)
- `rain_days_ge1mm`: Number of days with rainfall ≥ 1mm
- `temperature`: Average weekly temperature (°C)
- `humidity`: Average weekly relative humidity (%)
- `vpd_kpa`: Vapor pressure deficit (kPa)

## Setup and Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training Models

Place your training data CSV in the `data/` directory, then run:

```bash
python -m pipeline.preprocess
python -m pipeline.train
```

This will:
1. Preprocess the data (feature engineering, train/val/test splits)
2. Train 5 different forecasting models
3. Select the best model and save it to `models/`

### Running the Web Interface

```bash
streamlit run app/streamlit_app.py
```

The interface allows you to:
- Enter climate data for recent weeks
- Generate forecasts up to 12 weeks ahead
- View predictions with 95% confidence intervals
- Assess transmission risk levels

## Project Structure

```
├── app/
│   └── streamlit_app.py      # Web interface
├── pipeline/
│   ├── preprocess.py          # Data preprocessing and feature engineering
│   ├── train.py               # Model training and evaluation
│   └── forecast.py            # Forecasting with confidence intervals
├── data/                      # Training data (CSV files)
├── models/                    # Trained models
├── artifacts/                 # Preprocessing artifacts
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

## License

MIT

---
Developed by Mohamed Mustaf Ahmed, School of Global Health, Faculty of Medicine, Chulalongkorn University, Bangkok, Thailand
