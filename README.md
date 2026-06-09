# Energy Demand Forecasting

An end-to-end machine learning project for forecasting daily energy demand using historical consumption data and weather-related features.

## Project Overview

This project compares traditional time-series forecasting methods with machine learning approaches to predict daily energy demand.

Models evaluated:

* Seasonal Naive Baseline
* SARIMA (Auto ARIMA)
* XGBoost

## Results

| Model          | RMSE   |
| -------------- | ------ |
| Seasonal Naive | 66,106 |
| SARIMA         | 70,227 |
| XGBoost        | 58,431 |

**Best Model:** XGBoost

Additional evaluation:

* MAPE: 4.7%

## Project Structure

```text
data/
├── raw/
├── processed/

src/
├── ingest.py
├── validation.py
├── features.py
├── train.py
├── config.py

models/
app/
notebooks/
```

## Features

Feature engineering includes:

* Lag features
* Rolling statistics
* Calendar-based features
* Time-series preprocessing

## Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Statsmodels
* Pmdarima

## Future Improvements

* MLflow experiment tracking
* FastAPI deployment
* Docker containerization
* CI/CD pipeline
* Automated retraining

## Author

Gulnur Yildiz

