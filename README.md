# VIX Real Time Forecasting System 

## Overview

A production-grade machine learning system for VIX (CBOE Volatility Index) forecasting using ensemble methods and time series analysis. The system is designed to study volatility dynamics through complementary tasks: level behavior, regime identification, directional signals, and mean-reversion structure.

The system prioritizes **risk regime identification and interpretability**, rather than short-horizon point forecasting, reflecting the empirical structure of volatility dynamics.

---

## System Architecture

The forecasting pipeline consists of four core tasks:

1. **Baseline Models** – Establish statistical benchmarks for VIX level prediction
2. **Regime Classification** – Binary prediction of high-volatility states
3. **Directional Prediction** – Multi-horizon forecasting of VIX direction
4. **HAR Decomposition** – Structural modeling of volatility persistence and innovations

Each task is trained and evaluated independently to isolate sources of predictability.

---

## Installation

### Requirements

Python 3.8 or higher with the following packages:

```
pandas
numpy
yfinance
pandas-datareader
xgboost
scikit-learn
statsmodels
matplotlib
seaborn
```

### Setup

Install dependencies:

```bash
pip install pandas numpy yfinance pandas-datareader xgboost scikit-learn statsmodels matplotlib seaborn
```

Initialize system (first-time setup):

```bash
python main.py --initial-setup
```

This performs a full historical data download, feature engineering, model training, and report generation.

---

## File Structure

```
vix_forecasting_system/
│
├── config.py                      Configuration and system constants
├── data_fetcher.py                Market data acquisition (VIX, SPX, VVIX, SKEW)
├── feature_engineering.py         Feature creation with HAR, VRP, OHLC
│
├── task1_baselines.py             Baseline models (Naive, ARIMAX, XGBoost)
├── task2_regime.py                Regime classification (XGBoost vs Random Forest)
├── task3_direction.py             Directional prediction (XGBoost vs Random Forest)
├── task4_innovation.py            HAR decomposition and innovation analysis
│
├── vix_forecasting_pipeline.py    Pipeline orchestrator
├── main.py                        Command-line interface
├── visualizations.py              Plotting and analytics
│
├── data/                          Data storage (auto-created)
│   ├── raw_market_data.csv
│   ├── processed_features.csv
│   └── last_update.txt
│
├── models/                        Trained models (auto-created)
│   └── *.pkl
│
├── logs/                          Execution logs (auto-created)
│   └── vix_system_YYYYMMDD.log
│
├── results/                       Reports and outputs (auto-created)
│   ├── summary_report_*.txt
│   ├── daily_predictions.csv
│   └── plots/
│
└── README.md
```

---

## Usage

### Daily Execution

```bash
python main.py
```

Performs incremental data updates, feature computation, model execution, and report generation.

### System Status

```bash
python main.py --status
```

Displays data coverage, last update timestamp, and model availability.

---

## Data Sources

**Primary Sources (via yfinance):**
- VIX: CBOE Volatility Index (OHLC)
- SPX: S&P 500 Index (OHLC)

**Secondary Sources:**
- VVIX: VIX of VIX (CBOE API)
- SKEW: CBOE SKEW Index (CBOE API)
- Credit Spread: BAA-AAA corporate spread (FRED)

**Engineered Features:**
- Realized Volatility: Rolling windows (5, 10, 20 days)
- Variance Risk Premium: Squared VIX minus realized variance
- HAR Components: Daily, weekly, monthly averages
- VIX OHLC Dynamics: Overnight gaps, intraday ranges
- Asymmetric Returns: Separate up/down market effects


---

## Model Specifications

The system employs a **parsimonious and interpretable set of well-established models**, selected based on empirical relevance and robustness rather than model complexity.

**Task 1 – Baseline Level Prediction**

* Naive persistence model
* ARIMAX model with macro-volatility exogenous variables
* XGBoost regression model

**Task 2 – Volatility Regime Classification**

* Random Forest classifier
* XGBoost classifier

**Task 3 – Directional Prediction**

* Random Forest classifier
* XGBoost classifier

**Task 4 – HAR Decomposition**

* Linear HAR model (daily, weekly, monthly components)
* Innovation analysis using linear and tree-based regressors

---

## Output Files

### Summary Reports

Location: `results/summary_report_YYYYMMDD_HHMMSS.txt`

Contents:
- Complete model performance metrics
- Test period specification
- Feature importance rankings
- Trading signal recommendations

### Daily Predictions

Location: `results/daily_predictions.csv`

Columns:
- Date
- VIX_actual
- Regime_1d_prob, Regime_3d_prob, Regime_5d_prob
- Direction_1d_prob, Direction_3d_prob, Direction_5d_prob
- HAR_expected
- Best_model_1d, Best_model_3d, Best_model_5d
---

## Key Findings

* VIX level dynamics are dominated by strong mean reversion
* High-volatility regimes exhibit measurable short-horizon predictability
* Directional signals are weak but statistically consistent out-of-sample
* Deviations from mean reversion (innovations) display limited predictability

These findings are consistent with established results in the academic volatility literature.

---

## Limitations

* Daily frequency only; intraday dynamics are not modeled
* Structural breaks and extreme events remain difficult to forecast
* Feature set excludes futures, options, microstructure, and term structure data 

---

## Troubleshooting

* Review execution logs in the `logs/` directory for detailed diagnostics
* Re-run full system initialization if data ingestion or feature validation fails
* Negative out-of-sample R² for innovation forecasts is expected and reflects the absence of predictable structure

---

## License and Disclaimer

This system is intended for research and educational purposes only. It does not constitute investment advice. Past performance does not guarantee future results.

---

## References

Bollerslev, T., Tauchen, G., & Zhou, H. (2009). *Expected Stock Returns and Variance Risk Premia*. Review of Financial Studies.

Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics.

Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series*. Econometrica.
