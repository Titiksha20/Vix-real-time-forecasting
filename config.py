"""
VIX Forecasting System - Configuration Module (FIXED FOR WINDOWS UNICODE)
"""

from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import sys

# =========================================================
# SAFE Unicode handling for Windows (NO .buffer usage)
# =========================================================
if sys.platform == "win32":
    try:
        # Python 3.7+ (preferred)
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
        sys.stderr.reconfigure(encoding="utf-8", errors="strict")
    except AttributeError:
        # Fallback for wrapped stdout / older environments
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout, encoding="utf-8", errors="strict")
        sys.stderr = io.TextIOWrapper(sys.stderr, encoding="utf-8", errors="strict")

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_DIR / "raw_market_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_features.csv"
LAST_UPDATE_PATH = DATA_DIR / "last_update.txt"
PREDICTIONS_PATH = RESULTS_DIR / "daily_predictions.csv"

# =========================================================
# DATE SETTINGS
# =========================================================
INITIAL_START_DATE = "2007-01-01"
TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE = "2022-12-31"

# =========================================================
# MODEL PARAMETERS
# =========================================================
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

DIRECTION_CLASSIFIER_PARAMS = {
    'objective': 'binary:logistic',
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

ROLLING_WINDOWS = [5, 10, 20]
MA_WINDOWS = [5, 20]
FEATURE_LAGS = [1, 2, 3]
HIGH_VOL_THRESHOLD = 25

# =========================================================
# DATA SOURCES
# =========================================================
CBOE_VVIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv"
CBOE_SKEW_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"

# =========================================================
# LOGGING
# =========================================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging():
    """Configure logging with UTF-8 support."""
    log_file = LOGS_DIR / f"vix_system_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

# =========================================================
# SYSTEM CONSTANTS
# =========================================================
TRADING_DAYS_PER_YEAR = 252
MAX_FETCH_DAYS = 365
WEEKEND_DAYS = [5, 6]
MAX_LOOKBACK_DAYS = 10
MIN_DATA_COMPLETENESS = 0.80
SIGNIFICANT_CHANGE_THRESHOLD = 0.10
MIN_CONFIDENCE_THRESHOLD = 0.6

BASE_FEATURES = [
    'logVIX_t', 'logVIX_lag1', 'logVIX_lag2', 'logVIX_lag3',
    'logVIX_MA5', 'logVIX_MA20',
    'RV_5_t', 'RV_10_t', 'RV_20_t',
    'VVIX_t', 'SKEW_t', 'Credit_Spread_t',
    'SPX_return_t', 'neg_return_t',
    'logVIX_momentum', 'RV_trend'
]

CRITICAL_FEATURES = ['VIX', 'SPX', 'logVIX']
MIN_TRAINING_SAMPLES = 1000
MAX_MISSING_DATA_PCT = 0.05

DECIMAL_PLACES = {
    'accuracy': 3,
    'rmse': 4,
    'mae': 4,
    'vix_level': 2,
    'probability': 3
}

# =========================================================
# UTILITIES
# =========================================================
def get_last_processed_date():
    if LAST_UPDATE_PATH.exists():
        try:
            with open(LAST_UPDATE_PATH, 'r', encoding='utf-8') as f:
                return datetime.strptime(f.read().strip(), '%Y-%m-%d')
        except Exception:
            pass
    return datetime.strptime(INITIAL_START_DATE, '%Y-%m-%d')

def update_last_processed_date(date):
    with open(LAST_UPDATE_PATH, 'w', encoding='utf-8') as f:
        f.write(date.strftime('%Y-%m-%d'))

def get_latest_complete_date(df, required_cols=None):
    if required_cols is None:
        required_cols = CRITICAL_FEATURES

    available_cols = [c for c in required_cols if c in df.columns]
    if not available_cols:
        return None

    complete_mask = df[available_cols].notna().all(axis=1)
    dates = df.index[complete_mask]
    return dates[-1] if len(dates) else None

def is_trading_day(date):
    return date.weekday() < 5

def get_system_status():
    status = {
        'last_processed_date': get_last_processed_date(),
        'data_directory': str(DATA_DIR),
        'model_directory': str(MODEL_DIR),
        'raw_data_exists': RAW_DATA_PATH.exists(),
        'processed_data_exists': PROCESSED_DATA_PATH.exists(),
        'models_count': len(list(MODEL_DIR.glob('*.pkl')))
    }

    if PROCESSED_DATA_PATH.exists():
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
            status['total_records'] = len(df)
            status['date_range'] = f"{df.index.min()} → {df.index.max()}"
        except Exception:
            pass

    return status

def validate_config():
    train_date = datetime.strptime(TRAIN_END_DATE, '%Y-%m-%d')
    val_date = datetime.strptime(VAL_END_DATE, '%Y-%m-%d')

    if train_date >= val_date:
        raise ValueError("TRAIN_END_DATE must be before VAL_END_DATE")

    if HIGH_VOL_THRESHOLD <= 0:
        raise ValueError("HIGH_VOL_THRESHOLD must be positive")

    if not 0 <= MIN_CONFIDENCE_THRESHOLD <= 1:
        raise ValueError("MIN_CONFIDENCE_THRESHOLD must be between 0 and 1")

    if MIN_TRAINING_SAMPLES < 100:
        raise ValueError("MIN_TRAINING_SAMPLES should be at least 100")

validate_config()
