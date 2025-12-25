"""
VIX Forecasting System - Feature Engineering Module
ENHANCED with VIX OHLC, HAR components, VRP, and innovation features
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from datetime import datetime

from config import (
    PROCESSED_DATA_PATH, ROLLING_WINDOWS, MA_WINDOWS,
    FEATURE_LAGS, TRADING_DAYS_PER_YEAR, get_latest_complete_date,
    CRITICAL_FEATURES
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering for VIX forecasting models.
    ENHANCED with innovation prediction features.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.processed_data = None
        self._load_existing_features()
        logger.info("FeatureEngineer initialized")
    
    def _load_existing_features(self):
        """Load existing processed features if available."""
        if PROCESSED_DATA_PATH.exists():
            try:
                self.processed_data = pd.read_csv(
                    PROCESSED_DATA_PATH, 
                    index_col=0, 
                    parse_dates=True
                )
                logger.info(f"Loaded {len(self.processed_data)} existing feature records")
            except Exception as e:
                logger.warning(f"Could not load existing features: {e}")
                self.processed_data = None
        else:
            logger.info("No existing processed features found")
    
    def process_full_dataset(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Process complete dataset from scratch with enhanced features."""
        logger.info(f"Processing full dataset: {len(df_raw)} records")
        
        df = df_raw.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Create ALL features (original + enhanced)
        df = self._create_enhanced_features(df)
        
        # ONE simple dropna at the end
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        logger.info(f"After dropna: {len(df)} records (dropped {dropped} rows with NaN)")
        
        # Save processed features
        df.to_csv(PROCESSED_DATA_PATH)
        self.processed_data = df
        
        return df
    
    def process_incremental(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Process only new data rows incrementally."""
        if self.processed_data is None or self.processed_data.empty:
            logger.info("No existing features. Processing full dataset.")
            df = self.process_full_dataset(df_raw)
            return df, len(df)
        
        # Find latest complete date in existing features
        last_processed_date = self.processed_data.index.max()
        latest_complete_raw = get_latest_complete_date(df_raw, CRITICAL_FEATURES)
        
        if latest_complete_raw is None:
            logger.warning("No complete data in raw dataset")
            return self.processed_data, 0
        
        # Check if we have new complete data
        if latest_complete_raw <= last_processed_date:
            logger.info(f"No new complete data. Latest complete: {latest_complete_raw}, Last processed: {last_processed_date}")
            return self.processed_data, 0
        
        logger.info(f"Processing incremental data. Last processed: {last_processed_date}, New data through: {latest_complete_raw}")
        
        # Need historical context for rolling features
        context_start = last_processed_date - pd.Timedelta(days=60)  # Increased for HAR monthly
        context_data = df_raw[df_raw.index >= context_start].copy()
        
        # Process with context
        context_data.index = pd.to_datetime(context_data.index)
        context_data = context_data.sort_index()
        df_with_context = self._create_enhanced_features(context_data)
        df_with_context = df_with_context.dropna()
        
        # Extract only new rows
        df_new_features = df_with_context[df_with_context.index > last_processed_date]
        
        if df_new_features.empty:
            logger.info("No new valid features after processing")
            return self.processed_data, 0
        
        # Append to existing
        df_combined = pd.concat([self.processed_data, df_new_features])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()
        
        # Save updated features
        df_combined.to_csv(PROCESSED_DATA_PATH)
        self.processed_data = df_combined
        
        new_count = len(df_new_features)
        logger.info(f"Added {new_count} new feature rows. Total: {len(df_combined)}")
        
        return df_combined, new_count
    
    def _create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ALL features: Original Colab features + Enhanced innovation features.
        """
        # ========== ORIGINAL COLAB FEATURES ==========
        df['logVIX_t'] = df['logVIX']
        df['logVIX_lag1'] = df['logVIX'].shift(1)
        df['logVIX_lag2'] = df['logVIX'].shift(2)
        df['logVIX_lag3'] = df['logVIX'].shift(3)
        
        df['logVIX_MA5'] = df['logVIX'].rolling(window=5).mean()
        df['logVIX_MA20'] = df['logVIX'].rolling(window=20).mean()
        
        df['RV_5_t'] = df['RV_5']
        df['RV_10_t'] = df['RV_10']
        df['RV_20_t'] = df['RV_20']
        
        df['VVIX_t'] = df['VVIX']
        df['SKEW_t'] = df['SKEW']
        df['Credit_Spread_t'] = df['Credit_Spread']
        
        # Forward fill sparse data
        if 'Credit_Spread_t' in df.columns:
            df['Credit_Spread_t'] = df['Credit_Spread_t'].ffill().bfill()
        if 'VVIX_t' in df.columns:
            df['VVIX_t'] = df['VVIX_t'].ffill()
        if 'SKEW_t' in df.columns:
            df['SKEW_t'] = df['SKEW_t'].ffill()
        
        df['SPX_return_t'] = df['SPX_log_return']
        df['neg_return_t'] = df['neg_return']
        
        df['logVIX_momentum'] = df['logVIX_t'] - df['logVIX_lag1']
        df['logVIX_accel'] = df['logVIX_momentum'] - (df['logVIX_lag1'] - df['logVIX_lag2'])
        df['RV_trend'] = df['RV_20_t'] - df['RV_20'].shift(5)
        
        df['target'] = df['logVIX'].shift(-1)
        
        df['logVIX_1d_ahead'] = df['logVIX'].shift(-1)
        df['logVIX_3d_ahead'] = df['logVIX'].shift(-3)
        df['logVIX_5d_ahead'] = df['logVIX'].shift(-5)
        
        df['direction_1d'] = (df['logVIX_1d_ahead'] > df['logVIX_t']).astype(int)
        df['direction_3d'] = (df['logVIX_3d_ahead'] > df['logVIX_t']).astype(int)
        df['direction_5d'] = (df['logVIX_5d_ahead'] > df['logVIX_t']).astype(int)
        
        df['last_direction_1d'] = (df['logVIX_t'] > df['logVIX_lag1']).astype(int)
        df['last_direction_3d'] = (df['logVIX_t'] > df['logVIX'].shift(3)).astype(int)
        df['last_direction_5d'] = (df['logVIX_t'] > df['logVIX'].shift(5)).astype(int)
        
        df['VIX'] = np.exp(df['logVIX'])
        df['VIX_1d_ahead'] = df['VIX'].shift(-1)
        df['VIX_3d_ahead'] = df['VIX'].shift(-3)
        df['VIX_5d_ahead'] = df['VIX'].shift(-5)
        
        df['high_vol_1d'] = (df['VIX_1d_ahead'] > 20).astype(int)
        df['high_vol_3d'] = (df['VIX_3d_ahead'] > 20).astype(int)
        df['high_vol_5d'] = (df['VIX_5d_ahead'] > 20).astype(int)
        
        # ========== ENHANCED INNOVATION FEATURES ==========
        
        # 1. VIX OHLC Features
        if all(col in df.columns for col in ['VIX_Open', 'VIX_High', 'VIX_Low', 'VIX']):
            df['VIX_overnight_gap'] = df['VIX_Open'] - df['VIX'].shift(1)
            df['VIX_overnight_gap_pct'] = df['VIX_overnight_gap'] / df['VIX'].shift(1)
            df['VIX_intraday_range'] = (df['VIX_High'] - df['VIX_Low']) / df['VIX']
            df['VIX_gap_ratio'] = np.abs(df['VIX_overnight_gap']) / (df['VIX_High'] - df['VIX_Low'] + 1e-6)
        
        # 2. Variance Risk Premium
        if 'VIX' in df.columns and 'RV_20_t' in df.columns:
            vix_squared = (df['VIX'] / 100) ** 2
            expected_rv_squared = (df['RV_20_t']) ** 2
            df['variance_risk_premium'] = vix_squared - expected_rv_squared
            df['vrp_normalized'] = df['variance_risk_premium'] / (vix_squared + 1e-6)
        
        # 3. HAR Components
        df['logVIX_daily'] = df['logVIX_t']
        df['logVIX_weekly'] = df['logVIX_t'].rolling(window=5, min_periods=3).mean()
        df['logVIX_monthly'] = df['logVIX_t'].rolling(window=22, min_periods=10).mean()
        
        df['RV_daily'] = df['RV_5_t']
        df['RV_weekly'] = df['RV_5_t'].rolling(window=5, min_periods=3).mean()
        df['RV_monthly'] = df['RV_20_t']
        
        # 4. Asymmetric Return Effects
        df['return_down'] = np.where(df['SPX_return_t'] < 0, df['SPX_return_t'], 0)
        df['return_up'] = np.where(df['SPX_return_t'] > 0, df['SPX_return_t'], 0)
        df['extreme_down_day'] = (df['SPX_return_t'] < df['SPX_return_t'].rolling(252, min_periods=60).quantile(0.05)).astype(int)
        df['large_negative_return'] = (df['SPX_return_t'] < -0.02).astype(int)
        
        # 5. VIX Spike & Clustering
        df['vix_spike'] = (df['logVIX_momentum'] > df['logVIX_momentum'].rolling(60, min_periods=20).quantile(0.95)).astype(int)
        df['vix_abs_change'] = np.abs(df['logVIX_momentum'])
        df['vix_change_lag1'] = df['vix_abs_change'].shift(1)
        df['high_vol_yesterday'] = (df['VIX'].shift(1) > 20).astype(int)
        
        # 6. RV Surprise
        df['rv_surprise'] = df['RV_5_t'] - df['RV_20_t']
        df['rv_surprise_normalized'] = df['rv_surprise'] / (df['RV_20_t'] + 1e-6)
        
        return df
    
    def get_regression_features(self) -> List[str]:
        """Get list of features used by regression model."""
        return [
            'logVIX_t', 'logVIX_lag1', 'logVIX_lag2', 'logVIX_lag3',
            'logVIX_MA5', 'logVIX_MA20',
            'RV_5_t', 'RV_10_t', 'RV_20_t',
            'VVIX_t', 'SKEW_t', 'Credit_Spread_t',
            'SPX_return_t', 'neg_return_t',
            'logVIX_momentum', 'logVIX_accel', 'RV_trend'
        ]
    
    def get_classification_features(self) -> List[str]:
        """Get list of features used by classification model (no accel)."""
        return [
            'logVIX_t', 'logVIX_lag1', 'logVIX_lag2', 'logVIX_lag3',
            'logVIX_MA5', 'logVIX_MA20',
            'RV_5_t', 'RV_10_t', 'RV_20_t',
            'VVIX_t', 'SKEW_t', 'Credit_Spread_t',
            'SPX_return_t', 'neg_return_t',
            'logVIX_momentum', 'RV_trend'
        ]
    
    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate that all required features are present."""
        issues = []
        
        # Check regression features exist
        reg_features = self.get_regression_features()
        missing_reg = [f for f in reg_features if f not in df.columns]
        if missing_reg:
            issues.append(f"Missing regression features: {missing_reg}")
        
        # Check target
        if 'target' not in df.columns:
            issues.append("Missing target column")
        
        # Check for infinite values
        for col in df.columns:
            if np.isinf(df[col]).any():
                issues.append(f"Infinite values in {col}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Feature validation passed"
    
    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics for features."""
        reg_features = self.get_regression_features()
        
        summary = {
            'total_records': len(df),
            'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
            'feature_stats': {}
        }
        
        # Stats for available features
        for feature in reg_features:
            if feature in df.columns and df[feature].notna().any():
                summary['feature_stats'][feature] = {
                    'mean': round(df[feature].mean(), 4),
                    'std': round(df[feature].std(), 4),
                    'min': round(df[feature].min(), 4),
                    'max': round(df[feature].max(), 4),
                    'missing': df[feature].isna().sum()
                }
        
        return summary