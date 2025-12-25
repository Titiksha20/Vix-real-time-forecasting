"""
Task 4: Innovation (Deviation) Regression - ENHANCED WITH HAR (FIXED)
Model deviations from HAR expectation using VIX OHLC + VRP features.

CRITICAL FIX: HAR model now predicts TOMORROW's logVIX, not today's (prevents identity function)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


class InnovationRegressor:
    """Innovation regression - HAR model with enhanced features."""
    
    def __init__(self):
        self.har_model = None
        self.ar_alpha = None  # Unified naming
        self.ar_beta_daily = None
        self.ar_beta_weekly = None
        self.ar_beta_monthly = None
        self.ridge_model = None
        self.xgb_model = None
        self.results = {}
        self.used_har = False  # Track which model was used
        
        # Enhanced innovation features
        self.feature_list = [
            # Original features
            'logVIX_t',
            'VVIX_t',
            'SKEW_t',
            'Credit_Spread_t',
            'SPX_return_t',
            'neg_return_t',
            # VIX OHLC features (NEW!)
            'VIX_overnight_gap_pct',
            'VIX_intraday_range',
            'VIX_gap_ratio',
            # Variance Risk Premium (NEW!)
            'variance_risk_premium',
            'vrp_normalized',
            # HAR components (NEW!)
            'logVIX_weekly',
            'logVIX_monthly',
            'RV_weekly',
            'RV_monthly',
            # Asymmetric effects (NEW!)
            'return_down',
            'return_up',
            'extreme_down_day',
            # VIX dynamics (NEW!)
            'vix_spike',
            'vix_change_lag1',
            'high_vol_yesterday',
            # RV surprise (NEW!)
            'rv_surprise_normalized'
        ]
    
    def _fit_har(self, train_df):
        """Fit HAR (Heterogeneous AutoRegressive) model.
        
        CRITICAL FIX: Predict TOMORROW's logVIX using TODAY's multi-horizon features.
        This prevents the identity function (β_daily=1, others=0) that was happening before.
        """
        logger.info("\n4.1 HAR EXPECTATION MODEL (Multi-Horizon)")
        
        # Prepare HAR features - all from time t
        har_features = ['logVIX_t', 'logVIX_weekly', 'logVIX_monthly']
        
        # DIAGNOSTIC: Check what's in train_df
        logger.info(f"  train_df columns: {len(train_df.columns)}")
        logger.info(f"  train_df shape: {train_df.shape}")
        
        # Check all features exist
        missing = [f for f in har_features if f not in train_df.columns]
        if missing:
            logger.warning(f"  Missing HAR features: {missing}")
            logger.info(f"  Available columns: {list(train_df.columns)}")
            logger.warning(f"  → Falling back to AR(1)")
            return self._fit_ar1_fallback(train_df)
        
        logger.info(f"  ✓ All HAR features present: {har_features}")
        
        # CRITICAL FIX: Create target as TOMORROW's logVIX
        train_clean = train_df[['logVIX'] + har_features].copy()
        train_clean['logVIX_next'] = train_clean['logVIX'].shift(-1)  # Tomorrow's value
        train_clean = train_clean.dropna()
        
        logger.info(f"  Before dropna: {len(train_df)} rows")
        logger.info(f"  After dropna: {len(train_clean)} rows")
        
        if len(train_clean) < 100:
            logger.warning(f"  Insufficient data for HAR: {len(train_clean)} < 100")
            logger.warning(f"  → Falling back to AR(1)")
            return self._fit_ar1_fallback(train_df)
        
        # Features = today's values, Target = tomorrow's value
        X = train_clean[har_features].values
        y = train_clean['logVIX_next'].values
        
        # Fit HAR model
        self.har_model = LinearRegression()
        self.har_model.fit(X, y)
        
        self.ar_alpha = self.har_model.intercept_
        self.ar_beta_daily = self.har_model.coef_[0]
        self.ar_beta_weekly = self.har_model.coef_[1]
        self.ar_beta_monthly = self.har_model.coef_[2]
        self.used_har = True
        
        # Calculate in-sample fit quality
        y_pred = self.har_model.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        logger.info(f"  ✓ HAR model fitted successfully!")
        logger.info(f"  α (intercept):    {self.ar_alpha:.4f}")
        logger.info(f"  β_daily (t):      {self.ar_beta_daily:.4f}")
        logger.info(f"  β_weekly (5d):    {self.ar_beta_weekly:.4f}")
        logger.info(f"  β_monthly (22d):  {self.ar_beta_monthly:.4f}")
        logger.info(f"  In-sample R²:     {r2:.4f}")
        logger.info(f"  In-sample RMSE:   {rmse:.4f}")
        logger.info(f"  Purpose: Multi-horizon mean reversion (should beat AR(1))")
        
        # Sanity check: coefficients should NOT be [1, 0, 0]
        if abs(self.ar_beta_daily - 1.0) < 0.01 and abs(self.ar_beta_weekly) < 0.01:
            logger.warning("  ⚠ WARNING: HAR collapsed to identity! Check data.")
        
        return True
    
    def _fit_ar1_fallback(self, train_df):
        """Fallback to simple AR(1) if HAR fails."""
        from statsmodels.tsa.ar_model import AutoReg
        
        logger.info("\n4.1 AR(1) EXPECTATION MODEL (Fallback)")
        
        y = train_df['logVIX'].dropna().values
        model = AutoReg(y, lags=1, trend='c')
        fit = model.fit()
        
        self.ar_alpha = fit.params[0]
        self.ar_beta_daily = fit.params[1]
        self.ar_beta_weekly = 0
        self.ar_beta_monthly = 0
        self.used_har = False
        
        logger.info(f"  α (constant): {self.ar_alpha:.4f}")
        logger.info(f"  β (AR coef):  {self.ar_beta_daily:.4f}")
        logger.warning(f"  Using AR(1) fallback (HAR data unavailable)")
        
        return False
    
    def _create_innovation_features(self, df):
        """Create innovation features using HAR expectation.
        
        Innovation = Actual tomorrow - Expected tomorrow
        Expected tomorrow = HAR(today's features)
        """
        df = df.copy()
        
        # HAR expectation: use TODAY's features to predict TOMORROW
        if self.used_har and self.ar_beta_weekly != 0:
            # Full HAR model
            df['logVIX_expected'] = (self.ar_alpha + 
                                     self.ar_beta_daily * df['logVIX_t'] +
                                     self.ar_beta_weekly * df['logVIX_weekly'] +
                                     self.ar_beta_monthly * df['logVIX_monthly'])
        else:
            # AR(1) fallback
            df['logVIX_expected'] = self.ar_alpha + self.ar_beta_daily * df['logVIX_t']
        
        # Innovation = actual tomorrow - expected tomorrow
        df['innovation'] = df['logVIX'].shift(-1) - df['logVIX_expected']
        
        return df
    
    def run(self, train_df, val_df, test_df):
        """Train innovation regression models with HAR + enhanced features."""
        logger.info("="*80)
        logger.info("TASK 4: INNOVATION (DEVIATION) REGRESSION - ENHANCED")
        logger.info("="*80)
        
        # Fit HAR model (FIXED to predict tomorrow)
        har_success = self._fit_har(train_df)
        
        # Create innovation features
        train_df = self._create_innovation_features(train_df)
        val_df = self._create_innovation_features(val_df)
        test_df = self._create_innovation_features(test_df)
        
        # Check which features are available
        available_features = [f for f in self.feature_list if f in train_df.columns]
        missing_features = set(self.feature_list) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
            logger.info(f"Using {len(available_features)} available features")
        
        self.feature_list = available_features
        
        # Prepare data
        train_clean = train_df.dropna(subset=['innovation'] + self.feature_list)
        val_clean = val_df.dropna(subset=['innovation'] + self.feature_list)
        test_clean = test_df.dropna(subset=['innovation'] + self.feature_list)
        
        logger.info(f"\nData after feature creation:")
        logger.info(f"  Train: {len(train_clean)} samples")
        logger.info(f"  Val:   {len(val_clean)} samples")
        logger.info(f"  Test:  {len(test_clean)} samples")
        
        if len(train_clean) < 100:
            logger.error(f"Insufficient training data: {len(train_clean)}")
            return None
        
        X_train = train_clean[self.feature_list]
        y_train = train_clean['innovation']
        X_val = val_clean[self.feature_list]
        y_val = val_clean['innovation']
        X_test = test_clean[self.feature_list]
        y_test = test_clean['innovation']
        
        # Fill any remaining NaN with 0
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
        
        logger.info(f"\nInnovation statistics:")
        logger.info(f"  Train mean: {y_train.mean():.6f} (should be ≈0)")
        logger.info(f"  Train std:  {y_train.std():.4f}")
        logger.info(f"  Test mean:  {y_test.mean():.6f}")
        logger.info(f"  Test std:   {y_test.std():.4f}")
        
        # Ridge regression baseline
        logger.info("\n4.2 RIDGE REGRESSION (BASELINE)")
        self.ridge_model = Ridge(alpha=1.0, random_state=42)
        self.ridge_model.fit(X_train, y_train)
        
        ridge_pred = self.ridge_model.predict(X_test)
        ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
        ridge_mae = mean_absolute_error(y_test, ridge_pred)
        ridge_r2 = r2_score(y_test, ridge_pred)
        
        logger.info(f"  Test RMSE: {ridge_rmse:.4f}")
        logger.info(f"  Test MAE:  {ridge_mae:.4f}")
        logger.info(f"  Test R²:   {ridge_r2:.4f}")
        
        # XGBoost regression with grid search
        logger.info("\n4.3 XGBOOST REGRESSION (ENHANCED)")
        
        configs = [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.9},
            {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.02, 'subsample': 0.85},
        ]
        
        best_rmse = float('inf')
        best_model = None
        
        for config in configs:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                **config
            )
            model.fit(X_train, y_train, verbose=False)
            pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, pred))
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = model
        
        self.xgb_model = best_model
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        logger.info(f"  Test RMSE: {xgb_rmse:.4f}")
        logger.info(f"  Test MAE:  {xgb_mae:.4f}")
        logger.info(f"  Test R²:   {xgb_r2:.4f}")
        logger.info(f"  vs Ridge:  {((ridge_rmse-xgb_rmse)/ridge_rmse)*100:+.2f}%")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_list,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n  Top 10 features:")
        for idx, row in importance.head(10).iterrows():
            logger.info(f"    {row['feature']:30s}: {row['importance']:.4f}")
        
        # Store results
        
        # Generate HAR predictions for test set
        if har_success:
            test_har_predictions = (self.ar_alpha + 
                                   self.ar_beta_daily * test_clean['logVIX_t'] +
                                   self.ar_beta_weekly * test_clean['logVIX_weekly'] +
                                   self.ar_beta_monthly * test_clean['logVIX_monthly'])
        else:
            # Fallback to simple persistence if HAR failed
            test_har_predictions = test_clean['logVIX_t']
        
        self.results = {
            'har_used': har_success,
            'ridge': {
                'rmse': ridge_rmse,
                'mae': ridge_mae,
                'r2': ridge_r2,
                'predictions': ridge_pred
            },
            'xgboost': {
                'rmse': xgb_rmse,
                'mae': xgb_mae,
                'r2': xgb_r2,
                'predictions': xgb_pred,
                'importance': importance
            },
            'test_actual': y_test,
            'test_indices': test_clean.index,
            'test_har_pred': test_har_predictions.values,  # HAR predictions for test set
            'innovation_std': y_test.std()
        }
        
        # Interpretation
        logger.info("\n" + "="*80)
        if xgb_r2 > 0.10:
            logger.info("CONCLUSION: Innovation modeling captures predictable deviations!")
            logger.info(f"  R² = {xgb_r2:.3f} - Meaningful signal in VIX shocks")
        elif xgb_r2 > 0.05:
            logger.info("CONCLUSION: Innovation modeling finds weak but real signal")
            logger.info(f"  R² = {xgb_r2:.3f} - Some VIX shocks are predictable")
        else:
            logger.info("CONCLUSION: Innovations remain largely unpredictable")
            logger.info(f"  R² = {xgb_r2:.3f} - VIX shocks dominated by news/events")
        logger.info("="*80)
        
        return self.results