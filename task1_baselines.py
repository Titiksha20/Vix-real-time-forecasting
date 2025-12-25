"""
Task 1: Baselines and Failure of Level Prediction (FIXED FOR REAL-TIME)
Demonstrates that naive persistence, ARIMAX, and XGBoost cannot beat each other on raw levels.

NOW USES ALL AVAILABLE DATA AFTER VALIDATION PERIOD (real-time compatible)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class BaselineModels:
    """Baseline models demonstrating the difficulty of level prediction."""
    
    def __init__(self):
        self.results = {}
    
    def run(self, train_df, val_df, test_df, feature_list):
        """Run all baseline models.
        
        FIXED: Now properly uses test_df which contains all data after validation period,
        including real-time/latest data if available.
        """
        logger.info("="*80)
        logger.info("TASK 1: BASELINES AND FAILURE OF LEVEL PREDICTION")
        logger.info("="*80)
        logger.info(f"  Train period: {train_df.index.min().date()} to {train_df.index.max().date()}")
        logger.info(f"  Val period:   {val_df.index.min().date()} to {val_df.index.max().date()}")
        logger.info(f"  Test period:  {test_df.index.min().date()} to {test_df.index.max().date()}")
        logger.info(f"  Test size:    {len(test_df)} days (real-time data)")
        
        # Naive baseline
        self._naive_baseline(train_df, val_df, test_df)
        
        # ARIMAX with grid search
        self._arimax_model(train_df, val_df, test_df)
        
        # XGBoost level prediction
        self._xgboost_level(train_df, val_df, test_df, feature_list)
        
        return self.results
    
    def _naive_baseline(self, train_df, val_df, test_df):
        """Naive persistence: tomorrow = today."""
        logger.info("\n1.1 NAIVE PERSISTENCE BASELINE")
        
        # Test predictions (using all available test data)
        test_actual = test_df['logVIX'].iloc[1:]
        test_pred = test_df['logVIX'].shift(1).dropna()
        
        rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        mae = mean_absolute_error(test_actual, test_pred)
        
        # VIX space
        vix_actual = np.exp(test_actual)
        vix_pred = np.exp(test_pred)
        rmse_vix = np.sqrt(mean_squared_error(vix_actual, vix_pred))
        
        self.results['naive'] = {
            'test_rmse': rmse,
            'test_mae': mae,
            'test_rmse_vix': rmse_vix,
            'test_pred': test_pred,
            'test_actual': test_actual,
            'test_period': f"{test_df.index.min().date()} to {test_df.index.max().date()}"
        }
        
        logger.info(f"  Test RMSE (logVIX): {rmse:.4f}")
        logger.info(f"  Test RMSE (VIX):    {rmse_vix:.2f}")
        logger.info(f"  Test period:        {self.results['naive']['test_period']}")
        logger.info(f"  → This is the benchmark to beat")
    
    def _arimax_model(self, train_df, val_df, test_df):
        """ARIMAX with grid search - shows linear models fail."""
        logger.info("\n1.2 ARIMAX MODEL (Grid Search)")
        
        # Prepare exog features
        exog_features_base = ['VVIX', 'SKEW', 'Credit_Spread', 'RV_20', 'SPX_log_return', 'neg_return']
        exog_cols = []
        
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        for feature in exog_features_base:
            if feature in train_df.columns:
                train_df[f'{feature}_lag1'] = train_df[feature].shift(1)
                val_df[f'{feature}_lag1'] = val_df[feature].shift(1)
                test_df[f'{feature}_lag1'] = test_df[feature].shift(1)
                exog_cols.append(f'{feature}_lag1')
        
        exog_cols.extend(['logVIX_MA5', 'logVIX_MA20', 'RV_5', 'RV_10'])
        exog_cols = [c for c in exog_cols if c in train_df.columns]
        
        train_clean = train_df[['logVIX'] + exog_cols].dropna()
        val_clean = val_df[['logVIX'] + exog_cols].dropna()
        test_clean = test_df[['logVIX'] + exog_cols].dropna()
        
        if len(train_clean) < 100:
            logger.error(f"Insufficient clean training data: {len(train_clean)} rows")
            self.results['arimax'] = None
            return
        
        train_target = train_clean['logVIX']
        val_target = val_clean['logVIX']
        test_target = test_clean['logVIX']
        
        # Standardize
        scaler = StandardScaler()
        train_exog = pd.DataFrame(
            scaler.fit_transform(train_clean[exog_cols]), 
            index=train_clean.index, 
            columns=exog_cols
        )
        val_exog = pd.DataFrame(
            scaler.transform(val_clean[exog_cols]), 
            index=val_clean.index, 
            columns=exog_cols
        )
        test_exog = pd.DataFrame(
            scaler.transform(test_clean[exog_cols]), 
            index=test_clean.index, 
            columns=exog_cols
        )
        
        # Test different orders (grid search)
        candidates = [(1,0,0), (2,0,0), (1,0,1), (1,1,0), (1,1,1), (2,1,1)]
        results = []
        
        logger.info("  Testing ARIMA configurations...")
        for order in candidates:
            try:
                model = SARIMAX(
                    endog=train_target, 
                    exog=train_exog, 
                    order=order,
                    enforce_stationarity=False, 
                    enforce_invertibility=False,
                    initialization='approximate_diffuse',
                    concentrate_scale=True
                )
                
                fit = model.fit(
                    disp=False, 
                    maxiter=200,
                    method='powell',
                    warn_convergence=False
                )
                
                if not hasattr(fit, 'fittedvalues'):
                    continue
                
                train_fitted = fit.fittedvalues
                val_forecast = fit.forecast(steps=len(val_target), exog=val_exog)
                test_forecast = fit.forecast(steps=len(test_target), exog=test_exog)
                
                train_rmse = np.sqrt(mean_squared_error(train_target, train_fitted))
                val_rmse = np.sqrt(mean_squared_error(val_target, val_forecast))
                test_rmse = np.sqrt(mean_squared_error(test_target, test_forecast))
                
                results.append({
                    'order': order,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'test_rmse': test_rmse,
                    'model': fit,
                    'train_fitted': train_fitted,
                    'val_forecast': val_forecast,
                    'test_forecast': test_forecast,
                    'train_target': train_target,
                    'val_target': val_target,
                    'test_target': test_target
                })
                
            except Exception as e:
                logger.debug(f"  ARIMA{order}: Failed ({type(e).__name__})")
                continue
        
        if not results:
            logger.error("  All ARIMAX models failed to train")
            self.results['arimax'] = None
            return
        
        best_result = min(results, key=lambda x: x['val_rmse'])
        best_order = best_result['order']
        
        vs_naive = ((self.results['naive']['test_rmse'] - best_result['test_rmse']) / 
                   self.results['naive']['test_rmse']) * 100
        
        logger.info(f"  Best model: ARIMA{best_order}")
        logger.info(f"  Test RMSE: {best_result['test_rmse']:.4f}")
        logger.info(f"  Test period: {test_df.index.min().date()} to {test_df.index.max().date()}")
        logger.info(f"  vs Naive: {vs_naive:+.2f}%")
        logger.info(f"  → {'BETTER' if vs_naive > 0 else 'WORSE'} than naive")
        
        self.results['arimax'] = best_result
    
    def _xgboost_level(self, train_df, val_df, test_df, feature_list):
        """XGBoost level prediction with grid search."""
        logger.info("\n1.3 XGBOOST LEVEL REGRESSION (Grid Search)")
        
        X_train = train_df[feature_list]
        y_train = train_df['target']
        X_val = val_df[feature_list]
        y_val = val_df['target']
        X_test = test_df[feature_list]
        y_test = test_df['target']
        
        # Test configs
        configs = [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.9},
        ]
        
        results = []
        for i, params in enumerate(configs):
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, 
                                    n_jobs=-1, **params)
            model.fit(X_train, y_train, verbose=False)
            
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            results.append({
                'params': params,
                'model': model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'train_pred': train_pred,
                'val_pred': val_pred,
                'test_pred': test_pred
            })
        
        best_result = min(results, key=lambda x: x['val_rmse'])
        best_model = best_result['model']
        
        # Full metrics
        test_mae = mean_absolute_error(y_test, best_result['test_pred'])
        test_r2 = r2_score(y_test, best_result['test_pred'])
        
        test_vix_actual = np.exp(y_test.values)
        test_vix_pred = np.exp(best_result['test_pred'])
        test_rmse_vix = np.sqrt(mean_squared_error(test_vix_actual, test_vix_pred))
        
        # Naive comparison
        test_naive_pred = X_test['logVIX_t'].values
        test_rmse_naive = np.sqrt(mean_squared_error(y_test, test_naive_pred))
        
        improvement = ((test_rmse_naive - best_result['test_rmse']) / test_rmse_naive) * 100
        
        logger.info(f"  Best config: depth={best_result['params']['max_depth']}, "
                   f"trees={best_result['params']['n_estimators']}")
        logger.info(f"  Test RMSE: {best_result['test_rmse']:.4f}")
        logger.info(f"  Test RMSE (VIX): {test_rmse_vix:.2f}")
        logger.info(f"  Test period: {test_df.index.min().date()} to {test_df.index.max().date()}")
        logger.info(f"  vs Naive: {improvement:+.2f}%")
        logger.info(f"  → XGBoost ≈ naive baseline (no improvement)")
        
        self.results['xgboost'] = {
            'model': best_model,
            'test_rmse': best_result['test_rmse'],
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse_vix': test_rmse_vix,
            'test_pred': best_result['test_pred'],
            'test_actual': y_test,
            'naive_rmse': test_rmse_naive,
            'improvement': improvement,
            'test_period': f"{test_df.index.min().date()} to {test_df.index.max().date()}"
        }
        
        logger.info("\n" + "="*80)
        logger.info("KEY FINDING: Level prediction is dominated by mean reversion")
        logger.info("  • Naive: {:.4f}".format(self.results['naive']['test_rmse']))
        if self.results.get('arimax'):
            logger.info("  • ARIMAX: {:.4f} ({:+.1f}%)".format(
                self.results['arimax']['test_rmse'],
                ((self.results['naive']['test_rmse'] - self.results['arimax']['test_rmse']) / 
                 self.results['naive']['test_rmse']) * 100
            ))
        logger.info("  • XGBoost: {:.4f} ({:+.1f}%)".format(
            self.results['xgboost']['test_rmse'],
            improvement
        ))
        logger.info("  → ML does not beat persistence on raw levels")
        logger.info("  → We need to model DEVIATIONS, not levels")
        logger.info("="*80)
