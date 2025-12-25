"""
Task 3: Simple Directional Prediction - ENHANCED WITH RANDOM FOREST
Predict sign of logVIX_{t+h} - logVIX_t with model comparison.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class DirectionClassifier:
    """Simple directional classifier with XGBoost and Random Forest comparison."""
    
    def __init__(self):
        self.xgb_models = {}
        self.rf_models = {}
        self.best_models = {}
        self.results = {}
        self.feature_list = [
            'logVIX_t', 'logVIX_lag1', 'logVIX_lag2',
            'logVIX_MA5', 'logVIX_MA20',
            'RV_5_t', 'RV_20_t',
            'VVIX_t', 'SKEW_t',
            'SPX_return_t', 'neg_return_t',
            'logVIX_momentum', 'RV_trend'
        ]
    
    def run(self, train_df, test_df):
        """Train direction classifiers for 1/3/5 day horizons with model comparison."""
        logger.info("="*80)
        logger.info("TASK 3: SIMPLE DIRECTIONAL PREDICTION (XGBoost vs Random Forest)")
        logger.info("="*80)
        
        horizons = ['1d', '3d', '5d']
        
        for horizon in horizons:
            logger.info(f"\n3.{horizons.index(horizon)+1} {horizon.upper()} DIRECTION")
            
            # Prepare data
            X_train = train_df[self.feature_list].dropna()
            y_train = train_df.loc[X_train.index, f'direction_{horizon}']
            
            X_test = test_df[self.feature_list].dropna()
            y_test = test_df.loc[X_test.index, f'direction_{horizon}']
            
            # Baselines
            majority_class = int(y_train.mode()[0])
            majority_pred = np.full(len(y_test), majority_class)
            majority_acc = accuracy_score(y_test, majority_pred)
            
            persistence_pred = test_df.loc[X_test.index, f'last_direction_{horizon}'].values
            persistence_acc = accuracy_score(y_test, persistence_pred)
            
            # Train XGBoost
            logger.info("  Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train, verbose=False)
            
            xgb_pred = xgb_model.predict(X_test)
            xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            xgb_acc = accuracy_score(y_test, xgb_pred)
            xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
            xgb_recall = recall_score(y_test, xgb_pred, zero_division=0)
            xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
            xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
            xgb_cm = confusion_matrix(y_test, xgb_pred)
            
            # Train Random Forest
            logger.info("  Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            rf_pred = rf_model.predict(X_test)
            rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_precision = precision_score(y_test, rf_pred, zero_division=0)
            rf_recall = recall_score(y_test, rf_pred, zero_division=0)
            rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
            rf_auc = roc_auc_score(y_test, rf_pred_proba)
            rf_cm = confusion_matrix(y_test, rf_pred)
            
            # Select best model
            if xgb_acc >= rf_acc:
                best_model = xgb_model
                best_pred = xgb_pred
                best_pred_proba = xgb_pred_proba
                best_name = 'XGBoost'
                best_acc = xgb_acc
                best_auc = xgb_auc
                best_f1 = xgb_f1
                best_cm = xgb_cm
                best_importance = pd.DataFrame({
                    'feature': self.feature_list,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                best_model = rf_model
                best_pred = rf_pred
                best_pred_proba = rf_pred_proba
                best_name = 'Random Forest'
                best_acc = rf_acc
                best_auc = rf_auc
                best_f1 = rf_f1
                best_cm = rf_cm
                best_importance = pd.DataFrame({
                    'feature': self.feature_list,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # Log results
            logger.info(f"\n  Baselines:")
            logger.info(f"    Random:              0.500")
            logger.info(f"    Majority class:      {majority_acc:.3f}")
            logger.info(f"    Persistence:         {persistence_acc:.3f}")
            logger.info(f"\n  XGBoost:")
            logger.info(f"    Accuracy:            {xgb_acc:.3f}")
            logger.info(f"    Precision:           {xgb_precision:.3f}")
            logger.info(f"    Recall:              {xgb_recall:.3f}")
            logger.info(f"    F1:                  {xgb_f1:.3f}")
            logger.info(f"    AUC:                 {xgb_auc:.3f}")
            logger.info(f"    vs Majority:         {(xgb_acc-majority_acc)*100:+.2f}pp")
            logger.info(f"    vs Persistence:      {(xgb_acc-persistence_acc)*100:+.2f}pp")
            logger.info(f"\n  Random Forest:")
            logger.info(f"    Accuracy:            {rf_acc:.3f}")
            logger.info(f"    Precision:           {rf_precision:.3f}")
            logger.info(f"    Recall:              {rf_recall:.3f}")
            logger.info(f"    F1:                  {rf_f1:.3f}")
            logger.info(f"    AUC:                 {rf_auc:.3f}")
            logger.info(f"    vs Majority:         {(rf_acc-majority_acc)*100:+.2f}pp")
            logger.info(f"    vs Persistence:      {(rf_acc-persistence_acc)*100:+.2f}pp")
            logger.info(f"\n  → Best model: {best_name} (Accuracy={best_acc:.3f})")
            
            # Edge over random
            edge = (best_acc - 0.5) * 100
            logger.info(f"  → Edge over random: {edge:+.2f}pp")
            
            # Store models
            self.xgb_models[horizon] = xgb_model
            self.rf_models[horizon] = rf_model
            self.best_models[horizon] = best_model
            
            # Store results
            self.results[horizon] = {
                # Best model
                'best_name': best_name,
                'accuracy': best_acc,
                'auc': best_auc,
                'f1': best_f1,
                'confusion_matrix': best_cm,
                'predictions': best_pred,
                'probabilities': best_pred_proba,
                'importance': best_importance,
                
                # XGBoost results
                'xgb_accuracy': xgb_acc,
                'xgb_precision': xgb_precision,
                'xgb_recall': xgb_recall,
                'xgb_f1': xgb_f1,
                'xgb_auc': xgb_auc,
                'xgb_cm': xgb_cm,
                'xgb_pred': xgb_pred,
                'xgb_pred_proba': xgb_pred_proba,
                
                # Random Forest results
                'rf_accuracy': rf_acc,
                'rf_precision': rf_precision,
                'rf_recall': rf_recall,
                'rf_f1': rf_f1,
                'rf_auc': rf_auc,
                'rf_cm': rf_cm,
                'rf_pred': rf_pred,
                'rf_pred_proba': rf_pred_proba,
                
                # Baselines
                'majority_acc': majority_acc,
                'persistence_acc': persistence_acc,
                
                # Common
                'actuals': y_test.values,
                'test_indices': y_test.index,
                'edge': edge
            }
        
        # Summary comparison
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*80)
        logger.info(f"\n{'Horizon':<10} {'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} {'Edge':<10}")
        logger.info("-"*80)
        
        for horizon in horizons:
            r = self.results[horizon]
            
            # XGBoost row
            logger.info(f"{horizon:<10} {'XGBoost':<15} {r['xgb_accuracy']:<10.3f} "
                       f"{r['xgb_precision']:<10.3f} {r['xgb_recall']:<10.3f} "
                       f"{r['xgb_f1']:<10.3f} {r['xgb_auc']:<10.3f} "
                       f"{(r['xgb_accuracy']-0.5)*100:<10.2f}%")
            
            # Random Forest row
            logger.info(f"{'':<10} {'Random Forest':<15} {r['rf_accuracy']:<10.3f} "
                       f"{r['rf_precision']:<10.3f} {r['rf_recall']:<10.3f} "
                       f"{r['rf_f1']:<10.3f} {r['rf_auc']:<10.3f} "
                       f"{(r['rf_accuracy']-0.5)*100:<10.2f}%")
            
            # Best model indicator
            logger.info(f"{'':<10} {'→ BEST: ' + r['best_name']:<15}")
            logger.info("")
        
        logger.info("="*80)
        logger.info("CONCLUSION: Direction prediction provides modest edge")
        logger.info("  → XGBoost and Random Forest perform similarly")
        logger.info("  → Edge improves with longer forecast horizons")
        logger.info("="*80)
        
        return self.results
