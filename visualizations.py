"""
Visualization Module - Enhanced plots for all 4 tasks with model comparisons.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


class VIXVisualizer:
    """Creates publication-quality visualizations for all tasks."""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def plot_all(self, pipeline):
        """Generate all plots."""
        logger.info("Generating visualizations...")
        
        self.plot_task1_baselines(pipeline.all_results['task1_baselines'])
        self.plot_task2_regime(pipeline.all_results['task2_regime'])
        self.plot_task3_direction(pipeline.all_results['task3_direction'])
        self.plot_task4_innovation(pipeline.all_results['task4_innovation'])
        
        
        
        logger.info(f"All plots saved to {self.plots_dir}")
    
    def plot_task1_baselines(self, results):
        """Task 1: Baseline comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time series
        ax = axes[0, 0]
        test_actual = results['naive']['test_actual']
        test_pred = results['naive']['test_pred']
        ax.plot(test_actual.index, test_actual, label='Actual', linewidth=2)
        ax.plot(test_pred.index, test_pred, label='Naive', linewidth=2, alpha=0.7)
        ax.set_title('Naive Baseline (Persistence)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter
        ax = axes[0, 1]
        ax.scatter(test_actual, test_pred, alpha=0.5)
        lims = [min(test_actual.min(), test_pred.min()), max(test_actual.max(), test_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2)
        ax.set_xlabel('Actual logVIX')
        ax.set_ylabel('Predicted logVIX')
        ax.set_title('Actual vs Predicted', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Model comparison
        ax = axes[1, 0]
        models = ['Naive']
        rmses = [results['naive']['test_rmse']]
        if results.get('arimax'):
            models.append('ARIMAX')
            rmses.append(results['arimax']['test_rmse'])
        if results.get('xgboost'):
            models.append('XGBoost')
            rmses.append(results['xgboost']['test_rmse'])
        
        bars = ax.bar(models, rmses, color=['gray', 'steelblue', 'orange'][:len(models)], edgecolor='black')
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
                   ha='center', va='bottom', fontsize=11)
        
        # Text summary
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = (
            "KEY FINDING:\n\n"
            "Level prediction fails.\n"
            "Mean reversion dominates.\n\n"
            "Neither ARIMAX nor ML\n"
            "beats naive persistence.\n\n"
            "→ Need to model DEVIATIONS"
        )
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', 
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'task1_baselines_{self.timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_task2_regime(self, results):
        """Task 2: Regime classification with XGBoost vs Random Forest comparison."""
        # Main detailed plot (3 rows x 3 columns)
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        horizons = ['1d', '3d', '5d']
        for idx, horizon in enumerate(horizons):
            r = results[horizon]
            
            # Row 1: VIX level with regime predictions
            ax = axes[idx, 0]
            ax.plot(r['test_indices'], r['test_vix'], color='blue', linewidth=1, alpha=0.7, label='VIX')
            ax.axhline(r['threshold'], color='red', linestyle='--', linewidth=2, label='High Vol Threshold')
            high_vol_times = r['test_indices'][r['actuals'] == 1]
            pred_high_times = r['test_indices'][r['predictions'] == 1]
            ax.scatter(high_vol_times, r['test_vix'][r['actuals'] == 1], color='red', s=30, alpha=0.7, 
                      label='Actual High Vol', marker='x')
            ax.scatter(pred_high_times, r['test_vix'][r['predictions'] == 1], color='green', s=30, alpha=0.7, 
                      label='Predicted High Vol', marker='o')
            ax.set_title(f"{horizon.upper()} High Vol Predictions ({r['best_name']})", 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Date', fontsize=9)
            ax.set_ylabel('VIX Level')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            
            # Row 2: Confusion matrix (best model)
            ax = axes[idx, 1]
            cm = r['confusion_matrix']
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Low Vol', 'High Vol'])
            ax.set_yticklabels(['Low Vol', 'High Vol'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{horizon.upper()} Confusion Matrix ({r["best_name"]})', 
                        fontsize=11, fontweight='bold')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                           color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
            
            # Row 3: Feature importance (best model)
            ax = axes[idx, 2]
            top_10 = r['importance'].head(10)
            ax.barh(range(len(top_10)), top_10['importance'], color='steelblue', edgecolor='black')
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['feature'], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f"{horizon.upper()} Feature Importance", fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('TASK 2: REGIME PREDICTION - VIX Timeline, Confusion Matrix, Features', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'task2_regime_detailed_{self.timestamp}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Model comparison plot (2 rows x 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        horizons_names = [f'{h.upper()}' for h in horizons]
        
        # Accuracy comparison
        ax = axes[0, 0]
        xgb_accs = [results[h]['xgb_accuracy'] for h in horizons]
        rf_accs = [results[h]['rf_accuracy'] for h in horizons]
        baselines = [results[h]['baseline'] for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.25
        ax.bar(x - width, xgb_accs, width, label='XGBoost', color='steelblue', edgecolor='black')
        ax.bar(x, rf_accs, width, label='Random Forest', color='orange', edgecolor='black')
        ax.bar(x + width, baselines, width, label='Baseline', color='gray', edgecolor='black')
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison: XGBoost vs Random Forest', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # F1 Score comparison
        ax = axes[0, 1]
        xgb_f1s = [results[h]['xgb_f1'] for h in horizons]
        rf_f1s = [results[h]['rf_f1'] for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.35
        ax.bar(x - width/2, xgb_f1s, width, label='XGBoost', color='steelblue', edgecolor='black', alpha=0.7)
        ax.bar(x + width/2, rf_f1s, width, label='Random Forest', color='orange', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (name, xgb_val, rf_val) in enumerate(zip(horizons_names, xgb_f1s, rf_f1s)):
            ax.text(i - width/2, xgb_val, f'{xgb_val:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, rf_val, f'{rf_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Precision vs Recall
        ax = axes[1, 0]
        xgb_precisions = [results[h]['xgb_precision'] for h in horizons]
        xgb_recalls = [results[h]['xgb_recall'] for h in horizons]
        rf_precisions = [results[h]['rf_precision'] for h in horizons]
        rf_recalls = [results[h]['rf_recall'] for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.2
        ax.bar(x - 1.5*width, xgb_precisions, width, label='XGB Precision', color='steelblue', edgecolor='black')
        ax.bar(x - 0.5*width, xgb_recalls, width, label='XGB Recall', color='lightblue', edgecolor='black')
        ax.bar(x + 0.5*width, rf_precisions, width, label='RF Precision', color='orange', edgecolor='black')
        ax.bar(x + 1.5*width, rf_recalls, width, label='RF Recall', color='lightsalmon', edgecolor='black')
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Score')
        ax.set_title('Precision vs Recall Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # AUC comparison
        ax = axes[1, 1]
        xgb_aucs = [results[h]['xgb_auc'] for h in horizons]
        rf_aucs = [results[h]['rf_auc'] for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.35
        ax.bar(x - width/2, xgb_aucs, width, label='XGBoost', color='steelblue', edgecolor='black', alpha=0.7)
        ax.bar(x + width/2, rf_aucs, width, label='Random Forest', color='orange', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('AUC')
        ax.set_title('ROC AUC Comparison', fontweight='bold')
        ax.set_ylim([0.5, 1.0])
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (name, xgb_val, rf_val) in enumerate(zip(horizons_names, xgb_aucs, rf_aucs)):
            ax.text(i - width/2, xgb_val, f'{xgb_val:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, rf_val, f'{rf_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('TASK 2: MODEL COMPARISON - XGBoost vs Random Forest', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'task2_regime_comparison_{self.timestamp}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_task3_direction(self, results):
        """Task 3: Direction prediction with XGBoost vs Random Forest comparison."""
        # Main comparison plot (3 rows x 3 columns)
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        horizons = ['1d', '3d', '5d']
        for idx, horizon in enumerate(horizons):
            r = results[horizon]
            
            # Row 1: Direction predictions timeline
            ax = axes[idx, 0]
            correct = (r['predictions'] == r['actuals'])
            correct_times = r['test_indices'][correct]
            incorrect_times = r['test_indices'][~correct]
            ax.scatter(correct_times, [1]*sum(correct), color='green', s=20, alpha=0.5, label='Correct', marker='o')
            ax.scatter(incorrect_times, [0]*sum(~correct), color='red', s=20, alpha=0.5, label='Incorrect', marker='x')
            ax.set_ylim([-0.5, 1.5])
            ax.set_title(f"{horizon.upper()} Predictions ({r['best_name']}, Acc: {r['accuracy']:.3f})", 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Date', fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Wrong', 'Correct'])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Row 2: Confusion matrix (best model)
            ax = axes[idx, 1]
            cm = r['confusion_matrix']
            im = ax.imshow(cm, cmap='Greens', aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Down', 'Up'])
            ax.set_yticklabels(['Down', 'Up'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{horizon.upper()} Confusion Matrix ({r["best_name"]})', 
                        fontsize=11, fontweight='bold')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", 
                           color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
            
            # Row 3: Feature importance (best model)
            ax = axes[idx, 2]
            top_10 = r['importance'].head(10)
            ax.barh(range(len(top_10)), top_10['importance'], color='steelblue', edgecolor='black')
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['feature'], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f"{horizon.upper()} Feature Importance", fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('TASK 3: DIRECTIONAL PREDICTION - Timeline, Confusion Matrix, Features', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'task3_direction_detailed_{self.timestamp}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Model comparison plot (2 rows x 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        horizons_names = [f'{h.upper()}' for h in horizons]
        
        # Accuracy comparison
        ax = axes[0, 0]
        xgb_accs = [results[h]['xgb_accuracy'] for h in horizons]
        rf_accs = [results[h]['rf_accuracy'] for h in horizons]
        baselines = [results[h]['persistence_acc'] for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.25
        ax.bar(x - width, xgb_accs, width, label='XGBoost', color='steelblue', edgecolor='black')
        ax.bar(x, rf_accs, width, label='Random Forest', color='orange', edgecolor='black')
        ax.bar(x + width, baselines, width, label='Persistence', color='gray', edgecolor='black')
        ax.axhline(0.5, color='red', linestyle='--', label='Random', alpha=0.7)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison: XGBoost vs Random Forest', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Edge over random
        ax = axes[0, 1]
        xgb_edges = [(results[h]['xgb_accuracy'] - 0.5) * 100 for h in horizons]
        rf_edges = [(results[h]['rf_accuracy'] - 0.5) * 100 for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.35
        bars1 = ax.bar(x - width/2, xgb_edges, width, label='XGBoost', color='steelblue', edgecolor='black', alpha=0.7)
        bars2 = ax.bar(x + width/2, rf_edges, width, label='Random Forest', color='orange', edgecolor='black', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Edge over Random (%)')
        ax.set_title('Predictive Edge Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # AUC comparison
        ax = axes[1, 0]
        xgb_aucs = [results[h]['xgb_auc'] for h in horizons]
        rf_aucs = [results[h]['rf_auc'] for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.35
        ax.bar(x - width/2, xgb_aucs, width, label='XGBoost', color='steelblue', edgecolor='black', alpha=0.7)
        ax.bar(x + width/2, rf_aucs, width, label='Random Forest', color='orange', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('AUC')
        ax.set_title('ROC AUC Comparison', fontweight='bold')
        ax.set_ylim([0.5, 0.7])
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (name, xgb_val, rf_val) in enumerate(zip(horizons_names, xgb_aucs, rf_aucs)):
            ax.text(i - width/2, xgb_val, f'{xgb_val:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, rf_val, f'{rf_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1 Score comparison
        ax = axes[1, 1]
        xgb_f1s = [results[h]['xgb_f1'] for h in horizons]
        rf_f1s = [results[h]['rf_f1'] for h in horizons]
        
        x = np.arange(len(horizons_names))
        width = 0.35
        ax.bar(x - width/2, xgb_f1s, width, label='XGBoost', color='steelblue', edgecolor='black', alpha=0.7)
        ax.bar(x + width/2, rf_f1s, width, label='Random Forest', color='orange', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (name, xgb_val, rf_val) in enumerate(zip(horizons_names, xgb_f1s, rf_f1s)):
            ax.text(i - width/2, xgb_val, f'{xgb_val:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, rf_val, f'{rf_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('TASK 3: MODEL COMPARISON - XGBoost vs Random Forest', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'task3_direction_comparison_{self.timestamp}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_task4_innovation(self, results):
        """Task 4: Innovation regression."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        test_actual = results['test_actual']
        xgb_pred = results['xgboost']['predictions']
        ridge_pred = results['ridge']['predictions']
        
        # Time series
        ax = axes[0, 0]
        ax.plot(test_actual.index, test_actual, label='Actual', linewidth=2)
        ax.plot(test_actual.index, xgb_pred, label='XGBoost', linewidth=2, alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Innovation Over Time', fontsize=13, fontweight='bold')
        ax.set_ylabel('Innovation (deviation from HAR)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter
        ax = axes[0, 1]
        ax.scatter(test_actual, xgb_pred, alpha=0.5)
        lims = [min(test_actual.min(), xgb_pred.min()), max(test_actual.max(), xgb_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2)
        ax.set_xlabel('Actual Innovation')
        ax.set_ylabel('Predicted Innovation')
        ax.set_title(f"XGBoost R²={results['xgboost']['r2']:.3f}", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Feature importance
        ax = axes[0, 2]
        importance = results['xgboost']['importance'].head(8)
        ax.barh(range(len(importance)), importance['importance'], color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(importance['feature'], fontsize=10)
        ax.set_title('Feature Importance', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Model comparison
        ax = axes[1, 0]
        models = ['Ridge', 'XGBoost']
        rmses = [results['ridge']['rmse'], results['xgboost']['rmse']]
        bars = ax.bar(models, rmses, color=['gray', 'steelblue'], edgecolor='black')
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Model Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}',
                   ha='center', fontsize=11, fontweight='bold')
        
        # Residual distribution
        ax = axes[1, 1]
        residuals = test_actual.values - xgb_pred
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals')
        ax.set_title(f'Residual Distribution (mean={residuals.mean():.4f})', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Key insight
        ax = axes[1, 2]
        ax.axis('off')
        r2_val = results['xgboost']['r2']
        if r2_val > 0.05:
            insight = (
                "HAR Decomposition\n\n"
                "Innovation prediction\n"
                f"shows signal (R²={r2_val:.3f})\n\n"
                "Some VIX shocks are\n"
                "predictable beyond\n"
                "mean reversion"
            )
            bgcolor = 'lightgreen'
        else:
            insight = (
                "HAR Decomposition\n\n"
                "HAR explains 96%+ of\n"
                "VIX variance.\n\n"
                f"Innovations unpredictable\n"
                f"(R²={r2_val:.3f})\n\n"
                "Shocks are news-driven"
            )
            bgcolor = 'lightblue'
        
        ax.text(0.5, 0.5, insight, ha='center', va='center',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=0.5))
        
        plt.suptitle('TASK 4: HAR DECOMPOSITION & INNOVATION ANALYSIS', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'task4_innovation_{self.timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
    



