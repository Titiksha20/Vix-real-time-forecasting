"""
VIX Forecasting System - Main Script (FINAL VERSION)
Runs 4 core tasks: Baselines, Regime, Direction, HAR Decomposition
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

# Safe Unicode handling for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='strict')
        sys.stderr.reconfigure(encoding='utf-8', errors='strict')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout, encoding='utf-8', errors='strict')
        sys.stderr = io.TextIOWrapper(sys.stderr, encoding='utf-8', errors='strict')

from config import (
    setup_logging, get_system_status, RESULTS_DIR,
    get_latest_complete_date, CRITICAL_FEATURES
)
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from vix_forecasting_pipeline import run_pipeline
from visualizations import VIXVisualizer

logger = logging.getLogger(__name__)


class VIXSystem:
    """Main system orchestrator."""
    
    def __init__(self, initial_setup=False):
        self.initial_setup = initial_setup
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()
        
        logger.info("VIX Forecasting System initialized")
    
    def run_full_pipeline(self):
        """Run complete forecasting pipeline with 4 core tasks."""
        results = {
            'timestamp': datetime.now(),
            'success': False,
            'steps_completed': [],
            'errors': []
        }
        
        try:
            # Step 1: Data fetching
            logger.info("Step 1: Fetching market data")
            df_raw, data_updated = self._fetch_data()
            results['steps_completed'].append('data_fetch')
            results['data_updated'] = data_updated
            
            # Check data quality
            latest_complete = get_latest_complete_date(df_raw, CRITICAL_FEATURES)
            if latest_complete:
                days_behind = (datetime.now().date() - latest_complete.date()).days
                results['latest_complete_date'] = latest_complete.strftime('%Y-%m-%d')
                results['days_behind'] = days_behind
                
                if days_behind > 0:
                    logger.warning(f"Latest complete data is {days_behind} days old")
            
            # Step 2: Feature engineering
            logger.info("Step 2: Engineering features")
            df_features, new_feature_count = self._engineer_features(df_raw)
            results['steps_completed'].append('feature_engineering')
            results['new_features'] = new_feature_count
            
            if len(df_features) == 0:
                raise ValueError("No valid features generated")
            
            # Step 3: Run all 4 tasks
            logger.info("Step 3: Running all 4 forecasting tasks")
            pipeline, task_results, report = run_pipeline(df_features, RESULTS_DIR)
            results['steps_completed'].append('all_tasks_completed')
            results['task_results'] = task_results
            
            # Step 4: Generate visualizations
            logger.info("Step 4: Generating visualizations")
            visualizer = VIXVisualizer(RESULTS_DIR)
            visualizer.plot_all(pipeline)
            results['steps_completed'].append('visualizations')
            
            results['success'] = True
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['errors'].append(str(e))
            results['success'] = False
        
        return results
    
    def _fetch_data(self):
        """Fetch or update market data."""
        if self.initial_setup:
            logger.info("Performing initial data fetch")
            df = self.fetcher.fetch_initial_data()
            data_updated = True
        else:
            df, data_updated = self.fetcher.fetch_incremental_update()
        
        is_valid, message = self.fetcher.validate_data(df)
        if not is_valid:
            logger.warning(f"Data validation issues: {message}")
        
        logger.info(f"Data fetched: {len(df)} records, Updated: {data_updated}")
        return df, data_updated
    
    def _engineer_features(self, df_raw):
        """Engineer features for modeling."""
        if self.initial_setup:
            df_features = self.engineer.process_full_dataset(df_raw)
            new_count = len(df_features)
        else:
            df_features, new_count = self.engineer.process_incremental(df_raw)
        
        is_valid, message = self.engineer.validate_features(df_features)
        if not is_valid:
            logger.error(f"Feature validation failed: {message}")
            raise ValueError(f"Feature validation failed: {message}")
        
        logger.info(f"Features engineered: {new_count} new, {len(df_features)} total")
        return df_features, new_count
    
    def print_system_status(self):
        """Print current system status."""
        status = get_system_status()
        
        print("\n" + "=" * 80)
        print("SYSTEM STATUS")
        print("=" * 80)
        print(f"Last Processed Date:  {status['last_processed_date'].strftime('%Y-%m-%d')}")
        print(f"Data Directory:       {status['data_directory']}")
        print(f"Model Directory:      {status['model_directory']}")
        print(f"Raw Data Exists:      {status['raw_data_exists']}")
        print(f"Processed Data:       {status['processed_data_exists']}")
        
        if 'total_records' in status:
            print(f"Total Records:        {status['total_records']}")
            print(f"Date Range:           {status['date_range']}")
        
        print("=" * 80 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='VIX Forecasting System - 4 Core Tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                  # Run full pipeline
    python main.py --initial-setup  # First-time setup
    python main.py --status         # Check system status

Tasks:
    1. Baselines - Shows level prediction fails
    2. Regime Prediction - PRIMARY actionable model (AUC=0.943)
    3. Direction Prediction - Modest edge over baselines
    4. HAR Decomposition - Mean reversion analysis
"""
    )
    
    parser.add_argument('--initial-setup', action='store_true', 
                       help='Perform initial system setup (fetch all historical data)')
    parser.add_argument('--status', action='store_true', 
                       help='Show system status and exit')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    setup_logging()
    logger.info("=" * 80)
    logger.info("VIX FORECASTING SYSTEM - 4 CORE TASKS")
    logger.info("=" * 80)
    
    system = VIXSystem(initial_setup=args.initial_setup)
    
    if args.status:
        system.print_system_status()
        return 0
    
    system.print_system_status()
    
    try:
        results = system.run_full_pipeline()
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Status:          {'SUCCESS' if results['success'] else 'FAILED'}")
        print(f"Timestamp:       {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Steps Completed: {', '.join(results['steps_completed'])}")
        
        if results['success']:
            print(f"Data Updated:    {results.get('data_updated', False)}")
            print(f"New Features:    {results.get('new_features', 0)}")
            
            if 'latest_complete_date' in results:
                print(f"Latest Data:     {results['latest_complete_date']}")
            
            print("\nTASKS COMPLETED:")
            print("  1. Baselines (Naive, ARIMAX, XGBoost)")
            print("  2. Regime Prediction (PRIMARY MODEL)")
            print("  3. Direction Prediction")
            print("  4. HAR Decomposition")
            
            print(f"\nResults saved to: {RESULTS_DIR}")
            print(f"Plots saved to:   {RESULTS_DIR / 'plots'}")
        else:
            print(f"Errors: {'; '.join(results['errors'])}")
        
        print("=" * 80 + "\n")
        logger.info("Pipeline execution completed")
        
        return 0 if results['success'] else 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"\nERROR: Pipeline failed - {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())