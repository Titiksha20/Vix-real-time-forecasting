"""
VIX Forecasting System - Data Fetching Module (ENHANCED WITH VIX OHLC)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple

from config import (
    RAW_DATA_PATH, LAST_UPDATE_PATH, INITIAL_START_DATE,
    CBOE_VVIX_URL, CBOE_SKEW_URL, get_last_processed_date,
    update_last_processed_date, MAX_FETCH_DAYS, CRITICAL_FEATURES,
    get_latest_complete_date, is_trading_day
)

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Handles fetching and updating market data for VIX forecasting.
    NOW INCLUDES VIX OHLC for innovation prediction.
    """
    
    def __init__(self):
        """Initialize the DataFetcher."""
        self.last_processed = get_last_processed_date()
        logger.info(f"DataFetcher initialized. Last processed date: {self.last_processed}")
    
    def fetch_initial_data(self, start_date: str = INITIAL_START_DATE) -> pd.DataFrame:
        """
        Fetch complete historical dataset for initial setup.
        
        Args:
            start_date (str): Start date for data fetch in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Complete raw market data
        """
        logger.info(f"Fetching initial historical data from {start_date}")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = self._fetch_market_data(start_date, end_date)
        
        # Find latest complete date
        latest_complete = get_latest_complete_date(df, CRITICAL_FEATURES)
        
        if latest_complete:
            logger.info(f"Latest complete data date: {latest_complete}")
            update_last_processed_date(latest_complete)
        
        # Save raw data
        df.to_csv(RAW_DATA_PATH)
        
        logger.info(f"Initial data fetched: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def fetch_incremental_update(self) -> Tuple[pd.DataFrame, bool]:
        """
        Fetch only new data since last processed date.
        Handles partial data availability and timestamp misalignment.
        
        Returns:
            Tuple[pd.DataFrame, bool]: (Updated complete dataset, whether new data was added)
        """
        # Load existing data
        if not RAW_DATA_PATH.exists():
            logger.warning("No existing data found. Performing initial fetch.")
            df = self.fetch_initial_data()
            return df, True
        
        df_existing = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
        
        # Find latest complete date in existing data
        last_complete = get_latest_complete_date(df_existing, CRITICAL_FEATURES)
        
        if last_complete is None:
            logger.warning("No complete data found in existing dataset. Re-fetching.")
            df = self.fetch_initial_data()
            return df, True
        
        last_date = df_existing.index.max()
        logger.info(f"Last date in data: {last_date}, Last complete: {last_complete}")
        
        # Calculate next business day to fetch
        next_date = self._next_business_day(last_date)
        today = datetime.now()
        
        # Check if we need to fetch new data
        if next_date.date() >= today.date():
            logger.info("Data is already up to date. No fetch required.")
            return df_existing, False
        
        # Fetch new data with some overlap to handle updates
        start_fetch = last_complete.strftime('%Y-%m-%d')
        end_fetch = today.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching incremental data from {start_fetch} to {end_fetch}")
        
        try:
            df_new = self._fetch_market_data(start_fetch, end_fetch)
            
            if df_new.empty:
                logger.info("No new market data available.")
                return df_existing, False
            
            # Combine with existing data (new data overwrites old for same dates)
            df_combined = pd.concat([df_existing, df_new])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined = df_combined.sort_index()
            
            # Find new latest complete date
            new_latest_complete = get_latest_complete_date(df_combined, CRITICAL_FEATURES)
            
            # Check if we actually have new complete data
            if new_latest_complete and new_latest_complete > last_complete:
                update_last_processed_date(new_latest_complete)
                data_updated = True
                logger.info(f"New complete data through {new_latest_complete}")
            else:
                data_updated = False
                logger.info("Fetched data but no new complete records added")
            
            # Save updated data
            df_combined.to_csv(RAW_DATA_PATH)
            
            logger.info(f"Total records: {len(df_combined)}")
            return df_combined, data_updated
            
        except Exception as e:
            logger.error(f"Error during incremental fetch: {e}", exc_info=True)
            return df_existing, False
    
    def _fetch_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch all required market data from various sources.
        NOW INCLUDES VIX OHLC DATA.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Raw market data with all indicators
        """
        # 1. Fetch SPX with OHLC
        logger.debug("Fetching SPX OHLC data")
        try:
            spx_data = yf.download(
                "^GSPC",
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            
            df = pd.DataFrame(index=spx_data.index)
            df["SPX"] = spx_data["Close"]
            df["SPX_Open"] = spx_data["Open"]
            df["SPX_High"] = spx_data["High"]
            df["SPX_Low"] = spx_data["Low"]
            
        except Exception as e:
            logger.error(f"Failed to fetch SPX: {e}")
            raise ValueError("Cannot proceed without SPX data")
        
        # 2. Fetch VIX with OHLC (CRITICAL for innovation prediction)
        logger.debug("Fetching VIX OHLC data")
        try:
            vix_data = yf.download(
                "^VIX",
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            
            df["VIX"] = vix_data["Close"]
            df["VIX_Open"] = vix_data["Open"]
            df["VIX_High"] = vix_data["High"]
            df["VIX_Low"] = vix_data["Low"]
            
            logger.info(f"VIX OHLC fetched: {df['VIX'].notna().sum()} values")
            
        except Exception as e:
            logger.error(f"Failed to fetch VIX OHLC: {e}")
            raise ValueError("Cannot proceed without VIX data")
        
        # 3. Calculate SPX returns
        df["SPX_log_return"] = np.log(df["SPX"] / df["SPX"].shift(1))
        df["neg_return"] = np.where(df["SPX_log_return"] < 0, df["SPX_log_return"], 0.0)
        
        # 4. Calculate realized volatility
        for window in [5, 10, 20]:
            df[f"RV_{window}"] = df["SPX_log_return"].rolling(window).std() * np.sqrt(252)
        
        # 5. Log VIX and moving averages
        df["logVIX"] = np.log(df["VIX"])
        df["logVIX_MA5"] = df["logVIX"].rolling(5).mean()
        df["logVIX_MA20"] = df["logVIX"].rolling(20).mean()
        
        # 6. Fetch credit spread (BAA-AAA) - weekly data, forward fill
        logger.debug("Fetching credit spread data")
        try:
            baa = pdr.DataReader("BAA", "fred", start_date, end_date)
            aaa = pdr.DataReader("AAA", "fred", start_date, end_date)
            credit_spread = (baa["BAA"] - aaa["AAA"]).to_frame("Credit_Spread")
            credit_spread.index = pd.to_datetime(credit_spread.index)
            df = df.merge(credit_spread, how="left", left_index=True, right_index=True)
            df["Credit_Spread"] = df["Credit_Spread"].ffill().bfill()
            logger.info(f"Credit spread fetched: {df['Credit_Spread'].notna().sum()} values")
        except Exception as e:
            logger.warning(f"Failed to fetch credit spread: {e}. Using forward fill from existing.")
            df["Credit_Spread"] = np.nan
        
        # 7. Fetch VVIX (volatility of VIX)
        logger.debug("Fetching VVIX data")
        try:
            vvix = pd.read_csv(CBOE_VVIX_URL)
            logger.debug(f"VVIX columns: {vvix.columns.tolist()}")
            
            vvix["DATE"] = pd.to_datetime(vvix["DATE"])
            vvix = vvix.set_index("DATE")
            
            if "VVIX" in vvix.columns:
                df = df.merge(vvix[["VVIX"]], how="left", left_index=True, right_index=True)
                df["VVIX"] = df["VVIX"].ffill().bfill()
                vvix_count = df["VVIX"].notna().sum()
                logger.info(f"VVIX fetched successfully: {vvix_count} values")
            else:
                logger.error(f"VVIX column not found. Available columns: {vvix.columns.tolist()}")
                df["VVIX"] = np.nan
        except Exception as e:
            logger.error(f"Failed to fetch VVIX: {e}", exc_info=True)
            df["VVIX"] = np.nan
        
        # 8. Fetch SKEW
        logger.debug("Fetching SKEW data")
        try:
            skew = pd.read_csv(CBOE_SKEW_URL)
            logger.debug(f"SKEW columns: {skew.columns.tolist()}")
            
            skew["DATE"] = pd.to_datetime(skew["DATE"])
            skew = skew.set_index("DATE")
            
            if "SKEW" in skew.columns:
                df = df.merge(skew[["SKEW"]], how="left", left_index=True, right_index=True)
                df["SKEW"] = df["SKEW"].ffill().bfill()
                skew_count = df["SKEW"].notna().sum()
                logger.info(f"SKEW fetched successfully: {skew_count} values")
            else:
                logger.error(f"SKEW column not found. Available columns: {skew.columns.tolist()}")
                df["SKEW"] = np.nan
        except Exception as e:
            logger.error(f"Failed to fetch SKEW: {e}", exc_info=True)
            df["SKEW"] = np.nan
        
        # Filter to only business days
        df = self._filter_business_days(df)
        
        # Log data completeness
        self._log_data_completeness(df)
        
        return df
    
    def _filter_business_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to only include business days (Mon-Fri)."""
        df = df[df.index.dayofweek < 5]
        return df
    
    def _log_data_completeness(self, df: pd.DataFrame):
        """Log information about data completeness."""
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                logger.info(f"{col}: {missing_count} missing ({missing_pct:.1f}%)")
    
    def _next_business_day(self, date: datetime) -> datetime:
        """Get the next business day after given date."""
        next_day = date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data quality and completeness."""
        issues = []
        
        # Check for required columns (including OHLC now)
        required_cols = ['SPX', 'VIX', 'logVIX', 'VVIX', 'SKEW', 'Credit_Spread',
                        'VIX_Open', 'VIX_High', 'VIX_Low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for critical features
        for col in CRITICAL_FEATURES:
            if col not in df.columns:
                issues.append(f"Missing critical feature: {col}")
            elif df[col].isna().all():
                issues.append(f"Critical feature {col} is all NaN")
        
        # Check for excessive missing data in critical features
        for col in CRITICAL_FEATURES:
            if col in df.columns:
                missing_pct = df[col].isna().sum() / len(df)
                if missing_pct > 0.50:
                    issues.append(f"{col} has {missing_pct:.1%} missing data")
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(f"Found {dup_count} duplicate dates - will keep last")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Data validation passed"
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for the dataset."""
        latest_complete = get_latest_complete_date(df, CRITICAL_FEATURES)
        
        summary = {
            'total_records': len(df),
            'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
            'latest_complete_date': latest_complete.date() if latest_complete else None,
            'columns': list(df.columns),
            'missing_data': {},
            'vix_stats': {}
        }
        
        # Missing data summary
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                summary['missing_data'][col] = f"{missing_count} ({missing_count/len(df):.1%})"
        
        # VIX stats
        if 'VIX' in df.columns and df['VIX'].notna().any():
            summary['vix_stats'] = {
                'mean': round(df['VIX'].mean(), 2),
                'std': round(df['VIX'].std(), 2),
                'min': round(df['VIX'].min(), 2),
                'max': round(df['VIX'].max(), 2),
                'current': round(df['VIX'].iloc[-1], 2) if df['VIX'].notna().iloc[-1] else None
            }
        
        return summary