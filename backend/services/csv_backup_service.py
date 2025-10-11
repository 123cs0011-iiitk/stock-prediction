"""
CSV Backup Service for Stock Data Storage
Provides redundant storage of stock data in CSV format as backup to PostgreSQL
"""

import os
import csv
import logging
from datetime import datetime, date
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVBackupService:
    """CSV backup service for stock data with automatic fallback capabilities"""
    
    def __init__(self, backup_dir: str = "../backup"):
        """
        Initialize CSV backup service
        
        Args:
            backup_dir: Directory to store CSV backup files
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Subdirectories for different data types
        self.quotes_dir = self.backup_dir / "quotes"
        self.historical_dir = self.backup_dir / "historical"
        self.companies_dir = self.backup_dir / "companies"
        
        # Create subdirectories
        self.quotes_dir.mkdir(exist_ok=True)
        self.historical_dir.mkdir(exist_ok=True)
        self.companies_dir.mkdir(exist_ok=True)
        
        logger.info(f"CSV backup service initialized with backup directory: {self.backup_dir}")
    
    def _get_quotes_file_path(self, symbol: str) -> Path:
        """Get the file path for stock quotes CSV"""
        return self.quotes_dir / f"{symbol.upper()}_quotes.csv"
    
    def _get_historical_file_path(self, symbol: str) -> Path:
        """Get the file path for historical data CSV"""
        return self.historical_dir / f"{symbol.upper()}_historical.csv"
    
    def _get_company_file_path(self, symbol: str) -> Path:
        """Get the file path for company info CSV"""
        return self.companies_dir / f"{symbol.upper()}_company.csv"
    
    def store_stock_quote(self, symbol: str, quote_data: Dict, source: str = 'unknown') -> bool:
        """
        Store stock quote data in CSV format
        
        Args:
            symbol: Stock symbol
            quote_data: Quote data dictionary
            source: Data source name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            symbol = symbol.upper()
            file_path = self._get_quotes_file_path(symbol)
            
            # Prepare data for CSV
            csv_data = {
                'timestamp': quote_data.get('timestamp', datetime.now().isoformat()),
                'symbol': symbol,
                'price': quote_data.get('price', 0),
                'change': quote_data.get('change', 0),
                'change_percent': quote_data.get('change_percent', 0),
                'volume': quote_data.get('volume', 0),
                'high': quote_data.get('high', 0),
                'low': quote_data.get('low', 0),
                'open': quote_data.get('open', 0),
                'previous_close': quote_data.get('previous_close', 0),
                'market_cap': quote_data.get('market_cap', 'N/A'),
                'sector': quote_data.get('sector', 'N/A'),
                'industry': quote_data.get('industry', 'N/A'),
                'currency': quote_data.get('currency', 'USD'),
                'source': source
            }
            
            # Check if file exists to determine if we need headers
            file_exists = file_path.exists()
            
            # Write to CSV
            with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = csv_data.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_data)
            
            logger.info(f"Stored quote data for {symbol} in CSV backup")
            return True
            
        except Exception as e:
            logger.error(f"Error storing quote data for {symbol} in CSV: {e}")
            return False
    
    def store_historical_data(self, symbol: str, historical_data: List[Dict], source: str = 'unknown') -> bool:
        """
        Store historical stock data in CSV format
        
        Args:
            symbol: Stock symbol
            historical_data: List of historical data dictionaries
            source: Data source name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            symbol = symbol.upper()
            file_path = self._get_historical_file_path(symbol)
            
            # Prepare data for CSV
            csv_data_list = []
            for day_data in historical_data:
                csv_data = {
                    'date': day_data.get('date', ''),
                    'symbol': symbol,
                    'open': day_data.get('open', 0),
                    'high': day_data.get('high', 0),
                    'low': day_data.get('low', 0),
                    'close': day_data.get('close', 0),
                    'volume': day_data.get('volume', 0),
                    'source': source
                }
                csv_data_list.append(csv_data)
            
            # Check if file exists
            file_exists = file_path.exists()
            
            # Write to CSV
            with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
                if csv_data_list:
                    fieldnames = csv_data_list[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header if file is new
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerows(csv_data_list)
            
            logger.info(f"Stored {len(historical_data)} historical records for {symbol} in CSV backup")
            return True
            
        except Exception as e:
            logger.error(f"Error storing historical data for {symbol} in CSV: {e}")
            return False
    
    def store_company_info(self, symbol: str, company_data: Dict, source: str = 'unknown') -> bool:
        """
        Store company information in CSV format
        
        Args:
            symbol: Stock symbol
            company_data: Company data dictionary
            source: Data source name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            symbol = symbol.upper()
            file_path = self._get_company_file_path(symbol)
            
            # Prepare data for CSV
            csv_data = {
                'symbol': symbol,
                'name': company_data.get('name', ''),
                'description': company_data.get('description', ''),
                'sector': company_data.get('sector', ''),
                'industry': company_data.get('industry', ''),
                'market_cap': company_data.get('market_cap', ''),
                'pe_ratio': company_data.get('pe_ratio', ''),
                'dividend_yield': company_data.get('dividend_yield', ''),
                '52_week_high': company_data.get('52_week_high', 0),
                '52_week_low': company_data.get('52_week_low', 0),
                'currency': company_data.get('currency', 'USD'),
                'website': company_data.get('website', ''),
                'employees': company_data.get('employees', ''),
                'city': company_data.get('city', ''),
                'state': company_data.get('state', ''),
                'country': company_data.get('country', ''),
                'source': source,
                'last_updated': datetime.now().isoformat()
            }
            
            # Write to CSV (overwrite for company info as it should be unique)
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = csv_data.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(csv_data)
            
            logger.info(f"Stored company info for {symbol} in CSV backup")
            return True
            
        except Exception as e:
            logger.error(f"Error storing company info for {symbol} in CSV: {e}")
            return False
    
    def get_latest_quote(self, symbol: str, max_age_minutes: int = 5) -> Optional[Dict]:
        """
        Get the latest stock quote from CSV backup
        
        Args:
            symbol: Stock symbol
            max_age_minutes: Maximum age of data in minutes
            
        Returns:
            Dict: Latest quote data or None if not found/too old
        """
        try:
            symbol = symbol.upper()
            file_path = self._get_quotes_file_path(symbol)
            
            if not file_path.exists():
                return None
            
            # Read CSV file
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            
            # Get the latest record
            latest_record = df.iloc[-1]
            
            # Check if data is not too old
            try:
                record_time = datetime.fromisoformat(latest_record['timestamp'])
                cutoff_time = datetime.now() - pd.Timedelta(minutes=max_age_minutes)
                
                if record_time < cutoff_time:
                    logger.info(f"CSV quote data for {symbol} is too old")
                    return None
                
                # Convert to dictionary
                return latest_record.to_dict()
                
            except Exception as e:
                logger.warning(f"Could not parse timestamp for {symbol}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest quote for {symbol} from CSV: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 100) -> List[Dict]:
        """
        Get historical data from CSV backup
        
        Args:
            symbol: Stock symbol
            days: Number of days to retrieve
            
        Returns:
            List[Dict]: Historical data records
        """
        try:
            symbol = symbol.upper()
            file_path = self._get_historical_file_path(symbol)
            
            if not file_path.exists():
                return []
            
            # Read CSV file
            df = pd.read_csv(file_path)
            if df.empty:
                return []
            
            # Sort by date (newest first) and limit
            df = df.sort_values('date', ascending=False).head(days)
            
            # Convert to list of dictionaries
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol} from CSV: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """
        Get company information from CSV backup
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Company information or None if not found
        """
        try:
            symbol = symbol.upper()
            file_path = self._get_company_file_path(symbol)
            
            if not file_path.exists():
                return None
            
            # Read CSV file
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            
            # Return the first (and should be only) record
            return df.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error getting company info for {symbol} from CSV: {e}")
            return None
    
    def get_backup_statistics(self) -> Dict:
        """
        Get statistics about the CSV backup data
        
        Returns:
            Dict: Backup statistics
        """
        try:
            stats = {
                'quotes_files': len(list(self.quotes_dir.glob('*.csv'))),
                'historical_files': len(list(self.historical_dir.glob('*.csv'))),
                'companies_files': len(list(self.companies_dir.glob('*.csv'))),
                'backup_directory': str(self.backup_dir),
                'total_size_mb': 0
            }
            
            # Calculate total size
            total_size = 0
            for file_path in self.backup_dir.rglob('*.csv'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting backup statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up old data from CSV backups
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            # Clean old quotes
            for file_path in self.quotes_dir.glob('*.csv'):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        # Convert timestamp column to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        # Keep only recent data
                        df_filtered = df[df['timestamp'] >= cutoff_date]
                        # Write back to file
                        df_filtered.to_csv(file_path, index=False)
                        logger.info(f"Cleaned old quotes data in {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not clean {file_path.name}: {e}")
            
            logger.info(f"CSV backup cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during CSV backup cleanup: {e}")
    
    def health_check(self) -> Dict:
        """
        Check health of CSV backup system
        
        Returns:
            Dict: Health status
        """
        try:
            # Check if directories exist and are writable
            test_file = self.backup_dir / "health_check.tmp"
            
            # Test write
            test_file.write_text("health_check")
            test_file.unlink()  # Delete test file
            
            return {
                'status': 'healthy',
                'backup_directory': str(self.backup_dir),
                'writable': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backup_directory': str(self.backup_dir),
                'writable': False,
                'timestamp': datetime.now().isoformat()
            }

# Global CSV backup service instance
csv_backup = CSVBackupService()
