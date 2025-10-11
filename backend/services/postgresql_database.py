"""
PostgreSQL database service for persistent stock data storage
Uses SQLAlchemy ORM for better database management and connection pooling
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Text, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database base class
Base = declarative_base()

class StockQuote(Base):
    """Stock quotes table for real-time data"""
    __tablename__ = 'stock_quotes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    price = Column(Float, nullable=False)
    change = Column(Float, nullable=False)
    change_percent = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    previous_close = Column(Float, nullable=False)
    market_cap = Column(String(50))
    sector = Column(String(100))
    industry = Column(String(100))
    currency = Column(String(3), default='USD')
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='uq_stock_quotes_symbol_timestamp'),
        Index('idx_stock_quotes_symbol_timestamp', 'symbol', 'timestamp'),
    )

class HistoricalData(Base):
    """Historical data table"""
    __tablename__ = 'historical_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False)
    open_price = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_historical_symbol_date'),
        Index('idx_historical_symbol_date', 'symbol', 'date'),
    )

class CompanyInfo(Base):
    """Company information table"""
    __tablename__ = 'company_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(String(50))
    pe_ratio = Column(String(20))
    dividend_yield = Column(String(20))
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    currency = Column(String(3), default='USD')
    website = Column(String(200))
    employees = Column(String(20))
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    source = Column(String(50), nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow)

class PostgreSQLDatabase:
    """PostgreSQL database manager with connection pooling"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.init_database()
    
    def init_database(self):
        """Initialize database connection and create tables"""
        try:
            # Get database URL from environment
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                # Construct from individual components
                db_host = os.getenv('DB_HOST', 'localhost')
                db_port = os.getenv('DB_PORT', '5432')
                db_name = os.getenv('DB_NAME', 'stock_prediction_db')
                db_user = os.getenv('DB_USER', 'postgres')
                db_password = os.getenv('DB_PASSWORD', '')
                
                database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=int(os.getenv('DB_POOL_SIZE', 10)),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', 20)),
                pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', 30)),
                pool_recycle=3600,  # Recycle connections every hour
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            logger.info("PostgreSQL database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def store_stock_quote(self, symbol: str, quote_data: Dict, source: str = 'yahoo') -> bool:
        """Store real-time stock quote data"""
        try:
            with self.get_session() as session:
                # Check if quote already exists for this timestamp
                existing = session.query(StockQuote).filter(
                    StockQuote.symbol == symbol.upper(),
                    StockQuote.timestamp == quote_data.get('timestamp', datetime.now().isoformat())
                ).first()
                
                if existing:
                    # Update existing record
                    existing.price = quote_data.get('price', 0)
                    existing.change = quote_data.get('change', 0)
                    existing.change_percent = quote_data.get('change_percent', 0)
                    existing.volume = quote_data.get('volume', 0)
                    existing.high = quote_data.get('high', 0)
                    existing.low = quote_data.get('low', 0)
                    existing.open_price = quote_data.get('open', 0)
                    existing.previous_close = quote_data.get('previous_close', 0)
                    existing.market_cap = quote_data.get('market_cap', 'N/A')
                    existing.sector = quote_data.get('sector', 'N/A')
                    existing.industry = quote_data.get('industry', 'N/A')
                    existing.source = source
                else:
                    # Create new record
                    quote = StockQuote(
                        symbol=symbol.upper(),
                        price=quote_data.get('price', 0),
                        change=quote_data.get('change', 0),
                        change_percent=quote_data.get('change_percent', 0),
                        volume=quote_data.get('volume', 0),
                        high=quote_data.get('high', 0),
                        low=quote_data.get('low', 0),
                        open_price=quote_data.get('open', 0),
                        previous_close=quote_data.get('previous_close', 0),
                        market_cap=quote_data.get('market_cap', 'N/A'),
                        sector=quote_data.get('sector', 'N/A'),
                        industry=quote_data.get('industry', 'N/A'),
                        currency=quote_data.get('currency', 'USD'),
                        timestamp=datetime.fromisoformat(quote_data.get('timestamp', datetime.now().isoformat())),
                        source=source
                    )
                    session.add(quote)
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing stock quote for {symbol}: {e}")
            return False
    
    def store_historical_data(self, symbol: str, historical_data: List[Dict], source: str = 'yahoo') -> bool:
        """Store historical stock data"""
        try:
            with self.get_session() as session:
                for day_data in historical_data:
                    # Check if record already exists
                    existing = session.query(HistoricalData).filter(
                        HistoricalData.symbol == symbol.upper(),
                        HistoricalData.date == datetime.strptime(day_data['date'], '%Y-%m-%d').date()
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.open_price = day_data['open']
                        existing.high = day_data['high']
                        existing.low = day_data['low']
                        existing.close_price = day_data['close']
                        existing.volume = day_data['volume']
                        existing.source = source
                    else:
                        # Create new record
                        historical = HistoricalData(
                            symbol=symbol.upper(),
                            date=datetime.strptime(day_data['date'], '%Y-%m-%d').date(),
                            open_price=day_data['open'],
                            high=day_data['high'],
                            low=day_data['low'],
                            close_price=day_data['close'],
                            volume=day_data['volume'],
                            source=source
                        )
                        session.add(historical)
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing historical data for {symbol}: {e}")
            return False
    
    def store_company_info(self, symbol: str, company_data: Dict, source: str = 'yahoo') -> bool:
        """Store company information"""
        try:
            with self.get_session() as session:
                # Check if company info already exists
                existing = session.query(CompanyInfo).filter(
                    CompanyInfo.symbol == symbol.upper()
                ).first()
                
                if existing:
                    # Update existing record
                    existing.name = company_data.get('name', existing.name)
                    existing.description = company_data.get('description', existing.description)
                    existing.sector = company_data.get('sector', existing.sector)
                    existing.industry = company_data.get('industry', existing.industry)
                    existing.market_cap = company_data.get('market_cap', existing.market_cap)
                    existing.pe_ratio = company_data.get('pe_ratio', existing.pe_ratio)
                    existing.dividend_yield = company_data.get('dividend_yield', existing.dividend_yield)
                    existing.week_52_high = company_data.get('52_week_high', existing.week_52_high)
                    existing.week_52_low = company_data.get('52_week_low', existing.week_52_low)
                    existing.website = company_data.get('website', existing.website)
                    existing.employees = company_data.get('employees', existing.employees)
                    existing.city = company_data.get('city', existing.city)
                    existing.state = company_data.get('state', existing.state)
                    existing.country = company_data.get('country', existing.country)
                    existing.source = source
                    existing.last_updated = datetime.utcnow()
                else:
                    # Create new record
                    company = CompanyInfo(
                        symbol=symbol.upper(),
                        name=company_data.get('name', ''),
                        description=company_data.get('description', ''),
                        sector=company_data.get('sector', ''),
                        industry=company_data.get('industry', ''),
                        market_cap=company_data.get('market_cap', ''),
                        pe_ratio=company_data.get('pe_ratio', ''),
                        dividend_yield=company_data.get('dividend_yield', ''),
                        week_52_high=company_data.get('52_week_high', 0),
                        week_52_low=company_data.get('52_week_low', 0),
                        currency=company_data.get('currency', 'USD'),
                        website=company_data.get('website', ''),
                        employees=company_data.get('employees', ''),
                        city=company_data.get('city', ''),
                        state=company_data.get('state', ''),
                        country=company_data.get('country', ''),
                        source=source
                    )
                    session.add(company)
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing company info for {symbol}: {e}")
            return False
    
    def get_latest_quote(self, symbol: str, max_age_minutes: int = 5) -> Optional[Dict]:
        """Get the latest stock quote if it's not too old"""
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
                
                quote = session.query(StockQuote).filter(
                    StockQuote.symbol == symbol.upper(),
                    StockQuote.timestamp > cutoff_time
                ).order_by(StockQuote.timestamp.desc()).first()
                
                if quote:
                    return {
                        'symbol': quote.symbol,
                        'price': quote.price,
                        'change': quote.change,
                        'change_percent': quote.change_percent,
                        'volume': quote.volume,
                        'high': quote.high,
                        'low': quote.low,
                        'open': quote.open_price,
                        'previous_close': quote.previous_close,
                        'market_cap': quote.market_cap,
                        'sector': quote.sector,
                        'industry': quote.industry,
                        'currency': quote.currency,
                        'timestamp': quote.timestamp.isoformat(),
                        'source': quote.source
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest quote for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 100) -> List[Dict]:
        """Get historical data for a symbol"""
        try:
            with self.get_session() as session:
                historical_records = session.query(HistoricalData).filter(
                    HistoricalData.symbol == symbol.upper()
                ).order_by(HistoricalData.date.desc()).limit(days).all()
                
                return [
                    {
                        'date': record.date.strftime('%Y-%m-%d'),
                        'open': record.open_price,
                        'high': record.high,
                        'low': record.low,
                        'close': record.close_price,
                        'volume': record.volume
                    }
                    for record in historical_records
                ]
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information"""
        try:
            with self.get_session() as session:
                company = session.query(CompanyInfo).filter(
                    CompanyInfo.symbol == symbol.upper()
                ).first()
                
                if company:
                    return {
                        'symbol': company.symbol,
                        'name': company.name,
                        'description': company.description,
                        'sector': company.sector,
                        'industry': company.industry,
                        'market_cap': company.market_cap,
                        'pe_ratio': company.pe_ratio,
                        'dividend_yield': company.dividend_yield,
                        '52_week_high': company.week_52_high,
                        '52_week_low': company.week_52_low,
                        'currency': company.currency,
                        'website': company.website,
                        'employees': company.employees,
                        'city': company.city,
                        'state': company.state,
                        'country': company.country,
                        'source': company.source,
                        'last_updated': company.last_updated.isoformat()
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            return None
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to keep database size manageable"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
                
                # Clean old quotes (keep only recent ones)
                deleted_quotes = session.query(StockQuote).filter(
                    StockQuote.timestamp < cutoff_date
                ).delete()
                
                # Clean old historical data (keep more)
                cutoff_historical = datetime.utcnow() - timedelta(days=365)  # Keep 1 year
                deleted_historical = session.query(HistoricalData).filter(
                    HistoricalData.date < cutoff_historical.date()
                ).delete()
                
                logger.info(f"Cleaned up {deleted_quotes} old quotes and {deleted_historical} old historical records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_session() as session:
                stats = {}
                
                # Count records in each table
                stats['quotes_count'] = session.query(StockQuote).count()
                stats['historical_count'] = session.query(HistoricalData).count()
                stats['companies_count'] = session.query(CompanyInfo).count()
                
                # Get unique symbols
                stats['unique_symbols'] = session.query(StockQuote.symbol).distinct().count()
                
                # Get database size (PostgreSQL specific)
                try:
                    result = session.execute("""
                        SELECT pg_size_pretty(pg_database_size(current_database())) as size
                    """)
                    size_result = result.fetchone()
                    if size_result:
                        stats['db_size'] = size_result[0]
                except:
                    stats['db_size'] = 'Unknown'
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def health_check(self) -> Dict:
        """Check database health and connection"""
        try:
            with self.get_session() as session:
                # Simple query to test connection
                result = session.execute("SELECT 1 as health_check")
                health_check = result.fetchone()[0]
                
                return {
                    'status': 'healthy' if health_check == 1 else 'unhealthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'connection_pool_size': self.engine.pool.size(),
                    'checked_out_connections': self.engine.pool.checkedout(),
                    'overflow_connections': self.engine.pool.overflow()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Global PostgreSQL database instance
postgres_db = PostgreSQLDatabase()
