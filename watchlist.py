# config/watchlist.py

"""
198 high-liquidity stocks for the trading system
Selected based on:
- Market cap > $10B
- Average daily volume > 1M shares
- Listed on NYSE or NASDAQ
- Marginable on Alpaca
"""

WATCHLIST = [
    # Technology (40 stocks)
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE', 'CRM',
    'CSCO', 'ACN', 'TXN', 'QCOM', 'IBM', 'INTU', 'AMD', 'AMAT', 'MU', 'INTC',
    'NOW', 'UBER', 'SHOP', 'SQ', 'PLTR', 'SNOW', 'NET', 'CRWD', 'PANW', 'DDOG',
    'ZM', 'DOCU', 'TWLO', 'OKTA', 'ZS', 'TEAM', 'ABNB', 'COIN', 'RBLX', 'HOOD',

    # Healthcare (30 stocks)
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY', 'MRK', 'DHR', 'BMY',
    'AMGN', 'GILD', 'CVS', 'MDT', 'ISRG', 'VRTX', 'REGN', 'ZTS', 'BSX', 'EW',
    'BIIB', 'ILMN', 'HCA', 'CI', 'HUM', 'MRNA', 'A', 'IQV', 'DXCM', 'ALGN',

    # Financial (30 stocks)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC',
    'TFC', 'C', 'COF', 'SPGI', 'CME', 'ICE', 'MCO', 'MSCI', 'TRV', 'AIG',
    'MET', 'PRU', 'AFL', 'ALL', 'TROW', 'NTRS', 'FITB', 'KEY', 'RF', 'CFG',

    # Consumer Discretionary (25 stocks)
    'AMZN', 'HD', 'NKE', 'MCD', 'LOW', 'SBUX', 'TGT', 'BKNG', 'TJX', 'ORLY',
    'CMG', 'MAR', 'GM', 'F', 'AZO', 'YUM', 'DG', 'DLTR', 'ROST', 'BBY',
    'LVS', 'WYNN', 'MGM', 'CZR', 'DKNG',

    # Consumer Staples (20 stocks)
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'MDLZ', 'CL', 'GIS',
    'KMB', 'SYY', 'KHC', 'STZ', 'KDP', 'MNST', 'HSY', 'CAG', 'K', 'CPB',

    # Industrials (20 stocks)
    'BA', 'CAT', 'RTX', 'HON', 'UNP', 'LMT', 'DE', 'UPS', 'GE', 'MMM',
    'CSX', 'NSC', 'FDX', 'ETN', 'EMR', 'ITW', 'PH', 'GD', 'NOC', 'TXT',

    # Energy (15 stocks)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
    'DVN', 'HAL', 'BKR', 'FANG', 'HES',

    # Materials & Real Estate (10 stocks)
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'PLD', 'AMT', 'CCI',

    # Utilities & Communications (8 stocks)
    'NEE', 'SO', 'DUK', 'D', 'T', 'VZ', 'CMCSA', 'DIS'
]

# Sector mapping for diversification
SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'ABBV': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',

    # Financial
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
    'GS': 'Financial', 'MS': 'Financial', 'BLK': 'Financial',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples',

    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'RTX': 'Industrials',
    'HON': 'Industrials', 'UNP': 'Industrials',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'SLB': 'Energy', 'EOG': 'Energy',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',

    # Real Estate
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate',

    # Utilities
    'NEE': 'Utilities', 'SO': 'Utilities', 'DUK': 'Utilities',

    # Communications
    'T': 'Communications', 'VZ': 'Communications', 'CMCSA': 'Communications'
}

# High correlation pairs to avoid simultaneous positions
CORRELATION_PAIRS = [
    ('GOOGL', 'GOOG'),  # Same company different shares
    ('META', 'GOOGL'),  # Both ad-driven tech
    ('BAC', 'JPM'),  # Major banks
    ('XOM', 'CVX'),  # Oil majors
    ('HD', 'LOW'),  # Home improvement
    ('UPS', 'FDX'),  # Shipping
    ('V', 'MA'),  # Payment processors
    ('KO', 'PEP'),  # Beverages
]


def get_sector(symbol):
    """Get sector for a symbol"""
    return SECTOR_MAPPING.get(symbol, 'Unknown')


def validate_watchlist():
    """Validate watchlist integrity"""
    assert len(WATCHLIST) == 198, f"Watchlist should have 198 stocks, found {len(WATCHLIST)}"
    assert len(WATCHLIST) == len(set(WATCHLIST)), "Duplicate symbols in watchlist"
    return True


# Validate on import
validate_watchlist()