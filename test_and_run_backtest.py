# test_and_run_backtest.py
"""
Complete script to test setup and run full watchlist training and backtest
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        # Core modules
        from config.watchlist import WATCHLIST
        print(f"✓ Watchlist loaded: {len(WATCHLIST)} symbols")

        from models.enhanced_features import EnhancedFeatureEngineer
        print("✓ Enhanced features module loaded")

        from models.ensemble_gpu_hedge_fund import HedgeFundGPUEnsemble
        print("✓ Ensemble GPU module loaded")

        from hedge_fund_ml_backtest import HedgeFundBacktester, HedgeFundBacktestConfig
        print("✓ Backtest module loaded")

        from execution_simulator import ExecutionSimulator
        print("✓ Execution simulator loaded")

        # Check for ML libraries
        import xgboost
        print("✓ XGBoost available")

        import lightgbm
        print("✓ LightGBM available")

        try:
            import catboost
            print("✓ CatBoost available")
        except:
            print("⚠ CatBoost not available (optional)")

        import torch
        print(f"✓ PyTorch available - CUDA: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Data libraries
        import yfinance
        print("✓ yfinance available")

        import talib
        print("✓ TA-Lib available")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def quick_data_test():
    """Quick test to fetch data for a few symbols"""
    print("\nTesting data fetching...")

    import yfinance as yf
    from config.watchlist import WATCHLIST

    # Test first 3 symbols
    test_symbols = WATCHLIST[:3]

    for symbol in test_symbols:
        try:
            df = yf.download(symbol,
                             start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                             end=datetime.now().strftime('%Y-%m-%d'),
                             progress=False)
            if not df.empty:
                print(f"✓ {symbol}: {len(df)} days of data")
            else:
                print(f"✗ {symbol}: No data")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")


def run_minimal_backtest():
    """Run a minimal backtest to test the system"""
    print("\nRunning minimal backtest...")

    from hedge_fund_ml_backtest import HedgeFundBacktester, HedgeFundBacktestConfig
    from config.watchlist import WATCHLIST

    # Use minimal config for testing
    config = HedgeFundBacktestConfig(
        initial_capital=100000,
        max_positions=5,  # Small for testing
        min_training_samples=50,
        min_liquidity_usd=500_000
    )

    # Create backtester
    backtester = HedgeFundBacktester(config)

    # Run on subset of symbols for 3 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months

    print(f"Test period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Testing with first 10 symbols: {WATCHLIST[:10]}")

    results = backtester.run_backtest(
        symbols=WATCHLIST[:10],  # Test with 10 symbols
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    if 'error' not in results:
        print(f"\n✓ Minimal backtest successful!")
        print(f"  Total Return: {results['total_return'] * 100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Total Trades: {results['total_trades']}")
        return True
    else:
        print(f"\n✗ Backtest failed: {results['error']}")
        return False


def run_full_watchlist_backtest():
    """Run the full watchlist backtest"""
    print("\n" + "=" * 60)
    print("RUNNING FULL WATCHLIST BACKTEST")
    print("=" * 60)

    from run_hedge_fund_backtest import run_full_backtest

    # This will run the complete backtest with all validations
    results = run_full_backtest()

    if results:
        print("\n✓ Full backtest completed successfully!")

        # Show utilization stats
        if 'symbol_utilization' in results:
            util = results['symbol_utilization']
            print(f"\nWatchlist Utilization:")
            print(f"  Symbols traded: {util.get('symbols_traded', 0)} / {util.get('total_watchlist', 0)}")
            print(f"  Utilization rate: {util.get('utilization_rate', 0):.1f}%")
    else:
        print("\n✗ Full backtest failed!")

    return results


def main():
    """Main function to run everything"""

    print("ML Trading System - Full Watchlist Training and Backtest")
    print("=" * 60)

    # Step 1: Test imports
    if not test_imports():
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return

    # Step 2: Test data fetching
    quick_data_test()

    # Step 3: Ask user what to run
    print("\nOptions:")
    print("1. Run minimal test (10 symbols, 3 months)")
    print("2. Run full watchlist backtest (198 symbols, 1 year)")
    print("3. Run both (test first, then full)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        run_minimal_backtest()
    elif choice == '2':
        run_full_watchlist_backtest()
    elif choice == '3':
        if run_minimal_backtest():
            print("\nMinimal test passed! Proceeding to full backtest...")
            input("Press Enter to continue...")
            run_full_watchlist_backtest()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()