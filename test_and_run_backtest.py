# test_and_run_backtest.py
"""
Complete script to test setup and run full watchlist training and backtest
FIXED: Increased data period to ensure sufficient historical data for feature engineering
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
            # IMPORTANT: Fetch at least 1 year of data for feature engineering
            df = yf.download(symbol,
                             start=(datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d'),  # Extra data for 200-day features
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
    print("\nRunning minimal backtest with professional standards...")

    from hedge_fund_ml_backtest import HedgeFundBacktester, HedgeFundBacktestConfig
    from config.watchlist import WATCHLIST

    # Professional configuration
    config = HedgeFundBacktestConfig(
        initial_capital=100000,
        max_positions=5,  # Small for testing

        # Smaller windows for testing
        train_months=3,  # 3 months training
        validation_months=1,  # 1 month validation
        test_months=1,  # 1 month test
        buffer_days=5,  # 5 day buffer between periods

        # Lower thresholds for testing
        min_training_samples=50,
        min_liquidity_usd=500_000,
        feature_importance_threshold=0.01,
        min_validation_score=0.5,  # Lower for testing
        min_prediction_confidence=0.55  # Lower for testing
    )

    # Create backtester
    backtester = HedgeFundBacktester(config)

    # Use 2 years of data to ensure we have enough for walk-forward
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years

    print(f"Test period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Testing with first 10 symbols: {WATCHLIST[:10]}")
    print("\nNote: Using professional walk-forward optimization")
    print("This ensures no future data leakage\n")

    results = backtester.run_backtest(
        symbols=WATCHLIST[:10],  # Test with 10 symbols
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    if 'error' not in results:
        print(f"\n✓ Minimal backtest successful!")
        print(f"\nResults Summary:")
        print(f"  Total Return: {results.get('total_return', 0) * 100:.2f}%")
        print(f"  Annual Return: {results.get('annual_return', 0) * 100:.2f}%")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0) * 100:.2f}%")
        print(f"  Total Trades: {results.get('total_trades', 0)}")
        print(f"  Win Rate: {results.get('win_rate', 0) * 100:.1f}%")

        # Show data integrity
        if 'detailed_report' in results:
            report = results['detailed_report']
            print(f"\nData Integrity Check:")
            print(f"  {report['summary']['data_integrity']}")
            print(f"  Windows processed: {report['summary']['total_windows']}")
            print(f"  Average validation AUC: {report['summary']['avg_validation_auc']:.4f}")

            # Show window details
            print(f"\nWalk-Forward Windows:")
            for window in report['window_analysis'][:3]:  # Show first 3
                print(f"  Window {window['index'] + 1}:")
                print(f"    Train: {window['train_period']}")
                print(f"    Val: {window['val_period']}")
                print(f"    Test: {window['test_period']}")
                print(f"    Val AUC: {window['validation_auc']:.4f}")

        return True
    else:
        print(f"\n✗ Backtest failed: {results['error']}")

        # Show detailed report if available
        if 'detailed_report' in results:
            report = results['detailed_report']
            print(f"\nData Integrity Checks:")
            for check in report['data_integrity_checks'][:5]:  # Show first 5
                status = "✓" if check['passed'] else "✗"
                print(f"  {status} {check['check']}: {check['details']}")

        return False


def run_full_watchlist_backtest():
    """Run the full watchlist backtest"""
    print("\n" + "=" * 60)
    print("RUNNING FULL WATCHLIST BACKTEST")
    print("=" * 60)

    from hedge_fund_ml_backtest import HedgeFundBacktester, HedgeFundBacktestConfig
    from config.watchlist import WATCHLIST

    # Full configuration with proper settings
    config = HedgeFundBacktestConfig(
        # Capital settings
        initial_capital=100000,
        position_size_method="risk_parity",
        base_position_size=0.02,  # 2% per position
        max_position_size=0.05,  # 5% max
        max_positions=20,  # Reasonable for 200+ symbols
        max_sector_exposure=0.30,  # 30% sector limit

        # Risk management
        stop_loss_atr_multiplier=2.0,
        take_profit_atr_multiplier=4.0,
        max_portfolio_heat=0.06,  # 6% total portfolio risk
        correlation_threshold=0.70,

        # Walk-forward optimization
        train_months=12,  # 1 year training (ensures 250+ days)
        validation_months=3,  # 3 months validation
        test_months=1,  # 1 month test
        buffer_days=5,  # 5 day buffer
        retrain_frequency_days=21,  # Monthly retraining

        # ML thresholds
        min_prediction_confidence=0.65,  # Reasonable confidence
        ensemble_agreement_threshold=0.60,  # Reasonable agreement
        feature_importance_threshold=0.02,  # Capture interaction features

        # Execution realism
        execution_delay_minutes=5,
        use_vwap=True,
        max_spread_bps=20,

        # Performance filters
        min_sharpe_for_trading=0.5,  # More reasonable threshold
        min_training_samples=100,
        min_validation_score=0.55,  # More reasonable

        # Liquidity requirements
        min_liquidity_usd=1_000_000  # $1M daily volume
    )

    # Create backtester
    backtester = HedgeFundBacktester(config)

    # Use full year for proper walk-forward
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years for walk-forward

    print(f"Full backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Testing with full watchlist: {len(WATCHLIST)} symbols")

    # Ensure sufficient data lookback
    if hasattr(backtester, 'data_manager') and hasattr(backtester.data_manager, 'data_lookback_days'):
        backtester.data_manager.data_lookback_days = 300  # Extra buffer

    results = backtester.run_backtest(
        symbols=WATCHLIST,  # Full watchlist
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    if 'error' not in results:
        print("\n✓ Full backtest completed successfully!")

        # Show results
        print(f"\nBacktest Results:")
        print(f"  Total Return: {results['total_return'] * 100:.2f}%")
        print(f"  Annual Return: {results.get('annual_return', 0) * 100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results.get('win_rate', 0) * 100:.1f}%")

        # Show utilization stats
        if 'symbol_utilization' in results:
            util = results['symbol_utilization']
            print(f"\nWatchlist Utilization:")
            print(f"  Symbols traded: {util.get('symbols_traded', 0)} / {util.get('total_watchlist', 0)}")
            print(f"  Utilization rate: {util.get('utilization_rate', 0):.1f}%")

        return results
    else:
        print(f"\n✗ Full backtest failed: {results['error']}")
        return None


def verify_data_availability():
    """Verify that we can fetch sufficient data for all symbols"""
    print("\nVerifying data availability for feature engineering...")

    import yfinance as yf
    from config.watchlist import WATCHLIST

    # Check a sample of symbols
    sample_size = min(20, len(WATCHLIST))
    sample_symbols = WATCHLIST[:sample_size]

    issues = []
    for symbol in sample_symbols:
        try:
            # Need at least 300 days for 200-day features + buffer
            df = yf.download(
                symbol,
                start=(datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d'),
                progress=False
            )

            if len(df) < 250:
                issues.append(f"{symbol}: Only {len(df)} days available")
        except Exception as e:
            issues.append(f"{symbol}: Error - {e}")

    if issues:
        print(f"⚠ Data issues found:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")
    else:
        print(f"✓ All {sample_size} tested symbols have sufficient data")

    return len(issues) == 0


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

    # Step 3: Verify data availability
    if not verify_data_availability():
        print("\n⚠ Some symbols may not have sufficient historical data")
        print("The backtest will skip these symbols automatically")

    # Step 4: Ask user what to run
    print("\nOptions:")
    print("1. Run minimal test (10 symbols, 1 year)")
    print("2. Run full watchlist backtest (198 symbols, 2 years)")
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