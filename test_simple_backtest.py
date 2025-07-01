# test_simple_backtest.py
"""Simple test of the ML trading system with minimal configuration"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from models.ensemble_gpu_hedge_fund import HedgeFundGPUEnsemble
from models.enhanced_features import EnhancedFeatureEngineer
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


def fetch_data(symbol, start_date, end_date):
    """Fetch data for a symbol"""
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None

    # Fix column names - handle MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # For single symbol, just take the first level
        df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower()
                      for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    # Ensure timezone-naive index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df['symbol'] = symbol
    return df


def prepare_train_test_split(data, train_end_date):
    """Split data into train and test sets"""
    train_data = {}
    test_data = {}

    train_end = pd.to_datetime(train_end_date)

    for symbol, df in data.items():
        # Ensure timezone naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Split data
        train_df = df[df.index <= train_end]
        test_df = df[df.index > train_end]

        if len(train_df) > 100 and len(test_df) > 20:
            train_data[symbol] = train_df
            test_data[symbol] = test_df

    return train_data, test_data


def main():
    print("Simple ML Trading System Test")
    print("=" * 60)

    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']

    # Dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years
    train_end_date = end_date - timedelta(days=60)  # Last 2 months for testing

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Train/Test split: {train_end_date.date()}")
    print(f"Symbols: {symbols}")

    # Fetch data
    print("\nFetching data...")
    data = {}
    for symbol in symbols:
        try:
            df = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df is not None and len(df) > 0:
                data[symbol] = df
                print(f"  {symbol}: {len(df)} days")
            else:
                print(f"  {symbol}: No data")
        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    if not data:
        print("No data fetched!")
        return

    # Split data
    print("\nSplitting data...")
    train_data, test_data = prepare_train_test_split(data, train_end_date)
    print(f"  Training symbols: {len(train_data)}")
    print(f"  Test symbols: {len(test_data)}")

    if not train_data:
        print("No training data!")
        return

    # Create and train model
    print("\nTraining ensemble model...")
    ensemble = HedgeFundGPUEnsemble()

    # Prepare training data
    try:
        X_train, y_train, info = ensemble.prepare_training_data(train_data)
    except Exception as e:
        print(f"Error preparing training data: {e}")
        return

    if len(X_train) == 0:
        print("No training data prepared!")
        return

    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Positive class rate: {y_train.mean():.2%}")

    # For validation, use last 20% of training data
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_idx]
    y_train_split = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]

    print(f"  Train split: {len(X_train_split)} samples")
    print(f"  Val split: {len(X_val)} samples")

    # Train
    print("\nTraining models...")
    try:
        ensemble.train_combined(X_train_split, y_train_split, X_val, y_val)
    except Exception as e:
        print(f"Error training models: {e}")
        return

    # Validate
    val_score = ensemble.validate(X_val, y_val)
    print(f"\nValidation AUC: {val_score:.4f}")

    # Simple backtest on test data
    print("\nRunning simple backtest on test period...")

    # Initialize portfolio
    cash = 100000
    positions = {}
    trades = []

    # Get all test dates
    all_dates = set()
    for df in test_data.values():
        all_dates.update(df.index)
    test_dates = sorted(all_dates)

    print(f"Test dates: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

    # Feature engineer
    feature_engineer = EnhancedFeatureEngineer()

    # Process each date (last 20 days only for quick test)
    test_dates_subset = test_dates[-20:] if len(test_dates) > 20 else test_dates

    for i, date in enumerate(test_dates_subset):
        if i % 5 == 0:
            print(f"  Processing {date.date()}...")

        # Generate signals
        for symbol, df in test_data.items():
            if symbol in positions or date not in df.index:
                continue

            # Get historical data up to current date
            historical = data[symbol][data[symbol].index <= date]

            if len(historical) < 200:
                continue

            # Create features
            try:
                features = feature_engineer.create_all_features(historical, symbol)
                if features.empty or date not in features.index:
                    continue

                # Get prediction
                current_features = features.loc[[date]]
                prediction = ensemble.predict_proba(current_features)

                if len(prediction) > 0 and prediction[0] > 0.6:  # High confidence
                    # Buy signal
                    price = df.loc[date, 'close']
                    shares = int(10000 / price)  # $10k position

                    if shares > 0 and cash >= shares * price:
                        cash -= shares * price
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': date
                        }
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price
                        })
                        print(f"    {date.date()} BUY {shares} {symbol} @ ${price:.2f}")

            except Exception as e:
                # Silently continue
                pass

        # Check exits (simple 5% stop loss, 10% take profit)
        for symbol in list(positions.keys()):
            if date in test_data[symbol].index:
                current_price = test_data[symbol].loc[date, 'close']
                position = positions[symbol]
                entry_price = position['entry_price']

                # Check stop loss or take profit
                if (current_price < entry_price * 0.95 or
                        current_price > entry_price * 1.10 or
                        (date - position['entry_date']).days > 10):  # Time exit

                    # Sell
                    shares = position['shares']
                    cash += shares * current_price
                    pnl = shares * (current_price - entry_price)

                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares,
                        'price': current_price,
                        'pnl': pnl
                    })

                    del positions[symbol]
                    print(f"    {date.date()} SELL {shares} {symbol} @ ${current_price:.2f} (P&L: ${pnl:.2f})")

    # Calculate final value
    final_value = cash
    for symbol, position in positions.items():
        if test_dates[-1] in test_data[symbol].index:
            final_value += position['shares'] * test_data[symbol].loc[test_dates[-1], 'close']

    # Results
    print(f"\nSimple Backtest Results:")
    print(f"  Initial Capital: $100,000")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Return: {(final_value / 100000 - 1) * 100:.2f}%")
    print(f"  Total Trades: {len(trades)}")

    if trades:
        sells = [t for t in trades if t['action'] == 'SELL']
        if sells:
            total_pnl = sum(t['pnl'] for t in sells)
            wins = [t for t in sells if t['pnl'] > 0]
            print(f"  Closed Trades: {len(sells)}")
            print(f"  Win Rate: {len(wins) / len(sells) * 100:.1f}%")
            print(f"  Total P&L: ${total_pnl:.2f}")

        # Show recent trades
        print(f"\n  Recent trades:")
        for trade in trades[-5:]:
            action = trade['action']
            symbol = trade['symbol']
            shares = trade['shares']
            price = trade['price']
            date = trade['date'].date()
            if action == 'SELL' and 'pnl' in trade:
                print(f"    {date} {action} {shares} {symbol} @ ${price:.2f} (P&L: ${trade['pnl']:.2f})")
            else:
                print(f"    {date} {action} {shares} {symbol} @ ${price:.2f}")


if __name__ == "__main__":
    main()