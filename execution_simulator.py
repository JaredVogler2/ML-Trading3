# execution_simulator.py
"""
Execution Simulator for Realistic Trade Modeling
Supports both individual and vectorized operations
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation"""
    # Fixed costs
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission: float = 100.0

    # Slippage model
    base_slippage_bps: float = 5  # 5 basis points

    # Market impact model (square-root model)
    impact_coefficient: float = 0.1  # Price impact coefficient
    daily_volume_pct_limit: float = 0.10  # Max 10% of daily volume

    # Order types
    use_limit_orders: bool = True
    limit_offset_bps: float = 10  # Limit order offset

    # Execution probability (for limit orders)
    limit_fill_probability: float = 0.85

    # Time of day factors
    market_open_penalty: float = 1.5  # Higher costs at open
    market_close_penalty: float = 1.3  # Higher costs at close


class ExecutionSimulator:
    """Simulates realistic order execution"""

    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()

    def simulate_entry(self, symbol: str, price: float,
                       quantity: int, daily_volume: float,
                       time_of_day: str = 'mid') -> Tuple[float, float, float]:
        """
        Simulate entry execution

        Returns:
            (execution_price, slippage_cost, commission)
        """
        # Calculate commission
        commission = self._calculate_commission(quantity)

        # Calculate market impact
        volume_fraction = quantity / daily_volume if daily_volume > 0 else 0.01
        market_impact = self._calculate_market_impact(price, volume_fraction)

        # Calculate slippage
        slippage = self._calculate_slippage(price, 'buy', time_of_day)

        # Total execution price
        execution_price = price + slippage + market_impact

        # Total slippage cost
        slippage_cost = (slippage + market_impact) * quantity

        return execution_price, slippage_cost, commission

    def simulate_exit(self, symbol: str, price: float,
                      quantity: int, daily_volume: float,
                      time_of_day: str = 'mid') -> Tuple[float, float, float]:
        """
        Simulate exit execution

        Returns:
            (execution_price, slippage_cost, commission)
        """
        # Calculate commission
        commission = self._calculate_commission(quantity)

        # Calculate market impact
        volume_fraction = quantity / daily_volume if daily_volume > 0 else 0.01
        market_impact = self._calculate_market_impact(price, volume_fraction)

        # Calculate slippage
        slippage = self._calculate_slippage(price, 'sell', time_of_day)

        # Total execution price (subtract for sells)
        execution_price = price - slippage - market_impact

        # Total slippage cost
        slippage_cost = (slippage + market_impact) * quantity

        return execution_price, slippage_cost, commission

    def simulate_vectorized_entries(self, prices: np.ndarray,
                                    quantities: np.ndarray,
                                    volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized entry simulation for multiple symbols

        Args:
            prices: Array of current prices
            quantities: Array of order quantities
            volumes: Array of daily volumes

        Returns:
            Dictionary with execution_prices, slippage_costs, commissions
        """
        n = len(prices)

        # Calculate commissions (vectorized)
        commissions = np.maximum(
            self.config.commission_per_share * quantities,
            self.config.min_commission
        )
        commissions = np.minimum(commissions, self.config.max_commission)

        # Calculate volume fractions
        volume_fractions = np.divide(
            quantities,
            volumes,
            out=np.full(n, 0.01),
            where=volumes > 0
        )

        # Market impact (square-root model)
        market_impacts = prices * self.config.impact_coefficient * np.sqrt(volume_fractions)

        # Base slippage
        slippages = prices * self.config.base_slippage_bps / 10000

        # Execution prices
        execution_prices = prices + slippages + market_impacts

        # Slippage costs
        slippage_costs = (slippages + market_impacts) * quantities

        return {
            'execution_prices': execution_prices,
            'slippage_costs': slippage_costs,
            'commissions': commissions,
            'total_costs': slippage_costs + commissions
        }

    def simulate_vectorized_exits(self, prices: np.ndarray,
                                  quantities: np.ndarray,
                                  volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized exit simulation for multiple symbols

        Returns:
            Dictionary with execution_prices, slippage_costs, commissions
        """
        n = len(prices)

        # Calculate commissions (same as entries)
        commissions = np.maximum(
            self.config.commission_per_share * quantities,
            self.config.min_commission
        )
        commissions = np.minimum(commissions, self.config.max_commission)

        # Calculate volume fractions
        volume_fractions = np.divide(
            quantities,
            volumes,
            out=np.full(n, 0.01),
            where=volumes > 0
        )

        # Market impact
        market_impacts = prices * self.config.impact_coefficient * np.sqrt(volume_fractions)

        # Base slippage
        slippages = prices * self.config.base_slippage_bps / 10000

        # Execution prices (subtract for sells)
        execution_prices = prices - slippages - market_impacts

        # Slippage costs
        slippage_costs = (slippages + market_impacts) * quantities

        return {
            'execution_prices': execution_prices,
            'slippage_costs': slippage_costs,
            'commissions': commissions,
            'total_costs': slippage_costs + commissions
        }

    def _calculate_commission(self, quantity: int) -> float:
        """Calculate commission for given quantity"""
        commission = self.config.commission_per_share * quantity
        commission = max(commission, self.config.min_commission)
        commission = min(commission, self.config.max_commission)
        return commission

    def _calculate_slippage(self, price: float, side: str,
                            time_of_day: str) -> float:
        """Calculate slippage based on order side and time"""
        base_slippage = price * self.config.base_slippage_bps / 10000

        # Time of day adjustment
        if time_of_day == 'open':
            base_slippage *= self.config.market_open_penalty
        elif time_of_day == 'close':
            base_slippage *= self.config.market_close_penalty

        return base_slippage

    def _calculate_market_impact(self, price: float,
                                 volume_fraction: float) -> float:
        """
        Calculate market impact using square-root model
        Impact = price * coefficient * sqrt(volume_fraction)
        """
        # Cap volume fraction
        volume_fraction = min(volume_fraction, self.config.daily_volume_pct_limit)

        # Square-root market impact model
        impact = price * self.config.impact_coefficient * np.sqrt(volume_fraction)

        return impact

    def estimate_transaction_costs(self, trade_value: float,
                                   volume_fraction: float = 0.01) -> Dict[str, float]:
        """
        Estimate total transaction costs for a trade

        Useful for pre-trade analysis
        """
        # Rough estimates
        price = 100  # Normalized price
        quantity = int(trade_value / price)

        # Commission
        commission = self._calculate_commission(quantity)

        # Slippage
        slippage_pct = self.config.base_slippage_bps / 10000
        slippage_cost = trade_value * slippage_pct

        # Market impact
        impact_pct = self.config.impact_coefficient * np.sqrt(volume_fraction)
        impact_cost = trade_value * impact_pct

        # Total
        total_cost = commission + slippage_cost + impact_cost
        total_pct = total_cost / trade_value

        return {
            'commission': commission,
            'slippage_cost': slippage_cost,
            'impact_cost': impact_cost,
            'total_cost': total_cost,
            'total_pct': total_pct,
            'execution_price_adjustment': slippage_pct + impact_pct
        }


class AdvancedExecutionSimulator(ExecutionSimulator):
    """
    Advanced execution simulator with additional features
    """

    def __init__(self, config: ExecutionConfig = None):
        super().__init__(config)
        self.order_book_simulator = OrderBookSimulator()

    def simulate_limit_order(self, symbol: str, price: float,
                             limit_price: float, quantity: int,
                             market_data: pd.DataFrame) -> Dict:
        """
        Simulate limit order execution

        Returns execution details or None if not filled
        """
        # Check if limit order would fill
        if market_data is None or market_data.empty:
            return {'filled': False, 'reason': 'no_market_data'}

        # Get price range for the day
        high = market_data['high'].iloc[-1]
        low = market_data['low'].iloc[-1]
        close = market_data['close'].iloc[-1]

        # Buy limit order
        if limit_price >= low:
            # Order would have filled
            fill_probability = self._calculate_fill_probability(
                limit_price, low, high, close
            )

            if np.random.random() < fill_probability:
                # Simulate partial fills
                fill_pct = np.random.uniform(0.8, 1.0)
                filled_quantity = int(quantity * fill_pct)

                return {
                    'filled': True,
                    'fill_price': limit_price,
                    'filled_quantity': filled_quantity,
                    'remaining_quantity': quantity - filled_quantity,
                    'commission': self._calculate_commission(filled_quantity)
                }

        return {'filled': False, 'reason': 'price_not_reached'}

    def _calculate_fill_probability(self, limit_price: float,
                                    low: float, high: float,
                                    close: float) -> float:
        """Calculate probability of limit order fill"""
        price_range = high - low

        if price_range == 0:
            return 0.5

        # Distance from limit to low (for buy orders)
        distance = (limit_price - low) / price_range

        # Higher probability if limit is closer to the day's range
        base_prob = self.config.limit_fill_probability
        adjusted_prob = base_prob * (1 + distance * 0.2)

        return min(adjusted_prob, 0.95)


class OrderBookSimulator:
    """
    Simulates order book dynamics for more realistic execution
    """

    def __init__(self):
        self.spread_model = self._initialize_spread_model()

    def _initialize_spread_model(self) -> Dict:
        """Initialize spread model parameters"""
        return {
            'base_spread_bps': 10,  # 10 basis points base spread
            'volume_factor': 0.5,  # Spread widens with volume
            'volatility_factor': 2.0  # Spread widens with volatility
        }

    def estimate_spread(self, price: float, volume: float,
                        volatility: float) -> float:
        """Estimate bid-ask spread"""
        base_spread = price * self.spread_model['base_spread_bps'] / 10000

        # Volume adjustment (higher volume = tighter spread)
        volume_adj = 1.0 / (1.0 + np.log1p(volume / 1e6))

        # Volatility adjustment (higher vol = wider spread)
        vol_adj = 1.0 + volatility * self.spread_model['volatility_factor']

        spread = base_spread * volume_adj * vol_adj

        return spread

    def get_bid_ask(self, mid_price: float, volume: float,
                    volatility: float) -> Tuple[float, float]:
        """Get bid and ask prices"""
        spread = self.estimate_spread(mid_price, volume, volatility)

        bid = mid_price - spread / 2
        ask = mid_price + spread / 2

        return bid, ask


# Example usage and testing
if __name__ == "__main__":
    # Test execution simulator
    exec_sim = ExecutionSimulator()

    # Single trade simulation
    price = 100.0
    quantity = 1000
    daily_volume = 1_000_000

    exec_price, slippage, commission = exec_sim.simulate_entry(
        "AAPL", price, quantity, daily_volume
    )

    print("Single Trade Execution:")
    print(f"  Market Price: ${price:.2f}")
    print(f"  Execution Price: ${exec_price:.2f}")
    print(f"  Slippage Cost: ${slippage:.2f}")
    print(f"  Commission: ${commission:.2f}")
    print(f"  Total Cost: ${slippage + commission:.2f}")

    # Vectorized simulation
    n_symbols = 10
    prices = np.random.uniform(50, 200, n_symbols)
    quantities = np.random.randint(100, 1000, n_symbols)
    volumes = np.random.randint(100_000, 10_000_000, n_symbols)

    results = exec_sim.simulate_vectorized_entries(prices, quantities, volumes)

    print("\nVectorized Execution (10 symbols):")
    print(f"  Avg Execution Price Adjustment: {np.mean(results['execution_prices'] - prices):.3f}")
    print(f"  Total Slippage Costs: ${np.sum(results['slippage_costs']):.2f}")
    print(f"  Total Commissions: ${np.sum(results['commissions']):.2f}")

    # Transaction cost estimation
    trade_value = 50000
    est = exec_sim.estimate_transaction_costs(trade_value, volume_fraction=0.02)

    print("\nTransaction Cost Estimate ($50k trade):")
    print(f"  Commission: ${est['commission']:.2f}")
    print(f"  Slippage: ${est['slippage_cost']:.2f}")
    print(f"  Market Impact: ${est['impact_cost']:.2f}")
    print(f"  Total Cost: ${est['total_cost']:.2f} ({est['total_pct'] * 100:.2f}%)")