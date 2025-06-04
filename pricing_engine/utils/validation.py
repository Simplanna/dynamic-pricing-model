"""Safety Controls and Validation Utilities

Implements comprehensive validation and safety checks for pricing decisions.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np


class PricingValidator:
    """Validates pricing decisions against business rules and safety controls"""
    
    def __init__(self, safety_controls: Dict):
        """Initialize with safety control configuration"""
        self.safety_controls = safety_controls
        self.validation_history = []
        
    def validate_price_change(self, 
                            current_price: float,
                            new_price: float,
                            product_data: Dict) -> Tuple[bool, List[str]]:
        """Validate a proposed price change"""
        violations = []
        
        # Calculate price change
        price_change = (new_price - current_price) / current_price
        
        # Check daily change limit
        max_daily_change = self.safety_controls.get('max_daily_change', 0.05)
        if abs(price_change) > max_daily_change:
            violations.append(
                f"Price change {price_change:.1%} exceeds daily limit of {max_daily_change:.1%}"
            )
            
        # Check minimum margin
        cost = product_data.get('cost', 0)
        if cost > 0:
            margin = (new_price - cost) / new_price
            min_margin = self.safety_controls.get('min_margin', 0.15)
            if margin < min_margin:
                violations.append(
                    f"Margin {margin:.1%} below minimum of {min_margin:.1%}"
                )
                
        # Check maximum discount
        base_price = product_data.get('base_price', current_price)
        discount = (base_price - new_price) / base_price
        max_discount = self.safety_controls.get('max_discount', 0.40)
        if discount > max_discount:
            violations.append(
                f"Discount {discount:.1%} exceeds maximum of {max_discount:.1%}"
            )
            
        # Check absolute bounds
        floor_mult = self.safety_controls.get('price_floor_multiplier', 0.60)
        ceiling_mult = self.safety_controls.get('price_ceiling_multiplier', 1.50)
        
        if new_price < base_price * floor_mult:
            violations.append(
                f"Price below floor of {floor_mult:.0%} of base price"
            )
        elif new_price > base_price * ceiling_mult:
            violations.append(
                f"Price above ceiling of {ceiling_mult:.0%} of base price"
            )
            
        is_valid = len(violations) == 0
        return is_valid, violations
        
    def validate_factor_weights(self, weights: Dict) -> Tuple[bool, List[str]]:
        """Validate that factor weights are properly configured"""
        issues = []
        
        # Check sum equals 1.0
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            issues.append(f"Weights sum to {weight_sum:.3f}, should be 1.000")
            
        # Check individual weights are reasonable
        for factor, weight in weights.items():
            if weight < 0:
                issues.append(f"{factor} has negative weight: {weight}")
            elif weight > 0.5:
                issues.append(f"{factor} has excessive weight: {weight}")
                
        is_valid = len(issues) == 0
        return is_valid, issues
        
    def check_price_stability(self, 
                            price_history: List[Dict],
                            threshold: float = 0.15) -> Dict:
        """Check for price stability issues"""
        if len(price_history) < 2:
            return {'stable': True, 'volatility': 0, 'issues': []}
            
        # Convert to numpy array for calculations
        prices = np.array([p['price'] for p in price_history])
        timestamps = [p['timestamp'] for p in price_history]
        
        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Check for oscillations
        sign_changes = np.sum(np.diff(np.sign(returns)) != 0)
        oscillation_rate = sign_changes / len(returns)
        
        issues = []
        if volatility > threshold:
            issues.append(f"High price volatility: {volatility:.1%}")
            
        if oscillation_rate > 0.5:
            issues.append(f"Price oscillation detected: {oscillation_rate:.1%} direction changes")
            
        # Check for sudden jumps
        max_jump = np.max(np.abs(returns))
        if max_jump > 0.2:
            issues.append(f"Large price jump detected: {max_jump:.1%}")
            
        return {
            'stable': len(issues) == 0,
            'volatility': volatility,
            'oscillation_rate': oscillation_rate,
            'issues': issues
        }
        
    def validate_market_conditions(self, market_data: Dict) -> Tuple[bool, List[str]]:
        """Validate market data quality and conditions"""
        issues = []
        
        # Check data freshness
        if 'last_updated' in market_data:
            last_update = market_data['last_updated']
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update)
            age_hours = (datetime.now() - last_update).total_seconds() / 3600
            
            if age_hours > 24:
                issues.append(f"Market data is {age_hours:.1f} hours old")
                
        # Check competitor data
        competitors = market_data.get('competitors', [])
        if len(competitors) < 3:
            issues.append(f"Limited competitor data: only {len(competitors)} competitors")
            
        # Check sales history
        sales_history = market_data.get('sales_history', [])
        if len(sales_history) < 7:
            issues.append(f"Limited sales history: only {len(sales_history)} days")
            
        # Check inventory data
        inventory = market_data.get('inventory', {})
        if inventory.get('quantity', 0) < 0:
            issues.append("Invalid inventory quantity")
            
        is_valid = len(issues) == 0
        return is_valid, issues
        
    def suggest_price_adjustment(self,
                               current_price: float,
                               recommended_price: float,
                               violations: List[str]) -> float:
        """Suggest a compliant price adjustment"""
        if not violations:
            return recommended_price
            
        # Apply constraints to find compliant price
        adjusted_price = recommended_price
        
        # Apply daily change cap
        max_change = self.safety_controls.get('max_daily_change', 0.05)
        price_change = (recommended_price - current_price) / current_price
        
        if abs(price_change) > max_change:
            if price_change > 0:
                adjusted_price = current_price * (1 + max_change)
            else:
                adjusted_price = current_price * (1 - max_change)
                
        return adjusted_price


class ComplianceChecker:
    """Ensures pricing complies with state regulations"""
    
    # State-specific rules
    STATE_RULES = {
        'MA': {
            'max_thc_flower': 30.0,  # % THC limit
            'max_edible_package': 100,  # mg THC
            'max_concentrate_package': 5000,  # mg THC
            'required_taxes': ['excise', 'state', 'local'],
            'min_price_per_gram': 5.0  # Minimum to discourage diversion
        },
        'RI': {
            'max_thc_flower': 28.0,
            'max_edible_package': 100,
            'max_concentrate_package': 3000,
            'required_taxes': ['excise', 'state'],
            'min_price_per_gram': 4.0
        }
    }
    
    def check_compliance(self, 
                        product_data: Dict,
                        price: float,
                        state: str = 'MA') -> Tuple[bool, List[str]]:
        """Check pricing compliance with state regulations"""
        violations = []
        rules = self.STATE_RULES.get(state, self.STATE_RULES['MA'])
        
        category = product_data.get('category', 'flower')
        
        # Check minimum pricing (anti-diversion)
        if category == 'flower':
            size_grams = product_data.get('size', 1.0)
            if product_data.get('size_unit') == 'oz':
                size_grams *= 28.0
                
            price_per_gram = price / size_grams
            min_price = rules['min_price_per_gram']
            
            if price_per_gram < min_price:
                violations.append(
                    f"Price per gram ${price_per_gram:.2f} below minimum ${min_price:.2f}"
                )
                
        # Check THC limits
        thc_content = product_data.get('thc_percentage', 0)
        
        if category == 'flower' and thc_content > rules['max_thc_flower']:
            violations.append(
                f"THC content {thc_content}% exceeds limit of {rules['max_thc_flower']}%"
            )
            
        # Verify tax inclusion
        if not product_data.get('taxes_included', False):
            violations.append("Prices must include all applicable taxes")
            
        is_compliant = len(violations) == 0
        return is_compliant, violations
        
    def calculate_tax_inclusive_price(self,
                                    base_price: float,
                                    state: str = 'MA') -> Dict:
        """Calculate tax-inclusive pricing"""
        tax_rates = {
            'MA': {
                'excise': 0.1075,  # 10.75% excise
                'state': 0.0625,   # 6.25% state sales
                'local': 0.03      # Up to 3% local
            },
            'RI': {
                'excise': 0.10,    # 10% excise
                'state': 0.07,     # 7% state sales
                'local': 0.0       # No local cannabis tax
            }
        }
        
        rates = tax_rates.get(state, tax_rates['MA'])
        
        # Calculate cumulative tax
        total_tax_rate = sum(rates.values())
        tax_amount = base_price * total_tax_rate
        total_price = base_price + tax_amount
        
        return {
            'base_price': base_price,
            'tax_amount': tax_amount,
            'total_price': total_price,
            'tax_rate': total_tax_rate,
            'tax_breakdown': rates
        }