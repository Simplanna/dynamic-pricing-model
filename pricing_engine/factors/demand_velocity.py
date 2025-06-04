"""Demand Velocity Factor - 20% Weight

Calculates pricing adjustments based on sales velocity and demand patterns.
Uses 7-day rolling averages and price elasticity of -0.45 to -0.5.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class DemandVelocityFactor:
    """Manages demand-based pricing adjustments"""
    
    # Velocity bands (units per day)
    VELOCITY_BANDS = {
        'very_high': {'range': (10, float('inf')), 'multiplier': 1.12},
        'high': {'range': (5, 10), 'multiplier': 1.06},
        'normal': {'range': (1, 5), 'multiplier': 1.00},
        'low': {'range': (0.5, 1), 'multiplier': 0.94},
        'very_low': {'range': (0, 0.5), 'multiplier': 0.88}
    }
    
    # Category-specific thresholds (adjust bands by category)
    CATEGORY_MULTIPLIERS = {
        'flower': 1.2,      # Higher volume category
        'edibles': 0.8,     # Lower volume, higher margin
        'concentrates': 0.6,  # Premium, lower volume
        'prerolls': 1.5,    # High velocity impulse buys
        'vapes': 0.9,       # Moderate velocity
        'accessories': 0.4   # Low velocity, high margin
    }
    
    def __init__(self, elasticity_range: tuple = (-0.45, -0.5)):
        """Initialize with price elasticity range"""
        self.min_elasticity, self.max_elasticity = elasticity_range
        
    def calculate_rolling_velocity(self, 
                                 sales_history: List[Dict],
                                 days: int = 7) -> Dict:
        """Calculate rolling average velocity over specified days"""
        if not sales_history:
            return {'velocity': 0, 'trend': 0, 'volatility': 0}
            
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(sales_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate daily sales
        daily_sales = df.groupby(df['date'].dt.date)['quantity'].sum()
        
        # Fill missing dates with 0
        date_range = pd.date_range(
            start=daily_sales.index.min(),
            end=daily_sales.index.max(),
            freq='D'
        )
        daily_sales = daily_sales.reindex(date_range.date, fill_value=0)
        
        # Calculate metrics
        current_velocity = daily_sales.tail(days).mean()
        previous_velocity = daily_sales.tail(days * 2).head(days).mean()
        
        # Calculate trend
        if previous_velocity > 0:
            trend = (current_velocity - previous_velocity) / previous_velocity
        else:
            trend = 0
            
        # Calculate volatility (coefficient of variation)
        if current_velocity > 0:
            volatility = daily_sales.tail(days).std() / current_velocity
        else:
            volatility = 0
            
        return {
            'velocity': float(current_velocity),
            'trend': float(trend),
            'volatility': float(volatility)
        }
        
    def get_velocity_band(self, velocity: float, category: str) -> tuple:
        """Determine velocity band with category adjustment"""
        # Adjust velocity by category
        category_multiplier = self.CATEGORY_MULTIPLIERS.get(category, 1.0)
        adjusted_velocity = velocity / category_multiplier
        
        for band_name, band_config in self.VELOCITY_BANDS.items():
            min_vel, max_vel = band_config['range']
            if min_vel <= adjusted_velocity < max_vel:
                return band_name, band_config['multiplier']
                
        return 'very_low', self.VELOCITY_BANDS['very_low']['multiplier']
        
    def calculate_elasticity_adjustment(self, 
                                      velocity_trend: float,
                                      category: str) -> float:
        """Apply price elasticity to velocity trends"""
        # Get category-specific elasticity
        from ..import PRODUCT_CATEGORIES
        elasticity = PRODUCT_CATEGORIES.get(category, {}).get('elasticity', -0.45)
        
        # If velocity is increasing, we can afford to raise prices slightly
        # If velocity is decreasing, we should lower prices
        # The adjustment is dampened by elasticity
        
        if velocity_trend > 0:
            # Positive trend: can increase price
            # But be conservative - use lower elasticity magnitude
            adjustment = 1 + (velocity_trend * abs(self.min_elasticity) * 0.5)
        else:
            # Negative trend: should decrease price
            # Be more aggressive - use higher elasticity magnitude
            adjustment = 1 + (velocity_trend * abs(self.max_elasticity))
            
        # Cap adjustments
        return np.clip(adjustment, 0.9, 1.1)
        
    def calculate_factor_score(self,
                             product_data: Dict,
                             sales_history: List[Dict]) -> Dict:
        """Calculate demand velocity factor score and multiplier"""
        
        category = product_data.get('category', 'flower')
        
        # Calculate velocity metrics
        velocity_data = self.calculate_rolling_velocity(sales_history)
        current_velocity = velocity_data['velocity']
        velocity_trend = velocity_data['trend']
        volatility = velocity_data['volatility']
        
        # Get velocity band
        band_name, base_multiplier = self.get_velocity_band(current_velocity, category)
        
        # Apply elasticity adjustment
        elasticity_multiplier = self.calculate_elasticity_adjustment(
            velocity_trend, category
        )
        
        # Adjust for volatility (high volatility = be conservative)
        if volatility > 0.5:
            volatility_dampener = 0.95
        elif volatility > 0.3:
            volatility_dampener = 0.98
        else:
            volatility_dampener = 1.0
            
        # Combine multipliers
        final_multiplier = base_multiplier * elasticity_multiplier * volatility_dampener
        
        # Calculate confidence
        confidence = self._calculate_confidence(sales_history, volatility)
        
        return {
            'factor_name': 'demand_velocity',
            'weight': 0.20,
            'raw_score': current_velocity,
            'band': band_name,
            'multiplier': final_multiplier,
            'confidence': confidence,
            'details': {
                'daily_velocity': round(current_velocity, 2),
                'velocity_trend': round(velocity_trend * 100, 1),  # As percentage
                'volatility': round(volatility, 2),
                'base_multiplier': base_multiplier,
                'elasticity_multiplier': round(elasticity_multiplier, 3),
                'days_analyzed': len(sales_history) if sales_history else 0
            }
        }
        
    def _calculate_confidence(self, sales_history: List[Dict], volatility: float) -> float:
        """Calculate confidence based on data quality and consistency"""
        confidence = 1.0
        
        # Reduce confidence for limited data
        if not sales_history:
            return 0.3
            
        days_of_data = len(set(sale['date'] for sale in sales_history))
        if days_of_data < 7:
            confidence *= (days_of_data / 7)
        elif days_of_data < 3:
            confidence *= 0.5
            
        # Reduce confidence for high volatility
        if volatility > 0.5:
            confidence *= 0.8
        elif volatility > 0.8:
            confidence *= 0.6
            
        return confidence
        
    def get_seasonal_adjustment(self, date: datetime, category: str) -> float:
        """Apply seasonal adjustments to demand patterns"""
        month = date.month
        
        # Cannabis-specific seasonal patterns
        seasonal_factors = {
            'flower': {
                4: 1.15,   # 4/20 month
                7: 1.05,   # Summer
                8: 1.05,
                12: 1.10,  # Holidays
            },
            'edibles': {
                4: 1.20,   # 4/20 month
                10: 1.10,  # Halloween
                12: 1.15,  # Holidays
            },
            'vapes': {
                4: 1.10,
                6: 1.05,   # Summer travel
                7: 1.05,
                8: 1.05,
            }
        }
        
        category_seasons = seasonal_factors.get(category, {})
        return category_seasons.get(month, 1.0)
        
    def predict_future_demand(self, 
                            sales_history: List[Dict],
                            days_ahead: int = 7) -> Dict:
        """Predict future demand based on historical patterns"""
        if not sales_history or len(sales_history) < 14:
            return {'predicted_velocity': 0, 'confidence': 0}
            
        # Simple moving average prediction
        current_metrics = self.calculate_rolling_velocity(sales_history, days=7)
        trend_metrics = self.calculate_rolling_velocity(sales_history, days=14)
        
        # Project forward based on trend
        trend_rate = current_metrics['trend']
        current_velocity = current_metrics['velocity']
        
        predicted_velocity = current_velocity * (1 + trend_rate * (days_ahead / 7))
        
        # Confidence decreases with prediction distance
        confidence = 0.9 * (1 - days_ahead / 30)
        
        return {
            'predicted_velocity': max(0, predicted_velocity),
            'confidence': max(0.3, confidence)
        }