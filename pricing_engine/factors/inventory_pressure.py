"""Inventory Pressure Factor - 25% Weight

Calculates pricing adjustments based on inventory levels and days-on-hand.
Uses category-specific perishability and velocity bands.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class InventoryPressureFactor:
    """Manages inventory-based pricing adjustments"""
    
    # Inventory pressure bands and multipliers
    PRESSURE_BANDS = {
        'critical_low': {'days_on_hand': (0, 3), 'multiplier': 1.15},
        'low': {'days_on_hand': (3, 7), 'multiplier': 1.08},
        'target': {'days_on_hand': (7, 21), 'multiplier': 1.00},
        'high': {'days_on_hand': (21, 45), 'multiplier': 0.92},
        'critical_high': {'days_on_hand': (45, float('inf')), 'multiplier': 0.85}
    }
    
    # Category-specific adjustments
    CATEGORY_ADJUSTMENTS = {
        'flower': {'velocity_weight': 0.7, 'age_weight': 0.3},
        'edibles': {'velocity_weight': 0.6, 'age_weight': 0.4},
        'concentrates': {'velocity_weight': 0.5, 'age_weight': 0.5},
        'prerolls': {'velocity_weight': 0.8, 'age_weight': 0.2},
        'vapes': {'velocity_weight': 0.6, 'age_weight': 0.4},
        'accessories': {'velocity_weight': 1.0, 'age_weight': 0.0}  # Velocity only
    }
    
    def __init__(self, perishability_days: Dict[str, Optional[int]]):
        """Initialize with category-specific perishability data"""
        self.perishability_days = perishability_days
        
    def calculate_days_on_hand(self, 
                             inventory_qty: float, 
                             daily_velocity: float,
                             min_velocity: float = 0.1) -> float:
        """Calculate days-on-hand based on current inventory and velocity"""
        # Prevent division by zero
        velocity = max(daily_velocity, min_velocity)
        return inventory_qty / velocity
        
    def get_pressure_band(self, days_on_hand: float) -> Tuple[str, float]:
        """Determine pressure band and multiplier based on days-on-hand"""
        for band_name, band_config in self.PRESSURE_BANDS.items():
            min_days, max_days = band_config['days_on_hand']
            if min_days <= days_on_hand < max_days:
                return band_name, band_config['multiplier']
        return 'critical_high', self.PRESSURE_BANDS['critical_high']['multiplier']
        
    def calculate_perishability_factor(self, 
                                     category: str,
                                     days_until_expiry: Optional[float]) -> float:
        """Calculate additional pressure for perishable items"""
        if category == 'accessories' or days_until_expiry is None:
            return 1.0
            
        perishability = self.perishability_days.get(category)
        if not perishability:
            return 1.0
            
        # Progressive discount as expiry approaches
        if days_until_expiry <= 7:
            return 0.75  # 25% additional discount
        elif days_until_expiry <= 14:
            return 0.85  # 15% additional discount
        elif days_until_expiry <= 21:
            return 0.92  # 8% additional discount
        elif days_until_expiry <= perishability * 0.5:
            return 0.96  # 4% additional discount
        return 1.0
        
    def calculate_factor_score(self,
                             product_data: Dict,
                             inventory_data: Dict,
                             sales_velocity: float) -> Dict:
        """Calculate inventory pressure factor score and multiplier"""
        
        category = product_data.get('category', 'flower')
        inventory_qty = inventory_data.get('quantity', 0)
        
        # Calculate days on hand
        days_on_hand = self.calculate_days_on_hand(inventory_qty, sales_velocity)
        
        # Get base pressure band
        band_name, base_multiplier = self.get_pressure_band(days_on_hand)
        
        # Calculate perishability factor
        days_until_expiry = inventory_data.get('days_until_expiry')
        perishability_multiplier = self.calculate_perishability_factor(
            category, days_until_expiry
        )
        
        # Combine multipliers
        final_multiplier = base_multiplier * perishability_multiplier
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(inventory_data, sales_velocity)
        
        return {
            'factor_name': 'inventory_pressure',
            'weight': 0.25,
            'raw_score': days_on_hand,
            'band': band_name,
            'multiplier': final_multiplier,
            'confidence': confidence,
            'details': {
                'days_on_hand': round(days_on_hand, 1),
                'inventory_qty': inventory_qty,
                'daily_velocity': round(sales_velocity, 2),
                'base_multiplier': base_multiplier,
                'perishability_multiplier': perishability_multiplier,
                'days_until_expiry': days_until_expiry
            }
        }
        
    def _calculate_confidence(self, inventory_data: Dict, sales_velocity: float) -> float:
        """Calculate confidence score based on data quality"""
        confidence = 1.0
        
        # Reduce confidence for missing or suspect data
        if inventory_data.get('quantity', 0) < 0:
            confidence *= 0.5
            
        if sales_velocity <= 0:
            confidence *= 0.7
            
        if inventory_data.get('last_updated'):
            last_update = inventory_data['last_updated']
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update)
            hours_old = (datetime.now() - last_update).total_seconds() / 3600
            if hours_old > 24:
                confidence *= 0.8
            elif hours_old > 48:
                confidence *= 0.6
                
        return confidence
        
    def get_recommended_actions(self, factor_result: Dict) -> list:
        """Generate recommended actions based on inventory pressure"""
        actions = []
        band = factor_result['band']
        details = factor_result['details']
        
        if band == 'critical_low':
            actions.append({
                'type': 'urgent',
                'action': 'increase_price',
                'reason': f"Critical low inventory: {details['days_on_hand']:.1f} days on hand",
                'suggested_change': '+10-15%'
            })
            actions.append({
                'type': 'operational',
                'action': 'reorder',
                'reason': 'Inventory below critical threshold'
            })
            
        elif band == 'critical_high':
            actions.append({
                'type': 'urgent',
                'action': 'decrease_price',
                'reason': f"Excess inventory: {details['days_on_hand']:.1f} days on hand",
                'suggested_change': '-10-15%'
            })
            
            if details.get('days_until_expiry') and details['days_until_expiry'] < 14:
                actions.append({
                    'type': 'urgent',
                    'action': 'promotional_discount',
                    'reason': f"Product expires in {details['days_until_expiry']} days"
                })
                
        return actions