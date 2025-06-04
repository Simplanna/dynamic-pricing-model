"""Product Age Factor - 15% Weight

Manages age-based pricing with category-specific decay rates and discount ladders.
Includes potency degradation calculations for flower products.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np


class ProductAgeFactor:
    """Manages product age-based pricing adjustments"""
    
    # Age-based discount ladders by category
    AGE_DISCOUNT_LADDERS = {
        'flower': [
            {'days': 0, 'discount': 0.00},
            {'days': 30, 'discount': 0.05},
            {'days': 45, 'discount': 0.10},
            {'days': 60, 'discount': 0.20},
            {'days': 75, 'discount': 0.30},
            {'days': 90, 'discount': 0.40}  # Max discount
        ],
        'edibles': [
            {'days': 0, 'discount': 0.00},
            {'days': 60, 'discount': 0.05},
            {'days': 90, 'discount': 0.10},
            {'days': 120, 'discount': 0.15},
            {'days': 150, 'discount': 0.25},
            {'days': 180, 'discount': 0.35}
        ],
        'concentrates': [
            {'days': 0, 'discount': 0.00},
            {'days': 90, 'discount': 0.05},
            {'days': 180, 'discount': 0.10},
            {'days': 270, 'discount': 0.20},
            {'days': 365, 'discount': 0.30}
        ],
        'prerolls': [
            {'days': 0, 'discount': 0.00},
            {'days': 15, 'discount': 0.05},
            {'days': 30, 'discount': 0.15},
            {'days': 45, 'discount': 0.25},
            {'days': 60, 'discount': 0.40}
        ],
        'vapes': [
            {'days': 0, 'discount': 0.00},
            {'days': 90, 'discount': 0.05},
            {'days': 180, 'discount': 0.10},
            {'days': 270, 'discount': 0.15},
            {'days': 365, 'discount': 0.25}
        ],
        'accessories': [
            {'days': 0, 'discount': 0.00}  # No age discount
        ]
    }
    
    # Potency degradation rates (per month)
    POTENCY_DEGRADATION = {
        'flower': 0.03,        # 3% per month
        'prerolls': 0.04,      # 4% per month (more exposed)
        'edibles': 0.01,       # 1% per month
        'concentrates': 0.005, # 0.5% per month
        'vapes': 0.01,         # 1% per month
        'accessories': 0.0     # No degradation
    }
    
    # Category-specific aging start points
    AGING_START = {
        'flower': 'harvest_date',
        'edibles': 'manufacture_date',
        'concentrates': 'manufacture_date',
        'prerolls': 'manufacture_date',
        'vapes': 'manufacture_date',
        'accessories': None  # No aging
    }
    
    def __init__(self):
        """Initialize product age factor"""
        pass
        
    def calculate_product_age(self, 
                            product_data: Dict,
                            reference_date: Optional[datetime] = None) -> Optional[int]:
        """Calculate product age in days from appropriate start date"""
        category = product_data.get('category', 'flower')
        aging_field = self.AGING_START.get(category)
        
        if not aging_field or aging_field not in product_data:
            # Try fallback dates
            fallback_dates = ['harvest_date', 'manufacture_date', 'package_date', 'received_date']
            for date_field in fallback_dates:
                if date_field in product_data and product_data[date_field]:
                    aging_field = date_field
                    break
            else:
                return None
                
        start_date = product_data.get(aging_field)
        if not start_date:
            return None
            
        # Convert to datetime if string
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            
        # Calculate age
        reference = reference_date or datetime.now()
        age_days = (reference - start_date).days
        
        return max(0, age_days)  # Ensure non-negative
        
    def get_age_discount(self, age_days: int, category: str) -> float:
        """Get discount rate based on product age and category"""
        ladder = self.AGE_DISCOUNT_LADDERS.get(category, self.AGE_DISCOUNT_LADDERS['flower'])
        
        # Find applicable discount
        discount = 0.0
        for step in ladder:
            if age_days >= step['days']:
                discount = step['discount']
            else:
                break
                
        return discount
        
    def calculate_potency_factor(self, 
                               product_data: Dict,
                               age_days: int) -> float:
        """Calculate potency degradation impact on price"""
        category = product_data.get('category', 'flower')
        initial_thc = product_data.get('thc_percentage', 0)
        
        if initial_thc <= 0 or category == 'accessories':
            return 1.0
            
        # Get degradation rate
        monthly_degradation = self.POTENCY_DEGRADATION.get(category, 0.02)
        months_old = age_days / 30
        
        # Calculate degraded potency
        degradation_factor = (1 - monthly_degradation) ** months_old
        current_thc = initial_thc * degradation_factor
        
        # Price adjustment based on potency loss
        # For every 10% potency loss, reduce price by 5%
        potency_loss_pct = (initial_thc - current_thc) / initial_thc
        price_adjustment = 1 - (potency_loss_pct * 0.5)
        
        # Cap the adjustment
        return max(0.8, price_adjustment)  # No more than 20% reduction from potency
        
    def calculate_freshness_premium(self, age_days: int, category: str) -> float:
        """Calculate premium for very fresh products"""
        freshness_windows = {
            'flower': 7,       # Premium for <7 days old
            'prerolls': 3,     # Premium for <3 days old
            'edibles': 14,     # Premium for <14 days old
            'concentrates': 30, # Premium for <30 days old
            'vapes': 30,
            'accessories': None
        }
        
        fresh_window = freshness_windows.get(category)
        if not fresh_window or age_days >= fresh_window:
            return 1.0
            
        # Linear premium decay from 5% to 0%
        freshness_premium = 0.05 * (1 - age_days / fresh_window)
        return 1 + freshness_premium
        
    def calculate_factor_score(self,
                             product_data: Dict,
                             current_price: float) -> Dict:
        """Calculate product age factor score and multiplier"""
        
        category = product_data.get('category', 'flower')
        
        # Calculate product age
        age_days = self.calculate_product_age(product_data)
        
        if age_days is None:
            return self._no_age_data_result()
            
        # Get base age discount
        age_discount = self.get_age_discount(age_days, category)
        
        # Calculate potency degradation factor
        potency_factor = self.calculate_potency_factor(product_data, age_days)
        
        # Calculate freshness premium
        freshness_factor = self.calculate_freshness_premium(age_days, category)
        
        # Combine factors
        # Start with base multiplier of 1.0
        multiplier = 1.0
        
        # Apply age discount
        multiplier *= (1 - age_discount)
        
        # Apply potency factor
        multiplier *= potency_factor
        
        # Apply freshness premium
        multiplier *= freshness_factor
        
        # Calculate confidence
        confidence = self._calculate_confidence(product_data, age_days)
        
        # Determine age status
        age_status = self._get_age_status(age_days, category)
        
        return {
            'factor_name': 'product_age',
            'weight': 0.15,
            'raw_score': age_days,
            'age_status': age_status,
            'multiplier': multiplier,
            'confidence': confidence,
            'details': {
                'age_days': age_days,
                'age_discount': round(age_discount * 100, 1),
                'potency_factor': round(potency_factor, 3),
                'freshness_factor': round(freshness_factor, 3),
                'initial_thc': product_data.get('thc_percentage', 0),
                'category': category
            }
        }
        
    def _get_age_status(self, age_days: int, category: str) -> str:
        """Determine product age status"""
        # Category-specific thresholds
        status_thresholds = {
            'flower': {'fresh': 14, 'normal': 45, 'aging': 75},
            'prerolls': {'fresh': 7, 'normal': 30, 'aging': 45},
            'edibles': {'fresh': 30, 'normal': 90, 'aging': 150},
            'concentrates': {'fresh': 60, 'normal': 180, 'aging': 270},
            'vapes': {'fresh': 60, 'normal': 180, 'aging': 270},
            'accessories': {'fresh': float('inf'), 'normal': float('inf'), 'aging': float('inf')}
        }
        
        thresholds = status_thresholds.get(category, status_thresholds['flower'])
        
        if age_days <= thresholds['fresh']:
            return 'fresh'
        elif age_days <= thresholds['normal']:
            return 'normal'
        elif age_days <= thresholds['aging']:
            return 'aging'
        else:
            return 'old'
            
    def _calculate_confidence(self, product_data: Dict, age_days: int) -> float:
        """Calculate confidence based on data quality"""
        confidence = 1.0
        
        # Reduce confidence if using fallback dates
        if 'harvest_date' not in product_data and 'manufacture_date' not in product_data:
            confidence *= 0.7
            
        # Reduce confidence for very old products (data might be stale)
        if age_days > 180:
            confidence *= 0.8
        elif age_days > 365:
            confidence *= 0.6
            
        # Reduce confidence if no potency data for THC products
        category = product_data.get('category')
        if category in ['flower', 'prerolls', 'concentrates'] and not product_data.get('thc_percentage'):
            confidence *= 0.8
            
        return confidence
        
    def _no_age_data_result(self) -> Dict:
        """Return default result when no age data available"""
        return {
            'factor_name': 'product_age',
            'weight': 0.15,
            'raw_score': 0,
            'age_status': 'unknown',
            'multiplier': 1.0,  # No adjustment
            'confidence': 0.3,  # Low confidence
            'details': {
                'message': 'No age data available for product'
            }
        }
        
    def get_expiration_alert(self, product_data: Dict) -> Optional[Dict]:
        """Check if product is approaching expiration"""
        category = product_data.get('category')
        age_days = self.calculate_product_age(product_data)
        
        if not age_days:
            return None
            
        # Get max age for category
        from .. import PRODUCT_CATEGORIES
        max_age = PRODUCT_CATEGORIES.get(category, {}).get('max_age_days')
        
        if not max_age:
            return None
            
        days_until_expiry = max_age - age_days
        
        if days_until_expiry <= 7:
            return {
                'level': 'critical',
                'message': f'Product expires in {days_until_expiry} days',
                'recommended_action': 'immediate_discount'
            }
        elif days_until_expiry <= 14:
            return {
                'level': 'warning',
                'message': f'Product expires in {days_until_expiry} days',
                'recommended_action': 'promotional_pricing'
            }
        elif days_until_expiry <= 30:
            return {
                'level': 'notice',
                'message': f'Product expires in {days_until_expiry} days',
                'recommended_action': 'monitor_velocity'
            }
            
        return None