"""Minor Factors - Brand Equity (5%), Potency/Size (5%), Store Location (3%), Customer Segment (2%)

Implements the remaining pricing factors with lower weights but important adjustments.
"""

from typing import Dict, List, Optional
import numpy as np


class BrandEquityFactor:
    """Brand Equity Factor - 5% Weight"""
    
    # Premium brand indicators
    PREMIUM_BRANDS = [
        'cookies', 'raw garden', 'stiiizy', 'connected', 'alien labs',
        'jungle boys', 'wonderbrett', 'fig farms', 'cbx', 'papa & barkley'
    ]
    
    # Brand tier multipliers
    BRAND_TIERS = {
        'premium': 1.15,      # 15% premium
        'established': 1.05,  # 5% premium
        'standard': 1.00,     # No adjustment
        'value': 0.95,        # 5% discount
        'house': 0.90         # 10% discount
    }
    
    def calculate_factor_score(self, product_data: Dict) -> Dict:
        """Calculate brand equity factor score"""
        brand = product_data.get('brand', '').lower()
        brand_tier = product_data.get('brand_tier', 'standard')
        
        # Check if premium brand
        if any(premium in brand for premium in self.PREMIUM_BRANDS):
            brand_tier = 'premium'
            
        # Get multiplier
        multiplier = self.BRAND_TIERS.get(brand_tier, 1.0)
        
        # Higher confidence for known brands
        confidence = 0.9 if brand_tier != 'standard' else 0.7
        
        return {
            'factor_name': 'brand_equity',
            'weight': 0.05,
            'raw_score': list(self.BRAND_TIERS.keys()).index(brand_tier),
            'brand_tier': brand_tier,
            'multiplier': multiplier,
            'confidence': confidence,
            'details': {
                'brand': brand,
                'tier': brand_tier,
                'is_premium': brand_tier == 'premium'
            }
        }


class PotencySizeFactor:
    """Potency/Size Factor - 5% Weight"""
    
    # THC potency tiers
    THC_TIERS = {
        'ultra_high': {'range': (30, 100), 'multiplier': 1.20},
        'very_high': {'range': (25, 30), 'multiplier': 1.10},
        'high': {'range': (20, 25), 'multiplier': 1.05},
        'medium': {'range': (15, 20), 'multiplier': 1.00},
        'low': {'range': (10, 15), 'multiplier': 0.95},
        'very_low': {'range': (0, 10), 'multiplier': 0.90}
    }
    
    # Size-based discounts (bulk pricing)
    SIZE_DISCOUNTS = {
        'flower': {
            1.0: 1.00,   # 1g - no discount
            3.5: 0.98,   # Eighth - 2% discount
            7.0: 0.95,   # Quarter - 5% discount
            14.0: 0.92,  # Half - 8% discount
            28.0: 0.88   # Ounce - 12% discount
        },
        'edibles': {
            10: 1.00,    # Single dose
            50: 0.98,    # 5-pack
            100: 0.95,   # 10-pack
            200: 0.92    # 20-pack
        }
    }
    
    def calculate_factor_score(self, product_data: Dict) -> Dict:
        """Calculate potency/size factor score"""
        category = product_data.get('category', 'flower')
        thc_percentage = product_data.get('thc_percentage', 0)
        size = product_data.get('size', 0)
        size_unit = product_data.get('size_unit', 'g')
        
        # Calculate potency multiplier
        potency_mult = self._get_potency_multiplier(thc_percentage, category)
        
        # Calculate size multiplier
        size_mult = self._get_size_multiplier(size, category, size_unit)
        
        # Combine multipliers (average)
        final_multiplier = (potency_mult + size_mult) / 2
        
        # Confidence based on data availability
        confidence = 0.9 if thc_percentage > 0 and size > 0 else 0.6
        
        return {
            'factor_name': 'potency_size',
            'weight': 0.05,
            'raw_score': thc_percentage,
            'multiplier': final_multiplier,
            'confidence': confidence,
            'details': {
                'thc_percentage': thc_percentage,
                'size': size,
                'size_unit': size_unit,
                'potency_multiplier': round(potency_mult, 3),
                'size_multiplier': round(size_mult, 3)
            }
        }
        
    def _get_potency_multiplier(self, thc_pct: float, category: str) -> float:
        """Get multiplier based on THC potency"""
        if category == 'accessories' or thc_pct <= 0:
            return 1.0
            
        for tier, config in self.THC_TIERS.items():
            min_thc, max_thc = config['range']
            if min_thc <= thc_pct < max_thc:
                return config['multiplier']
                
        return 1.0
        
    def _get_size_multiplier(self, size: float, category: str, unit: str) -> float:
        """Get bulk discount multiplier based on size"""
        size_discounts = self.SIZE_DISCOUNTS.get(category, {})
        
        if not size_discounts:
            return 1.0
            
        # Convert to standard units if needed
        if unit == 'oz' and category == 'flower':
            size = size * 28.0  # Convert to grams
            
        # Find applicable discount
        applicable_discount = 1.0
        for threshold_size, discount in sorted(size_discounts.items()):
            if size >= threshold_size:
                applicable_discount = discount
                
        return applicable_discount


class StoreLocationFactor:
    """Store Location Factor - 3% Weight"""
    
    # Location-based adjustments
    LOCATION_MULTIPLIERS = {
        'downtown': 1.10,      # High rent, tourist area
        'suburban': 1.00,      # Standard pricing
        'rural': 0.95,         # Lower overhead
        'border': 1.05,        # Near state lines
        'campus': 0.98,        # Price-sensitive students
        'medical': 0.97        # Medical discount zones
    }
    
    # State-specific adjustments
    STATE_ADJUSTMENTS = {
        'MA': {
            'boston': 1.08,
            'cambridge': 1.06,
            'worcester': 1.00,
            'springfield': 0.98
        },
        'RI': {
            'providence': 1.05,
            'newport': 1.08,
            'warwick': 1.00,
            'pawtucket': 0.98
        }
    }
    
    def calculate_factor_score(self, 
                             product_data: Dict,
                             store_data: Dict) -> Dict:
        """Calculate store location factor score"""
        location_type = store_data.get('location_type', 'suburban')
        city = store_data.get('city', '').lower()
        state = store_data.get('state', 'MA')
        
        # Get base location multiplier
        base_mult = self.LOCATION_MULTIPLIERS.get(location_type, 1.0)
        
        # Apply city-specific adjustment
        city_adjustments = self.STATE_ADJUSTMENTS.get(state, {})
        city_mult = city_adjustments.get(city, 1.0)
        
        # Combine multipliers
        final_multiplier = base_mult * city_mult
        
        return {
            'factor_name': 'store_location',
            'weight': 0.03,
            'raw_score': list(self.LOCATION_MULTIPLIERS.keys()).index(location_type) if location_type in self.LOCATION_MULTIPLIERS else 0,
            'location_type': location_type,
            'multiplier': final_multiplier,
            'confidence': 0.85,
            'details': {
                'city': city,
                'state': state,
                'location_type': location_type,
                'base_multiplier': base_mult,
                'city_multiplier': city_mult
            }
        }


class CustomerSegmentFactor:
    """Customer Segment Factor - 2% Weight"""
    
    # Segment-based adjustments
    SEGMENT_MULTIPLIERS = {
        'premium': 1.08,       # Price insensitive
        'regular': 1.00,       # Standard pricing
        'value': 0.95,         # Price sensitive
        'medical': 0.93,       # Medical discount
        'senior': 0.95,        # Senior discount
        'veteran': 0.95,       # Veteran discount
        'employee': 0.90,      # Employee discount
        'wholesale': 0.85      # B2B pricing
    }
    
    # Product preferences by segment
    SEGMENT_PREFERENCES = {
        'premium': ['concentrates', 'vapes'],
        'regular': ['flower', 'prerolls'],
        'value': ['flower', 'edibles'],
        'medical': ['edibles', 'vapes'],
        'senior': ['edibles', 'topicals'],
        'veteran': ['flower', 'prerolls']
    }
    
    def calculate_factor_score(self,
                             product_data: Dict,
                             segment: str = 'regular',
                             segment_mix: Optional[Dict] = None) -> Dict:
        """Calculate customer segment factor score"""
        category = product_data.get('category', 'flower')
        
        if segment_mix:
            # Weighted average of segment multipliers
            weighted_mult = 0
            total_weight = 0
            
            for seg, weight in segment_mix.items():
                mult = self.SEGMENT_MULTIPLIERS.get(seg, 1.0)
                
                # Boost if product matches segment preference
                if category in self.SEGMENT_PREFERENCES.get(seg, []):
                    mult *= 1.02
                    
                weighted_mult += mult * weight
                total_weight += weight
                
            final_multiplier = weighted_mult / total_weight if total_weight > 0 else 1.0
            primary_segment = max(segment_mix.items(), key=lambda x: x[1])[0]
        else:
            # Single segment
            final_multiplier = self.SEGMENT_MULTIPLIERS.get(segment, 1.0)
            
            # Boost if product matches segment preference
            if category in self.SEGMENT_PREFERENCES.get(segment, []):
                final_multiplier *= 1.02
                
            primary_segment = segment
            
        return {
            'factor_name': 'customer_segment',
            'weight': 0.02,
            'raw_score': list(self.SEGMENT_MULTIPLIERS.keys()).index(primary_segment) if primary_segment in self.SEGMENT_MULTIPLIERS else 0,
            'primary_segment': primary_segment,
            'multiplier': final_multiplier,
            'confidence': 0.75,
            'details': {
                'segment': segment,
                'segment_mix': segment_mix,
                'category': category,
                'preference_match': category in self.SEGMENT_PREFERENCES.get(primary_segment, [])
            }
        }