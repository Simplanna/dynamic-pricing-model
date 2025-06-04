"""Cannabis Dynamic Pricing Engine - Core Module"""

from typing import Dict, Any

__version__ = "1.0.0"

# Factor weights configuration
FACTOR_WEIGHTS = {
    'inventory_pressure': 0.25,
    'demand_velocity': 0.20,
    'competition': 0.15,
    'product_age': 0.15,
    'market_events': 0.10,
    'brand_equity': 0.05,
    'potency_size': 0.05,
    'store_location': 0.03,
    'customer_segment': 0.02
}

# Validate weights sum to 1.0
assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 0.001, "Factor weights must sum to 100%"

# Cannabis product categories
PRODUCT_CATEGORIES = {
    'flower': {
        'perishability_days': 45,
        'max_age_days': 90,
        'elasticity': -0.45
    },
    'edibles': {
        'perishability_days': 60,
        'max_age_days': 180,
        'elasticity': -0.48
    },
    'concentrates': {
        'perishability_days': 90,
        'max_age_days': 365,
        'elasticity': -0.50
    },
    'prerolls': {
        'perishability_days': 30,
        'max_age_days': 60,
        'elasticity': -0.43
    },
    'vapes': {
        'perishability_days': 180,
        'max_age_days': 365,
        'elasticity': -0.47
    },
    'accessories': {
        'perishability_days': None,  # Non-perishable
        'max_age_days': None,
        'elasticity': -0.35
    }
}

# Price safety controls
SAFETY_CONTROLS = {
    'max_daily_change': 0.05,  # ±5% daily
    'min_margin': 0.15,  # 15% minimum margin
    'max_discount': 0.40,  # 40% maximum discount
    'confidence_threshold': 0.70,  # 70% confidence required
    'price_floor_multiplier': 0.60,  # 60% of base price minimum
    'price_ceiling_multiplier': 1.50  # 150% of base price maximum
}