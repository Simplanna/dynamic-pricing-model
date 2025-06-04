"""Master Pricing Orchestrator

Coordinates all nine pricing factors, applies weights, and ensures safety controls.
Generates final pricing recommendations with full audit trails.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from dataclasses import dataclass, asdict

from . import FACTOR_WEIGHTS, SAFETY_CONTROLS, PRODUCT_CATEGORIES
from .factors.inventory_pressure import InventoryPressureFactor
from .factors.demand_velocity import DemandVelocityFactor
from .factors.competition import CompetitionFactor
from .factors.product_age import ProductAgeFactor
from .factors.market_events import MarketEventsFactor
from .factors.minor_factors import (
    BrandEquityFactor, PotencySizeFactor, 
    StoreLocationFactor, CustomerSegmentFactor
)


@dataclass
class PricingDecision:
    """Structured pricing decision with full audit trail"""
    product_id: str
    timestamp: datetime
    current_price: float
    recommended_price: float
    final_price: float
    price_change_pct: float
    confidence_score: float
    factors: Dict
    safety_overrides: List[str]
    metadata: Dict
    

class PricingOrchestrator:
    """Master orchestrator for dynamic pricing decisions"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize orchestrator with optional configuration"""
        self.config = config or {}
        self.factor_weights = FACTOR_WEIGHTS.copy()
        self.safety_controls = SAFETY_CONTROLS.copy()
        
        # Update with any custom configuration
        if 'factor_weights' in self.config:
            self.factor_weights.update(self.config['factor_weights'])
        if 'safety_controls' in self.config:
            self.safety_controls.update(self.config['safety_controls'])
            
        # Validate weights sum to 1.0
        weight_sum = sum(self.factor_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Factor weights must sum to 1.0, got {weight_sum}")
            
        # Initialize factors
        self._initialize_factors()
        
    def _initialize_factors(self):
        """Initialize all pricing factor instances"""
        self.factors = {
            'inventory_pressure': InventoryPressureFactor(
                {cat: config['perishability_days'] 
                 for cat, config in PRODUCT_CATEGORIES.items()}
            ),
            'demand_velocity': DemandVelocityFactor(),
            'competition': CompetitionFactor(),
            'product_age': ProductAgeFactor(),
            'market_events': MarketEventsFactor(),
            'brand_equity': BrandEquityFactor(),
            'potency_size': PotencySizeFactor(),
            'store_location': StoreLocationFactor(),
            'customer_segment': CustomerSegmentFactor()
        }
        
    def calculate_price(self,
                       product_data: Dict,
                       market_data: Dict,
                       override_weights: Optional[Dict] = None) -> PricingDecision:
        """Calculate optimal price using all factors"""
        
        # Use override weights if provided
        weights = override_weights or self.factor_weights
        
        # Current price
        current_price = product_data.get('current_price', 0)
        if current_price <= 0:
            raise ValueError("Current price must be positive")
            
        # Calculate all factor scores
        factor_results = self._calculate_all_factors(product_data, market_data)
        
        # Calculate weighted price multiplier
        weighted_multiplier = self._calculate_weighted_multiplier(
            factor_results, weights
        )
        
        # Calculate recommended price
        recommended_price = current_price * weighted_multiplier
        
        # Apply safety controls
        final_price, safety_overrides = self._apply_safety_controls(
            current_price, recommended_price, product_data
        )
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(factor_results)
        
        # Calculate price change percentage
        price_change_pct = ((final_price - current_price) / current_price) * 100
        
        # Create pricing decision
        decision = PricingDecision(
            product_id=product_data.get('id', 'unknown'),
            timestamp=datetime.now(),
            current_price=current_price,
            recommended_price=recommended_price,
            final_price=final_price,
            price_change_pct=price_change_pct,
            confidence_score=confidence_score,
            factors=factor_results,
            safety_overrides=safety_overrides,
            metadata={
                'category': product_data.get('category'),
                'brand': product_data.get('brand'),
                'weighted_multiplier': weighted_multiplier
            }
        )
        
        return decision
        
    def _calculate_all_factors(self, 
                             product_data: Dict,
                             market_data: Dict) -> Dict:
        """Calculate scores for all pricing factors"""
        results = {}
        
        # Inventory Pressure
        if 'inventory_pressure' in self.factor_weights:
            results['inventory_pressure'] = self.factors['inventory_pressure'].calculate_factor_score(
                product_data,
                market_data.get('inventory', {}),
                market_data.get('sales_velocity', 0)
            )
            
        # Demand Velocity
        if 'demand_velocity' in self.factor_weights:
            results['demand_velocity'] = self.factors['demand_velocity'].calculate_factor_score(
                product_data,
                market_data.get('sales_history', [])
            )
            
        # Competition
        if 'competition' in self.factor_weights:
            results['competition'] = self.factors['competition'].calculate_factor_score(
                product_data,
                product_data.get('current_price', 0),
                market_data.get('competitors', []),
                market_data.get('market', 'MA')
            )
            
        # Product Age
        if 'product_age' in self.factor_weights:
            results['product_age'] = self.factors['product_age'].calculate_factor_score(
                product_data,
                product_data.get('current_price', 0)
            )
            
        # Market Events
        if 'market_events' in self.factor_weights:
            results['market_events'] = self.factors['market_events'].calculate_factor_score(
                product_data,
                market_data.get('pricing_date')
            )
            
        # Brand Equity
        if 'brand_equity' in self.factor_weights:
            results['brand_equity'] = self.factors['brand_equity'].calculate_factor_score(
                product_data
            )
            
        # Potency/Size
        if 'potency_size' in self.factor_weights:
            results['potency_size'] = self.factors['potency_size'].calculate_factor_score(
                product_data
            )
            
        # Store Location
        if 'store_location' in self.factor_weights:
            results['store_location'] = self.factors['store_location'].calculate_factor_score(
                product_data,
                market_data.get('store', {})
            )
            
        # Customer Segment
        if 'customer_segment' in self.factor_weights:
            results['customer_segment'] = self.factors['customer_segment'].calculate_factor_score(
                product_data,
                market_data.get('primary_segment', 'regular'),
                market_data.get('segment_mix')
            )
            
        return results
        
    def _calculate_weighted_multiplier(self,
                                     factor_results: Dict,
                                     weights: Dict) -> float:
        """Calculate weighted average of all factor multipliers"""
        weighted_sum = 0
        weight_sum = 0
        
        for factor_name, result in factor_results.items():
            weight = weights.get(factor_name, 0)
            multiplier = result.get('multiplier', 1.0)
            confidence = result.get('confidence', 1.0)
            
            # Weight by both factor weight and confidence
            effective_weight = weight * confidence
            weighted_sum += multiplier * effective_weight
            weight_sum += effective_weight
            
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 1.0  # No change if no weights
            
    def _apply_safety_controls(self,
                             current_price: float,
                             recommended_price: float,
                             product_data: Dict) -> Tuple[float, List[str]]:
        """Apply safety controls to prevent extreme price changes"""
        overrides = []
        final_price = recommended_price
        
        # Daily change cap
        max_change = self.safety_controls['max_daily_change']
        price_change = (recommended_price - current_price) / current_price
        
        if abs(price_change) > max_change:
            # Cap the change
            if price_change > 0:
                final_price = current_price * (1 + max_change)
                overrides.append(f"Daily increase capped at {max_change*100}%")
            else:
                final_price = current_price * (1 - max_change)
                overrides.append(f"Daily decrease capped at {max_change*100}%")
                
        # Price floor (minimum margin)
        cost = product_data.get('cost', 0)
        if cost > 0:
            min_price = cost * (1 + self.safety_controls['min_margin'])
            if final_price < min_price:
                final_price = min_price
                overrides.append(f"Price floor applied (min margin {self.safety_controls['min_margin']*100}%)")
                
        # Maximum discount
        base_price = product_data.get('base_price', current_price)
        max_discount = self.safety_controls['max_discount']
        min_allowed_price = base_price * (1 - max_discount)
        
        if final_price < min_allowed_price:
            final_price = min_allowed_price
            overrides.append(f"Maximum discount capped at {max_discount*100}%")
            
        # Absolute price bounds
        floor_mult = self.safety_controls['price_floor_multiplier']
        ceiling_mult = self.safety_controls['price_ceiling_multiplier']
        
        absolute_floor = base_price * floor_mult
        absolute_ceiling = base_price * ceiling_mult
        
        if final_price < absolute_floor:
            final_price = absolute_floor
            overrides.append(f"Absolute price floor at {floor_mult*100}% of base")
        elif final_price > absolute_ceiling:
            final_price = absolute_ceiling
            overrides.append(f"Absolute price ceiling at {ceiling_mult*100}% of base")
            
        return final_price, overrides
        
    def _calculate_overall_confidence(self, factor_results: Dict) -> float:
        """Calculate overall confidence score for the pricing decision"""
        if not factor_results:
            return 0.0
            
        # Weighted average of factor confidences
        weighted_confidence = 0
        weight_sum = 0
        
        for factor_name, result in factor_results.items():
            weight = self.factor_weights.get(factor_name, 0)
            confidence = result.get('confidence', 0)
            
            weighted_confidence += confidence * weight
            weight_sum += weight
            
        if weight_sum > 0:
            overall_confidence = weighted_confidence / weight_sum
        else:
            overall_confidence = 0.0
            
        # Apply confidence threshold check
        if overall_confidence < self.safety_controls['confidence_threshold']:
            # Low confidence indicator
            return overall_confidence * 0.8  # Further reduce
            
        return overall_confidence
        
    def generate_audit_log(self, decision: PricingDecision) -> Dict:
        """Generate detailed audit log for pricing decision"""
        audit_log = {
            'decision_id': f"{decision.product_id}_{decision.timestamp.isoformat()}",
            'timestamp': decision.timestamp.isoformat(),
            'product_id': decision.product_id,
            'price_change': {
                'from': decision.current_price,
                'to': decision.final_price,
                'change_pct': decision.price_change_pct,
                'recommended': decision.recommended_price
            },
            'confidence': decision.confidence_score,
            'factors': {}
        }
        
        # Add detailed factor information
        for factor_name, result in decision.factors.items():
            audit_log['factors'][factor_name] = {
                'weight': result.get('weight', 0),
                'multiplier': result.get('multiplier', 1.0),
                'confidence': result.get('confidence', 0),
                'details': result.get('details', {})
            }
            
        # Add safety overrides
        if decision.safety_overrides:
            audit_log['safety_overrides'] = decision.safety_overrides
            
        # Add metadata
        audit_log['metadata'] = decision.metadata
        
        return audit_log
        
    def validate_pricing_decision(self, decision: PricingDecision) -> Tuple[bool, List[str]]:
        """Validate pricing decision against business rules"""
        issues = []
        
        # Check confidence threshold
        if decision.confidence_score < self.safety_controls['confidence_threshold']:
            issues.append(f"Low confidence score: {decision.confidence_score:.2f}")
            
        # Check price change magnitude
        if abs(decision.price_change_pct) > 20:
            issues.append(f"Large price change: {decision.price_change_pct:.1f}%")
            
        # Check for multiple safety overrides
        if len(decision.safety_overrides) > 2:
            issues.append(f"Multiple safety overrides applied: {len(decision.safety_overrides)}")
            
        # Check for factor anomalies
        for factor_name, result in decision.factors.items():
            if result.get('confidence', 1.0) < 0.5:
                issues.append(f"Low confidence in {factor_name}: {result['confidence']:.2f}")
                
        is_valid = len(issues) == 0
        return is_valid, issues
        
    def batch_calculate_prices(self, 
                             products: List[Dict],
                             market_data: Dict) -> List[PricingDecision]:
        """Calculate prices for multiple products efficiently"""
        decisions = []
        
        for product in products:
            try:
                decision = self.calculate_price(product, market_data)
                decisions.append(decision)
            except Exception as e:
                # Log error but continue with other products
                print(f"Error pricing product {product.get('id')}: {e}")
                
        return decisions