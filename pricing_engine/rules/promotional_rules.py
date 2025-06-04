"""Manager for promotional pricing rules and campaigns."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging
from enum import Enum
import json
from pathlib import Path

from .business_rules import BusinessRule, BusinessRulesEngine, RuleType, RulePriority
from .rule_builder import RuleBuilder
from ..core.models import Product, Market, ProductCategory

logger = logging.getLogger(__name__)


class PromotionType(Enum):
    """Types of promotions."""
    PERCENTAGE_OFF = "percentage_off"
    FIXED_AMOUNT_OFF = "fixed_amount_off"
    BOGO = "buy_one_get_one"
    BUNDLE = "bundle_discount"
    TIERED = "tiered_discount"
    FLASH_SALE = "flash_sale"
    LOYALTY = "loyalty_discount"
    FIRST_TIME = "first_time_customer"
    SEASONAL = "seasonal"
    CLEARANCE = "clearance"


@dataclass
class Promotion:
    """Represents a promotional campaign."""
    promotion_id: str
    name: str
    description: str
    promotion_type: PromotionType
    start_date: datetime
    end_date: datetime
    discount_value: Decimal  # Percentage or fixed amount
    conditions: Dict[str, Any] = field(default_factory=dict)
    applicable_products: Optional[Set[str]] = None  # None means all products
    applicable_categories: Optional[Set[ProductCategory]] = None
    applicable_markets: Optional[Set[Market]] = None
    max_uses: Optional[int] = None
    max_uses_per_customer: Optional[int] = None
    min_purchase_amount: Optional[Decimal] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if promotion is currently active."""
        if not self.enabled:
            return False
        now = datetime.now()
        return self.start_date <= now <= self.end_date
    
    def is_applicable_to_product(self, product: Product) -> bool:
        """Check if promotion applies to a specific product."""
        # Check product SKU
        if self.applicable_products and product.sku not in self.applicable_products:
            return False
        
        # Check category
        if self.applicable_categories and product.category not in self.applicable_categories:
            return False
        
        # Check market
        if self.applicable_markets and product.market not in self.applicable_markets:
            return False
        
        return True


@dataclass
class PromotionStack:
    """Rules for how promotions can be combined."""
    allow_stacking: bool = False
    max_stack_count: int = 2
    excluded_combinations: Set[Tuple[str, str]] = field(default_factory=set)
    priority_order: List[PromotionType] = field(default_factory=lambda: [
        PromotionType.CLEARANCE,
        PromotionType.FLASH_SALE,
        PromotionType.SEASONAL,
        PromotionType.PERCENTAGE_OFF,
        PromotionType.LOYALTY,
        PromotionType.FIRST_TIME
    ])


class PromotionalRulesManager:
    """Manages promotional pricing rules and campaigns."""
    
    def __init__(self, business_rules_engine: BusinessRulesEngine,
                 promotions_file: Optional[Path] = None):
        self.rules_engine = business_rules_engine
        self.promotions_file = promotions_file or Path("promotions.json")
        self.promotions: Dict[str, Promotion] = {}
        self.stack_config = PromotionStack()
        
        # Usage tracking
        self._usage_count: Dict[str, int] = {}
        self._customer_usage: Dict[Tuple[str, str], int] = {}  # (promotion_id, customer_id) -> count
        
        # Load promotions if file exists
        if self.promotions_file.exists():
            self.load_promotions()
        
        logger.info(f"Promotional rules manager initialized with {len(self.promotions)} promotions")
    
    def create_promotion(self, promotion: Promotion) -> str:
        """Create a new promotion and associated business rule."""
        # Store promotion
        self.promotions[promotion.promotion_id] = promotion
        
        # Create corresponding business rule
        rule = self._create_rule_for_promotion(promotion)
        self.rules_engine.add_rule(rule)
        
        logger.info(f"Created promotion: {promotion.name}")
        
        return promotion.promotion_id
    
    def update_promotion(self, promotion_id: str, updates: Dict[str, Any]):
        """Update an existing promotion."""
        if promotion_id not in self.promotions:
            raise ValueError(f"Promotion {promotion_id} not found")
        
        promotion = self.promotions[promotion_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(promotion, key):
                setattr(promotion, key, value)
        
        # Update corresponding rule
        self.rules_engine.remove_rule(f"promo_{promotion_id}")
        rule = self._create_rule_for_promotion(promotion)
        self.rules_engine.add_rule(rule)
        
        logger.info(f"Updated promotion: {promotion_id}")
    
    def deactivate_promotion(self, promotion_id: str):
        """Deactivate a promotion."""
        if promotion_id in self.promotions:
            self.promotions[promotion_id].enabled = False
            
            # Disable corresponding rule
            rule_id = f"promo_{promotion_id}"
            if rule_id in self.rules_engine.rules:
                self.rules_engine.rules[rule_id].enabled = False
    
    def get_active_promotions(self, product: Optional[Product] = None) -> List[Promotion]:
        """Get all active promotions, optionally filtered by product."""
        active_promotions = [
            p for p in self.promotions.values()
            if p.is_active()
        ]
        
        if product:
            active_promotions = [
                p for p in active_promotions
                if p.is_applicable_to_product(product)
            ]
        
        return active_promotions
    
    def calculate_promotional_price(self, product: Product, base_price: Decimal,
                                  customer_id: Optional[str] = None) -> Tuple[Decimal, List[str]]:
        """Calculate promotional price for a product."""
        applicable_promotions = self.get_active_promotions(product)
        
        if not applicable_promotions:
            return base_price, []
        
        # Filter by usage limits
        if customer_id:
            applicable_promotions = self._filter_by_usage_limits(
                applicable_promotions, customer_id
            )
        
        # Sort by priority if stacking is allowed
        if self.stack_config.allow_stacking:
            applicable_promotions = self._sort_by_priority(applicable_promotions)
            applicable_promotions = applicable_promotions[:self.stack_config.max_stack_count]
        else:
            # Take only the best promotion
            best_promotion = self._select_best_promotion(applicable_promotions, product, base_price)
            applicable_promotions = [best_promotion] if best_promotion else []
        
        # Apply promotions
        final_price = base_price
        applied_promotions = []
        
        for promotion in applicable_promotions:
            new_price = self._apply_promotion(promotion, product, final_price)
            if new_price < final_price:
                final_price = new_price
                applied_promotions.append(promotion.promotion_id)
                
                # Track usage
                self._track_usage(promotion.promotion_id, customer_id)
        
        return final_price, applied_promotions
    
    def create_seasonal_campaign(self, name: str, season: str,
                               discount: Decimal, start_date: date,
                               duration_days: int = 30) -> str:
        """Create a seasonal promotional campaign."""
        promotion_id = f"seasonal_{season}_{start_date.strftime('%Y%m%d')}"
        
        promotion = Promotion(
            promotion_id=promotion_id,
            name=name,
            description=f"{season} seasonal promotion - {discount:.0%} off",
            promotion_type=PromotionType.SEASONAL,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(start_date + timedelta(days=duration_days), datetime.max.time()),
            discount_value=discount,
            conditions={'season': season},
            metadata={'auto_created': True, 'campaign_type': 'seasonal'}
        )
        
        return self.create_promotion(promotion)
    
    def create_flash_sale(self, name: str, products: List[str],
                         discount: Decimal, duration_hours: int = 4) -> str:
        """Create a flash sale promotion."""
        promotion_id = f"flash_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        promotion = Promotion(
            promotion_id=promotion_id,
            name=name,
            description=f"Flash Sale - {discount:.0%} off for {duration_hours} hours",
            promotion_type=PromotionType.FLASH_SALE,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(hours=duration_hours),
            discount_value=discount,
            applicable_products=set(products),
            metadata={'auto_created': True, 'urgency': 'high'}
        )
        
        return self.create_promotion(promotion)
    
    def create_loyalty_tier(self, tier_name: str, discount: Decimal,
                          min_lifetime_spend: Decimal) -> str:
        """Create a loyalty tier promotion."""
        promotion_id = f"loyalty_{tier_name.lower().replace(' ', '_')}"
        
        promotion = Promotion(
            promotion_id=promotion_id,
            name=f"{tier_name} Loyalty Discount",
            description=f"{discount:.0%} off for {tier_name} members",
            promotion_type=PromotionType.LOYALTY,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=365),  # 1 year
            discount_value=discount,
            conditions={'min_lifetime_spend': float(min_lifetime_spend)},
            metadata={'tier': tier_name}
        )
        
        return self.create_promotion(promotion)
    
    def get_promotion_analytics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get analytics for promotions within date range."""
        analytics = {
            'total_promotions': 0,
            'active_promotions': 0,
            'total_uses': 0,
            'by_type': {},
            'top_promotions': [],
            'revenue_impact_estimate': Decimal('0')
        }
        
        for promotion in self.promotions.values():
            # Check if promotion was active during period
            if promotion.end_date >= start_date and promotion.start_date <= end_date:
                analytics['total_promotions'] += 1
                
                if promotion.is_active():
                    analytics['active_promotions'] += 1
                
                # Count by type
                promo_type = promotion.promotion_type.value
                analytics['by_type'][promo_type] = analytics['by_type'].get(promo_type, 0) + 1
                
                # Usage count
                uses = self._usage_count.get(promotion.promotion_id, 0)
                analytics['total_uses'] += uses
                
                # Top promotions by usage
                if uses > 0:
                    analytics['top_promotions'].append({
                        'promotion_id': promotion.promotion_id,
                        'name': promotion.name,
                        'uses': uses,
                        'discount_value': float(promotion.discount_value)
                    })
        
        # Sort top promotions
        analytics['top_promotions'].sort(key=lambda x: x['uses'], reverse=True)
        analytics['top_promotions'] = analytics['top_promotions'][:10]
        
        return analytics
    
    def save_promotions(self):
        """Save promotions to file."""
        promotions_data = []
        
        for promotion in self.promotions.values():
            promo_dict = {
                'promotion_id': promotion.promotion_id,
                'name': promotion.name,
                'description': promotion.description,
                'promotion_type': promotion.promotion_type.value,
                'start_date': promotion.start_date.isoformat(),
                'end_date': promotion.end_date.isoformat(),
                'discount_value': str(promotion.discount_value),
                'conditions': promotion.conditions,
                'applicable_products': list(promotion.applicable_products) if promotion.applicable_products else None,
                'applicable_categories': [c.value for c in promotion.applicable_categories] if promotion.applicable_categories else None,
                'applicable_markets': [m.value for m in promotion.applicable_markets] if promotion.applicable_markets else None,
                'max_uses': promotion.max_uses,
                'max_uses_per_customer': promotion.max_uses_per_customer,
                'min_purchase_amount': str(promotion.min_purchase_amount) if promotion.min_purchase_amount else None,
                'enabled': promotion.enabled,
                'metadata': promotion.metadata
            }
            promotions_data.append(promo_dict)
        
        with open(self.promotions_file, 'w') as f:
            json.dump({
                'promotions': promotions_data,
                'usage_count': self._usage_count,
                'stack_config': {
                    'allow_stacking': self.stack_config.allow_stacking,
                    'max_stack_count': self.stack_config.max_stack_count
                }
            }, f, indent=2)
    
    def load_promotions(self):
        """Load promotions from file."""
        with open(self.promotions_file, 'r') as f:
            data = json.load(f)
        
        self.promotions.clear()
        
        for promo_dict in data.get('promotions', []):
            # Reconstruct promotion
            promotion = Promotion(
                promotion_id=promo_dict['promotion_id'],
                name=promo_dict['name'],
                description=promo_dict['description'],
                promotion_type=PromotionType(promo_dict['promotion_type']),
                start_date=datetime.fromisoformat(promo_dict['start_date']),
                end_date=datetime.fromisoformat(promo_dict['end_date']),
                discount_value=Decimal(promo_dict['discount_value']),
                conditions=promo_dict.get('conditions', {}),
                applicable_products=set(promo_dict['applicable_products']) if promo_dict.get('applicable_products') else None,
                applicable_categories=set(ProductCategory[c] for c in promo_dict['applicable_categories']) if promo_dict.get('applicable_categories') else None,
                applicable_markets=set(Market[m] for m in promo_dict['applicable_markets']) if promo_dict.get('applicable_markets') else None,
                max_uses=promo_dict.get('max_uses'),
                max_uses_per_customer=promo_dict.get('max_uses_per_customer'),
                min_purchase_amount=Decimal(promo_dict['min_purchase_amount']) if promo_dict.get('min_purchase_amount') else None,
                enabled=promo_dict.get('enabled', True),
                metadata=promo_dict.get('metadata', {})
            )
            
            self.promotions[promotion.promotion_id] = promotion
            
            # Create corresponding rule
            rule = self._create_rule_for_promotion(promotion)
            self.rules_engine.add_rule(rule)
        
        # Load usage counts
        self._usage_count = data.get('usage_count', {})
        
        # Load stack config
        stack_data = data.get('stack_config', {})
        self.stack_config.allow_stacking = stack_data.get('allow_stacking', False)
        self.stack_config.max_stack_count = stack_data.get('max_stack_count', 2)
    
    def _create_rule_for_promotion(self, promotion: Promotion) -> BusinessRule:
        """Create a business rule for a promotion."""
        builder = (RuleBuilder()
                   .with_id(f"promo_{promotion.promotion_id}")
                   .with_name(promotion.name)
                   .with_description(promotion.description)
                   .with_type(RuleType.PROMOTIONAL)
                   .with_priority(RulePriority.LOW)
                   .valid_between(promotion.start_date, promotion.end_date)
                   .enabled(promotion.enabled))
        
        # Add conditions based on promotion type
        if promotion.applicable_categories:
            for category in promotion.applicable_categories:
                builder.when_product_category(category)
        
        if promotion.applicable_markets:
            for market in promotion.applicable_markets:
                builder.when_market(market)
        
        # Add promotion-specific conditions
        if promotion.promotion_type == PromotionType.CLEARANCE:
            builder.when_inventory_above(300)
        elif promotion.promotion_type == PromotionType.FIRST_TIME:
            builder.when_custom("customer.is_first_purchase", "==", True)
        
        # Add action
        if promotion.promotion_type in [PromotionType.PERCENTAGE_OFF, PromotionType.SEASONAL,
                                      PromotionType.FLASH_SALE, PromotionType.LOYALTY]:
            builder.then_apply_discount(promotion.discount_value)
        elif promotion.promotion_type == PromotionType.FIXED_AMOUNT_OFF:
            builder.then_set_max_price(f"current_price - {promotion.discount_value}")
        
        return builder.build()
    
    def _apply_promotion(self, promotion: Promotion, product: Product,
                        current_price: Decimal) -> Decimal:
        """Apply a single promotion to a price."""
        if promotion.promotion_type in [PromotionType.PERCENTAGE_OFF, PromotionType.SEASONAL,
                                      PromotionType.FLASH_SALE, PromotionType.LOYALTY,
                                      PromotionType.FIRST_TIME, PromotionType.CLEARANCE]:
            # Percentage discount
            return current_price * (Decimal('1') - promotion.discount_value)
        
        elif promotion.promotion_type == PromotionType.FIXED_AMOUNT_OFF:
            # Fixed amount discount
            return max(Decimal('0'), current_price - promotion.discount_value)
        
        elif promotion.promotion_type == PromotionType.BOGO:
            # Buy one get one (50% off when buying 2)
            return current_price * Decimal('0.5')
        
        elif promotion.promotion_type == PromotionType.BUNDLE:
            # Bundle discount based on quantity
            quantity = promotion.conditions.get('quantity', 1)
            if quantity >= promotion.conditions.get('min_quantity', 3):
                return current_price * (Decimal('1') - promotion.discount_value)
        
        elif promotion.promotion_type == PromotionType.TIERED:
            # Tiered discount based on purchase amount
            if promotion.min_purchase_amount and current_price >= promotion.min_purchase_amount:
                return current_price * (Decimal('1') - promotion.discount_value)
        
        return current_price
    
    def _filter_by_usage_limits(self, promotions: List[Promotion],
                              customer_id: str) -> List[Promotion]:
        """Filter promotions by usage limits."""
        filtered = []
        
        for promotion in promotions:
            # Check overall usage limit
            if promotion.max_uses:
                uses = self._usage_count.get(promotion.promotion_id, 0)
                if uses >= promotion.max_uses:
                    continue
            
            # Check per-customer limit
            if promotion.max_uses_per_customer:
                customer_uses = self._customer_usage.get(
                    (promotion.promotion_id, customer_id), 0
                )
                if customer_uses >= promotion.max_uses_per_customer:
                    continue
            
            filtered.append(promotion)
        
        return filtered
    
    def _sort_by_priority(self, promotions: List[Promotion]) -> List[Promotion]:
        """Sort promotions by priority order."""
        def priority_key(promotion: Promotion) -> int:
            try:
                return self.stack_config.priority_order.index(promotion.promotion_type)
            except ValueError:
                return len(self.stack_config.priority_order)
        
        return sorted(promotions, key=priority_key)
    
    def _select_best_promotion(self, promotions: List[Promotion],
                             product: Product, base_price: Decimal) -> Optional[Promotion]:
        """Select the best promotion for a product."""
        if not promotions:
            return None
        
        best_promotion = None
        best_price = base_price
        
        for promotion in promotions:
            price = self._apply_promotion(promotion, product, base_price)
            if price < best_price:
                best_price = price
                best_promotion = promotion
        
        return best_promotion
    
    def _track_usage(self, promotion_id: str, customer_id: Optional[str]):
        """Track promotion usage."""
        self._usage_count[promotion_id] = self._usage_count.get(promotion_id, 0) + 1
        
        if customer_id:
            key = (promotion_id, customer_id)
            self._customer_usage[key] = self._customer_usage.get(key, 0) + 1