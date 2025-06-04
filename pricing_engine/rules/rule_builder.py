"""Builder pattern for creating business rules dynamically."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
import logging

from .business_rules import BusinessRule, RuleType, RulePriority, RuleCondition, RuleAction
from ..core.models import Market, ProductCategory

logger = logging.getLogger(__name__)


class RuleBuilder:
    """Fluent builder for creating business rules."""
    
    def __init__(self):
        self._reset()
    
    def _reset(self):
        """Reset builder state."""
        self._rule_id: Optional[str] = None
        self._name: Optional[str] = None
        self._description: Optional[str] = None
        self._rule_type: Optional[RuleType] = None
        self._priority: RulePriority = RulePriority.MEDIUM
        self._conditions: List[RuleCondition] = []
        self._actions: List[RuleAction] = []
        self._enabled: bool = True
        self._valid_from: Optional[datetime] = None
        self._valid_until: Optional[datetime] = None
        self._metadata: Dict[str, Any] = {}
    
    def with_id(self, rule_id: str) -> 'RuleBuilder':
        """Set rule ID."""
        self._rule_id = rule_id
        return self
    
    def with_name(self, name: str) -> 'RuleBuilder':
        """Set rule name."""
        self._name = name
        return self
    
    def with_description(self, description: str) -> 'RuleBuilder':
        """Set rule description."""
        self._description = description
        return self
    
    def with_type(self, rule_type: RuleType) -> 'RuleBuilder':
        """Set rule type."""
        self._rule_type = rule_type
        return self
    
    def with_priority(self, priority: RulePriority) -> 'RuleBuilder':
        """Set rule priority."""
        self._priority = priority
        return self
    
    def when_product_category(self, category: ProductCategory) -> 'RuleBuilder':
        """Add condition for product category."""
        self._conditions.append(
            RuleCondition("product.category", "==", category)
        )
        return self
    
    def when_market(self, market: Market) -> 'RuleBuilder':
        """Add condition for market."""
        self._conditions.append(
            RuleCondition("product.market", "==", market)
        )
        return self
    
    def when_inventory_above(self, threshold: int) -> 'RuleBuilder':
        """Add condition for high inventory."""
        self._conditions.append(
            RuleCondition("inventory_level", ">", threshold)
        )
        return self
    
    def when_inventory_below(self, threshold: int) -> 'RuleBuilder':
        """Add condition for low inventory."""
        self._conditions.append(
            RuleCondition("inventory_level", "<", threshold)
        )
        return self
    
    def when_margin_below(self, threshold: Decimal) -> 'RuleBuilder':
        """Add condition for low margin."""
        self._conditions.append(
            RuleCondition("current_margin", "<", threshold)
        )
        return self
    
    def when_price_above(self, price: Decimal) -> 'RuleBuilder':
        """Add condition for price above threshold."""
        self._conditions.append(
            RuleCondition("current_price", ">", price)
        )
        return self
    
    def when_price_below(self, price: Decimal) -> 'RuleBuilder':
        """Add condition for price below threshold."""
        self._conditions.append(
            RuleCondition("current_price", "<", price)
        )
        return self
    
    def when_time_between(self, start_hour: int, end_hour: int) -> 'RuleBuilder':
        """Add condition for time range."""
        self._conditions.append(
            RuleCondition("timestamp.hour", "between", [start_hour, end_hour])
        )
        return self
    
    def when_day_of_week(self, days: List[int]) -> 'RuleBuilder':
        """Add condition for specific days of week (0=Monday, 6=Sunday)."""
        self._conditions.append(
            RuleCondition("timestamp.weekday()", "in", days)
        )
        return self
    
    def when_custom(self, field: str, operator: str, value: Any) -> 'RuleBuilder':
        """Add custom condition."""
        self._conditions.append(
            RuleCondition(field, operator, value)
        )
        return self
    
    def then_set_min_price(self, price: Union[Decimal, str]) -> 'RuleBuilder':
        """Action to set minimum price."""
        self._actions.append(
            RuleAction("set_min_price", {"value": price})
        )
        return self
    
    def then_set_max_price(self, price: Union[Decimal, str]) -> 'RuleBuilder':
        """Action to set maximum price."""
        self._actions.append(
            RuleAction("set_max_price", {"value": price})
        )
        return self
    
    def then_apply_discount(self, percentage: Decimal) -> 'RuleBuilder':
        """Action to apply discount."""
        self._actions.append(
            RuleAction("apply_discount", {"percentage": percentage})
        )
        return self
    
    def then_enforce_margin(self, min_margin: Decimal) -> 'RuleBuilder':
        """Action to enforce minimum margin."""
        self._actions.append(
            RuleAction("enforce_margin", {"min_margin": min_margin})
        )
        return self
    
    def then_block_change(self, reason: str) -> 'RuleBuilder':
        """Action to block price change."""
        self._actions.append(
            RuleAction("block_price_change", {"reason": reason})
        )
        return self
    
    def valid_between(self, start: datetime, end: datetime) -> 'RuleBuilder':
        """Set validity period."""
        self._valid_from = start
        self._valid_until = end
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'RuleBuilder':
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def enabled(self, is_enabled: bool = True) -> 'RuleBuilder':
        """Set enabled status."""
        self._enabled = is_enabled
        return self
    
    def build(self) -> BusinessRule:
        """Build the business rule."""
        # Validate required fields
        if not self._rule_id:
            raise ValueError("Rule ID is required")
        if not self._name:
            raise ValueError("Rule name is required")
        if not self._rule_type:
            raise ValueError("Rule type is required")
        if not self._conditions:
            raise ValueError("At least one condition is required")
        if not self._actions:
            raise ValueError("At least one action is required")
        
        # Create rule
        rule = BusinessRule(
            rule_id=self._rule_id,
            name=self._name,
            description=self._description or "",
            rule_type=self._rule_type,
            priority=self._priority,
            conditions=self._conditions,
            actions=self._actions,
            enabled=self._enabled,
            valid_from=self._valid_from,
            valid_until=self._valid_until,
            metadata=self._metadata
        )
        
        # Reset builder
        self._reset()
        
        return rule


class RuleTemplates:
    """Pre-built rule templates for common scenarios."""
    
    @staticmethod
    def margin_protection_rule(min_margin: Decimal = Decimal('0.15')) -> BusinessRule:
        """Create margin protection rule."""
        return (RuleBuilder()
                .with_id(f"margin_protection_{int(min_margin * 100)}")
                .with_name(f"Margin Protection {min_margin:.0%}")
                .with_description(f"Ensure minimum {min_margin:.0%} margin")
                .with_type(RuleType.MARGIN)
                .with_priority(RulePriority.CRITICAL)
                .when_custom("product.cost", ">", 0)
                .then_enforce_margin(min_margin)
                .build())
    
    @staticmethod
    def inventory_clearance_rule(inventory_threshold: int = 500,
                               discount: Decimal = Decimal('0.15')) -> BusinessRule:
        """Create inventory clearance rule."""
        return (RuleBuilder()
                .with_id(f"inventory_clearance_{inventory_threshold}")
                .with_name(f"Clearance for inventory > {inventory_threshold}")
                .with_description(f"Apply {discount:.0%} discount for excess inventory")
                .with_type(RuleType.INVENTORY)
                .with_priority(RulePriority.MEDIUM)
                .when_inventory_above(inventory_threshold)
                .then_apply_discount(discount)
                .build())
    
    @staticmethod
    def happy_hour_rule(start_hour: int = 16, end_hour: int = 18,
                       discount: Decimal = Decimal('0.10')) -> BusinessRule:
        """Create happy hour promotional rule."""
        return (RuleBuilder()
                .with_id(f"happy_hour_{start_hour}_{end_hour}")
                .with_name(f"Happy Hour {start_hour}:00-{end_hour}:00")
                .with_description(f"{discount:.0%} discount during happy hour")
                .with_type(RuleType.TIME_BASED)
                .with_priority(RulePriority.LOW)
                .when_time_between(start_hour, end_hour)
                .then_apply_discount(discount)
                .build())
    
    @staticmethod
    def medical_pricing_rule(max_markup: Decimal = Decimal('0.50')) -> BusinessRule:
        """Create medical product pricing rule."""
        return (RuleBuilder()
                .with_id("medical_pricing_cap")
                .with_name("Medical Product Pricing Cap")
                .with_description(f"Limit medical products to {max_markup:.0%} markup")
                .with_type(RuleType.CUSTOMER_TYPE)
                .with_priority(RulePriority.CRITICAL)
                .when_product_category(ProductCategory.MEDICAL)
                .when_custom("product.cost", ">", 0)
                .then_set_max_price(f"product.cost * {1 + max_markup}")
                .build())
    
    @staticmethod
    def weekend_pricing_rule(adjustment: Decimal = Decimal('0.05')) -> BusinessRule:
        """Create weekend pricing adjustment rule."""
        return (RuleBuilder()
                .with_id("weekend_pricing")
                .with_name("Weekend Price Adjustment")
                .with_description(f"{adjustment:.0%} price increase on weekends")
                .with_type(RuleType.TIME_BASED)
                .with_priority(RulePriority.LOW)
                .when_day_of_week([5, 6])  # Saturday, Sunday
                .then_apply_discount(-adjustment)  # Negative discount = increase
                .build())
    
    @staticmethod
    def competitive_pricing_rule(market: Market, max_above_competitor: Decimal = Decimal('0.10')) -> BusinessRule:
        """Create competitive pricing rule."""
        return (RuleBuilder()
                .with_id(f"competitive_pricing_{market.value}")
                .with_name(f"Competitive Pricing {market.value}")
                .with_description(f"Stay within {max_above_competitor:.0%} of competitors")
                .with_type(RuleType.COMPETITIVE)
                .with_priority(RulePriority.HIGH)
                .when_market(market)
                .when_custom("competitor_avg_price", ">", 0)
                .then_set_max_price(f"competitor_avg_price * {1 + max_above_competitor}")
                .build())
    
    @staticmethod
    def bundle_discount_rule(bundle_size: int = 3, discount: Decimal = Decimal('0.20')) -> BusinessRule:
        """Create bundle discount rule."""
        return (RuleBuilder()
                .with_id(f"bundle_discount_{bundle_size}")
                .with_name(f"Bundle Discount (Buy {bundle_size})")
                .with_description(f"{discount:.0%} off when buying {bundle_size} or more")
                .with_type(RuleType.BUNDLE)
                .with_priority(RulePriority.MEDIUM)
                .when_custom("quantity", ">=", bundle_size)
                .then_apply_discount(discount)
                .build())