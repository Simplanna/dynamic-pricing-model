"""Core business rules engine for pricing decisions."""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
from enum import Enum
import json
from pathlib import Path

from ..core.models import Product, Market, ProductCategory, PricingDecision

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of business rules."""
    MARGIN = "margin"
    PROMOTIONAL = "promotional"
    INVENTORY = "inventory"
    COMPETITIVE = "competitive"
    TIME_BASED = "time_based"
    BUNDLE = "bundle"
    CUSTOMER_TYPE = "customer_type"
    CROSS_BORDER = "cross_border"
    VOLUME = "volume"
    CLEARANCE = "clearance"


class RulePriority(Enum):
    """Rule execution priority."""
    CRITICAL = 1  # Must be applied
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


@dataclass
class RuleCondition:
    """Condition for rule application."""
    field: str  # e.g., "product.category", "product.inventory_level"
    operator: str  # e.g., "==", ">", "<", "in", "between"
    value: Any  # The value to compare against
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        # Navigate through nested fields
        field_value = context
        for part in self.field.split('.'):
            if isinstance(field_value, dict):
                field_value = field_value.get(part)
            elif hasattr(field_value, part):
                field_value = getattr(field_value, part)
            else:
                return False
        
        # Evaluate based on operator
        if self.operator == "==":
            return field_value == self.value
        elif self.operator == "!=":
            return field_value != self.value
        elif self.operator == ">":
            return field_value > self.value
        elif self.operator == "<":
            return field_value < self.value
        elif self.operator == ">=":
            return field_value >= self.value
        elif self.operator == "<=":
            return field_value <= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "between":
            return self.value[0] <= field_value <= self.value[1]
        elif self.operator == "contains":
            return self.value in str(field_value)
        else:
            logger.warning(f"Unknown operator: {self.operator}")
            return False


@dataclass
class RuleAction:
    """Action to take when rule matches."""
    action_type: str  # e.g., "set_price", "apply_discount", "add_constraint"
    parameters: Dict[str, Any]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action and return modified context."""
        if self.action_type == "set_min_price":
            context['constraints']['min_price'] = Decimal(str(self.parameters['value']))
        elif self.action_type == "set_max_price":
            context['constraints']['max_price'] = Decimal(str(self.parameters['value']))
        elif self.action_type == "apply_discount":
            discount = Decimal(str(self.parameters['percentage']))
            context['adjustments']['discount'] = discount
        elif self.action_type == "enforce_margin":
            min_margin = Decimal(str(self.parameters['min_margin']))
            context['constraints']['min_margin'] = min_margin
        elif self.action_type == "block_price_change":
            context['constraints']['allow_change'] = False
            context['constraints']['reason'] = self.parameters.get('reason', 'Rule blocked change')
        
        return context


@dataclass
class BusinessRule:
    """A business rule for pricing."""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    priority: RulePriority
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    enabled: bool = True
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if rule is applicable in given context."""
        if not self.enabled:
            return False
        
        # Check validity period
        now = datetime.now()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        
        # Check all conditions (AND logic)
        for condition in self.conditions:
            if not condition.evaluate(context):
                return False
        
        return True
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule actions to context."""
        for action in self.actions:
            context = action.execute(context)
        
        # Record rule application
        if 'applied_rules' not in context:
            context['applied_rules'] = []
        context['applied_rules'].append(self.rule_id)
        
        return context


class BusinessRulesEngine:
    """Engine for managing and applying business rules."""
    
    def __init__(self, rules_file: Optional[Path] = None):
        self.rules: Dict[str, BusinessRule] = {}
        self.rules_file = rules_file or Path("business_rules.json")
        
        # Rule execution metrics
        self._execution_count: Dict[str, int] = {}
        self._execution_time: Dict[str, float] = {}
        
        # Load rules from file if exists
        if self.rules_file.exists():
            self.load_rules()
        else:
            self._initialize_default_rules()
        
        logger.info(f"Business rules engine initialized with {len(self.rules)} rules")
    
    def add_rule(self, rule: BusinessRule):
        """Add a business rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rule: {rule.name} (priority: {rule.priority.name})")
    
    def remove_rule(self, rule_id: str):
        """Remove a business rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
    
    def apply_rules(self, product: Product, pricing_decision: PricingDecision) -> PricingDecision:
        """Apply all applicable rules to a pricing decision."""
        # Create context
        context = self._create_context(product, pricing_decision)
        
        # Get applicable rules sorted by priority
        applicable_rules = [
            rule for rule in self.rules.values()
            if rule.is_applicable(context)
        ]
        applicable_rules.sort(key=lambda r: r.priority.value)
        
        # Apply rules in priority order
        for rule in applicable_rules:
            start_time = datetime.now()
            
            try:
                context = rule.apply(context)
                
                # Track metrics
                self._execution_count[rule.rule_id] = self._execution_count.get(rule.rule_id, 0) + 1
                execution_time = (datetime.now() - start_time).total_seconds()
                self._execution_time[rule.rule_id] = execution_time
                
                logger.debug(f"Applied rule {rule.rule_id} to {product.sku}")
                
            except Exception as e:
                logger.error(f"Error applying rule {rule.rule_id}: {str(e)}")
        
        # Apply constraints to pricing decision
        return self._apply_constraints(pricing_decision, context)
    
    def validate_pricing_decision(self, product: Product, 
                                proposed_price: Decimal) -> Tuple[bool, List[str]]:
        """Validate if a proposed price meets all business rules."""
        # Create minimal context for validation
        context = {
            'product': product,
            'proposed_price': proposed_price,
            'current_price': product.current_price,
            'constraints': {},
            'validation_mode': True
        }
        
        violations = []
        
        # Check each rule
        for rule in self.rules.values():
            if rule.is_applicable(context):
                # Simulate rule application
                test_context = context.copy()
                test_context = rule.apply(test_context)
                
                # Check for violations
                constraints = test_context.get('constraints', {})
                
                if 'min_price' in constraints and proposed_price < constraints['min_price']:
                    violations.append(
                        f"{rule.name}: Price ${proposed_price} below minimum ${constraints['min_price']}"
                    )
                
                if 'max_price' in constraints and proposed_price > constraints['max_price']:
                    violations.append(
                        f"{rule.name}: Price ${proposed_price} above maximum ${constraints['max_price']}"
                    )
                
                if not constraints.get('allow_change', True):
                    violations.append(
                        f"{rule.name}: {constraints.get('reason', 'Price change not allowed')}"
                    )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def get_price_boundaries(self, product: Product) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Get min/max price boundaries based on business rules."""
        context = self._create_context(product, None)
        
        # Apply all rules to get constraints
        for rule in self.rules.values():
            if rule.is_applicable(context):
                context = rule.apply(context)
        
        constraints = context.get('constraints', {})
        return constraints.get('min_price'), constraints.get('max_price')
    
    def save_rules(self):
        """Save rules to file."""
        rules_data = []
        
        for rule in self.rules.values():
            rule_dict = {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'rule_type': rule.rule_type.value,
                'priority': rule.priority.value,
                'conditions': [
                    {
                        'field': c.field,
                        'operator': c.operator,
                        'value': c.value
                    }
                    for c in rule.conditions
                ],
                'actions': [
                    {
                        'action_type': a.action_type,
                        'parameters': a.parameters
                    }
                    for a in rule.actions
                ],
                'enabled': rule.enabled,
                'valid_from': rule.valid_from.isoformat() if rule.valid_from else None,
                'valid_until': rule.valid_until.isoformat() if rule.valid_until else None,
                'metadata': rule.metadata
            }
            rules_data.append(rule_dict)
        
        with open(self.rules_file, 'w') as f:
            json.dump(rules_data, f, indent=2, default=str)
    
    def load_rules(self):
        """Load rules from file."""
        with open(self.rules_file, 'r') as f:
            rules_data = json.load(f)
        
        self.rules.clear()
        
        for rule_dict in rules_data:
            # Reconstruct conditions
            conditions = [
                RuleCondition(
                    field=c['field'],
                    operator=c['operator'],
                    value=c['value']
                )
                for c in rule_dict['conditions']
            ]
            
            # Reconstruct actions
            actions = [
                RuleAction(
                    action_type=a['action_type'],
                    parameters=a['parameters']
                )
                for a in rule_dict['actions']
            ]
            
            # Create rule
            rule = BusinessRule(
                rule_id=rule_dict['rule_id'],
                name=rule_dict['name'],
                description=rule_dict['description'],
                rule_type=RuleType(rule_dict['rule_type']),
                priority=RulePriority(rule_dict['priority']),
                conditions=conditions,
                actions=actions,
                enabled=rule_dict.get('enabled', True),
                valid_from=datetime.fromisoformat(rule_dict['valid_from']) if rule_dict.get('valid_from') else None,
                valid_until=datetime.fromisoformat(rule_dict['valid_until']) if rule_dict.get('valid_until') else None,
                metadata=rule_dict.get('metadata', {})
            )
            
            self.add_rule(rule)
    
    def get_rule_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for rules."""
        metrics = {
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
            'execution_stats': {}
        }
        
        for rule_id, count in self._execution_count.items():
            rule = self.rules.get(rule_id)
            if rule:
                metrics['execution_stats'][rule_id] = {
                    'name': rule.name,
                    'execution_count': count,
                    'avg_execution_time': self._execution_time.get(rule_id, 0),
                    'rule_type': rule.rule_type.value,
                    'priority': rule.priority.name
                }
        
        return metrics
    
    def _create_context(self, product: Product, 
                       pricing_decision: Optional[PricingDecision]) -> Dict[str, Any]:
        """Create context for rule evaluation."""
        context = {
            'product': product,
            'current_price': product.current_price,
            'market': product.market,
            'timestamp': datetime.now(),
            'constraints': {},
            'adjustments': {}
        }
        
        # Add pricing decision if available
        if pricing_decision:
            context['pricing_decision'] = pricing_decision
            context['proposed_price'] = pricing_decision.final_price
            context['base_price'] = pricing_decision.base_price
        
        # Add derived fields
        if hasattr(product, 'cost') and product.cost:
            context['current_margin'] = (product.current_price - product.cost) / product.current_price
        
        if hasattr(product, 'inventory_level'):
            context['inventory_level'] = product.inventory_level
        
        return context
    
    def _apply_constraints(self, pricing_decision: PricingDecision,
                         context: Dict[str, Any]) -> PricingDecision:
        """Apply accumulated constraints to pricing decision."""
        constraints = context.get('constraints', {})
        adjustments = context.get('adjustments', {})
        
        # Apply price boundaries
        if 'min_price' in constraints:
            if pricing_decision.final_price < constraints['min_price']:
                pricing_decision.final_price = constraints['min_price']
                pricing_decision.add_metadata('price_floor_applied', True)
        
        if 'max_price' in constraints:
            if pricing_decision.final_price > constraints['max_price']:
                pricing_decision.final_price = constraints['max_price']
                pricing_decision.add_metadata('price_ceiling_applied', True)
        
        # Apply margin constraints
        if 'min_margin' in constraints and hasattr(pricing_decision.product, 'cost'):
            cost = pricing_decision.product.cost
            min_price_for_margin = cost / (Decimal('1') - constraints['min_margin'])
            if pricing_decision.final_price < min_price_for_margin:
                pricing_decision.final_price = min_price_for_margin
                pricing_decision.add_metadata('margin_floor_applied', True)
        
        # Apply adjustments
        if 'discount' in adjustments:
            discount = adjustments['discount']
            pricing_decision.final_price *= (Decimal('1') - discount)
            pricing_decision.add_metadata('rule_discount_applied', float(discount))
        
        # Block change if not allowed
        if not constraints.get('allow_change', True):
            pricing_decision.final_price = pricing_decision.base_price
            pricing_decision.add_metadata('price_change_blocked', True)
            pricing_decision.add_metadata('block_reason', constraints.get('reason', 'Business rule'))
        
        # Add applied rules to metadata
        if 'applied_rules' in context:
            pricing_decision.add_metadata('applied_business_rules', context['applied_rules'])
        
        return pricing_decision
    
    def _initialize_default_rules(self):
        """Initialize with default business rules."""
        
        # Minimum margin rule
        self.add_rule(BusinessRule(
            rule_id="margin_floor",
            name="Minimum Margin Protection",
            description="Ensure minimum 15% margin on all products",
            rule_type=RuleType.MARGIN,
            priority=RulePriority.CRITICAL,
            conditions=[
                RuleCondition("product.cost", ">", 0)
            ],
            actions=[
                RuleAction("enforce_margin", {"min_margin": 0.15})
            ]
        ))
        
        # High inventory discount rule
        self.add_rule(BusinessRule(
            rule_id="high_inventory_discount",
            name="High Inventory Clearance",
            description="Apply discount for overstocked items",
            rule_type=RuleType.INVENTORY,
            priority=RulePriority.MEDIUM,
            conditions=[
                RuleCondition("inventory_level", ">", 500)
            ],
            actions=[
                RuleAction("apply_discount", {"percentage": 0.10})
            ]
        ))
        
        # No price changes during peak hours
        self.add_rule(BusinessRule(
            rule_id="peak_hours_freeze",
            name="Peak Hours Price Freeze",
            description="No price changes during peak business hours",
            rule_type=RuleType.TIME_BASED,
            priority=RulePriority.HIGH,
            conditions=[
                RuleCondition("timestamp.hour", "between", [17, 20])  # 5 PM - 8 PM
            ],
            actions=[
                RuleAction("block_price_change", {"reason": "Peak hours price freeze"})
            ]
        ))
        
        # Medical product pricing cap
        self.add_rule(BusinessRule(
            rule_id="medical_price_cap",
            name="Medical Product Price Cap",
            description="Medical cannabis products have maximum markup",
            rule_type=RuleType.CUSTOMER_TYPE,
            priority=RulePriority.CRITICAL,
            conditions=[
                RuleCondition("product.category", "==", ProductCategory.MEDICAL)
            ],
            actions=[
                RuleAction("set_max_price", {"value": "product.cost * 1.5"})
            ]
        ))
        
        # Cross-border pricing adjustment
        self.add_rule(BusinessRule(
            rule_id="cross_border_adjustment",
            name="Cross-Border Price Adjustment",
            description="Adjust prices for cross-border considerations",
            rule_type=RuleType.CROSS_BORDER,
            priority=RulePriority.MEDIUM,
            conditions=[
                RuleCondition("product.market", "==", Market.MASSACHUSETTS),
                RuleCondition("product.border_distance", "<", 10)  # Within 10 miles of RI border
            ],
            actions=[
                RuleAction("set_max_price", {"value": "competitor_avg * 1.05"})
            ]
        ))
        
        logger.info("Initialized default business rules")