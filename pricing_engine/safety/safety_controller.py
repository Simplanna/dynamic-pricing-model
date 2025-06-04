"""Master safety controller enforcing pricing constraints and limits."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging
from enum import Enum
from collections import deque

from ..core.models import Product, PricingDecision, Market
from ..utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class SafetyViolationType(Enum):
    """Types of safety violations."""
    MAX_DAILY_CHANGE = "max_daily_change"
    MAX_SINGLE_CHANGE = "max_single_change"
    MAPE_THRESHOLD = "mape_threshold"
    VELOCITY_LIMIT = "velocity_limit"
    MARGIN_FLOOR = "margin_floor"
    COMPETITOR_DEVIATION = "competitor_deviation"
    INVENTORY_CONSTRAINT = "inventory_constraint"


@dataclass
class SafetyViolation:
    """Details of a safety violation."""
    violation_type: SafetyViolationType
    product: Product
    current_value: Decimal
    limit_value: Decimal
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyConfig:
    """Configuration for safety controls."""
    # Price change limits
    max_daily_price_change: Decimal = Decimal('0.10')  # 10%
    max_single_price_change: Decimal = Decimal('0.05')  # 5%
    max_hourly_changes: int = 3
    
    # Performance thresholds
    mape_threshold: Decimal = Decimal('0.02')  # 2%
    acceptable_mape_duration: timedelta = timedelta(hours=1)
    
    # Business constraints
    min_margin_threshold: Decimal = Decimal('0.15')  # 15%
    max_competitor_deviation: Decimal = Decimal('0.20')  # 20%
    
    # Inventory constraints
    low_inventory_threshold: int = 10
    high_inventory_threshold: int = 500
    inventory_price_adjustment: Decimal = Decimal('0.03')  # 3%
    
    # Safety modes
    enforce_hard_limits: bool = True
    allow_emergency_override: bool = False
    require_dual_approval: bool = True


@dataclass
class PriceChangeHistory:
    """Track price change history for safety monitoring."""
    timestamp: datetime
    old_price: Decimal
    new_price: Decimal
    change_percent: Decimal
    reason: str


class SafetyController:
    """Enforces safety constraints on pricing decisions."""
    
    def __init__(self, config: Optional[SafetyConfig] = None,
                 audit_logger: Optional[AuditLogger] = None):
        self.config = config or SafetyConfig()
        self.audit_logger = audit_logger or AuditLogger()
        
        # Price history tracking
        self._price_history: Dict[str, deque] = {}  # SKU -> history
        self._daily_changes: Dict[str, List[PriceChangeHistory]] = {}
        self._performance_metrics: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        
        # Emergency overrides
        self._emergency_overrides: Set[str] = set()
        self._override_expiry: Dict[str, datetime] = {}
        
        logger.info("Safety controller initialized with hard limits: "
                   f"{self.config.enforce_hard_limits}")
    
    def validate_price_change(self, product: Product, current_price: Decimal,
                            proposed_price: Decimal, reason: str = "") -> Tuple[bool, List[SafetyViolation]]:
        """Validate a proposed price change against safety constraints."""
        violations = []
        
        # Calculate change percentage
        if current_price > 0:
            change_pct = abs((proposed_price - current_price) / current_price)
        else:
            change_pct = Decimal('1.0')  # 100% change if no current price
        
        # Check single change limit
        if change_pct > self.config.max_single_price_change:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.MAX_SINGLE_CHANGE,
                product=product,
                current_value=change_pct,
                limit_value=self.config.max_single_price_change,
                description=f"Price change {change_pct:.1%} exceeds single change limit",
                severity="high",
                recommended_action="Reduce price change or enable graduated pricing"
            ))
        
        # Check daily cumulative changes
        daily_change = self._calculate_daily_change(product.sku, proposed_price)
        if daily_change > self.config.max_daily_price_change:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.MAX_DAILY_CHANGE,
                product=product,
                current_value=daily_change,
                limit_value=self.config.max_daily_price_change,
                description=f"Daily change {daily_change:.1%} exceeds limit",
                severity="critical",
                recommended_action="Postpone price change to next day"
            ))
        
        # Check velocity limits (changes per hour)
        hourly_changes = self._count_hourly_changes(product.sku)
        if hourly_changes >= self.config.max_hourly_changes:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.VELOCITY_LIMIT,
                product=product,
                current_value=Decimal(str(hourly_changes)),
                limit_value=Decimal(str(self.config.max_hourly_changes)),
                description=f"Too many changes ({hourly_changes}) in the last hour",
                severity="medium",
                recommended_action="Wait before next price change"
            ))
        
        # Check margin constraints
        if hasattr(product, 'cost') and product.cost:
            margin = (proposed_price - product.cost) / proposed_price
            if margin < self.config.min_margin_threshold:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.MARGIN_FLOOR,
                    product=product,
                    current_value=margin,
                    limit_value=self.config.min_margin_threshold,
                    description=f"Margin {margin:.1%} below minimum threshold",
                    severity="high",
                    recommended_action="Increase price to maintain minimum margin"
                ))
        
        # Check inventory constraints
        inventory_violations = self._check_inventory_constraints(product, proposed_price)
        violations.extend(inventory_violations)
        
        # Check if product has emergency override
        is_overridden = self._check_emergency_override(product.sku)
        
        # Determine if change is allowed
        is_allowed = True
        if self.config.enforce_hard_limits and not is_overridden:
            critical_violations = [v for v in violations if v.severity == "critical"]
            if critical_violations:
                is_allowed = False
        
        # Log the validation
        self._log_validation(product, current_price, proposed_price, 
                           violations, is_allowed, reason)
        
        return is_allowed, violations
    
    def check_mape_threshold(self, product_sku: str, mape: Decimal) -> Optional[SafetyViolation]:
        """Check if MAPE exceeds threshold."""
        # Track MAPE history
        if product_sku not in self._performance_metrics:
            self._performance_metrics[product_sku] = []
        
        self._performance_metrics[product_sku].append((datetime.now(), mape))
        
        # Clean old metrics
        cutoff = datetime.now() - self.config.acceptable_mape_duration
        self._performance_metrics[product_sku] = [
            (t, m) for t, m in self._performance_metrics[product_sku]
            if t > cutoff
        ]
        
        # Check if MAPE consistently exceeds threshold
        recent_mapes = [m for _, m in self._performance_metrics[product_sku]]
        if recent_mapes and all(m > self.config.mape_threshold for m in recent_mapes):
            return SafetyViolation(
                violation_type=SafetyViolationType.MAPE_THRESHOLD,
                product=Product(sku=product_sku, name="", market=Market.MASSACHUSETTS),
                current_value=mape,
                limit_value=self.config.mape_threshold,
                description=f"MAPE {mape:.1%} consistently exceeds threshold",
                severity="high",
                recommended_action="Review and retrain pricing model"
            )
        
        return None
    
    def apply_safety_adjustments(self, decision: PricingDecision) -> PricingDecision:
        """Apply safety adjustments to a pricing decision."""
        original_price = decision.final_price
        adjusted_price = original_price
        adjustments = []
        
        # Get current price
        current_price = decision.base_price
        
        # Validate the proposed change
        is_allowed, violations = self.validate_price_change(
            decision.product, current_price, original_price, 
            decision.explanation
        )
        
        if not is_allowed:
            # Calculate safe price
            adjusted_price = self._calculate_safe_price(
                decision.product, current_price, original_price, violations
            )
            adjustments.append(f"Safety limit: {original_price} -> {adjusted_price}")
        
        # Apply inventory-based adjustments
        if decision.product.inventory_level:
            if decision.product.inventory_level < self.config.low_inventory_threshold:
                # Increase price for low inventory
                adjusted_price *= (Decimal('1') + self.config.inventory_price_adjustment)
                adjustments.append("Low inventory premium applied")
            elif decision.product.inventory_level > self.config.high_inventory_threshold:
                # Decrease price for high inventory
                adjusted_price *= (Decimal('1') - self.config.inventory_price_adjustment)
                adjustments.append("High inventory discount applied")
        
        # Update decision if adjusted
        if adjusted_price != original_price:
            decision.final_price = adjusted_price
            decision.add_metadata("safety_adjusted", True)
            decision.add_metadata("original_price", float(original_price))
            decision.add_metadata("safety_adjustments", adjustments)
            decision.explanation += f" [Safety: {', '.join(adjustments)}]"
        
        # Record the price change
        self._record_price_change(
            decision.product.sku, current_price, decision.final_price,
            decision.explanation
        )
        
        return decision
    
    def enable_emergency_override(self, product_sku: str, duration_hours: int = 1,
                                authorized_by: str = "system") -> bool:
        """Enable emergency override for a product."""
        if not self.config.allow_emergency_override:
            logger.warning(f"Emergency override requested but not allowed for {product_sku}")
            return False
        
        self._emergency_overrides.add(product_sku)
        self._override_expiry[product_sku] = datetime.now() + timedelta(hours=duration_hours)
        
        # Audit log
        self.audit_logger.log_emergency_override(
            product_sku=product_sku,
            duration_hours=duration_hours,
            authorized_by=authorized_by,
            reason="Emergency pricing adjustment required"
        )
        
        logger.info(f"Emergency override enabled for {product_sku} for {duration_hours} hours")
        return True
    
    def _calculate_daily_change(self, sku: str, proposed_price: Decimal) -> Decimal:
        """Calculate cumulative daily price change."""
        if sku not in self._daily_changes:
            return Decimal('0')
        
        # Get changes from today
        today = datetime.now().date()
        today_changes = [
            ch for ch in self._daily_changes[sku]
            if ch.timestamp.date() == today
        ]
        
        if not today_changes:
            return Decimal('0')
        
        # Calculate cumulative change from first price of the day
        first_price = today_changes[0].old_price
        if first_price > 0:
            return abs((proposed_price - first_price) / first_price)
        
        return Decimal('0')
    
    def _count_hourly_changes(self, sku: str) -> int:
        """Count price changes in the last hour."""
        if sku not in self._price_history:
            return 0
        
        cutoff = datetime.now() - timedelta(hours=1)
        recent_changes = [
            ch for ch in self._price_history[sku]
            if ch.timestamp > cutoff
        ]
        
        return len(recent_changes)
    
    def _check_inventory_constraints(self, product: Product, 
                                   proposed_price: Decimal) -> List[SafetyViolation]:
        """Check inventory-based pricing constraints."""
        violations = []
        
        if not product.inventory_level:
            return violations
        
        # Low inventory but price decrease
        if (product.inventory_level < self.config.low_inventory_threshold and
            proposed_price < product.current_price):
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.INVENTORY_CONSTRAINT,
                product=product,
                current_value=Decimal(str(product.inventory_level)),
                limit_value=Decimal(str(self.config.low_inventory_threshold)),
                description="Price decrease not recommended for low inventory",
                severity="medium",
                recommended_action="Maintain or increase price due to scarcity"
            ))
        
        # High inventory but price increase
        if (product.inventory_level > self.config.high_inventory_threshold and
            proposed_price > product.current_price):
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.INVENTORY_CONSTRAINT,
                product=product,
                current_value=Decimal(str(product.inventory_level)),
                limit_value=Decimal(str(self.config.high_inventory_threshold)),
                description="Price increase not recommended for high inventory",
                severity="medium",
                recommended_action="Consider price reduction to move inventory"
            ))
        
        return violations
    
    def _check_emergency_override(self, sku: str) -> bool:
        """Check if SKU has active emergency override."""
        if sku not in self._emergency_overrides:
            return False
        
        # Check if override expired
        expiry = self._override_expiry.get(sku)
        if expiry and datetime.now() > expiry:
            self._emergency_overrides.remove(sku)
            del self._override_expiry[sku]
            return False
        
        return True
    
    def _calculate_safe_price(self, product: Product, current_price: Decimal,
                            proposed_price: Decimal, violations: List[SafetyViolation]) -> Decimal:
        """Calculate a safe price that satisfies constraints."""
        safe_price = proposed_price
        
        # Apply constraints based on violations
        for violation in violations:
            if violation.violation_type == SafetyViolationType.MAX_SINGLE_CHANGE:
                # Limit single change
                max_change = current_price * self.config.max_single_price_change
                if proposed_price > current_price:
                    safe_price = min(safe_price, current_price + max_change)
                else:
                    safe_price = max(safe_price, current_price - max_change)
            
            elif violation.violation_type == SafetyViolationType.MAX_DAILY_CHANGE:
                # Limit daily change
                if product.sku in self._daily_changes:
                    today_changes = [ch for ch in self._daily_changes[product.sku]
                                   if ch.timestamp.date() == datetime.now().date()]
                    if today_changes:
                        first_price = today_changes[0].old_price
                        max_price = first_price * (Decimal('1') + self.config.max_daily_price_change)
                        min_price = first_price * (Decimal('1') - self.config.max_daily_price_change)
                        safe_price = max(min_price, min(safe_price, max_price))
            
            elif violation.violation_type == SafetyViolationType.MARGIN_FLOOR:
                # Ensure minimum margin
                if hasattr(product, 'cost') and product.cost:
                    min_price = product.cost / (Decimal('1') - self.config.min_margin_threshold)
                    safe_price = max(safe_price, min_price)
        
        return safe_price
    
    def _record_price_change(self, sku: str, old_price: Decimal, 
                           new_price: Decimal, reason: str):
        """Record a price change in history."""
        if old_price == new_price:
            return
        
        change = PriceChangeHistory(
            timestamp=datetime.now(),
            old_price=old_price,
            new_price=new_price,
            change_percent=abs((new_price - old_price) / old_price) if old_price > 0 else Decimal('0'),
            reason=reason
        )
        
        # Initialize if needed
        if sku not in self._price_history:
            self._price_history[sku] = deque(maxlen=100)
        if sku not in self._daily_changes:
            self._daily_changes[sku] = []
        
        # Add to history
        self._price_history[sku].append(change)
        self._daily_changes[sku].append(change)
        
        # Clean old daily changes
        cutoff = datetime.now() - timedelta(days=7)
        self._daily_changes[sku] = [
            ch for ch in self._daily_changes[sku]
            if ch.timestamp > cutoff
        ]
    
    def _log_validation(self, product: Product, current_price: Decimal,
                      proposed_price: Decimal, violations: List[SafetyViolation],
                      is_allowed: bool, reason: str):
        """Log safety validation results."""
        self.audit_logger.log_safety_check(
            product_sku=product.sku,
            current_price=float(current_price),
            proposed_price=float(proposed_price),
            is_allowed=is_allowed,
            violations=[{
                'type': v.violation_type.value,
                'severity': v.severity,
                'description': v.description,
                'current_value': float(v.current_value),
                'limit_value': float(v.limit_value)
            } for v in violations],
            reason=reason
        )
    
    def get_safety_metrics(self) -> Dict:
        """Get safety controller metrics."""
        total_products = len(self._price_history)
        total_changes = sum(len(history) for history in self._price_history.values())
        
        # Count violations by type
        violation_counts = {vtype.value: 0 for vtype in SafetyViolationType}
        
        return {
            'total_products_monitored': total_products,
            'total_price_changes': total_changes,
            'active_overrides': len(self._emergency_overrides),
            'products_at_daily_limit': sum(
                1 for sku in self._daily_changes
                if self._calculate_daily_change(sku, Decimal('0')) >= self.config.max_daily_price_change * Decimal('0.9')
            ),
            'config': {
                'max_daily_change': float(self.config.max_daily_price_change),
                'max_single_change': float(self.config.max_single_price_change),
                'mape_threshold': float(self.config.mape_threshold),
                'enforce_hard_limits': self.config.enforce_hard_limits
            }
        }