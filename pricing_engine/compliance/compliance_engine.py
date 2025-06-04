"""Master compliance engine coordinating all compliance checks."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging
from collections import defaultdict

from .state_compliance import (
    StateCompliance, MassachusettsCompliance, RhodeIslandCompliance,
    ComplianceViolation, ComplianceRule
)
from ..core.models import Product, Market
from ..utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check."""
    product: Product
    proposed_price: Decimal
    is_compliant: bool
    violations: List[ComplianceViolation] = field(default_factory=list)
    warnings: List[ComplianceViolation] = field(default_factory=list)
    tax_calculation: Dict[str, Decimal] = field(default_factory=dict)
    adjusted_price: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations."""
        return any(v.rule.severity == 'critical' for v in self.violations)
    
    @property
    def violation_summary(self) -> str:
        """Get a summary of violations."""
        if not self.violations:
            return "No violations found"
        
        by_severity = defaultdict(int)
        for v in self.violations:
            by_severity[v.rule.severity] += 1
        
        parts = []
        for severity in ['critical', 'warning', 'info']:
            if by_severity[severity] > 0:
                parts.append(f"{by_severity[severity]} {severity}")
        
        return ", ".join(parts)


@dataclass
class MunicipalOverride:
    """Municipal-specific compliance overrides."""
    municipality: str
    state: str
    local_tax_rate: Optional[Decimal] = None
    additional_rules: List[ComplianceRule] = field(default_factory=list)
    operating_hours: Optional[Tuple[datetime.time, datetime.time]] = None
    max_discount_override: Optional[Decimal] = None


class ComplianceEngine:
    """Master compliance engine for multi-state operations."""
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.state_engines: Dict[Market, StateCompliance] = {
            Market.MASSACHUSETTS: MassachusettsCompliance(),
            Market.RHODE_ISLAND: RhodeIslandCompliance()
        }
        self.municipal_overrides: Dict[str, MunicipalOverride] = {}
        self.audit_logger = audit_logger or AuditLogger()
        self._price_change_history: Dict[str, List[datetime]] = defaultdict(list)
        
        # Configuration
        self.max_daily_price_changes = 2  # RI rule
        self.require_audit_trail = True
        
        logger.info("Compliance engine initialized for MA and RI markets")
    
    def validate_pricing(self, product: Product, proposed_price: Decimal,
                        current_price: Optional[Decimal] = None,
                        is_promotional: bool = False) -> ComplianceCheckResult:
        """Comprehensive pricing validation."""
        
        # Get appropriate state engine
        state_engine = self.state_engines.get(product.market)
        if not state_engine:
            raise ValueError(f"No compliance engine for market: {product.market}")
        
        result = ComplianceCheckResult(
            product=product,
            proposed_price=proposed_price,
            is_compliant=True
        )
        
        # Basic price validation
        violations = state_engine.validate_price(product, proposed_price)
        
        # Promotional pricing validation
        if is_promotional and current_price:
            promo_violations = state_engine.validate_promotional_pricing(
                product, proposed_price, current_price
            )
            violations.extend(promo_violations)
        
        # Check daily price change limits
        daily_violations = self._check_daily_price_changes(product)
        violations.extend(daily_violations)
        
        # Apply municipal overrides
        if product.location:
            municipal_violations = self._apply_municipal_rules(product, proposed_price)
            violations.extend(municipal_violations)
        
        # Calculate taxes
        result.tax_calculation = state_engine.calculate_taxes(product, proposed_price)
        
        # Separate violations by severity
        for violation in violations:
            if violation.rule.severity == 'critical':
                result.violations.append(violation)
                result.is_compliant = False
            else:
                result.warnings.append(violation)
        
        # Suggest adjusted price if needed
        if not result.is_compliant:
            result.adjusted_price = self._calculate_compliant_price(
                product, proposed_price, violations
            )
        
        # Audit logging
        if self.require_audit_trail:
            self._log_compliance_check(result)
        
        return result
    
    def batch_validate(self, pricing_updates: List[Tuple[Product, Decimal]]) -> Dict[str, ComplianceCheckResult]:
        """Validate multiple pricing updates at once."""
        results = {}
        
        for product, proposed_price in pricing_updates:
            try:
                result = self.validate_pricing(product, proposed_price)
                results[product.sku] = result
            except Exception as e:
                logger.error(f"Compliance check failed for {product.sku}: {str(e)}")
                # Create failed result
                results[product.sku] = ComplianceCheckResult(
                    product=product,
                    proposed_price=proposed_price,
                    is_compliant=False,
                    violations=[ComplianceViolation(
                        rule=ComplianceRule("SYS001", "System error during validation", 
                                          "system", "critical"),
                        product=product,
                        violation_details=str(e)
                    )]
                )
        
        return results
    
    def get_price_boundaries(self, product: Product) -> Tuple[Decimal, Decimal]:
        """Get compliant price boundaries for a product."""
        state_engine = self.state_engines.get(product.market)
        if not state_engine:
            raise ValueError(f"No compliance engine for market: {product.market}")
        
        min_price, max_price = state_engine.get_price_constraints(product)
        
        # Apply municipal overrides
        if product.location and product.location in self.municipal_overrides:
            override = self.municipal_overrides[product.location]
            # Municipal rules might further restrict pricing
            # Implementation depends on specific municipal requirements
        
        # Ensure boundaries are valid
        if min_price is None:
            min_price = Decimal('0.01')
        if max_price is None:
            max_price = Decimal('9999.99')
        
        return min_price, max_price
    
    def add_municipal_override(self, override: MunicipalOverride):
        """Add municipal-specific compliance rules."""
        key = f"{override.state}:{override.municipality}"
        self.municipal_overrides[key] = override
        logger.info(f"Added municipal override for {key}")
    
    def _check_daily_price_changes(self, product: Product) -> List[ComplianceViolation]:
        """Check daily price change limits."""
        violations = []
        
        # Track price changes
        today = datetime.now().date()
        change_times = self._price_change_history.get(product.sku, [])
        today_changes = [t for t in change_times if t.date() == today]
        
        # RI specific rule: max 2 price changes per day
        if product.market == Market.RHODE_ISLAND and len(today_changes) >= self.max_daily_price_changes:
            violations.append(ComplianceViolation(
                rule=ComplianceRule("RI007", "Daily price changes limited to twice per product",
                                  "operational", "warning"),
                product=product,
                violation_details=f"Already changed price {len(today_changes)} times today"
            ))
        
        return violations
    
    def _apply_municipal_rules(self, product: Product, proposed_price: Decimal) -> List[ComplianceViolation]:
        """Apply municipal-specific rules."""
        violations = []
        
        key = f"{product.market.value}:{product.location}"
        override = self.municipal_overrides.get(key)
        
        if override:
            # Check operating hours
            if override.operating_hours:
                current_time = datetime.now().time()
                start, end = override.operating_hours
                if not (start <= current_time <= end):
                    violations.append(ComplianceViolation(
                        rule=ComplianceRule("MUN001", f"{override.municipality} operating hours",
                                          "operational", "critical"),
                        product=product,
                        violation_details=f"Outside operating hours {start}-{end}"
                    ))
            
            # Apply additional municipal rules
            for rule in override.additional_rules:
                # Custom rule validation logic would go here
                pass
        
        return violations
    
    def _calculate_compliant_price(self, product: Product, proposed_price: Decimal,
                                 violations: List[ComplianceViolation]) -> Decimal:
        """Calculate a compliant price based on violations."""
        # Start with proposed price
        adjusted_price = proposed_price
        
        # Apply suggested prices from violations
        for violation in violations:
            if violation.suggested_price:
                if violation.rule.severity == 'critical':
                    # For critical violations, must use suggested price
                    adjusted_price = violation.suggested_price
                    break
                else:
                    # For warnings, take the most conservative price
                    adjusted_price = max(adjusted_price, violation.suggested_price)
        
        # Ensure within boundaries
        min_price, max_price = self.get_price_boundaries(product)
        adjusted_price = max(min_price, min(adjusted_price, max_price))
        
        return adjusted_price
    
    def _log_compliance_check(self, result: ComplianceCheckResult):
        """Log compliance check to audit trail."""
        self.audit_logger.log_compliance_check(
            product_sku=result.product.sku,
            proposed_price=float(result.proposed_price),
            is_compliant=result.is_compliant,
            violations=[{
                'rule_id': v.rule.rule_id,
                'description': v.rule.description,
                'details': v.violation_details,
                'severity': v.rule.severity
            } for v in result.violations],
            warnings=[{
                'rule_id': w.rule.rule_id,
                'description': w.rule.description,
                'details': w.violation_details,
                'severity': w.rule.severity
            } for w in result.warnings],
            adjusted_price=float(result.adjusted_price) if result.adjusted_price else None,
            tax_calculation=result.tax_calculation
        )
    
    def record_price_change(self, product: Product):
        """Record that a price change occurred."""
        self._price_change_history[product.sku].append(datetime.now())
        
        # Clean up old history (keep only last 7 days)
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - 7)
        
        self._price_change_history[product.sku] = [
            t for t in self._price_change_history[product.sku]
            if t > cutoff
        ]
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate compliance report for date range."""
        # This would query the audit logs for compliance data
        return self.audit_logger.generate_compliance_report(start_date, end_date)