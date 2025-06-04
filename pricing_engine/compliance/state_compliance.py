"""State-specific compliance rules for Massachusetts and Rhode Island."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import logging

from ..core.models import Product, Market, ProductCategory

logger = logging.getLogger(__name__)


@dataclass
class ComplianceRule:
    """Represents a single compliance rule."""
    rule_id: str
    description: str
    category: str
    severity: str  # 'critical', 'warning', 'info'
    
    
@dataclass
class ComplianceViolation:
    """Details of a compliance violation."""
    rule: ComplianceRule
    product: Product
    violation_details: str
    suggested_price: Optional[Decimal] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StateCompliance(ABC):
    """Base class for state-specific compliance engines."""
    
    def __init__(self):
        self.rules: List[ComplianceRule] = []
        self._load_rules()
    
    @abstractmethod
    def _load_rules(self):
        """Load state-specific rules."""
        pass
    
    @abstractmethod
    def validate_price(self, product: Product, proposed_price: Decimal) -> List[ComplianceViolation]:
        """Validate a proposed price against state rules."""
        pass
    
    @abstractmethod
    def calculate_taxes(self, product: Product, price: Decimal) -> Dict[str, Decimal]:
        """Calculate state and local taxes."""
        pass
    
    @abstractmethod
    def get_price_constraints(self, product: Product) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Get min/max price constraints for a product."""
        pass
    
    @abstractmethod
    def validate_promotional_pricing(self, product: Product, promo_price: Decimal, 
                                   regular_price: Decimal) -> List[ComplianceViolation]:
        """Validate promotional pricing rules."""
        pass


class MassachusettsCompliance(StateCompliance):
    """Massachusetts-specific compliance rules and regulations."""
    
    # MA Cannabis Tax Rates
    MA_EXCISE_TAX = Decimal('0.1075')  # 10.75% excise tax
    MA_STATE_TAX = Decimal('0.0625')   # 6.25% state tax
    MA_LOCAL_TAX_MAX = Decimal('0.03') # Up to 3% local tax
    
    # MA Pricing Constraints
    MAX_THC_PRICE_PER_MG = Decimal('2.00')  # Example constraint
    MIN_PRICE_PER_GRAM = Decimal('5.00')    # Minimum pricing to prevent loss leaders
    
    def _load_rules(self):
        """Load Massachusetts-specific compliance rules."""
        self.rules = [
            ComplianceRule(
                rule_id="MA001",
                description="Cannabis excise tax must be included in pricing calculations",
                category="tax",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="MA002",
                description="Price per mg of THC cannot exceed state maximum",
                category="pricing",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="MA003",
                description="Promotional discounts cannot exceed 30% of regular price",
                category="promotion",
                severity="warning"
            ),
            ComplianceRule(
                rule_id="MA004",
                description="Medical cannabis products must have lower tax rate",
                category="tax",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="MA005",
                description="No sales before 8:00 AM or after 10:00 PM",
                category="operational",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="MA006",
                description="Bulk discounts must be documented and justified",
                category="pricing",
                severity="warning"
            ),
            ComplianceRule(
                rule_id="MA007",
                description="Price changes must be reported to state tracking system",
                category="reporting",
                severity="critical"
            )
        ]
    
    def validate_price(self, product: Product, proposed_price: Decimal) -> List[ComplianceViolation]:
        """Validate proposed price against MA regulations."""
        violations = []
        
        # Check minimum price per gram
        if product.unit == "gram" and proposed_price < self.MIN_PRICE_PER_GRAM:
            violations.append(ComplianceViolation(
                rule=self._get_rule("MA002"),
                product=product,
                violation_details=f"Price ${proposed_price} below minimum ${self.MIN_PRICE_PER_GRAM}/gram",
                suggested_price=self.MIN_PRICE_PER_GRAM
            ))
        
        # Check THC pricing limits (if THC content is available)
        if hasattr(product, 'thc_mg') and product.thc_mg:
            price_per_mg = proposed_price / Decimal(str(product.thc_mg))
            if price_per_mg > self.MAX_THC_PRICE_PER_MG:
                violations.append(ComplianceViolation(
                    rule=self._get_rule("MA002"),
                    product=product,
                    violation_details=f"Price per mg THC ${price_per_mg} exceeds maximum ${self.MAX_THC_PRICE_PER_MG}",
                    suggested_price=Decimal(str(product.thc_mg)) * self.MAX_THC_PRICE_PER_MG
                ))
        
        # Check operating hours
        current_time = datetime.now().time()
        if current_time < time(8, 0) or current_time > time(22, 0):
            violations.append(ComplianceViolation(
                rule=self._get_rule("MA005"),
                product=product,
                violation_details="Price changes not allowed outside operating hours (8AM-10PM)"
            ))
        
        return violations
    
    def calculate_taxes(self, product: Product, price: Decimal) -> Dict[str, Decimal]:
        """Calculate MA cannabis taxes."""
        taxes = {}
        
        # Medical vs recreational tax rates
        if product.category == ProductCategory.MEDICAL:
            # Medical cannabis is exempt from excise tax in MA
            taxes['state_tax'] = price * self.MA_STATE_TAX
            taxes['excise_tax'] = Decimal('0')
        else:
            taxes['state_tax'] = price * self.MA_STATE_TAX
            taxes['excise_tax'] = price * self.MA_EXCISE_TAX
        
        # Local tax (varies by municipality, using max for safety)
        taxes['local_tax'] = price * self.MA_LOCAL_TAX_MAX
        
        taxes['total_tax'] = sum(taxes.values())
        taxes['total_with_tax'] = price + taxes['total_tax']
        
        return taxes
    
    def get_price_constraints(self, product: Product) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Get MA-specific price constraints."""
        min_price = None
        max_price = None
        
        # Minimum pricing
        if product.unit == "gram":
            min_price = self.MIN_PRICE_PER_GRAM
        elif product.unit == "each":
            min_price = Decimal('10.00')  # Minimum for individual items
        
        # Maximum pricing based on THC content
        if hasattr(product, 'thc_mg') and product.thc_mg:
            max_price = Decimal(str(product.thc_mg)) * self.MAX_THC_PRICE_PER_MG
        
        return min_price, max_price
    
    def validate_promotional_pricing(self, product: Product, promo_price: Decimal, 
                                   regular_price: Decimal) -> List[ComplianceViolation]:
        """Validate promotional pricing against MA rules."""
        violations = []
        
        # Calculate discount percentage
        discount_pct = (regular_price - promo_price) / regular_price
        
        # MA limits promotional discounts to 30%
        if discount_pct > Decimal('0.30'):
            violations.append(ComplianceViolation(
                rule=self._get_rule("MA003"),
                product=product,
                violation_details=f"Discount {discount_pct:.1%} exceeds 30% maximum",
                suggested_price=regular_price * Decimal('0.70')
            ))
        
        # Still must meet minimum pricing
        base_violations = self.validate_price(product, promo_price)
        violations.extend(base_violations)
        
        return violations
    
    def _get_rule(self, rule_id: str) -> ComplianceRule:
        """Get rule by ID."""
        return next((r for r in self.rules if r.rule_id == rule_id), None)


class RhodeIslandCompliance(StateCompliance):
    """Rhode Island-specific compliance rules and regulations."""
    
    # RI Cannabis Tax Rates
    RI_EXCISE_TAX = Decimal('0.10')    # 10% excise tax
    RI_STATE_TAX = Decimal('0.07')     # 7% state tax
    RI_LOCAL_TAX = Decimal('0.03')     # 3% local tax
    RI_WEIGHT_TAX = Decimal('0.0125')  # $0.0125 per gram additional
    
    # RI Pricing Constraints
    MAX_DISCOUNT_MEDICAL = Decimal('0.20')   # 20% max discount for medical
    MAX_DISCOUNT_REC = Decimal('0.25')       # 25% max discount for recreational
    MIN_MARGIN_REQUIREMENT = Decimal('0.15') # 15% minimum margin
    
    def _load_rules(self):
        """Load Rhode Island-specific compliance rules."""
        self.rules = [
            ComplianceRule(
                rule_id="RI001",
                description="Weight-based tax must be added per gram",
                category="tax",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="RI002",
                description="Medical cannabis discounts limited to 20%",
                category="promotion",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="RI003",
                description="Recreational cannabis discounts limited to 25%",
                category="promotion",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="RI004",
                description="Minimum 15% profit margin required",
                category="pricing",
                severity="warning"
            ),
            ComplianceRule(
                rule_id="RI005",
                description="Price matching competitors requires documentation",
                category="pricing",
                severity="info"
            ),
            ComplianceRule(
                rule_id="RI006",
                description="Compassion center pricing must be lower than retail",
                category="pricing",
                severity="critical"
            ),
            ComplianceRule(
                rule_id="RI007",
                description="Daily price changes limited to twice per product",
                category="operational",
                severity="warning"
            )
        ]
    
    def validate_price(self, product: Product, proposed_price: Decimal) -> List[ComplianceViolation]:
        """Validate proposed price against RI regulations."""
        violations = []
        
        # Check minimum margin requirement
        if hasattr(product, 'cost') and product.cost:
            margin = (proposed_price - product.cost) / proposed_price
            if margin < self.MIN_MARGIN_REQUIREMENT:
                violations.append(ComplianceViolation(
                    rule=self._get_rule("RI004"),
                    product=product,
                    violation_details=f"Margin {margin:.1%} below required 15%",
                    suggested_price=product.cost / (Decimal('1') - self.MIN_MARGIN_REQUIREMENT)
                ))
        
        # Check compassion center pricing
        if product.market == Market.RHODE_ISLAND and hasattr(product, 'dispensary_type'):
            if product.dispensary_type == 'compassion_center':
                # Should be lower than average retail price
                # This would need actual market data in production
                pass
        
        return violations
    
    def calculate_taxes(self, product: Product, price: Decimal) -> Dict[str, Decimal]:
        """Calculate RI cannabis taxes."""
        taxes = {}
        
        # Standard percentage-based taxes
        taxes['state_tax'] = price * self.RI_STATE_TAX
        taxes['excise_tax'] = price * self.RI_EXCISE_TAX
        taxes['local_tax'] = price * self.RI_LOCAL_TAX
        
        # Weight-based tax (convert to grams if needed)
        weight_in_grams = self._convert_to_grams(product)
        taxes['weight_tax'] = weight_in_grams * self.RI_WEIGHT_TAX
        
        taxes['total_tax'] = sum(taxes.values())
        taxes['total_with_tax'] = price + taxes['total_tax']
        
        return taxes
    
    def get_price_constraints(self, product: Product) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Get RI-specific price constraints."""
        min_price = None
        max_price = None
        
        # Minimum based on cost + margin
        if hasattr(product, 'cost') and product.cost:
            min_price = product.cost / (Decimal('1') - self.MIN_MARGIN_REQUIREMENT)
        
        # No specific maximum in RI, market-driven
        return min_price, max_price
    
    def validate_promotional_pricing(self, product: Product, promo_price: Decimal, 
                                   regular_price: Decimal) -> List[ComplianceViolation]:
        """Validate promotional pricing against RI rules."""
        violations = []
        
        # Calculate discount percentage
        discount_pct = (regular_price - promo_price) / regular_price
        
        # Different limits for medical vs recreational
        if product.category == ProductCategory.MEDICAL:
            if discount_pct > self.MAX_DISCOUNT_MEDICAL:
                violations.append(ComplianceViolation(
                    rule=self._get_rule("RI002"),
                    product=product,
                    violation_details=f"Medical discount {discount_pct:.1%} exceeds 20% maximum",
                    suggested_price=regular_price * (Decimal('1') - self.MAX_DISCOUNT_MEDICAL)
                ))
        else:
            if discount_pct > self.MAX_DISCOUNT_REC:
                violations.append(ComplianceViolation(
                    rule=self._get_rule("RI003"),
                    product=product,
                    violation_details=f"Recreational discount {discount_pct:.1%} exceeds 25% maximum",
                    suggested_price=regular_price * (Decimal('1') - self.MAX_DISCOUNT_REC)
                ))
        
        # Still must meet margin requirements
        base_violations = self.validate_price(product, promo_price)
        violations.extend(base_violations)
        
        return violations
    
    def _get_rule(self, rule_id: str) -> ComplianceRule:
        """Get rule by ID."""
        return next((r for r in self.rules if r.rule_id == rule_id), None)
    
    def _convert_to_grams(self, product: Product) -> Decimal:
        """Convert product weight to grams for tax calculation."""
        if product.unit == "gram":
            return Decimal(str(product.size))
        elif product.unit == "ounce":
            return Decimal(str(product.size)) * Decimal('28.3495')
        elif product.unit == "each":
            # Estimate weight for edibles/pre-rolls
            return Decimal('1.0')  # Default 1g for individual items
        else:
            return Decimal('1.0')  # Default