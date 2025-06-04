"""Edge case validation for compliance and safety systems."""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import logging
import random

from ..core.models import Product, Market, ProductCategory
from ..compliance.compliance_engine import ComplianceEngine
from ..safety.safety_controller import SafetyController

logger = logging.getLogger(__name__)


@dataclass
class EdgeCase:
    """Represents an edge case test."""
    case_id: str
    name: str
    description: str
    category: str  # 'boundary', 'null', 'extreme', 'concurrent', 'state'
    test_data: Dict[str, Any]
    expected_behavior: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class EdgeCaseResult:
    """Result of an edge case test."""
    edge_case: EdgeCase
    handled_correctly: bool
    actual_behavior: str
    error_occurred: bool
    error_message: Optional[str] = None
    system_state_stable: bool = True
    performance_impact: Optional[float] = None


class EdgeCaseValidator:
    """Validates system behavior with edge cases."""
    
    def __init__(self, compliance_engine: ComplianceEngine,
                 safety_controller: SafetyController):
        self.compliance_engine = compliance_engine
        self.safety_controller = safety_controller
        self.edge_cases: List[EdgeCase] = []
        self.results: List[EdgeCaseResult] = []
        
        # Initialize edge cases
        self._init_edge_cases()
        
        logger.info(f"Edge case validator initialized with {len(self.edge_cases)} cases")
    
    def validate_all_edge_cases(self) -> Dict[str, Any]:
        """Validate all edge cases."""
        self.results.clear()
        start_time = datetime.now()
        
        for edge_case in self.edge_cases:
            logger.debug(f"Validating edge case: {edge_case.name}")
            result = self._validate_edge_case(edge_case)
            self.results.append(result)
        
        # Generate report
        report = self._generate_validation_report(start_time)
        return report
    
    def validate_category(self, category: str) -> List[EdgeCaseResult]:
        """Validate edge cases of specific category."""
        category_cases = [ec for ec in self.edge_cases if ec.category == category]
        results = []
        
        for edge_case in category_cases:
            result = self._validate_edge_case(edge_case)
            results.append(result)
        
        return results
    
    def add_custom_edge_case(self, edge_case: EdgeCase):
        """Add a custom edge case."""
        self.edge_cases.append(edge_case)
        logger.info(f"Added custom edge case: {edge_case.case_id}")
    
    def _validate_edge_case(self, edge_case: EdgeCase) -> EdgeCaseResult:
        """Validate a single edge case."""
        try:
            if edge_case.category == 'boundary':
                result = self._validate_boundary_case(edge_case)
            elif edge_case.category == 'null':
                result = self._validate_null_case(edge_case)
            elif edge_case.category == 'extreme':
                result = self._validate_extreme_case(edge_case)
            elif edge_case.category == 'concurrent':
                result = self._validate_concurrent_case(edge_case)
            elif edge_case.category == 'state':
                result = self._validate_state_case(edge_case)
            else:
                raise ValueError(f"Unknown edge case category: {edge_case.category}")
            
            return result
            
        except Exception as e:
            logger.error(f"Edge case {edge_case.case_id} raised exception: {str(e)}")
            
            return EdgeCaseResult(
                edge_case=edge_case,
                handled_correctly=False,
                actual_behavior=f"Exception: {type(e).__name__}",
                error_occurred=True,
                error_message=str(e),
                system_state_stable=self._check_system_stability()
            )
    
    def _validate_boundary_case(self, edge_case: EdgeCase) -> EdgeCaseResult:
        """Validate boundary value edge cases."""
        test_data = edge_case.test_data
        
        if 'price' in test_data:
            # Price boundary testing
            product = test_data['product']
            price = Decimal(str(test_data['price']))
            
            # Test compliance
            compliance_result = self.compliance_engine.validate_pricing(product, price)
            
            # Test safety
            if 'current_price' in test_data:
                current = Decimal(str(test_data['current_price']))
                is_allowed, violations = self.safety_controller.validate_price_change(
                    product, current, price, "Edge case test"
                )
            else:
                is_allowed = True
                violations = []
            
            actual_behavior = f"Compliant: {compliance_result.is_compliant}, Allowed: {is_allowed}"
            handled_correctly = self._matches_expected_behavior(
                actual_behavior, edge_case.expected_behavior
            )
            
            return EdgeCaseResult(
                edge_case=edge_case,
                handled_correctly=handled_correctly,
                actual_behavior=actual_behavior,
                error_occurred=False,
                system_state_stable=True
            )
    
    def _validate_null_case(self, edge_case: EdgeCase) -> EdgeCaseResult:
        """Validate null/missing value edge cases."""
        test_data = edge_case.test_data
        
        # Test with None values
        try:
            if test_data.get('test_type') == 'null_product':
                # This should raise an exception
                self.compliance_engine.validate_pricing(None, Decimal('50'))
                actual_behavior = "No exception raised"
            
            elif test_data.get('test_type') == 'null_price':
                product = test_data['product']
                self.compliance_engine.validate_pricing(product, None)
                actual_behavior = "No exception raised"
            
            elif test_data.get('test_type') == 'missing_required_field':
                # Product with missing required fields
                product = test_data['product']
                result = self.compliance_engine.validate_pricing(product, Decimal('50'))
                actual_behavior = f"Validation completed: {result.is_compliant}"
            
        except (AttributeError, TypeError, ValueError) as e:
            actual_behavior = f"Exception raised: {type(e).__name__}"
        
        handled_correctly = self._matches_expected_behavior(
            actual_behavior, edge_case.expected_behavior
        )
        
        return EdgeCaseResult(
            edge_case=edge_case,
            handled_correctly=handled_correctly,
            actual_behavior=actual_behavior,
            error_occurred='Exception' in actual_behavior,
            system_state_stable=True
        )
    
    def _validate_extreme_case(self, edge_case: EdgeCase) -> EdgeCaseResult:
        """Validate extreme value edge cases."""
        test_data = edge_case.test_data
        
        if test_data.get('test_type') == 'extreme_price':
            product = test_data['product']
            price = Decimal(str(test_data['price']))
            
            # Measure performance impact
            start_time = datetime.now()
            result = self.compliance_engine.validate_pricing(product, price)
            duration = (datetime.now() - start_time).total_seconds()
            
            actual_behavior = f"Handled price {price}, compliant: {result.is_compliant}"
            
            return EdgeCaseResult(
                edge_case=edge_case,
                handled_correctly=True,  # If no exception
                actual_behavior=actual_behavior,
                error_occurred=False,
                performance_impact=duration
            )
        
        elif test_data.get('test_type') == 'extreme_volume':
            # Test with large number of products
            num_products = test_data['num_products']
            products = self._generate_products(num_products)
            
            start_time = datetime.now()
            results = self.compliance_engine.batch_validate([
                (p, p.current_price * Decimal('1.05'))
                for p in products
            ])
            duration = (datetime.now() - start_time).total_seconds()
            
            actual_behavior = f"Processed {num_products} products in {duration:.2f}s"
            
            return EdgeCaseResult(
                edge_case=edge_case,
                handled_correctly=len(results) == num_products,
                actual_behavior=actual_behavior,
                error_occurred=False,
                performance_impact=duration
            )
    
    def _validate_concurrent_case(self, edge_case: EdgeCase) -> EdgeCaseResult:
        """Validate concurrent access edge cases."""
        test_data = edge_case.test_data
        
        if test_data.get('test_type') == 'concurrent_modifications':
            # Simulate concurrent price changes
            product = test_data['product']
            
            # This would need actual threading in production
            # For now, simulate sequential rapid changes
            changes_applied = 0
            errors = 0
            
            for i in range(test_data['num_concurrent']):
                try:
                    new_price = product.current_price * Decimal(str(1 + i * 0.01))
                    is_allowed, _ = self.safety_controller.validate_price_change(
                        product, product.current_price, new_price, f"Concurrent {i}"
                    )
                    if is_allowed:
                        product.current_price = new_price
                        changes_applied += 1
                except Exception:
                    errors += 1
            
            actual_behavior = f"Applied {changes_applied}/{test_data['num_concurrent']} changes, {errors} errors"
            
            return EdgeCaseResult(
                edge_case=edge_case,
                handled_correctly=errors == 0,
                actual_behavior=actual_behavior,
                error_occurred=errors > 0,
                system_state_stable=True
            )
    
    def _validate_state_case(self, edge_case: EdgeCase) -> EdgeCaseResult:
        """Validate system state edge cases."""
        test_data = edge_case.test_data
        
        if test_data.get('test_type') == 'invalid_state_transition':
            # Test invalid state transitions
            # For example, rollback without snapshot
            try:
                # This should fail gracefully
                from ..safety.rollback_manager import RollbackManager, RollbackReason
                rollback_mgr = RollbackManager()
                
                # Try to rollback without any snapshots
                products = [test_data['product']]
                plan = rollback_mgr.create_rollback_plan(
                    products, RollbackReason.MANUAL_REQUEST, hours_ago=24
                )
                actual_behavior = "Rollback plan created (unexpected)"
                
            except ValueError as e:
                actual_behavior = f"Correctly rejected: {str(e)}"
            
            handled_correctly = 'Correctly rejected' in actual_behavior
            
            return EdgeCaseResult(
                edge_case=edge_case,
                handled_correctly=handled_correctly,
                actual_behavior=actual_behavior,
                error_occurred=False,
                system_state_stable=True
            )
    
    def _init_edge_cases(self):
        """Initialize standard edge cases."""
        
        # Boundary cases
        self.edge_cases.extend([
            EdgeCase(
                case_id="BOUNDARY_001",
                name="Zero Price",
                description="Test handling of zero price",
                category="boundary",
                test_data={
                    'product': Product(
                        sku="EDGE_ZERO",
                        name="Zero Price Test",
                        market=Market.MASSACHUSETTS,
                        category=ProductCategory.FLOWER,
                        current_price=Decimal("50.00")
                    ),
                    'price': 0
                },
                expected_behavior="Compliant: False",
                risk_level="high"
            ),
            
            EdgeCase(
                case_id="BOUNDARY_002",
                name="Maximum Price",
                description="Test extremely high price",
                category="boundary",
                test_data={
                    'product': Product(
                        sku="EDGE_MAX",
                        name="Max Price Test",
                        market=Market.MASSACHUSETTS,
                        category=ProductCategory.FLOWER,
                        current_price=Decimal("50.00")
                    ),
                    'price': 999999.99
                },
                expected_behavior="Compliant: False",
                risk_level="medium"
            ),
            
            EdgeCase(
                case_id="BOUNDARY_003",
                name="Minimum Margin",
                description="Test price at exact minimum margin",
                category="boundary",
                test_data={
                    'product': Product(
                        sku="EDGE_MARGIN",
                        name="Min Margin Test",
                        market=Market.RHODE_ISLAND,
                        category=ProductCategory.EDIBLES,
                        current_price=Decimal("20.00"),
                        cost=Decimal("17.00")  # 15% margin exactly
                    ),
                    'price': 20.00
                },
                expected_behavior="Compliant: True",
                risk_level="low"
            )
        ])
        
        # Null/missing cases
        self.edge_cases.extend([
            EdgeCase(
                case_id="NULL_001",
                name="Null Product",
                description="Test with null product",
                category="null",
                test_data={
                    'test_type': 'null_product'
                },
                expected_behavior="Exception raised: AttributeError",
                risk_level="critical"
            ),
            
            EdgeCase(
                case_id="NULL_002",
                name="Null Price",
                description="Test with null price",
                category="null",
                test_data={
                    'test_type': 'null_price',
                    'product': Product(
                        sku="EDGE_NULL_PRICE",
                        name="Null Price Test",
                        market=Market.MASSACHUSETTS,
                        category=ProductCategory.VAPES,
                        current_price=Decimal("75.00")
                    )
                },
                expected_behavior="Exception raised: TypeError",
                risk_level="critical"
            ),
            
            EdgeCase(
                case_id="NULL_003",
                name="Missing Cost",
                description="Test product without cost field",
                category="null",
                test_data={
                    'test_type': 'missing_required_field',
                    'product': Product(
                        sku="EDGE_NO_COST",
                        name="No Cost Test",
                        market=Market.RHODE_ISLAND,
                        category=ProductCategory.CONCENTRATES,
                        current_price=Decimal("40.00")
                        # cost field missing
                    )
                },
                expected_behavior="Validation completed",
                risk_level="medium"
            )
        ])
        
        # Extreme cases
        self.edge_cases.extend([
            EdgeCase(
                case_id="EXTREME_001",
                name="Very Large Price",
                description="Test with extremely large price value",
                category="extreme",
                test_data={
                    'test_type': 'extreme_price',
                    'product': Product(
                        sku="EDGE_EXTREME_PRICE",
                        name="Extreme Price Test",
                        market=Market.MASSACHUSETTS,
                        category=ProductCategory.FLOWER,
                        current_price=Decimal("50.00")
                    ),
                    'price': Decimal('1000000000.00')  # 1 billion
                },
                expected_behavior="Handled price",
                risk_level="medium"
            ),
            
            EdgeCase(
                case_id="EXTREME_002",
                name="High Volume Batch",
                description="Test with large batch processing",
                category="extreme",
                test_data={
                    'test_type': 'extreme_volume',
                    'num_products': 10000
                },
                expected_behavior="Processed 10000 products",
                risk_level="high"
            ),
            
            EdgeCase(
                case_id="EXTREME_003",
                name="Rapid Price Changes",
                description="Test rapid sequential price changes",
                category="extreme",
                test_data={
                    'test_type': 'rapid_changes',
                    'product': Product(
                        sku="EDGE_RAPID",
                        name="Rapid Change Test",
                        market=Market.MASSACHUSETTS,
                        category=ProductCategory.EDIBLES,
                        current_price=Decimal("30.00")
                    ),
                    'num_changes': 100,
                    'change_interval_ms': 10
                },
                expected_behavior="Rate limited after threshold",
                risk_level="medium"
            )
        ])
        
        # Concurrent access cases
        self.edge_cases.extend([
            EdgeCase(
                case_id="CONCURRENT_001",
                name="Concurrent Price Updates",
                description="Test concurrent price modifications",
                category="concurrent",
                test_data={
                    'test_type': 'concurrent_modifications',
                    'product': Product(
                        sku="EDGE_CONCURRENT",
                        name="Concurrent Test",
                        market=Market.RHODE_ISLAND,
                        category=ProductCategory.FLOWER,
                        current_price=Decimal("60.00")
                    ),
                    'num_concurrent': 10
                },
                expected_behavior="Applied",
                risk_level="high"
            )
        ])
        
        # State cases
        self.edge_cases.extend([
            EdgeCase(
                case_id="STATE_001",
                name="Invalid Rollback",
                description="Test rollback without valid snapshot",
                category="state",
                test_data={
                    'test_type': 'invalid_state_transition',
                    'product': Product(
                        sku="EDGE_STATE",
                        name="State Test",
                        market=Market.MASSACHUSETTS,
                        category=ProductCategory.VAPES,
                        current_price=Decimal("80.00")
                    )
                },
                expected_behavior="Correctly rejected",
                risk_level="medium"
            )
        ])
    
    def _generate_products(self, count: int) -> List[Product]:
        """Generate test products."""
        products = []
        for i in range(count):
            products.append(Product(
                sku=f"EDGE_GEN_{i:06d}",
                name=f"Generated Product {i}",
                market=random.choice(list(Market)),
                category=random.choice(list(ProductCategory)),
                current_price=Decimal(str(random.uniform(10, 200))),
                cost=Decimal(str(random.uniform(5, 100)))
            ))
        return products
    
    def _matches_expected_behavior(self, actual: str, expected: str) -> bool:
        """Check if actual behavior matches expected."""
        # Simple string matching for now
        # Could be made more sophisticated
        return expected.lower() in actual.lower()
    
    def _check_system_stability(self) -> bool:
        """Check if system is still stable after edge case."""
        # Basic stability check
        try:
            # Try a simple operation
            test_product = Product(
                sku="STABILITY_CHECK",
                name="Stability Check",
                market=Market.MASSACHUSETTS,
                category=ProductCategory.FLOWER,
                current_price=Decimal("50.00")
            )
            
            result = self.compliance_engine.validate_pricing(
                test_product, Decimal("55.00")
            )
            
            return result is not None
            
        except Exception:
            return False
    
    def _generate_validation_report(self, start_time: datetime) -> Dict[str, Any]:
        """Generate edge case validation report."""
        total_cases = len(self.results)
        handled_correctly = sum(1 for r in self.results if r.handled_correctly)
        errors = sum(1 for r in self.results if r.error_occurred)
        unstable = sum(1 for r in self.results if not r.system_state_stable)
        
        # Group by category
        by_category = {}
        for result in self.results:
            category = result.edge_case.category
            if category not in by_category:
                by_category[category] = {'total': 0, 'passed': 0, 'failed': 0}
            
            by_category[category]['total'] += 1
            if result.handled_correctly:
                by_category[category]['passed'] += 1
            else:
                by_category[category]['failed'] += 1
        
        # Find critical failures
        critical_failures = [
            r for r in self.results
            if not r.handled_correctly and r.edge_case.risk_level == 'critical'
        ]
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'summary': {
                'total_cases': total_cases,
                'handled_correctly': handled_correctly,
                'errors': errors,
                'system_unstable': unstable,
                'success_rate': handled_correctly / total_cases if total_cases > 0 else 0,
                'duration': duration
            },
            'by_category': by_category,
            'critical_failures': [
                {
                    'case_id': r.edge_case.case_id,
                    'name': r.edge_case.name,
                    'expected': r.edge_case.expected_behavior,
                    'actual': r.actual_behavior,
                    'error': r.error_message
                }
                for r in critical_failures
            ],
            'performance_impact': {
                'cases_with_impact': sum(1 for r in self.results if r.performance_impact),
                'max_impact': max(
                    (r.performance_impact for r in self.results if r.performance_impact),
                    default=0
                )
            }
        }