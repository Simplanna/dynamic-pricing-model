"""Comprehensive test suite for compliance validation."""

from dataclasses import dataclass, field
from datetime import datetime, time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json

from ..core.models import Product, Market, ProductCategory
from ..compliance.compliance_engine import ComplianceEngine, ComplianceCheckResult
from ..compliance.compliance_validator import ComplianceValidator, ValidationMode

logger = logging.getLogger(__name__)


@dataclass
class ComplianceTestCase:
    """A single compliance test case."""
    test_id: str
    name: str
    description: str
    product: Product
    proposed_price: Decimal
    expected_compliant: bool
    expected_violations: List[str]
    test_conditions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'description': self.description,
            'product_sku': self.product.sku,
            'market': self.product.market.value,
            'proposed_price': str(self.proposed_price),
            'expected_compliant': self.expected_compliant,
            'expected_violations': self.expected_violations,
            'test_conditions': self.test_conditions
        }


@dataclass
class ComplianceTestResult:
    """Result of a compliance test."""
    test_case: ComplianceTestCase
    passed: bool
    actual_compliant: bool
    actual_violations: List[str]
    compliance_result: Optional[ComplianceCheckResult]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    @property
    def violation_match(self) -> bool:
        """Check if violations match expected."""
        return set(self.actual_violations) == set(self.test_case.expected_violations)


class ComplianceTestSuite:
    """Comprehensive test suite for compliance validation."""
    
    def __init__(self, compliance_engine: ComplianceEngine):
        self.compliance_engine = compliance_engine
        self.test_cases: List[ComplianceTestCase] = []
        self.test_results: List[ComplianceTestResult] = []
        
        # Initialize standard test cases
        self._init_standard_tests()
        
        logger.info(f"Compliance test suite initialized with {len(self.test_cases)} test cases")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all compliance test cases."""
        self.test_results.clear()
        start_time = datetime.now()
        
        for test_case in self.test_cases:
            result = self._run_single_test(test_case)
            self.test_results.append(result)
        
        # Calculate summary
        total_time = (datetime.now() - start_time).total_seconds()
        summary = self._generate_summary(total_time)
        
        return summary
    
    def run_market_tests(self, market: Market) -> Dict[str, Any]:
        """Run tests specific to a market."""
        market_tests = [tc for tc in self.test_cases if tc.product.market == market]
        results = []
        
        for test_case in market_tests:
            result = self._run_single_test(test_case)
            results.append(result)
        
        # Generate market-specific summary
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        return {
            'market': market.value,
            'total_tests': len(results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(results) if results else 0,
            'failed_tests': [r.test_case.test_id for r in results if not r.passed]
        }
    
    def run_violation_type_tests(self, violation_type: str) -> List[ComplianceTestResult]:
        """Run tests for specific violation types."""
        relevant_tests = [
            tc for tc in self.test_cases
            if violation_type in tc.expected_violations
        ]
        
        results = []
        for test_case in relevant_tests:
            result = self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    def add_custom_test(self, test_case: ComplianceTestCase):
        """Add a custom test case."""
        self.test_cases.append(test_case)
        logger.info(f"Added custom test case: {test_case.test_id}")
    
    def export_test_results(self, output_path: Path) -> Dict[str, Any]:
        """Export test results to file."""
        export_data = {
            'test_run': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.test_results),
                'passed': sum(1 for r in self.test_results if r.passed),
                'failed': sum(1 for r in self.test_results if not r.passed)
            },
            'test_cases': [tc.to_dict() for tc in self.test_cases],
            'results': [
                {
                    'test_id': r.test_case.test_id,
                    'passed': r.passed,
                    'actual_compliant': r.actual_compliant,
                    'actual_violations': r.actual_violations,
                    'error_message': r.error_message,
                    'execution_time': r.execution_time
                }
                for r in self.test_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data['test_run']
    
    def _run_single_test(self, test_case: ComplianceTestCase) -> ComplianceTestResult:
        """Run a single test case."""
        start_time = datetime.now()
        
        try:
            # Apply test conditions (e.g., specific time)
            self._apply_test_conditions(test_case.test_conditions)
            
            # Run compliance check
            result = self.compliance_engine.validate_pricing(
                product=test_case.product,
                proposed_price=test_case.proposed_price
            )
            
            # Extract violations
            actual_violations = [v.rule.rule_id for v in result.violations]
            
            # Determine if test passed
            passed = (
                result.is_compliant == test_case.expected_compliant and
                set(actual_violations) == set(test_case.expected_violations)
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ComplianceTestResult(
                test_case=test_case,
                passed=passed,
                actual_compliant=result.is_compliant,
                actual_violations=actual_violations,
                compliance_result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Test {test_case.test_id} failed with error: {str(e)}")
            
            return ComplianceTestResult(
                test_case=test_case,
                passed=False,
                actual_compliant=False,
                actual_violations=[],
                compliance_result=None,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _init_standard_tests(self):
        """Initialize standard compliance test cases."""
        
        # MA Tax Compliance Tests
        self.test_cases.extend([
            ComplianceTestCase(
                test_id="MA_TAX_001",
                name="MA Excise Tax Calculation",
                description="Verify MA excise tax is properly calculated",
                product=Product(
                    sku="TEST_MA_001",
                    name="Test Product MA",
                    market=Market.MASSACHUSETTS,
                    category=ProductCategory.FLOWER,
                    current_price=Decimal("50.00")
                ),
                proposed_price=Decimal("55.00"),
                expected_compliant=True,
                expected_violations=[]
            ),
            
            ComplianceTestCase(
                test_id="MA_MIN_PRICE_001",
                name="MA Minimum Price Violation",
                description="Test price below minimum for MA",
                product=Product(
                    sku="TEST_MA_002",
                    name="Test Product MA Low Price",
                    market=Market.MASSACHUSETTS,
                    category=ProductCategory.FLOWER,
                    current_price=Decimal("10.00"),
                    unit="gram"
                ),
                proposed_price=Decimal("3.00"),  # Below $5/gram minimum
                expected_compliant=False,
                expected_violations=["MA002"]
            ),
            
            ComplianceTestCase(
                test_id="MA_PROMO_001",
                name="MA Excessive Promotional Discount",
                description="Test promotional discount exceeding 30% limit",
                product=Product(
                    sku="TEST_MA_003",
                    name="Test Product MA Promo",
                    market=Market.MASSACHUSETTS,
                    category=ProductCategory.EDIBLES,
                    current_price=Decimal("100.00")
                ),
                proposed_price=Decimal("65.00"),  # 35% discount
                expected_compliant=False,
                expected_violations=["MA003"],
                test_conditions={"is_promotional": True}
            ),
            
            ComplianceTestCase(
                test_id="MA_HOURS_001",
                name="MA Operating Hours Violation",
                description="Test price change outside operating hours",
                product=Product(
                    sku="TEST_MA_004",
                    name="Test Product MA Hours",
                    market=Market.MASSACHUSETTS,
                    category=ProductCategory.VAPES,
                    current_price=Decimal("75.00")
                ),
                proposed_price=Decimal("80.00"),
                expected_compliant=False,
                expected_violations=["MA005"],
                test_conditions={"test_time": time(23, 0)}  # 11 PM
            )
        ])
        
        # RI Tax and Margin Tests
        self.test_cases.extend([
            ComplianceTestCase(
                test_id="RI_TAX_001",
                name="RI Weight-Based Tax",
                description="Verify RI weight-based tax calculation",
                product=Product(
                    sku="TEST_RI_001",
                    name="Test Product RI",
                    market=Market.RHODE_ISLAND,
                    category=ProductCategory.FLOWER,
                    current_price=Decimal("60.00"),
                    size=Decimal("3.5"),  # 3.5 grams
                    unit="gram"
                ),
                proposed_price=Decimal("65.00"),
                expected_compliant=True,
                expected_violations=[]
            ),
            
            ComplianceTestCase(
                test_id="RI_MARGIN_001",
                name="RI Margin Requirement Violation",
                description="Test price below minimum margin requirement",
                product=Product(
                    sku="TEST_RI_002",
                    name="Test Product RI Margin",
                    market=Market.RHODE_ISLAND,
                    category=ProductCategory.CONCENTRATES,
                    current_price=Decimal("40.00"),
                    cost=Decimal("35.00")
                ),
                proposed_price=Decimal("38.00"),  # Less than 15% margin
                expected_compliant=False,
                expected_violations=["RI004"]
            ),
            
            ComplianceTestCase(
                test_id="RI_MEDICAL_DISCOUNT_001",
                name="RI Medical Discount Limit",
                description="Test medical discount exceeding 20% limit",
                product=Product(
                    sku="TEST_RI_003",
                    name="Test Product RI Medical",
                    market=Market.RHODE_ISLAND,
                    category=ProductCategory.MEDICAL,
                    current_price=Decimal("50.00")
                ),
                proposed_price=Decimal("38.00"),  # 24% discount
                expected_compliant=False,
                expected_violations=["RI002"],
                test_conditions={"is_promotional": True}
            )
        ])
        
        # Edge Cases
        self.test_cases.extend([
            ComplianceTestCase(
                test_id="EDGE_ZERO_PRICE_001",
                name="Zero Price Test",
                description="Test handling of zero price",
                product=Product(
                    sku="TEST_EDGE_001",
                    name="Test Product Edge",
                    market=Market.MASSACHUSETTS,
                    category=ProductCategory.FLOWER,
                    current_price=Decimal("50.00")
                ),
                proposed_price=Decimal("0.00"),
                expected_compliant=False,
                expected_violations=["MA002"]  # Below minimum
            ),
            
            ComplianceTestCase(
                test_id="EDGE_HIGH_PRICE_001",
                name="Extremely High Price",
                description="Test handling of very high price",
                product=Product(
                    sku="TEST_EDGE_002",
                    name="Test Product High Price",
                    market=Market.MASSACHUSETTS,
                    category=ProductCategory.FLOWER,
                    current_price=Decimal("50.00"),
                    thc_mg=100
                ),
                proposed_price=Decimal("500.00"),  # $5/mg THC
                expected_compliant=False,
                expected_violations=["MA002"]  # Exceeds THC price limit
            )
        ])
    
    def _apply_test_conditions(self, conditions: Dict[str, Any]):
        """Apply test conditions to simulate specific scenarios."""
        # This would modify system state or mock certain conditions
        # For example, mocking current time for operating hours tests
        pass
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate test run summary."""
        passed = sum(1 for r in self.test_results if r.passed)
        failed = len(self.test_results) - passed
        
        # Group failures by type
        failures_by_market = {}
        failures_by_violation = {}
        
        for result in self.test_results:
            if not result.passed:
                # By market
                market = result.test_case.product.market.value
                if market not in failures_by_market:
                    failures_by_market[market] = []
                failures_by_market[market].append(result.test_case.test_id)
                
                # By violation type
                for violation in result.actual_violations:
                    if violation not in failures_by_violation:
                        failures_by_violation[violation] = []
                    failures_by_violation[violation].append(result.test_case.test_id)
        
        # Performance metrics
        avg_execution_time = sum(r.execution_time for r in self.test_results) / len(self.test_results)
        max_execution_time = max(r.execution_time for r in self.test_results)
        
        return {
            'summary': {
                'total_tests': len(self.test_results),
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / len(self.test_results),
                'total_execution_time': total_time,
                'avg_test_time': avg_execution_time,
                'max_test_time': max_execution_time
            },
            'failures': {
                'by_market': failures_by_market,
                'by_violation': failures_by_violation,
                'details': [
                    {
                        'test_id': r.test_case.test_id,
                        'name': r.test_case.name,
                        'expected_compliant': r.test_case.expected_compliant,
                        'actual_compliant': r.actual_compliant,
                        'expected_violations': r.test_case.expected_violations,
                        'actual_violations': r.actual_violations,
                        'error': r.error_message
                    }
                    for r in self.test_results if not r.passed
                ]
            },
            'performance': {
                'tests_per_second': len(self.test_results) / total_time,
                'slowest_tests': sorted([
                    {'test_id': r.test_case.test_id, 'time': r.execution_time}
                    for r in self.test_results
                ], key=lambda x: x['time'], reverse=True)[:5]
            }
        }