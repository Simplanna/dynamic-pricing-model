"""Test scenarios for safety control systems."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import logging
import random

from ..core.models import Product, Market, ProductCategory, PricingDecision
from ..safety.safety_controller import SafetyController, SafetyConfig
from ..safety.canary_testing import CanaryTester, CanaryConfig
from ..safety.rollback_manager import RollbackManager, RollbackReason

logger = logging.getLogger(__name__)


@dataclass
class SafetyTestScenario:
    """A safety system test scenario."""
    scenario_id: str
    name: str
    description: str
    test_type: str  # 'price_limit', 'canary', 'rollback', 'mape'
    products: List[Product]
    price_changes: List[Tuple[str, Decimal, Decimal]]  # SKU, old_price, new_price
    expected_outcomes: Dict[str, Any]
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyTestResult:
    """Result of a safety test scenario."""
    scenario: SafetyTestScenario
    passed: bool
    actual_outcomes: Dict[str, Any]
    violations_detected: List[str]
    actions_taken: List[str]
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None


class SafetyTestScenarios:
    """Comprehensive test scenarios for safety systems."""
    
    def __init__(self, safety_controller: SafetyController,
                 canary_tester: CanaryTester,
                 rollback_manager: RollbackManager):
        self.safety_controller = safety_controller
        self.canary_tester = canary_tester
        self.rollback_manager = rollback_manager
        
        self.scenarios: List[SafetyTestScenario] = []
        self.test_results: List[SafetyTestResult] = []
        
        # Initialize test scenarios
        self._init_test_scenarios()
        
        logger.info(f"Safety test scenarios initialized with {len(self.scenarios)} scenarios")
    
    def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all safety test scenarios."""
        self.test_results.clear()
        start_time = datetime.now()
        
        for scenario in self.scenarios:
            logger.info(f"Running scenario: {scenario.name}")
            result = self._run_scenario(scenario)
            self.test_results.append(result)
        
        # Generate comprehensive report
        report = self._generate_test_report(start_time)
        return report
    
    def run_scenario_type(self, test_type: str) -> List[SafetyTestResult]:
        """Run scenarios of a specific type."""
        type_scenarios = [s for s in self.scenarios if s.test_type == test_type]
        results = []
        
        for scenario in type_scenarios:
            result = self._run_scenario(scenario)
            results.append(result)
        
        return results
    
    def stress_test_safety_limits(self, num_products: int = 100,
                                num_changes: int = 1000) -> Dict[str, Any]:
        """Stress test safety systems with high volume."""
        logger.info(f"Starting stress test: {num_products} products, {num_changes} changes")
        
        # Generate test products
        test_products = self._generate_test_products(num_products)
        
        # Simulate rapid price changes
        start_time = datetime.now()
        violations = 0
        blocked_changes = 0
        
        for i in range(num_changes):
            product = random.choice(test_products)
            current_price = product.current_price
            
            # Generate random price change
            change_pct = Decimal(str(random.uniform(-0.15, 0.15)))  # -15% to +15%
            new_price = current_price * (Decimal('1') + change_pct)
            
            # Test safety validation
            is_allowed, safety_violations = self.safety_controller.validate_price_change(
                product, current_price, new_price, f"Stress test change {i}"
            )
            
            if not is_allowed:
                blocked_changes += 1
            if safety_violations:
                violations += len(safety_violations)
            
            # Update price if allowed
            if is_allowed:
                product.current_price = new_price
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'test_type': 'stress_test',
            'num_products': num_products,
            'num_changes': num_changes,
            'duration_seconds': duration,
            'changes_per_second': num_changes / duration,
            'blocked_changes': blocked_changes,
            'block_rate': blocked_changes / num_changes,
            'total_violations': violations,
            'avg_violations_per_change': violations / num_changes
        }
    
    def test_canary_rollout(self, num_products: int = 50) -> Dict[str, Any]:
        """Test canary deployment process."""
        # Create products and pricing decisions
        products = self._generate_test_products(num_products)
        pricing_decisions = []
        
        for product in products:
            # Generate price change
            change_pct = Decimal(str(random.uniform(-0.10, 0.10)))
            new_price = product.current_price * (Decimal('1') + change_pct)
            
            decision = PricingDecision(
                product=product,
                base_price=product.current_price,
                recommended_price=new_price,
                final_price=new_price,
                factors={},
                explanation="Canary test price change"
            )
            pricing_decisions.append(decision)
        
        # Create canary test
        canary_test = self.canary_tester.create_canary_test(
            pricing_decisions, "Safety test canary"
        )
        
        if not canary_test:
            return {'error': 'Failed to create canary test'}
        
        # Start canary
        canary_decisions = self.canary_tester.start_test(canary_test.test_id)
        
        # Simulate monitoring
        monitoring_results = []
        for i in range(5):  # 5 monitoring cycles
            status, metrics = self.canary_tester.monitor_test(canary_test.test_id)
            monitoring_results.append({
                'cycle': i + 1,
                'status': status.value,
                'metrics': {k: float(v) for k, v in metrics.items()}
            })
        
        # Complete test
        final_result = self.canary_tester.complete_test(canary_test.test_id)
        
        return {
            'test_type': 'canary_rollout',
            'total_products': num_products,
            'canary_size': len(canary_test.products),
            'control_size': len(canary_test.control_products),
            'success': final_result.success,
            'rollback_required': final_result.rollback_required,
            'monitoring_results': monitoring_results,
            'recommendations': final_result.recommendations
        }
    
    def test_rollback_scenarios(self) -> Dict[str, Any]:
        """Test various rollback scenarios."""
        results = []
        
        # Scenario 1: Compliance violation rollback
        products = self._generate_test_products(20)
        
        # Create snapshot before changes
        snapshot = self.rollback_manager.create_snapshot(
            products, {'scenario': 'compliance_rollback_test'}
        )
        
        # Simulate price changes
        for product in products:
            product.current_price *= Decimal('1.15')  # 15% increase
        
        # Create rollback plan
        plan = self.rollback_manager.create_rollback_plan(
            products,
            RollbackReason.COMPLIANCE_VIOLATION,
            hours_ago=0  # Use most recent snapshot
        )
        
        # Execute rollback
        executed_plan = self.rollback_manager.execute_rollback(plan.plan_id)
        
        results.append({
            'scenario': 'compliance_violation',
            'affected_products': len(plan.affected_products),
            'rollback_state': executed_plan.state.value,
            'success_rate': sum(executed_plan.results.values()) / len(executed_plan.results)
        })
        
        # Scenario 2: Performance degradation rollback
        products = self._generate_test_products(30)
        snapshot = self.rollback_manager.create_snapshot(
            products, {'scenario': 'performance_rollback_test'}
        )
        
        # Simulate problematic changes
        for product in products:
            product.current_price *= Decimal('0.75')  # 25% decrease
        
        plan = self.rollback_manager.create_rollback_plan(
            products,
            RollbackReason.PERFORMANCE_DEGRADATION,
            hours_ago=0
        )
        
        # Test batch execution
        executed_plan = self.rollback_manager.execute_rollback(plan.plan_id, batch_size=10)
        
        results.append({
            'scenario': 'performance_degradation',
            'affected_products': len(plan.affected_products),
            'rollback_state': executed_plan.state.value,
            'success_rate': sum(executed_plan.results.values()) / len(executed_plan.results),
            'batch_execution': True
        })
        
        return {
            'test_type': 'rollback_scenarios',
            'scenarios_tested': len(results),
            'results': results,
            'rollback_history': [
                {
                    'plan_id': plan.plan_id,
                    'reason': plan.reason.value,
                    'created_at': plan.created_at.isoformat(),
                    'state': plan.state.value
                }
                for plan in self.rollback_manager.get_rollback_history(hours=1)
            ]
        }
    
    def _run_scenario(self, scenario: SafetyTestScenario) -> SafetyTestResult:
        """Run a single test scenario."""
        start_time = datetime.now()
        
        try:
            if scenario.test_type == 'price_limit':
                result = self._run_price_limit_scenario(scenario)
            elif scenario.test_type == 'canary':
                result = self._run_canary_scenario(scenario)
            elif scenario.test_type == 'rollback':
                result = self._run_rollback_scenario(scenario)
            elif scenario.test_type == 'mape':
                result = self._run_mape_scenario(scenario)
            else:
                raise ValueError(f"Unknown test type: {scenario.test_type}")
            
            performance_metrics = {
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Check if outcomes match expected
            passed = self._compare_outcomes(
                scenario.expected_outcomes,
                result['actual_outcomes']
            )
            
            return SafetyTestResult(
                scenario=scenario,
                passed=passed,
                actual_outcomes=result['actual_outcomes'],
                violations_detected=result.get('violations', []),
                actions_taken=result.get('actions', []),
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Scenario {scenario.scenario_id} failed: {str(e)}")
            
            return SafetyTestResult(
                scenario=scenario,
                passed=False,
                actual_outcomes={},
                violations_detected=[],
                actions_taken=[],
                performance_metrics={
                    'execution_time': (datetime.now() - start_time).total_seconds()
                },
                error_message=str(e)
            )
    
    def _run_price_limit_scenario(self, scenario: SafetyTestScenario) -> Dict[str, Any]:
        """Run price limit test scenario."""
        violations = []
        actions = []
        blocked_count = 0
        
        for sku, old_price, new_price in scenario.price_changes:
            product = next((p for p in scenario.products if p.sku == sku), None)
            if not product:
                continue
            
            # Test price change
            is_allowed, safety_violations = self.safety_controller.validate_price_change(
                product, old_price, new_price, "Test scenario"
            )
            
            if not is_allowed:
                blocked_count += 1
                actions.append(f"Blocked price change for {sku}")
            
            for violation in safety_violations:
                violations.append(f"{sku}: {violation.violation_type.value}")
        
        return {
            'actual_outcomes': {
                'blocked_changes': blocked_count,
                'total_violations': len(violations),
                'violation_types': list(set(v.split(':')[1] for v in violations))
            },
            'violations': violations,
            'actions': actions
        }
    
    def _run_canary_scenario(self, scenario: SafetyTestScenario) -> Dict[str, Any]:
        """Run canary test scenario."""
        # Create pricing decisions from scenario
        decisions = []
        for sku, old_price, new_price in scenario.price_changes:
            product = next((p for p in scenario.products if p.sku == sku), None)
            if product:
                decision = PricingDecision(
                    product=product,
                    base_price=old_price,
                    recommended_price=new_price,
                    final_price=new_price,
                    factors={},
                    explanation="Canary test"
                )
                decisions.append(decision)
        
        # Create and run canary test
        canary_test = self.canary_tester.create_canary_test(decisions, scenario.name)
        
        if not canary_test:
            return {
                'actual_outcomes': {'canary_created': False},
                'violations': [],
                'actions': ['Failed to create canary test']
            }
        
        # Start test
        canary_decisions = self.canary_tester.start_test(canary_test.test_id)
        
        # Monitor
        final_status, metrics = self.canary_tester.monitor_test(canary_test.test_id)
        
        return {
            'actual_outcomes': {
                'canary_created': True,
                'canary_size': len(canary_test.products),
                'final_status': final_status.value,
                'metrics': {k: float(v) for k, v in metrics.items()}
            },
            'violations': [],
            'actions': [f"Created canary test {canary_test.test_id}"]
        }
    
    def _run_rollback_scenario(self, scenario: SafetyTestScenario) -> Dict[str, Any]:
        """Run rollback test scenario."""
        # Create snapshot
        snapshot = self.rollback_manager.create_snapshot(
            scenario.products,
            {'scenario': scenario.scenario_id}
        )
        
        # Apply price changes
        for sku, old_price, new_price in scenario.price_changes:
            product = next((p for p in scenario.products if p.sku == sku), None)
            if product:
                product.current_price = new_price
        
        # Create rollback plan
        reason = RollbackReason[scenario.conditions.get('reason', 'MANUAL_REQUEST')]
        plan = self.rollback_manager.create_rollback_plan(
            scenario.products,
            reason,
            hours_ago=0
        )
        
        # Execute rollback
        executed_plan = self.rollback_manager.execute_rollback(plan.plan_id)
        
        success_count = sum(executed_plan.results.values())
        
        return {
            'actual_outcomes': {
                'rollback_created': True,
                'rollback_executed': True,
                'rollback_state': executed_plan.state.value,
                'success_count': success_count,
                'success_rate': success_count / len(executed_plan.results)
            },
            'violations': [],
            'actions': [
                f"Created snapshot",
                f"Created rollback plan {plan.plan_id}",
                f"Executed rollback with {success_count}/{len(executed_plan.results)} successes"
            ]
        }
    
    def _run_mape_scenario(self, scenario: SafetyTestScenario) -> Dict[str, Any]:
        """Run MAPE threshold test scenario."""
        violations = []
        
        # Simulate MAPE values
        for condition in scenario.conditions.get('mape_values', []):
            sku = condition['sku']
            mape = Decimal(str(condition['mape']))
            
            violation = self.safety_controller.check_mape_threshold(sku, mape)
            if violation:
                violations.append(f"{sku}: MAPE {mape:.1%}")
        
        return {
            'actual_outcomes': {
                'mape_violations': len(violations),
                'violated_skus': [v.split(':')[0] for v in violations]
            },
            'violations': violations,
            'actions': []
        }
    
    def _init_test_scenarios(self):
        """Initialize standard test scenarios."""
        
        # Price limit scenarios
        self.scenarios.append(SafetyTestScenario(
            scenario_id="SAFETY_001",
            name="Daily Price Change Limit",
            description="Test 10% daily price change limit",
            test_type="price_limit",
            products=[
                Product(
                    sku=f"LIMIT_TEST_{i}",
                    name=f"Limit Test Product {i}",
                    market=Market.MASSACHUSETTS,
                    category=ProductCategory.FLOWER,
                    current_price=Decimal("50.00")
                )
                for i in range(5)
            ],
            price_changes=[
                ("LIMIT_TEST_0", Decimal("50.00"), Decimal("54.00")),  # 8% - OK
                ("LIMIT_TEST_1", Decimal("50.00"), Decimal("56.00")),  # 12% - Violation
                ("LIMIT_TEST_2", Decimal("50.00"), Decimal("45.00")),  # -10% - OK
                ("LIMIT_TEST_3", Decimal("50.00"), Decimal("43.00")),  # -14% - Violation
                ("LIMIT_TEST_4", Decimal("50.00"), Decimal("52.50")),  # 5% - OK
            ],
            expected_outcomes={
                'blocked_changes': 2,
                'total_violations': 2,
                'violation_types': ['max_daily_change']
            }
        ))
        
        # Canary test scenarios
        self.scenarios.append(SafetyTestScenario(
            scenario_id="CANARY_001",
            name="Standard Canary Deployment",
            description="Test standard canary rollout process",
            test_type="canary",
            products=[
                Product(
                    sku=f"CANARY_TEST_{i}",
                    name=f"Canary Test Product {i}",
                    market=Market.MASSACHUSETTS,
                    category=random.choice(list(ProductCategory)),
                    current_price=Decimal(str(random.uniform(20, 100)))
                )
                for i in range(30)
            ],
            price_changes=[
                (f"CANARY_TEST_{i}", 
                 Decimal(str(random.uniform(20, 100))),
                 Decimal(str(random.uniform(20, 100))))
                for i in range(30)
            ],
            expected_outcomes={
                'canary_created': True,
                'canary_size': 3,  # 10% of 30
                'final_status': 'monitoring'
            }
        ))
        
        # Rollback scenarios
        self.scenarios.append(SafetyTestScenario(
            scenario_id="ROLLBACK_001",
            name="Emergency Rollback",
            description="Test emergency rollback functionality",
            test_type="rollback",
            products=[
                Product(
                    sku=f"ROLLBACK_TEST_{i}",
                    name=f"Rollback Test Product {i}",
                    market=Market.RHODE_ISLAND,
                    category=ProductCategory.EDIBLES,
                    current_price=Decimal("30.00")
                )
                for i in range(10)
            ],
            price_changes=[
                (f"ROLLBACK_TEST_{i}", Decimal("30.00"), Decimal("45.00"))  # 50% increase
                for i in range(10)
            ],
            expected_outcomes={
                'rollback_created': True,
                'rollback_executed': True,
                'rollback_state': 'completed',
                'success_rate': 1.0
            },
            conditions={'reason': 'SAFETY_LIMIT'}
        ))
        
        # MAPE threshold scenarios
        self.scenarios.append(SafetyTestScenario(
            scenario_id="MAPE_001",
            name="MAPE Threshold Violations",
            description="Test MAPE threshold detection",
            test_type="mape",
            products=[],  # Not needed for MAPE tests
            price_changes=[],  # Not needed for MAPE tests
            expected_outcomes={
                'mape_violations': 2,
                'violated_skus': ['MAPE_HIGH_1', 'MAPE_HIGH_2']
            },
            conditions={
                'mape_values': [
                    {'sku': 'MAPE_OK_1', 'mape': 0.015},  # 1.5% - OK
                    {'sku': 'MAPE_HIGH_1', 'mape': 0.035},  # 3.5% - Violation
                    {'sku': 'MAPE_OK_2', 'mape': 0.018},  # 1.8% - OK
                    {'sku': 'MAPE_HIGH_2', 'mape': 0.025},  # 2.5% - Violation
                ]
            }
        ))
    
    def _generate_test_products(self, count: int) -> List[Product]:
        """Generate test products with random attributes."""
        products = []
        
        for i in range(count):
            products.append(Product(
                sku=f"TEST_PRODUCT_{i:04d}",
                name=f"Test Product {i}",
                market=random.choice(list(Market)),
                category=random.choice(list(ProductCategory)),
                current_price=Decimal(str(random.uniform(10, 200))),
                cost=Decimal(str(random.uniform(5, 100))),
                inventory_level=random.randint(0, 1000)
            ))
        
        return products
    
    def _compare_outcomes(self, expected: Dict[str, Any], 
                         actual: Dict[str, Any]) -> bool:
        """Compare expected and actual outcomes."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            
            actual_value = actual[key]
            
            # Handle different types of comparisons
            if isinstance(expected_value, (int, float, Decimal)):
                # Numeric comparison with tolerance
                if abs(expected_value - actual_value) > 0.01:
                    return False
            elif isinstance(expected_value, list):
                # List comparison (order doesn't matter)
                if set(expected_value) != set(actual_value):
                    return False
            else:
                # Direct comparison
                if expected_value != actual_value:
                    return False
        
        return True
    
    def _generate_test_report(self, start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed = sum(1 for r in self.test_results if r.passed)
        failed = len(self.test_results) - passed
        
        # Group by test type
        by_type = {}
        for result in self.test_results:
            test_type = result.scenario.test_type
            if test_type not in by_type:
                by_type[test_type] = {'passed': 0, 'failed': 0, 'scenarios': []}
            
            if result.passed:
                by_type[test_type]['passed'] += 1
            else:
                by_type[test_type]['failed'] += 1
            
            by_type[test_type]['scenarios'].append({
                'scenario_id': result.scenario.scenario_id,
                'name': result.scenario.name,
                'passed': result.passed,
                'execution_time': result.performance_metrics.get('execution_time', 0)
            })
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'summary': {
                'total_scenarios': len(self.test_results),
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / len(self.test_results),
                'total_execution_time': total_time
            },
            'by_type': by_type,
            'failed_scenarios': [
                {
                    'scenario_id': r.scenario.scenario_id,
                    'name': r.scenario.name,
                    'expected': r.scenario.expected_outcomes,
                    'actual': r.actual_outcomes,
                    'error': r.error_message
                }
                for r in self.test_results if not r.passed
            ],
            'performance': {
                'avg_execution_time': sum(
                    r.performance_metrics.get('execution_time', 0)
                    for r in self.test_results
                ) / len(self.test_results)
            }
        }