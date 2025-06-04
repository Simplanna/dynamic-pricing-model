"""Canary testing framework for gradual price rollouts."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging
import random
from enum import Enum

from ..core.models import Product, PricingDecision, Market
from ..utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class CanaryStatus(Enum):
    """Status of a canary test."""
    PENDING = "pending"
    ACTIVE = "active"
    MONITORING = "monitoring"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class CanaryMetric(Enum):
    """Metrics to monitor during canary testing."""
    REVENUE = "revenue"
    UNITS_SOLD = "units_sold"
    CONVERSION_RATE = "conversion_rate"
    CUSTOMER_COMPLAINTS = "customer_complaints"
    COMPETITOR_RESPONSE = "competitor_response"
    MARGIN = "margin"


@dataclass
class CanaryConfig:
    """Configuration for canary testing."""
    canary_percentage: Decimal = Decimal('0.10')  # 10% of SKUs
    min_canary_size: int = 5
    max_canary_size: int = 100
    monitoring_duration: timedelta = timedelta(hours=24)
    success_threshold: Decimal = Decimal('0.95')  # 95% of baseline
    failure_threshold: Decimal = Decimal('0.90')  # 90% of baseline
    auto_rollback: bool = True
    stratified_sampling: bool = True  # Sample across categories


@dataclass
class CanaryTest:
    """Represents a canary test."""
    test_id: str
    name: str
    products: List[Product]
    control_products: List[Product]
    start_time: datetime
    end_time: datetime
    status: CanaryStatus
    config: CanaryConfig
    baseline_metrics: Dict[str, Decimal] = field(default_factory=dict)
    current_metrics: Dict[str, Decimal] = field(default_factory=dict)
    price_changes: Dict[str, Tuple[Decimal, Decimal]] = field(default_factory=dict)  # SKU -> (old, new)


@dataclass
class CanaryResult:
    """Result of a canary test."""
    test: CanaryTest
    success: bool
    metrics_comparison: Dict[str, Dict[str, Decimal]]
    recommendations: List[str]
    rollback_required: bool
    completion_time: datetime = field(default_factory=datetime.now)


class CanaryTester:
    """Manages canary testing for pricing changes."""
    
    def __init__(self, config: Optional[CanaryConfig] = None,
                 audit_logger: Optional[AuditLogger] = None):
        self.config = config or CanaryConfig()
        self.audit_logger = audit_logger or AuditLogger()
        self.active_tests: Dict[str, CanaryTest] = {}
        self.completed_tests: List[CanaryResult] = []
        self._test_counter = 0
        
        logger.info(f"Canary tester initialized with {self.config.canary_percentage:.0%} canary size")
    
    def create_canary_test(self, pricing_decisions: List[PricingDecision],
                          name: str = "") -> Optional[CanaryTest]:
        """Create a new canary test from pricing decisions."""
        
        # Filter decisions with actual price changes
        significant_changes = [
            d for d in pricing_decisions
            if abs(d.final_price - d.base_price) / d.base_price > Decimal('0.01')  # >1% change
        ]
        
        if not significant_changes:
            logger.info("No significant price changes to test")
            return None
        
        # Select canary products
        canary_products, control_products = self._select_canary_products(significant_changes)
        
        if len(canary_products) < self.config.min_canary_size:
            logger.warning(f"Insufficient products for canary test: {len(canary_products)}")
            return None
        
        # Create test
        self._test_counter += 1
        test_id = f"canary_{self._test_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test = CanaryTest(
            test_id=test_id,
            name=name or f"Canary Test {self._test_counter}",
            products=canary_products,
            control_products=control_products,
            start_time=datetime.now(),
            end_time=datetime.now() + self.config.monitoring_duration,
            status=CanaryStatus.PENDING,
            config=self.config,
            price_changes={
                d.product.sku: (d.base_price, d.final_price)
                for d in significant_changes
                if d.product in canary_products
            }
        )
        
        # Store test
        self.active_tests[test_id] = test
        
        # Log creation
        self._log_test_creation(test, significant_changes)
        
        return test
    
    def start_test(self, test_id: str) -> List[PricingDecision]:
        """Start a canary test and return decisions to apply."""
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        if test.status != CanaryStatus.PENDING:
            raise ValueError(f"Test {test_id} is not in pending state")
        
        # Collect baseline metrics
        test.baseline_metrics = self._collect_baseline_metrics(test.products + test.control_products)
        
        # Update status
        test.status = CanaryStatus.ACTIVE
        test.start_time = datetime.now()
        
        # Create pricing decisions for canary products only
        canary_decisions = []
        for product in test.products:
            old_price, new_price = test.price_changes.get(product.sku, (product.current_price, product.current_price))
            
            decision = PricingDecision(
                product=product,
                base_price=old_price,
                recommended_price=new_price,
                final_price=new_price,
                factors={},
                explanation=f"Canary test {test.name}"
            )
            decision.add_metadata("canary_test_id", test_id)
            decision.add_metadata("is_canary", True)
            
            canary_decisions.append(decision)
        
        # Log test start
        self.audit_logger.log_canary_test_start(
            test_id=test_id,
            test_name=test.name,
            canary_size=len(test.products),
            control_size=len(test.control_products),
            duration_hours=self.config.monitoring_duration.total_seconds() / 3600
        )
        
        logger.info(f"Started canary test {test_id} with {len(canary_decisions)} products")
        
        return canary_decisions
    
    def monitor_test(self, test_id: str) -> Tuple[CanaryStatus, Dict[str, Decimal]]:
        """Monitor an active canary test."""
        test = self.active_tests.get(test_id)
        if not test or test.status != CanaryStatus.ACTIVE:
            raise ValueError(f"No active test found: {test_id}")
        
        # Collect current metrics
        current_metrics = self._collect_current_metrics(test)
        test.current_metrics = current_metrics
        
        # Compare with baseline
        performance = self._calculate_performance(test)
        
        # Check if monitoring period is complete
        if datetime.now() >= test.end_time:
            test.status = CanaryStatus.MONITORING
            return self._evaluate_test(test)
        
        # Check for early failure
        if self._check_early_failure(performance):
            test.status = CanaryStatus.FAILED
            if self.config.auto_rollback:
                return CanaryStatus.FAILED, performance
        
        return test.status, performance
    
    def complete_test(self, test_id: str) -> CanaryResult:
        """Complete a canary test and evaluate results."""
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        # Final evaluation
        final_status, performance = self._evaluate_test(test)
        test.status = final_status
        
        # Determine success
        success = final_status == CanaryStatus.SUCCESS
        rollback_required = final_status == CanaryStatus.FAILED and self.config.auto_rollback
        
        # Generate recommendations
        recommendations = self._generate_recommendations(test, performance)
        
        # Create result
        result = CanaryResult(
            test=test,
            success=success,
            metrics_comparison=self._detailed_metrics_comparison(test),
            recommendations=recommendations,
            rollback_required=rollback_required
        )
        
        # Archive test
        del self.active_tests[test_id]
        self.completed_tests.append(result)
        
        # Log completion
        self._log_test_completion(result)
        
        return result
    
    def get_rollback_decisions(self, test_id: str) -> List[PricingDecision]:
        """Get pricing decisions to rollback a failed test."""
        # Find test in active or completed
        test = self.active_tests.get(test_id)
        if not test:
            # Check completed tests
            completed = next((r.test for r in self.completed_tests if r.test.test_id == test_id), None)
            if completed:
                test = completed
        
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        # Create rollback decisions
        rollback_decisions = []
        for product in test.products:
            old_price, _ = test.price_changes.get(product.sku, (product.current_price, product.current_price))
            
            decision = PricingDecision(
                product=product,
                base_price=product.current_price,
                recommended_price=old_price,
                final_price=old_price,
                factors={},
                explanation=f"Rollback canary test {test.name}"
            )
            decision.add_metadata("canary_test_id", test_id)
            decision.add_metadata("is_rollback", True)
            
            rollback_decisions.append(decision)
        
        logger.info(f"Generated {len(rollback_decisions)} rollback decisions for test {test_id}")
        
        return rollback_decisions
    
    def _select_canary_products(self, decisions: List[PricingDecision]) -> Tuple[List[Product], List[Product]]:
        """Select products for canary and control groups."""
        # Calculate canary size
        total_products = len(decisions)
        canary_size = max(
            self.config.min_canary_size,
            min(
                int(total_products * float(self.config.canary_percentage)),
                self.config.max_canary_size
            )
        )
        
        if self.config.stratified_sampling:
            # Group by category/market
            by_category = {}
            for decision in decisions:
                key = (decision.product.category, decision.product.market)
                if key not in by_category:
                    by_category[key] = []
                by_category[key].append(decision.product)
            
            # Sample proportionally from each category
            canary_products = []
            control_products = []
            
            for category_products in by_category.values():
                category_canary_size = max(1, int(len(category_products) * float(self.config.canary_percentage)))
                
                # Random selection within category
                random.shuffle(category_products)
                canary_products.extend(category_products[:category_canary_size])
                control_products.extend(category_products[category_canary_size:])
            
            # Trim to exact size if needed
            if len(canary_products) > canary_size:
                excess = canary_products[canary_size:]
                canary_products = canary_products[:canary_size]
                control_products.extend(excess)
        
        else:
            # Simple random sampling
            all_products = [d.product for d in decisions]
            random.shuffle(all_products)
            canary_products = all_products[:canary_size]
            control_products = all_products[canary_size:]
        
        return canary_products, control_products
    
    def _collect_baseline_metrics(self, products: List[Product]) -> Dict[str, Decimal]:
        """Collect baseline metrics for products."""
        # In production, this would query actual sales/performance data
        # For now, using simulated metrics
        metrics = {}
        
        for metric in CanaryMetric:
            if metric == CanaryMetric.REVENUE:
                # Simulate based on price * estimated sales
                total_revenue = sum(
                    p.current_price * Decimal('10')  # Assumed 10 units/day
                    for p in products
                )
                metrics[metric.value] = total_revenue
            
            elif metric == CanaryMetric.UNITS_SOLD:
                metrics[metric.value] = Decimal(str(len(products) * 10))
            
            elif metric == CanaryMetric.CONVERSION_RATE:
                metrics[metric.value] = Decimal('0.15')  # 15% baseline
            
            elif metric == CanaryMetric.MARGIN:
                # Average margin
                margins = []
                for p in products:
                    if hasattr(p, 'cost') and p.cost:
                        margin = (p.current_price - p.cost) / p.current_price
                        margins.append(margin)
                metrics[metric.value] = sum(margins) / len(margins) if margins else Decimal('0.20')
            
            else:
                metrics[metric.value] = Decimal('0')
        
        return metrics
    
    def _collect_current_metrics(self, test: CanaryTest) -> Dict[str, Decimal]:
        """Collect current metrics during test."""
        # In production, would query real-time data
        # Simulating with some variance
        
        current = {}
        baseline = test.baseline_metrics
        
        for metric_name, baseline_value in baseline.items():
            # Simulate performance impact
            if metric_name == CanaryMetric.REVENUE.value:
                # Revenue might increase with optimized pricing
                variance = Decimal(str(random.uniform(0.95, 1.08)))
            elif metric_name == CanaryMetric.UNITS_SOLD.value:
                # Units might decrease with higher prices
                variance = Decimal(str(random.uniform(0.92, 1.02)))
            elif metric_name == CanaryMetric.CONVERSION_RATE.value:
                # Conversion might be affected
                variance = Decimal(str(random.uniform(0.90, 1.05)))
            else:
                variance = Decimal(str(random.uniform(0.95, 1.05)))
            
            current[metric_name] = baseline_value * variance
        
        return current
    
    def _calculate_performance(self, test: CanaryTest) -> Dict[str, Decimal]:
        """Calculate performance metrics vs baseline."""
        performance = {}
        
        for metric_name, current_value in test.current_metrics.items():
            baseline_value = test.baseline_metrics.get(metric_name, Decimal('1'))
            if baseline_value > 0:
                performance[metric_name] = current_value / baseline_value
            else:
                performance[metric_name] = Decimal('1')
        
        return performance
    
    def _check_early_failure(self, performance: Dict[str, Decimal]) -> bool:
        """Check if test should fail early."""
        # Fail if any critical metric below failure threshold
        critical_metrics = [CanaryMetric.REVENUE.value, CanaryMetric.CONVERSION_RATE.value]
        
        for metric in critical_metrics:
            if metric in performance and performance[metric] < self.config.failure_threshold:
                logger.warning(f"Canary test failing early: {metric} = {performance[metric]:.2%}")
                return True
        
        return False
    
    def _evaluate_test(self, test: CanaryTest) -> Tuple[CanaryStatus, Dict[str, Decimal]]:
        """Evaluate test results."""
        performance = self._calculate_performance(test)
        
        # Check overall performance
        key_metrics = [CanaryMetric.REVENUE.value, CanaryMetric.MARGIN.value]
        key_performance = [performance.get(m, Decimal('1')) for m in key_metrics]
        
        if all(p >= self.config.success_threshold for p in key_performance):
            return CanaryStatus.SUCCESS, performance
        elif any(p < self.config.failure_threshold for p in key_performance):
            return CanaryStatus.FAILED, performance
        else:
            return CanaryStatus.MONITORING, performance
    
    def _generate_recommendations(self, test: CanaryTest, 
                                performance: Dict[str, Decimal]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Revenue performance
        revenue_perf = performance.get(CanaryMetric.REVENUE.value, Decimal('1'))
        if revenue_perf >= Decimal('1.05'):
            recommendations.append("Strong revenue increase - consider full rollout")
        elif revenue_perf < Decimal('0.95'):
            recommendations.append("Revenue declined - review pricing strategy")
        
        # Units sold
        units_perf = performance.get(CanaryMetric.UNITS_SOLD.value, Decimal('1'))
        if units_perf < Decimal('0.90'):
            recommendations.append("Significant drop in units - price elasticity may be high")
        
        # Margin
        margin_perf = performance.get(CanaryMetric.MARGIN.value, Decimal('1'))
        if margin_perf >= Decimal('1.02'):
            recommendations.append("Improved margins - pricing optimization successful")
        
        # Test size
        if len(test.products) < 20:
            recommendations.append("Consider larger test size for more reliable results")
        
        return recommendations
    
    def _detailed_metrics_comparison(self, test: CanaryTest) -> Dict[str, Dict[str, Decimal]]:
        """Create detailed metrics comparison."""
        comparison = {}
        
        for metric_name in test.baseline_metrics:
            comparison[metric_name] = {
                'baseline': test.baseline_metrics[metric_name],
                'current': test.current_metrics.get(metric_name, Decimal('0')),
                'change_pct': (
                    (test.current_metrics.get(metric_name, Decimal('0')) - test.baseline_metrics[metric_name])
                    / test.baseline_metrics[metric_name]
                    if test.baseline_metrics[metric_name] > 0 else Decimal('0')
                )
            }
        
        return comparison
    
    def _log_test_creation(self, test: CanaryTest, decisions: List[PricingDecision]):
        """Log canary test creation."""
        avg_price_change = sum(
            abs(d.final_price - d.base_price) / d.base_price
            for d in decisions
            if d.product in test.products
        ) / len(test.products)
        
        self.audit_logger.log_event(
            event_type="canary_test_created",
            details={
                'test_id': test.test_id,
                'test_name': test.name,
                'canary_size': len(test.products),
                'control_size': len(test.control_products),
                'avg_price_change': float(avg_price_change),
                'duration_hours': test.config.monitoring_duration.total_seconds() / 3600
            }
        )
    
    def _log_test_completion(self, result: CanaryResult):
        """Log canary test completion."""
        self.audit_logger.log_canary_test_complete(
            test_id=result.test.test_id,
            test_name=result.test.name,
            success=result.success,
            metrics_comparison=result.metrics_comparison,
            recommendations=result.recommendations,
            rollback_required=result.rollback_required
        )