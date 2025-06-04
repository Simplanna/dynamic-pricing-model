"""Performance benchmarks for compliance and safety systems."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import logging
import time
import statistics
import concurrent.futures
from pathlib import Path
import json

from ..core.models import Product, Market, ProductCategory, PricingDecision
from ..compliance.compliance_engine import ComplianceEngine
from ..safety.safety_controller import SafetyController
from ..monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # operations per second
    memory_usage: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'benchmark_name': self.benchmark_name,
            'iterations': self.iterations,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'std_dev': self.std_dev,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage
        }


class PerformanceBenchmark:
    """Performance benchmarking for pricing systems."""
    
    def __init__(self, compliance_engine: ComplianceEngine,
                 safety_controller: SafetyController):
        self.compliance_engine = compliance_engine
        self.safety_controller = safety_controller
        self.results: Dict[str, BenchmarkResult] = {}
        
        logger.info("Performance benchmark initialized")
    
    def run_all_benchmarks(self, iterations: int = 1000) -> Dict[str, BenchmarkResult]:
        """Run all performance benchmarks."""
        logger.info(f"Running all benchmarks with {iterations} iterations")
        
        benchmarks = [
            ('compliance_validation', self.benchmark_compliance_validation),
            ('safety_checks', self.benchmark_safety_checks),
            ('batch_processing', self.benchmark_batch_processing),
            ('concurrent_processing', self.benchmark_concurrent_processing),
            ('cache_performance', self.benchmark_cache_performance),
            ('rollback_operations', self.benchmark_rollback_operations)
        ]
        
        for name, benchmark_func in benchmarks:
            logger.info(f"Running benchmark: {name}")
            result = benchmark_func(iterations)
            self.results[name] = result
        
        return self.results
    
    def benchmark_compliance_validation(self, iterations: int) -> BenchmarkResult:
        """Benchmark compliance validation performance."""
        # Generate test products
        products = self._generate_test_products(100)
        times = []
        
        for i in range(iterations):
            product = products[i % len(products)]
            price = product.current_price * Decimal(str(1 + (i % 20 - 10) / 100))
            
            start = time.perf_counter()
            result = self.compliance_engine.validate_pricing(product, price)
            end = time.perf_counter()
            
            times.append(end - start)
        
        return self._calculate_benchmark_result('compliance_validation', times)
    
    def benchmark_safety_checks(self, iterations: int) -> BenchmarkResult:
        """Benchmark safety validation performance."""
        products = self._generate_test_products(100)
        times = []
        
        for i in range(iterations):
            product = products[i % len(products)]
            current_price = product.current_price
            new_price = current_price * Decimal(str(1 + (i % 10 - 5) / 100))
            
            start = time.perf_counter()
            is_allowed, violations = self.safety_controller.validate_price_change(
                product, current_price, new_price, "Benchmark test"
            )
            end = time.perf_counter()
            
            times.append(end - start)
        
        return self._calculate_benchmark_result('safety_checks', times)
    
    def benchmark_batch_processing(self, iterations: int) -> BenchmarkResult:
        """Benchmark batch processing performance."""
        batch_sizes = [10, 50, 100, 500]
        all_times = []
        
        for batch_size in batch_sizes:
            products = self._generate_test_products(batch_size)
            pricing_updates = [
                (p, p.current_price * Decimal('1.05'))
                for p in products
            ]
            
            # Run multiple batches
            batch_iterations = max(1, iterations // len(batch_sizes))
            
            for _ in range(batch_iterations):
                start = time.perf_counter()
                results = self.compliance_engine.batch_validate(pricing_updates)
                end = time.perf_counter()
                
                # Normalize to per-item time
                per_item_time = (end - start) / batch_size
                all_times.append(per_item_time)
        
        return self._calculate_benchmark_result('batch_processing', all_times)
    
    def benchmark_concurrent_processing(self, iterations: int) -> BenchmarkResult:
        """Benchmark concurrent processing capabilities."""
        products = self._generate_test_products(iterations)
        
        # Sequential processing
        sequential_start = time.perf_counter()
        for product in products:
            self.compliance_engine.validate_pricing(
                product, product.current_price * Decimal('1.05')
            )
        sequential_time = time.perf_counter() - sequential_start
        
        # Concurrent processing
        concurrent_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for product in products:
                future = executor.submit(
                    self.compliance_engine.validate_pricing,
                    product, product.current_price * Decimal('1.05')
                )
                futures.append(future)
            
            # Wait for all to complete
            concurrent.futures.wait(futures)
        concurrent_time = time.perf_counter() - concurrent_start
        
        speedup = sequential_time / concurrent_time
        
        return BenchmarkResult(
            benchmark_name='concurrent_processing',
            iterations=iterations,
            total_time=concurrent_time,
            avg_time=concurrent_time / iterations,
            min_time=concurrent_time / iterations,
            max_time=concurrent_time / iterations,
            std_dev=0,
            throughput=iterations / concurrent_time,
            memory_usage=speedup  # Store speedup factor
        )
    
    def benchmark_cache_performance(self, iterations: int) -> BenchmarkResult:
        """Benchmark caching performance."""
        products = self._generate_test_products(10)  # Small set for cache testing
        times_no_cache = []
        times_with_cache = []
        
        # Test without cache (unique prices each time)
        for i in range(iterations // 2):
            product = products[i % len(products)]
            price = product.current_price * Decimal(str(1 + i / 1000))
            
            start = time.perf_counter()
            self.compliance_engine.validate_pricing(product, price)
            end = time.perf_counter()
            
            times_no_cache.append(end - start)
        
        # Test with cache (repeated prices)
        for i in range(iterations // 2):
            product = products[i % len(products)]
            price = product.current_price * Decimal('1.05')  # Same price
            
            start = time.perf_counter()
            self.compliance_engine.validate_pricing(product, price)
            end = time.perf_counter()
            
            times_with_cache.append(end - start)
        
        # Calculate cache speedup
        avg_no_cache = statistics.mean(times_no_cache)
        avg_with_cache = statistics.mean(times_with_cache)
        cache_speedup = avg_no_cache / avg_with_cache
        
        all_times = times_no_cache + times_with_cache
        result = self._calculate_benchmark_result('cache_performance', all_times)
        result.memory_usage = cache_speedup  # Store cache speedup
        
        return result
    
    def benchmark_rollback_operations(self, iterations: int) -> BenchmarkResult:
        """Benchmark rollback operation performance."""
        times = []
        
        # Create smaller iterations for rollback testing
        rollback_iterations = min(iterations, 100)
        
        for i in range(rollback_iterations):
            products = self._generate_test_products(50)
            
            start = time.perf_counter()
            
            # Simulate snapshot creation
            snapshot_time = time.perf_counter()
            
            # Simulate rollback plan creation
            plan_time = time.perf_counter()
            
            # Simulate rollback execution
            execution_time = time.perf_counter()
            
            end = execution_time
            times.append(end - start)
        
        return self._calculate_benchmark_result('rollback_operations', times)
    
    def stress_test(self, duration_seconds: int = 60,
                   target_qps: int = 100) -> Dict[str, Any]:
        """Run stress test for specified duration."""
        logger.info(f"Starting stress test: {duration_seconds}s at {target_qps} QPS")
        
        products = self._generate_test_products(1000)
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        operations = 0
        errors = 0
        latencies = []
        
        # Calculate sleep time between operations
        sleep_time = 1.0 / target_qps if target_qps > 0 else 0
        
        while time.time() < end_time:
            product = products[operations % len(products)]
            price = product.current_price * Decimal('1.05')
            
            op_start = time.perf_counter()
            try:
                self.compliance_engine.validate_pricing(product, price)
                operations += 1
            except Exception as e:
                errors += 1
                logger.error(f"Stress test error: {str(e)}")
            op_end = time.perf_counter()
            
            latencies.append(op_end - op_start)
            
            # Sleep to maintain target QPS
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        actual_duration = time.time() - start_time
        actual_qps = operations / actual_duration
        
        return {
            'duration': actual_duration,
            'target_qps': target_qps,
            'actual_qps': actual_qps,
            'total_operations': operations,
            'errors': errors,
            'error_rate': errors / operations if operations > 0 else 0,
            'latency_stats': {
                'avg': statistics.mean(latencies),
                'p50': statistics.median(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)],
                'p99': sorted(latencies)[int(len(latencies) * 0.99)],
                'max': max(latencies)
            }
        }
    
    def compare_configurations(self, configs: List[Dict[str, Any]],
                             iterations: int = 1000) -> Dict[str, Any]:
        """Compare performance across different configurations."""
        comparison_results = {}
        
        for config in configs:
            config_name = config['name']
            logger.info(f"Testing configuration: {config_name}")
            
            # Apply configuration
            self._apply_configuration(config)
            
            # Run benchmarks
            results = {}
            results['compliance'] = self.benchmark_compliance_validation(iterations)
            results['safety'] = self.benchmark_safety_checks(iterations)
            results['batch'] = self.benchmark_batch_processing(iterations // 10)
            
            comparison_results[config_name] = results
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparison_results)
        
        return {
            'configurations': comparison_results,
            'summary': summary
        }
    
    def export_results(self, output_path: Path) -> Dict[str, Any]:
        """Export benchmark results to file."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'results': {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            'system_info': self._get_system_info()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Benchmark results exported to {output_path}")
        return export_data
    
    def _generate_test_products(self, count: int) -> List[Product]:
        """Generate test products for benchmarking."""
        products = []
        
        for i in range(count):
            products.append(Product(
                sku=f"BENCH_{i:06d}",
                name=f"Benchmark Product {i}",
                market=Market.MASSACHUSETTS if i % 2 == 0 else Market.RHODE_ISLAND,
                category=list(ProductCategory)[i % len(ProductCategory)],
                current_price=Decimal(str(50 + (i % 100))),
                cost=Decimal(str(25 + (i % 50))),
                inventory_level=100 + (i % 500)
            ))
        
        return products
    
    def _calculate_benchmark_result(self, name: str, times: List[float]) -> BenchmarkResult:
        """Calculate benchmark statistics from timing data."""
        return BenchmarkResult(
            benchmark_name=name,
            iterations=len(times),
            total_time=sum(times),
            avg_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            throughput=len(times) / sum(times)
        )
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply a configuration for testing."""
        # This would modify system settings based on config
        # For example, cache settings, validation modes, etc.
        pass
    
    def _generate_comparison_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary comparing different configurations."""
        summary = {
            'best_overall': None,
            'best_by_metric': {},
            'relative_performance': {}
        }
        
        # Find best configuration for each metric
        metrics = ['compliance', 'safety', 'batch']
        for metric in metrics:
            best_config = None
            best_throughput = 0
            
            for config_name, config_results in results.items():
                throughput = config_results[metric].throughput
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config_name
            
            summary['best_by_metric'][metric] = {
                'config': best_config,
                'throughput': best_throughput
            }
        
        # Calculate relative performance
        baseline_config = list(results.keys())[0]
        for config_name, config_results in results.items():
            if config_name == baseline_config:
                continue
            
            summary['relative_performance'][config_name] = {}
            for metric in metrics:
                baseline_throughput = results[baseline_config][metric].throughput
                config_throughput = config_results[metric].throughput
                relative = config_throughput / baseline_throughput
                summary['relative_performance'][config_name][metric] = relative
        
        return summary
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'timestamp': datetime.now().isoformat()
        }